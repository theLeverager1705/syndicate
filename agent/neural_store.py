"""Evorozen Neural DB backend for learned policy.

Design note that matters more than the code: only RULES go to the hosted
store. A rule records a field name, a decision, a confidence and the layout
anchors that located the field -- never the document, never the OCR text,
never the identifier itself. Images and extracted values stay on the machine.

A privacy tool that uploaded people's scorecards to a third party in order to
remember their preferences would have defeated its own purpose.

The free tier allows 50 calls, and `all_rules()` is consulted many times per
run, so reads are cached in memory and only writes go over the wire.
"""
from __future__ import annotations

import json
import os
from typing import Any

ENDPOINT = "https://pulse.evorozen.com/api/neural"
TABLE = "policy_rules"

# Columns are flat text; structured fields are JSON-encoded into *_json.
_JSON_FIELDS = ("context", "support", "contradictions", "anchors")


class NeuralPulseError(RuntimeError):
    pass


class NeuralPulseStore:
    """RuleStore backed by Evorozen Neural DB."""

    def __init__(self, api_key: str | None = None, table: str = TABLE) -> None:
        from .config import load_env

        load_env()
        self.api_key = api_key or os.environ.get("EVOROZEN_API_KEY", "").strip()
        if not self.api_key:
            raise NeuralPulseError("EVOROZEN_API_KEY is not set")
        self.table = table
        self._cache: list[dict[str, Any]] | None = None
        self.calls = 0

    # ---------- transport ----------

    def _post(self, action: str, prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        from .config import _ssl_context

        ctx = _ssl_context()
        body = {"action_type": action, "prompt": prompt, "data_payload": payload}
        with httpx.Client(verify=ctx, timeout=45.0) as client:
            resp = client.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json=body,
            )
        self.calls += 1
        try:
            data = resp.json()
        except ValueError:
            raise NeuralPulseError(f"non-JSON response ({resp.status_code})")
        if isinstance(data, dict) and data.get("error"):
            raise NeuralPulseError(f"{data['error']} (trace {data.get('trace_id')})")
        return data

    # ---------- schema ----------

    def ensure_schema(self) -> None:
        columns = [{"name": "rule_id", "type": "text"},
                   {"name": "field_name", "type": "text"},
                   {"name": "decision", "type": "text"},
                   {"name": "rationale", "type": "text"},
                   {"name": "created_at", "type": "text"},
                   {"name": "updated_at", "type": "text"}]
        columns += [{"name": f"{f}_json", "type": "text"} for f in _JSON_FIELDS]
        self._post("create_schema", "policy rule storage",
                   {"tables": [{"name": self.table, "columns": columns}]})

    # ---------- encoding ----------

    @staticmethod
    def _to_row(payload: dict[str, Any]) -> dict[str, Any]:
        row = {
            "rule_id": payload["id"],
            "field_name": payload["field_name"],
            "decision": payload["decision"],
            "rationale": payload.get("rationale", ""),
            "created_at": str(payload.get("created_at", "")),
            "updated_at": str(payload.get("updated_at", "")),
        }
        for f in _JSON_FIELDS:
            row[f"{f}_json"] = json.dumps(payload.get(f) or ([] if f != "context" else {}))
        return row

    @staticmethod
    def _from_row(row: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": row.get("rule_id"),
            "field_name": row.get("field_name"),
            "decision": row.get("decision"),
            "rationale": row.get("rationale", ""),
        }
        for key in ("created_at", "updated_at"):
            try:
                out[key] = float(row.get(key) or 0.0)
            except (TypeError, ValueError):
                out[key] = 0.0
        for f in _JSON_FIELDS:
            try:
                out[f] = json.loads(row.get(f"{f}_json") or "null")
            except (TypeError, ValueError):
                out[f] = None
            if out[f] is None:
                out[f] = {} if f == "context" else []
        return out

    # ---------- RuleStore ----------

    def load_all(self) -> list[dict[str, Any]]:
        """Cached. Call refresh() to force a re-read."""
        if self._cache is None:
            data = self._post("select_data", "read all policy rules",
                              {"table": self.table})
            rows = data.get("data") or []
            self._cache = [self._from_row(r) for r in rows if r.get("rule_id")]
        return list(self._cache)

    def refresh(self) -> None:
        self._cache = None

    def put(self, rule_id: str, payload: dict[str, Any]) -> None:
        row = self._to_row(payload)
        self._post("upsert_data", f"store rule {rule_id}",
                   {"table": self.table, "where": {"rule_id": rule_id},
                    "record": row})
        if self._cache is None:
            self._cache = []
        decoded = self._from_row(row)
        self._cache = [r for r in self._cache if r.get("id") != rule_id] + [decoded]

    def delete(self, rule_id: str) -> None:
        self._post("delete_data", f"delete rule {rule_id}",
                   {"table": self.table, "where": {"rule_id": rule_id}})
        if self._cache is not None:
            self._cache = [r for r in self._cache if r.get("id") != rule_id]
