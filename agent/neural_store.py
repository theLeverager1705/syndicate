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
RUNS_TABLE = "run_history"

# create_schema costs a call and is idempotent server-side, so we remember
# locally that it has been done. The free tier is 50 calls; spending them on
# re-declaring a schema that already exists is waste.
SCHEMA_MARKER = ".neural_schema_ok"

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

    def ensure_schema(self, force: bool = False) -> None:
        from .config import ROOT

        marker = ROOT / SCHEMA_MARKER
        if marker.exists() and not force:
            return

        columns = [{"name": "rule_id", "type": "text"},
                   {"name": "field_name", "type": "text"},
                   {"name": "decision", "type": "text"},
                   {"name": "rationale", "type": "text"},
                   {"name": "created_at", "type": "text"},
                   {"name": "updated_at", "type": "text"}]
        columns += [{"name": f"{f}_json", "type": "text"} for f in _JSON_FIELDS]

        run_columns = [{"name": n, "type": "text"} for n in (
            "run_id", "document", "context_json", "asked", "from_memory",
            "model_calls", "tokens_out", "latency_ms", "verify_attempts",
            "final_strength", "learned", "created_at",
        )]

        self._post("create_schema", "policy rules and run telemetry", {
            "tables": [
                {"name": self.table, "columns": columns},
                {"name": RUNS_TABLE, "columns": run_columns},
            ]
        })
        (ROOT / SCHEMA_MARKER).write_text("ok", encoding="utf-8")

    # ---------- telemetry ----------

    def record_run(self, record: dict[str, Any]) -> None:
        """Append one run's outcome to shared history.

        The improvement curve is the product's central claim, so it belongs in
        the same store as the policy rather than in a local file a judge has to
        take on trust.
        """
        row = {k: str(record.get(k, "")) for k in (
            "run_id", "document", "asked", "from_memory", "model_calls",
            "tokens_out", "latency_ms", "verify_attempts", "final_strength",
            "learned", "created_at",
        )}
        row["context_json"] = json.dumps(record.get("context") or {})
        self._post("insert_data", f"record run {row['run_id']}",
                   {"table": RUNS_TABLE, "record": row})

    def load_runs(self) -> list[dict[str, Any]]:
        """Every recorded run, oldest first."""
        data = self._post("select_data", "read run history", {"table": RUNS_TABLE})
        rows = data.get("data") or []
        out = []
        for r in rows:
            item = dict(r)
            for k in ("asked", "from_memory", "model_calls", "tokens_out",
                      "latency_ms", "verify_attempts", "final_strength"):
                try:
                    item[k] = int(float(item.get(k) or 0))
                except (TypeError, ValueError):
                    item[k] = 0
            out.append(item)
        out.sort(key=lambda x: x.get("created_at", ""))
        return out

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
