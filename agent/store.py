"""Where learned rules live.

The memory layer only ever needs three operations: read every rule, write one,
delete one. Keeping that surface explicit means the backing store is a
substitution rather than a refactor -- a local directory today, a hosted
virtual-database service tomorrow, with `PolicyMemory` unchanged either way.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RuleStore(Protocol):
    """The complete persistence contract for learned policy."""

    def load_all(self) -> list[dict[str, Any]]:
        """Every stored rule, as raw dicts."""
        ...

    def put(self, rule_id: str, payload: dict[str, Any]) -> None:
        """Insert or replace one rule."""
        ...

    def delete(self, rule_id: str) -> None:
        """Remove one rule. Silent if it does not exist."""
        ...


class LocalJSONStore:
    """One JSON file per rule, in a directory.

    Chosen deliberately over a vector database: with rules numbering in the
    tens, embeddings would add infrastructure and remove the ability to explain
    why a rule fired. Each rule is also a file you can open and read, which
    matters for a system whose whole claim is that its decisions are auditable.
    """

    def __init__(self, directory: Path | str) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in sorted(self.dir.glob("rule_*.json")):
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                # A corrupt rule file must not take down the whole memory.
                continue
        return out

    def put(self, rule_id: str, payload: dict[str, Any]) -> None:
        (self.dir / f"{rule_id}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def delete(self, rule_id: str) -> None:
        (self.dir / f"{rule_id}.json").unlink(missing_ok=True)


def default_store(prefer_local: bool = False) -> RuleStore:
    """The backing store this installation should use.

    Evorozen Neural DB is the primary backend: learned policy is shared state
    that should outlive one machine, and keeping it in a hosted store is what
    lets the same person's preferences follow them across devices.

    Local JSON remains the fallback so the pipeline still runs with no
    credentials and no network -- which matters, because the parts that touch
    the actual document are local by design and should not stop working just
    because a remote service is unreachable.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    local = LocalJSONStore(root / "memory")

    if prefer_local:
        return local

    try:
        from .neural_store import NeuralPulseStore

        primary = NeuralPulseStore()
        primary.ensure_schema()
        return ResilientStore(primary, local)
    except Exception:
        # No credentials, or the service is unreachable at startup.
        return local


class ResilientStore:
    """Primary store with a local mirror that takes over when it fails.

    A hosted dependency will be unavailable sometimes -- quota exhausted, DNS
    down, a bad deploy at the provider. None of that is a reason for a user's
    redaction to fail, so:

      * every write goes to the local mirror as well as the primary, which
        keeps the fallback warm rather than empty at the moment it is needed;
      * the first primary failure flips the store to degraded and everything
        continues locally;
      * `degraded` and `reason` are exposed so the UI can say what happened
        instead of pretending nothing did.
    """

    def __init__(self, primary: RuleStore, fallback: RuleStore) -> None:
        self.primary = primary
        self.fallback = fallback
        self.degraded = False
        self.reason = ""

    def _degrade(self, exc: Exception) -> None:
        if not self.degraded:
            self.degraded = True
            self.reason = str(exc)

    def load_all(self) -> list[dict[str, Any]]:
        if not self.degraded:
            try:
                return self.primary.load_all()
            except Exception as exc:
                self._degrade(exc)
        return self.fallback.load_all()

    def put(self, rule_id: str, payload: dict[str, Any]) -> None:
        self.fallback.put(rule_id, payload)          # mirror first, always
        if not self.degraded:
            try:
                self.primary.put(rule_id, payload)
            except Exception as exc:
                self._degrade(exc)

    def delete(self, rule_id: str) -> None:
        self.fallback.delete(rule_id)
        if not self.degraded:
            try:
                self.primary.delete(rule_id)
            except Exception as exc:
                self._degrade(exc)

    # Telemetry is best-effort and primary-only; losing it must never surface
    # to the user as a failed redaction.
    def record_run(self, record: dict[str, Any]) -> None:
        if self.degraded:
            return
        recorder = getattr(self.primary, "record_run", None)
        if recorder is None:
            return
        try:
            recorder(record)
        except Exception as exc:
            self._degrade(exc)

    def load_runs(self) -> list[dict[str, Any]]:
        loader = getattr(self.primary, "load_runs", None)
        if self.degraded or loader is None:
            return []
        try:
            return loader()
        except Exception as exc:
            self._degrade(exc)
            return []
