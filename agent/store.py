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

    if not prefer_local:
        try:
            from .neural_store import NeuralPulseStore

            store = NeuralPulseStore()
            store.ensure_schema()
            return store
        except Exception:
            # Falling back is deliberate and silent-ish: the caller reports it.
            pass

    root = Path(__file__).resolve().parent.parent
    return LocalJSONStore(root / "memory")
