"""Which store versions a report was drawn from, checked before anything is built.

Every real figure's I7 names the store versions it was drawn from, in
``provenance.store_refs`` (a list of :class:`af.store.StoreRef` JSON objects). From those the
builder:

- refuses a report whose figures disagree about a dataset (two maps drawn from two versions of
  the same chain would put two different analyses side by side);
- checks every ref resolves in the store, with its manifest hash (``af.store.check_refs``);
- follows each version's provenance upstream (a chain to its tables, and so on) and refuses a
  **stale** figure: one not pinned in the config resting on any version that is not its
  dataset's CURRENT. However recently it was drawn, it shows old data (today's stale-figure
  failures, B-report-layer §4.1);
- writes the report manifest in the store's snapshot format (``af.store.write_manifest``), which
  is what "reproduce this report" starts from.

A figure with no store refs (a stand-in drawn from elsewhere) is allowed only in bring-up mode,
like a placeholder, and is counted on the cover.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from af.report.figures import Resolved
from af.store import Store, StoreError, StoreRef, check_refs
from af.store.manifest import PROVENANCE


class ProvenanceError(RuntimeError):
    """The figures' store provenance is inconsistent, unresolvable or stale."""


def figure_refs(r: Resolved) -> list[StoreRef]:
    """The store refs a figure's I7 names; malformed refs are an error, naming the figure."""
    raw = r.figure.doc["provenance"].get("store_refs", [])
    if not isinstance(raw, list):
        raise ProvenanceError(f"slot {r.slot}: provenance.store_refs is not a list")
    try:
        return [StoreRef.from_json(item) for item in raw]
    except (StoreError, KeyError, TypeError) as err:
        raise ProvenanceError(f"slot {r.slot}: bad store ref: {err}") from err


@dataclass
class StoreUse:
    """The refs a report uses, and which slots were not drawn from the store."""

    refs: list[StoreRef] = field(default_factory=list)
    used_by: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    not_from_store: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "refs": [ref.to_json() for ref in self.refs],
            "used_by": {f"{k}/{d}": slots for (k, d), slots in sorted(self.used_by.items())},
            "not_from_store": self.not_from_store,
        }


def collect(figs: dict[str, Resolved]) -> StoreUse:
    """Gather refs across figures; two versions of one dataset is an error listing the slots."""
    use = StoreUse()
    by_dataset: dict[tuple[str, str], dict[StoreRef, list[str]]] = {}
    for slot, r in figs.items():
        if r.figure.placeholder:
            continue
        refs = figure_refs(r)
        if not refs:
            use.not_from_store.append(slot)
        for ref in refs:
            by_dataset.setdefault((ref.kind, ref.dataset), {}).setdefault(ref, []).append(slot)
    conflicts = [
        f"{kind}/{dataset}: "
        + "; ".join(f"{ref.version} in {', '.join(slots)}" for ref, slots in versions.items())
        for (kind, dataset), versions in sorted(by_dataset.items())
        if len(versions) > 1
    ]
    if conflicts:
        raise ProvenanceError(
            "figures drawn from different versions of the same dataset:\n  "
            + "\n  ".join(conflicts)
        )
    for key, versions in sorted(by_dataset.items()):
        ((ref, slots),) = versions.items()
        use.refs.append(ref)
        use.used_by[key] = sorted(slots)
    return use


def upstream_refs(store: Store, ref: StoreRef) -> list[StoreRef]:
    """The store refs a version was built from, as its PROVENANCE.json records them."""
    record = json.loads((store.resolve(ref) / PROVENANCE).read_text())
    return [StoreRef.from_json(item["store"]) for item in record["inputs"] if "store" in item]


def expand_upstream(use: StoreUse, store: Store) -> None:
    """Add every version the figures' versions were built from, transitively (in place).

    A chain that is CURRENT but built from tables that are not shows old data just the same;
    the report manifest must also say which tables and sequences sit under each figure
    (STORE-LAYOUT: "which trees, chains and tables it was built from").
    """
    by_dataset: dict[tuple[str, str], dict[StoreRef, set[str]]] = {}
    queue = [(ref, slot) for ref in use.refs for slot in use.used_by[(ref.kind, ref.dataset)]]
    seen: set[tuple[StoreRef, str]] = set()
    while queue:
        ref, slot = queue.pop()
        if (ref, slot) in seen:
            continue
        seen.add((ref, slot))
        by_dataset.setdefault((ref.kind, ref.dataset), {}).setdefault(ref, set()).add(slot)
        queue.extend((up, slot) for up in upstream_refs(store, ref))
    conflicts = [
        f"{kind}/{dataset}: "
        + "; ".join(f"{ref.version} under {', '.join(sorted(s))}" for ref, s in versions.items())
        for (kind, dataset), versions in sorted(by_dataset.items())
        if len(versions) > 1
    ]
    if conflicts:
        raise ProvenanceError(
            "figures rest on different versions of the same dataset:\n  " + "\n  ".join(conflicts)
        )
    use.refs = []
    use.used_by = {}
    for key, versions in sorted(by_dataset.items()):
        ((ref, slots),) = versions.items()
        use.refs.append(ref)
        use.used_by[key] = sorted(slots)


def check_against_store(
    use: StoreUse, figs: dict[str, Resolved], store: Store, *, deep: bool
) -> None:
    """Every ref and everything under it resolves (``deep`` re-hashes) and is CURRENT.

    A ref used only by pinned slots may be old: pinning is how a report keeps an older figure
    on purpose.
    """
    problems = check_refs(store, use.refs, deep=deep)
    if problems:
        raise ProvenanceError(f"{len(problems)} store problem(s):\n  " + "\n  ".join(problems))
    expand_upstream(use, store)
    problems = check_refs(store, use.refs, deep=deep)
    pinned = {slot for slot, r in figs.items() if r.pinned}
    for ref in use.refs:
        try:
            current = store.current(ref.kind, ref.dataset)
        except StoreError as err:
            problems.append(str(err))
            continue
        if current != ref:
            stale = [s for s in use.used_by[(ref.kind, ref.dataset)] if s not in pinned]
            if stale:
                problems.append(
                    f"stale: {', '.join(stale)} rest on {ref.kind}/{ref.dataset} "
                    f"{ref.version}, but CURRENT is {current.version} (rebuild and redraw, "
                    "or pin the slot)"
                )
    if problems:
        raise ProvenanceError(f"{len(problems)} store problem(s):\n  " + "\n  ".join(problems))
