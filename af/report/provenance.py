"""Which store versions a report was drawn from, checked before anything is built.

Every real figure's I7 names the store versions it was drawn from, in
``provenance.store_refs`` (a list of :class:`af.store.StoreRef` JSON objects). From those the
builder:

- refuses a report whose figures disagree about a dataset (two maps drawn from two versions of
  the same chain would put two different analyses side by side);
- checks every ref resolves in the store, with its manifest hash (``af.store.check_refs``);
- refuses a **stale** figure: one not pinned in the config whose refs are not the dataset's
  CURRENT version. However recently it was drawn, it shows old data (today's stale-figure
  failures, B-report-layer §4.1);
- writes the report manifest in the store's snapshot format (``af.store.write_manifest``), which
  is what "reproduce this report" starts from.

A figure with no store refs (a stand-in drawn from elsewhere) is allowed only in bring-up mode,
like a placeholder, and is counted on the cover.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from af.report.figures import Resolved
from af.store import Store, StoreError, StoreRef, check_refs


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


def check_against_store(
    use: StoreUse, figs: dict[str, Resolved], store: Store, *, deep: bool
) -> None:
    """Every ref resolves (and, with ``deep``, re-hashes); every unpinned figure is CURRENT."""
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
                    f"stale: {', '.join(stale)} drawn from {ref.kind}/{ref.dataset} "
                    f"{ref.version}, but CURRENT is {current.version} (redraw, or pin the slot)"
                )
    if problems:
        raise ProvenanceError(f"{len(problems)} store problem(s):\n  " + "\n  ".join(problems))
