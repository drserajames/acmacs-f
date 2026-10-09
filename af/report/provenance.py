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
- compares a clades table with the sequences a figure read by **content**, not version label
  (``af.clades.coverage``): labelled from another sequences version whose clade-call inputs are
  identical, it is recorded as ``clades_same_content`` and the older label is not followed;
  behind in content, or not labelled from exactly one version, it is refused;
- writes the report manifest in the store's snapshot format (``af.store.write_manifest``), which
  is what "reproduce this report" starts from.

A figure with no store refs (a stand-in drawn from elsewhere) is allowed only in bring-up mode,
like a placeholder, and is counted on the cover.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.clades.coverage import clades_behind_in_content
from af.clades.store import CladeStoreError
from af.report.figures import Resolved
from af.store import Store, StoreError, StoreRef, check_refs
from af.store.manifest import PROVENANCE
from af.store.snapshot import refs_document


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
    # clade dataset -> (sequences version it was labelled from, sequences version the figures
    # read), for clade tables labelled from another version whose clade calls' inputs are
    # identical: not behind, the versions only differ (af.serology.joins uses the same name)
    clades_same_content: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Older versions of a dataset reached only through PINNED slots (the report's own version of
    # it is in refs): kept on purpose, so recorded and listed in the manifest to reproduce it.
    pinned_versions: dict[StoreRef, list[str]] = field(default_factory=dict)
    # Datasets every one of whose versions is under pinned slots: the report's own is CURRENT.
    pinned_resolved_by_current: list[str] = field(default_factory=list)

    def all_refs(self) -> list[StoreRef]:
        """Every version the report rests on: its own, then the pinned-only older ones."""
        return [*self.refs, *sorted(self.pinned_versions, key=str)]

    def to_json(self) -> dict[str, Any]:
        return {
            "refs": [ref.to_json() for ref in self.refs],
            "pinned_resolved_by_current": self.pinned_resolved_by_current,
            "pinned_versions": {
                f"{ref.kind}/{ref.dataset}@{ref.version}": slots
                for ref, slots in sorted(self.pinned_versions.items(), key=lambda kv: str(kv[0]))
            },
            "used_by": {f"{k}/{d}": slots for (k, d), slots in sorted(self.used_by.items())},
            "not_from_store": self.not_from_store,
            "clades_same_content": {
                d: list(pair) for d, pair in sorted(self.clades_same_content.items())
            },
        }


Versions = dict[tuple[str, str], dict[StoreRef, set[str]]]  # (kind, dataset) -> version -> slots
Chosen = dict[tuple[str, str], tuple[StoreRef, set[str]]]


def write_report_manifest(path: Path, use: StoreUse, description: str) -> Path:
    """The report manifest: the store's snapshot document of the report's own versions, plus
    ``pinned``: each older version reached only through pinned slots, with those slots.

    A snapshot names one version per dataset (af.store refuses repeats), so the pinned older
    versions sit beside ``refs``, not in it; together they are every version the report rests
    on. With no pins the document is exactly the snapshot format.
    """
    document = refs_document(use.refs, description)
    if use.pinned_versions:
        document["pinned"] = [
            {"store": ref.to_json(), "slots": slots}
            for ref, slots in sorted(use.pinned_versions.items(), key=lambda kv: str(kv[0]))
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def one_version_each(
    by_dataset: Versions,
    pinned: set[str],
    verb: str,
    current: Callable[[str, str], StoreRef] | None = None,
    resolved_by_current: list[str] | None = None,
) -> tuple[Chosen, dict[StoreRef, list[str]]]:
    """The report's one version of each dataset, and the older ones only pinned slots reach.

    A report rests on one version of each dataset (two maps on two versions of a chain would
    put two analyses side by side). The exception is a pin: a slot pinned to keep an older
    figure may rest on an older version. So a dataset may appear in more than one version only
    if every version but one is reached solely through pinned slots; two versions under
    unpinned slots are refused, naming every version and its slots.

    When EVERY version is under pinned slots, none is the report's by use, and version ids are
    hashes whose order means nothing: the report's own is the dataset's CURRENT if it is among
    them (named in ``resolved_by_current``), otherwise the report is refused until one slot is
    unpinned. Without ``current`` (before the store is read) the choice is provisional, made
    again when the store is (:func:`expand_upstream` re-reads every version).
    """
    chosen: Chosen = {}
    extra: dict[StoreRef, list[str]] = {}
    conflicts = []
    for (kind, dataset), versions in sorted(by_dataset.items()):
        unpinned = [ref for ref, slots in versions.items() if slots - pinned]
        if len(unpinned) > 1:
            conflicts.append(
                f"{kind}/{dataset}: "
                + "; ".join(
                    f"{ref.version} {verb} {', '.join(sorted(s))}" for ref, s in versions.items()
                )  # fmt: skip
            )
            continue
        if len(versions) == 1:
            ((main, _),) = versions.items()  # one version: pinned or not, nothing to choose
        elif unpinned:
            main = unpinned[0]
        elif current is None:
            main = next(iter(versions))  # provisional: decided against the store later
        else:
            now = current(kind, dataset)
            if now not in versions:
                conflicts.append(
                    f"{kind}/{dataset}: every version is under pinned slots and none is CURRENT "
                    f"({now.version}): unpin the slots that should carry the report's own; "
                    + "; ".join(
                        f"{ref.version} {verb} {', '.join(sorted(s))}"
                        for ref, s in versions.items()
                    )  # fmt: skip
                )
                continue
            main = now
            if resolved_by_current is not None:
                resolved_by_current.append(f"{kind}/{dataset}")
        chosen[(kind, dataset)] = (main, versions[main])
        extra.update({ref: sorted(s) for ref, s in versions.items() if ref != main})
    if conflicts:
        raise ProvenanceError(
            f"figures {'drawn from' if verb == 'in' else 'rest on'} different versions of the "
            "same dataset (pin the slots that should keep an older version):\n  "
            + "\n  ".join(conflicts)
        )
    return chosen, extra


def collect(figs: dict[str, Resolved]) -> StoreUse:
    """Gather refs across figures, one version of each dataset unless pinned slots keep another."""
    use = StoreUse()
    by_dataset: dict[tuple[str, str], dict[StoreRef, set[str]]] = {}
    for slot, r in figs.items():
        if r.figure.placeholder:
            continue
        refs = figure_refs(r)
        if not refs:
            use.not_from_store.append(slot)
        for ref in refs:
            by_dataset.setdefault((ref.kind, ref.dataset), {}).setdefault(ref, set()).add(slot)
    pinned = {slot for slot, r in figs.items() if r.pinned}
    chosen, use.pinned_versions = one_version_each(by_dataset, pinned, "in")
    for key, (ref, slots) in chosen.items():
        use.refs.append(ref)
        use.used_by[key] = sorted(slots)
    return use


def upstream_refs(store: Store, ref: StoreRef) -> list[StoreRef]:
    """The store refs a version was built from, as its PROVENANCE.json records them."""
    record = json.loads((store.resolve(ref) / PROVENANCE).read_text())
    return [StoreRef.from_json(item["store"]) for item in record["inputs"] if "store" in item]


def clades_label_only(
    store: Store, clades: StoreRef, read: dict[str, StoreRef], use: StoreUse
) -> StoreRef | None:
    """The sequences version ``clades`` cites if it differs from the figures' only in label.

    Checked when the figures read a version of the sequences dataset the table is named for,
    by content (:func:`af.clades.coverage.clades_behind_in_content`). Same content: recorded in
    ``use.clades_same_content`` and returned, so the walk does not follow the older label.
    Behind in content, or a table whose provenance names no sequences version or several, is
    refused. Same version, or no sequences read: None.
    """
    sequences = read.get(clades.dataset)
    if sequences is None:
        return None
    try:
        result = clades_behind_in_content(store, clades, sequences)
    except CladeStoreError as err:
        raise ProvenanceError(str(err)) from err
    if result.behind:
        raise ProvenanceError(
            f"clades/{clades.dataset} {clades.version} was labelled from "
            f"sequences/{clades.dataset} {result.labelled.version} and is behind "
            f"{sequences.version}, which the figures read, in content: its clade calls would "
            "change (refresh the clades table and redraw)"
        )
    if result.same_version:
        return None
    use.clades_same_content[clades.dataset] = (result.labelled.version, sequences.version)
    return result.labelled


def expand_upstream(use: StoreUse, store: Store, pinned: set[str] | None = None) -> None:
    """Add every version the figures' versions were built from, transitively (in place).

    A chain that is CURRENT but built from tables that are not shows old data just the same;
    the report manifest must also say which tables and sequences sit under each figure
    (STORE-LAYOUT: "which trees, chains and tables it was built from").
    """
    by_dataset: dict[tuple[str, str], dict[StoreRef, set[str]]] = {}
    queue = [(ref, slot) for ref in use.refs for slot in use.used_by[(ref.kind, ref.dataset)]]
    queue += [(ref, slot) for ref, slots in use.pinned_versions.items() for slot in slots]
    read = {ref.dataset: ref for ref in use.refs if ref.kind == "sequences"}
    labels: dict[StoreRef, StoreRef | None] = {}
    seen: set[tuple[StoreRef, str]] = set()
    while queue:
        ref, slot = queue.pop()
        if (ref, slot) in seen:
            continue
        seen.add((ref, slot))
        by_dataset.setdefault((ref.kind, ref.dataset), {}).setdefault(ref, set()).add(slot)
        if ref.kind == "clades" and ref not in labels:
            labels[ref] = clades_label_only(store, ref, read, use)
        label_only = labels.get(ref)
        queue.extend((up, slot) for up in upstream_refs(store, ref) if up != label_only)
    use.pinned_resolved_by_current = []
    chosen, use.pinned_versions = one_version_each(
        by_dataset, pinned or set(), "under", store.current, use.pinned_resolved_by_current
    )
    use.refs = []
    use.used_by = {}
    for key, (ref, slots) in chosen.items():
        use.refs.append(ref)
        use.used_by[key] = sorted(slots)


def check_against_store(
    use: StoreUse, figs: dict[str, Resolved], store: Store, *, deep: bool
) -> None:
    """Every ref and everything under it resolves (``deep`` re-hashes) and is CURRENT.

    A ref used only by pinned slots may be old: pinning is how a report keeps an older figure
    on purpose.
    """
    problems = check_refs(store, use.all_refs(), deep=deep)
    if problems:
        raise ProvenanceError(f"{len(problems)} store problem(s):\n  " + "\n  ".join(problems))
    pinned = {slot for slot, r in figs.items() if r.pinned}
    expand_upstream(use, store, pinned)
    problems = check_refs(store, use.all_refs(), deep=deep)
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
