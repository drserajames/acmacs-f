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
- compares a tree with the sequences the figures read by the **export it was built from**. A
  tree cites the sequences version it was exported from and its export files' sha256. When that
  version is not the one the figures read, the read version's export record decides:
  ``<export records>/<dataset>/<alignment sha256[:16]>/<version>.export.json``, written by
  af.tree (``export`` is its export.json verbatim). Both file hashes the same is
  ``trees_same_content``, and the older label is not followed; a record of that version under
  another alignment is behind in content; no record, or a malformed one, is refused. The check
  never computes an export, and runs for slots that are not pinned (a pinned slot keeps its
  older inputs on purpose);
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


# A figure input that is only compared with, not drawn: af.map.build records each new-antigen
# outline tier's previous map as "previous_map <dataset>@<version>" (Q264 b). The figure's colours
# and points come from its own chain; the previous map only decides which antigens are new. So
# it is not an input the report rests on: it is recorded as a comparison, with the slots that
# cite it, and what it was built from (older tables, as the previous report's maps were) is not
# held to the report's one version of each dataset.
COMPARISON_INPUT = "previous_map "


REF_KEYS = {"kind", "dataset", "version", "manifest_sha256"}


def comparison_refs(r: Resolved) -> list[StoreRef]:
    """The store refs a figure's I7 names only as comparisons (:data:`COMPARISON_INPUT`).

    A ref another input also names is an input, not only a comparison, and is left out: an
    outline tier's previous map may be the very chain the figure draws (a quiet month, the
    previous interim's version still CURRENT; 08, 24 Oct 2026).
    """
    inputs = r.figure.doc["provenance"].get("inputs") or {}
    try:
        compared = [
            StoreRef.from_json({k: v for k, v in item.items() if k != "sha256"})
            for name, item in inputs.items()
            if name.startswith(COMPARISON_INPUT)
        ]
    except (StoreError, AttributeError, TypeError) as err:
        raise ProvenanceError(f"slot {r.slot}: bad comparison input: {err}") from err
    drawn = {
        StoreRef.from_json({k: item[k] for k in REF_KEYS})
        for name, item in inputs.items()
        if not name.startswith(COMPARISON_INPUT)
        and isinstance(item, dict)
        and set(item) >= REF_KEYS
    }
    return [ref for ref in compared if ref not in drawn]


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
    # tree dataset -> (sequences version it was exported from, sequences version the figures
    # read, the export record that shows both exports identical, how that record was written)
    trees_same_content: dict[str, tuple[str, str, str, str]] = field(default_factory=dict)
    # Versions a figure only compares with (:func:`comparison_refs`) -> the slots citing them:
    # not inputs, so neither in refs nor followed upstream, and need not be CURRENT.
    comparison_refs: dict[StoreRef, list[str]] = field(default_factory=dict)

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
            "trees_same_content": {
                d: list(triple) for d, triple in sorted(self.trees_same_content.items())
            },
            "comparison_refs": {
                str(ref): slots
                for ref, slots in sorted(self.comparison_refs.items(), key=lambda kv: str(kv[0]))
            },
        }


Versions = dict[tuple[str, str], dict[StoreRef, set[str]]]  # (kind, dataset) -> version -> slots
Chosen = dict[tuple[str, str], tuple[StoreRef, set[str]]]


def write_report_manifest(path: Path, use: StoreUse, description: str) -> Path:
    """The report manifest: the store's snapshot document of the report's own versions, plus
    ``pinned``: each older version reached only through pinned slots, with those slots, and
    ``compared``: each version a figure only compares with, with the slots that cite it.

    A snapshot names one version per dataset (af.store refuses repeats), so the pinned older
    versions sit beside ``refs``, not in it; together they are every version the report rests
    on. The compared versions are not inputs, so they are listed apart, where nobody takes them
    for one. With no pins and no comparisons the document is exactly the snapshot format.
    """
    document = refs_document(use.refs, description)
    if use.pinned_versions:
        document["pinned"] = [
            {"store": ref.to_json(), "slots": slots}
            for ref, slots in sorted(use.pinned_versions.items(), key=lambda kv: str(kv[0]))
        ]
    if use.comparison_refs:
        document["compared"] = [
            {"store": ref.to_json(), "slots": slots}
            for ref, slots in sorted(use.comparison_refs.items(), key=lambda kv: str(kv[0]))
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
        compared = comparison_refs(r)
        for ref in compared:
            use.comparison_refs.setdefault(ref, []).append(slot)
        refs = [ref for ref in figure_refs(r) if ref not in compared]
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


EXPORT_FILES = ("alignment.fasta", "leaves.parquet")


def tree_export(store: Store, tree: StoreRef) -> dict[str, str]:
    """The sha256 of each export file a tree version cites as an external input, by file name."""
    record = json.loads((store.resolve(tree) / PROVENANCE).read_text())
    files = {
        Path(item["external"]["path"]).name: item["external"]["sha256"]
        for item in record["inputs"]
        if "external" in item and Path(item["external"].get("path", "")).name in EXPORT_FILES
    }
    if sorted(files) != sorted(EXPORT_FILES):
        raise ProvenanceError(
            f"{tree}: its provenance cites export files {sorted(files)}, not {list(EXPORT_FILES)}; "
            "the export it was built from cannot be compared"
        )
    return files


EXPORT_HOW = ("at export", "back-filled")


def _export_record(path: Path, wanted: StoreRef) -> dict[str, Any]:
    """An export record, verified before it is trusted: it parses, its embedded export.json
    names the version in its file name, both file hashes are there, and it says how it was
    written. Anything else is refused, never read as a pass."""
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as err:
        raise ProvenanceError(f"{path}: unreadable export record ({err})") from err
    if not isinstance(record, dict):
        raise ProvenanceError(f"{path}: malformed export record: not a JSON object")
    named = (record.get("export") or {}).get("sequences") or {}
    files = record.get("files")
    problems = []
    if named.get("dataset") != wanted.dataset or named.get("version") != wanted.version:
        problems.append(f"export.sequences is {named or 'missing'}, not {wanted}")
    hashes = files.values() if isinstance(files, dict) else []
    if (
        not isinstance(files, dict)
        or sorted(files) != sorted(EXPORT_FILES)
        or not all(isinstance(v, str) and len(v) == 64 for v in hashes)
    ):
        problems.append(f"files {files!r} are not the sha256 of {list(EXPORT_FILES)}")
    if record.get("how") not in EXPORT_HOW:
        problems.append(f"how is {record.get('how')!r}, not one of {EXPORT_HOW}")
    if problems:
        raise ProvenanceError(f"{path}: malformed export record: " + "; ".join(problems))
    return record


def tree_label_only(
    store: Store, tree: StoreRef, read: dict[str, StoreRef], use: StoreUse,
    export_records: Path | None,
) -> StoreRef | None:  # fmt: skip
    """The sequences version ``tree`` cites if it differs from the figures' only in label.

    Same content means the read version's export record, filed under the tree's alignment hash,
    lists exactly the tree's export files (sha256). Recorded in ``use.trees_same_content`` and
    returned, so the walk does not follow the older label. A record under another alignment,
    no record, or a malformed one is refused. Same version, or no sequences of that dataset
    read: None.
    """
    cited = [up for up in upstream_refs(store, tree) if up.kind == "sequences"]
    differs = [up for up in cited if up.dataset in read and up != read[up.dataset]]
    if not differs:
        return None
    if len(cited) != 1:
        raise ProvenanceError(f"{tree}: cites {len(cited)} sequences versions; one is needed")
    exported, wanted = differs[0], read[differs[0].dataset]
    where = f"sequences/{wanted.dataset} {wanted.version}"
    if export_records is None:
        raise ProvenanceError(
            f"{tree} was exported from sequences/{exported.dataset} {exported.version}, the "
            f"figures read {wanted.version}: an export record is needed to compare them "
            "(--export-records)"
        )
    files = tree_export(store, tree)
    root = export_records / wanted.dataset
    path = root / files["alignment.fasta"][:16] / f"{wanted.version}.export.json"
    if not path.is_file():
        other = sorted(p.parent.name for p in root.glob(f"*/{wanted.version}.export.json"))
        if other:
            raise ProvenanceError(
                f"{tree} was exported from sequences/{exported.dataset} {exported.version} "
                f"(alignment.fasta {files['alignment.fasta'][:16]}); {where} exports to "
                f"{', '.join(other)} ({root}): the tree is behind in content (rebuild the tree)"
            )
        raise ProvenanceError(
            f"{tree} was exported from sequences/{exported.dataset} {exported.version}, the "
            f"figures read {wanted.version}, and {where} has no export record ({path}): "
            "write it, or rebuild the tree"
        )
    record = _export_record(path, wanted)
    if record["files"] != files:
        raise ProvenanceError(f"{path}: its files {record['files']} are not {tree}'s {files}")
    use.trees_same_content[tree.dataset] = (
        exported.version, wanted.version, str(path), record["how"]
    )  # fmt: skip
    return exported


def expand_upstream(
    use: StoreUse, store: Store, pinned: set[str] | None = None,
    export_records: Path | None = None,
) -> None:  # fmt: skip
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
    exports: dict[StoreRef, StoreRef | None] = {}
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
        if ref.kind == "trees" and slot not in (pinned or set()):
            if ref not in exports:
                exports[ref] = tree_label_only(store, ref, read, use, export_records)
            label_only = exports[ref]
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
    use: StoreUse, figs: dict[str, Resolved], store: Store, *, deep: bool,
    export_records: Path | None = None,
) -> None:  # fmt: skip
    """Every ref and everything under it resolves (``deep`` re-hashes) and is CURRENT.

    A ref used only by pinned slots may be old: pinning is how a report keeps an older figure
    on purpose.
    """
    problems = check_refs(store, [*use.all_refs(), *use.comparison_refs], deep=deep)
    if problems:
        raise ProvenanceError(f"{len(problems)} store problem(s):\n  " + "\n  ".join(problems))
    pinned = {slot for slot, r in figs.items() if r.pinned}
    expand_upstream(use, store, pinned, export_records)
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
