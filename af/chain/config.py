"""Chain configuration: which tables, in what order, and how each step is mapped.

A chain is data, not code (today's chains are Python files that glob a directory). The
config is TOML read through `af.util.config`; the table list is resolved once into
(table id, path, date, suffix) and the pipeline hashes each table when it runs, which is
what makes restart-from-change work.

Example::

    name = "labx-h9-hi"
    seed = 1

    [tables]                      # read from the run config's [paths] store
    dataset = "labx/h9-hi-turkey-labx"
    date_from = "2020-04-21"
    exclude = []

    [options]
    scratch_starts = 1000
    incremental_starts = 1000

    [[select.remove]]             # optional, repeatable: points left out (af.chain.select)
    what = "antigens"
    passage = "egg"
    reason = "egg-free map"
"""

from __future__ import annotations

import datetime
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from af.chain.adjust import ColumnBasisAdjustment
from af.chain.append import ORDERS, Order, Rerun, SettingsRerun
from af.chain.backend import DEFAULT_MOVE_LOWER_RULE, MOVE_LOWER_RULES
from af.chain.select import DropCell, RemoveRule, Selection, SeraPolicy
from af.chain.sequence_select import SequenceRemovals
from af.chart.merge import ColumnBasisConvention
from af.chart.sera import read_markers
from af.util.artefacts import sha256_path
from af.util.config import load_config

CHOSEN_PROVENANCE = "chosen.ace.provenance.json"  # a step's build record (af.chain.engine CHOSEN)


class ChainConfigError(ValueError):
    pass


@dataclass(frozen=True)
class MapOptions:
    """How every step's maps are made. A pipeline parameter: changing it remaps every step."""

    dimensions: int = 2
    minimum_column_basis: str = "none"
    scratch_starts: int = 1000
    incremental_starts: int = 1000
    projections_to_keep: int = 10  # distinct basins kept per map (Procrustes RMSD > 0.5)
    disconnect_threshold: int | None = 3  # fewer REGULAR titres than this -> disconnected
    # A table whose reference block repeats an earlier table's titres exactly merges only its
    # test antigens (ae's "cheating assay"; renamed by Sarah, 6 Oct 2026).
    combine_copied_references: bool = True
    column_bases: str = ColumnBasisConvention.ADJUST_TO_NEXT.value
    sd_limit: float = 1.0
    grid_test: bool = True
    code_version: int = 1  # bump to force every step to be remade after a code change
    # Group-aware trap pass after the trapped-point loop (af.map.optimise.resolve_trapped;
    # Sarah, Q61, 26 Sep 2026). Moves points stuck together in a worse place; changes maps.
    move_groups: bool = False
    # One step instead of a chain: merge every selected table in order (simple merge, no layout
    # carried), then map the result from scratch. The round's own download_full does this (ae:
    # merge_type "simple", 10,000 optimisations); Sarah, 1 Oct 2026: all 18 maps from scratch
    # until chains' composition and selection issues are settled. Set scratch_starts with it.
    merge_all: bool = False
    # How a map from scratch is relaxed. "rough": every start rough, then the best 5 fine, as AD's
    # chart-relax-grid (and af's incremental maps) do; 2.8x cheaper per start on a 7,094-point H1
    # merge with the same best (06-optimiser). "fine": every start fine, af's method before
    # 2 Oct 2026. Sarah, 2 Oct 2026: "Yes, match AD".
    scratch_precision: str = "rough"
    # How many of a map's best refined layouts the trapped-point loop resolves; the lowest is
    # kept. 1 (the default) resolves only the best, as every chain before 12 Oct 2026. A layout
    # that refines worse can resolve far lower (06-optimiser, gp-cnic: the best refined layout
    # resolved to 26,882, the next four to ~26,236). A search setting: the problem is unchanged.
    # At most 5, the layouts that are refined fine. Opt-in; using it is Sarah's decision (Q155).
    resolve_best: int = 1
    # The trapped-point loop's move-lower rule (af.map.optimise.MOVE_LOWER_RULE): after the loop,
    # move every point whose other position is lower, jointly, while the stress falls. The loop
    # alone leaves hemisphering points that each gain little but together gain more (06, Q195,
    # 14 Oct 2026). A search setting: the problem is unchanged. Opt-in; using it is Q195, Sarah's.
    move_lower: bool = False
    # Which move-lower rule, by the name it records (af.chain.backend.MOVE_LOWER_RULES):
    # "move-lower-1" (rounds until nothing is lower, at most 20) or "ad-last-round-1", AD's one
    # joint round after every map's trapped-point loop (Sarah, Q198 a, 16 Oct 2026: the chains
    # move to it once reports A and B are final). Only with move_lower. A search setting. With
    # resolve_best > 1 it runs on each of the best maps and the lowest is kept; AD ran it on
    # projection 0 only, which is resolve_best = 1, the default.
    move_lower_rule: str = DEFAULT_MOVE_LOWER_RULE

    def __post_init__(self) -> None:
        ColumnBasisConvention(self.column_bases)  # raises on an unknown convention
        if self.scratch_precision not in ("rough", "fine"):
            raise ChainConfigError(
                f"scratch_precision must be rough or fine: {self.scratch_precision!r}"
            )
        if not 1 <= self.resolve_best <= 5:
            raise ChainConfigError(f"resolve_best must be 1 to 5: {self.resolve_best!r}")
        if self.move_lower_rule not in MOVE_LOWER_RULES:
            raise ChainConfigError(
                f"move_lower_rule must be one of {sorted(MOVE_LOWER_RULES)}:"
                f" {self.move_lower_rule!r}"
            )
        if self.move_lower_rule != DEFAULT_MOVE_LOWER_RULE and not self.move_lower:
            raise ChainConfigError(
                f"move_lower_rule = {self.move_lower_rule!r} needs move_lower = true: alone it"
                " would change nothing"
            )


# Options added after chains were first published, with the value that reproduces them. They
# enter a step's parameters (and chain.json) only when set otherwise, so adding one neither
# reruns nor republishes existing chains.
LATER_OPTIONS = {
    "move_groups": False,
    "merge_all": False,
    "scratch_precision": "fine",
    "resolve_best": 1,
    "move_lower": False,
    "move_lower_rule": DEFAULT_MOVE_LOWER_RULE,
}


# Options renamed after chains were published: they enter a step's parameters (and chain.json)
# under the name they were published with, in the same place, so a rename reruns nothing.
PUBLISHED_NAMES = {"combine_copied_references": "combine_cheating_assays"}


def option_parameters(options: MapOptions) -> dict[str, Any]:
    params = asdict(options)
    for name, old in LATER_OPTIONS.items():
        if params[name] == old:
            del params[name]
    return {PUBLISHED_NAMES.get(k, k): v for k, v in params.items()}


@dataclass(frozen=True)
class TableSelection:
    """Either a tables-store dataset (`dataset`, the normal case) or a directory of `.ace`
    tables (`directory` + `group`, for comparisons with ae-era tables).

    The tables store is not named here: where it lives is a fact about the machine, so it
    comes from the run config (`[paths] store`) and one chain file runs anywhere. `store` is
    still parsed only so that a chain file that sets it is refused with a clear message.
    """

    store: Path | None = None  # refused: see load_chain_config
    dataset: str | None = None  # e.g. "labx/h3-hi-turkey-labx"
    directory: Path | None = None
    group: str | None = None
    date_from: str | None = None  # ISO date, inclusive
    date_to: str | None = None
    exclude: list[str] = field(default_factory=list)  # table ids; each must match a table
    # "date" (every table in test-date order) or "append-to-published": the published chain's
    # order kept and new tables appended, late ones recorded (af.chain.append; Q164, Q224)
    order: str = "date"
    rerun_from: list[Rerun] = field(default_factory=list)  # revised or dropped tables, decided
    rerun_settings: list[SettingsRerun] = field(default_factory=list)  # changed parameters, decided

    def __post_init__(self) -> None:
        if self.order not in ORDERS:
            raise ChainConfigError(f"[tables] order must be one of {ORDERS}: {self.order!r}")
        if (self.rerun_from or self.rerun_settings) and self.order != "append-to-published":
            raise ChainConfigError(
                '[tables] rerun_from and rerun_settings go with order = "append-to-published"'
            )


@dataclass(frozen=True)
class Reference:
    """A reference map of the same data to measure the finished map against (af.chain.reference):
    `chart` is relative to the run config's `references` directory (where reference maps live
    is a fact about the machine). Measurement only: nothing from it enters the map. A chain may
    name several (`[[reference]]`), e.g. a round's unadjusted output and its adjusted map: Sarah,
    2 Oct 2026, "Can not expect af to find the result of an adjustment"."""

    label: str  # how the review and the provenance name it, e.g. "downloaded.ace (the ae round)"
    chart: str
    starts: int = 50  # seeded relaxations for the basin measurement


@dataclass(frozen=True)
class ChainSettings:
    """The TOML schema."""

    name: str
    tables: TableSelection
    seed: int
    first_map: Path | None = None  # a seed chart with a projection (today's `first_source`)
    options: MapOptions = field(default_factory=MapOptions)
    select: Selection = field(default_factory=Selection)
    reference: list[Reference] = field(default_factory=list)
    # named changes to the merged chart's column bases before optimising (af.chain.adjust)
    adjust_column_bases: list[ColumnBasisAdjustment] = field(default_factory=list)


@dataclass(frozen=True)
class TableRef:
    table_id: str  # e.g. <group>-20260904 or <group>-20240617.2
    path: Path
    date: datetime.date
    suffix: int  # 1 for the first table of a date
    warnings: tuple[str, ...] = ()  # the table's own warnings (tables store); not map content
    # Cells the table's own repeat merge turned into `*` (af.chain.tables.repeat_drops): these
    # vanish before the chain sees them, so they are carried here to be reported.
    repeat_drops: tuple[str, ...] = ()
    # Different samples sharing an identity key in the test; later ones marked DISTINCT
    distinct_marked: tuple[str, ...] = ()


@dataclass
class ChainConfig:
    """A resolved chain: settings plus the ordered table list."""

    name: str
    tables: list[TableRef]
    options: MapOptions = field(default_factory=MapOptions)
    seed: int = 0
    first_map: Path | None = None
    tables_source: dict[str, str] | None = None  # the store ref the tables came from
    remove: list[RemoveRule] = field(default_factory=list)  # applied to each table (select)
    # How the tables were chosen (dataset or directory, date_from/date_to, exclude), as written
    # in the chain file. For a merge_all map it IS the map's definition, so it is published.
    selection: dict[str, Any] | None = None
    # Ferret sera only (af.chart.sera), applied to every table before the merge
    sera_policy: SeraPolicy = field(default_factory=SeraPolicy)
    # measured against after the run; not a step parameter
    references: list[Reference] = field(default_factory=list)
    column_basis_adjustments: list[ColumnBasisAdjustment] = field(default_factory=list)
    drop_cells: list[DropCell] = field(default_factory=list)  # readings kept out of the map
    # points left out by their sequence, decided when the chain is loaded (af.chain.sequence_select)
    sequence_removals: SequenceRemovals | None = None
    # set when the order is "append-to-published" (af.chain.append): the published version kept,
    # late tables and acknowledged re-runs, for the records
    append: dict[str, Any] | None = None
    # the published version that order keeps, and the settings re-runs accepted: checked against
    # the steps' parameters when they are built (af.chain.append.check_settings); not a record
    append_published: Any = None
    rerun_settings: list[SettingsRerun] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.column_basis_adjustments and not self.options.merge_all:
            raise ChainConfigError(
                f"{self.name}: column-basis adjustments apply to a merge_all map only (in a chain"
                " each step's merge would meet them again)"
            )
        if self.options.merge_all and self.first_map is not None:
            raise ChainConfigError(
                f"{self.name}: merge_all maps the tables from scratch; no first_map"
            )
        if self.sequence_removals is not None and self.first_map is not None:
            raise ChainConfigError(
                f"{self.name}: select.remove_by_sequence cannot be used with first_map"
            )
        if self.remove and self.first_map is not None:
            # the seed map would lose its stress (Chart.select unsets it: stale), which step 0
            # records; the round's selections apply to tables, so rules go with tables only
            raise ChainConfigError(f"{self.name}: select.remove cannot be used with first_map")
        labels = [r.label for r in self.references]
        if len(set(labels)) != len(labels):
            raise ChainConfigError(f"{self.name}: two [[reference]] entries share a label")
        if not self.tables:
            raise ChainConfigError(f"{self.name}: no tables")
        ids = [t.table_id for t in self.tables]
        if len(set(ids)) != len(ids):
            raise ChainConfigError(f"{self.name}: duplicate table ids")
        order = [(t.date, t.suffix) for t in self.tables]
        if self.append is None and order != sorted(order):
            raise ChainConfigError(f"{self.name}: tables are not in (date, suffix) order")
        if self.append is not None and (self.options.merge_all or self.first_map is not None):
            raise ChainConfigError(
                f"{self.name}: append-to-published is for a chain of steps, not merge_all or"
                " first_map"
            )

    def late(self, table_id: str) -> dict[str, Any] | None:
        """A late table's record (af.chain.append), or None."""
        return None if self.append is None else self.append["late_tables"].get(table_id)


# Options a chain file must state: their defaults are not neutral. Sarah, 2 Oct 2026: a map built
# with the silent default minimum_column_basis "none" differed from the round's (1280) unnoticed.
REQUIRED_OPTIONS = ("minimum_column_basis",)


def _require_explicit(path: Path) -> None:
    options = tomllib.loads(Path(path).read_text()).get("options", {})
    for new, old in PUBLISHED_NAMES.items():
        if old in options:
            raise ChainConfigError(f"{path}: [options] {old} is now called {new}")
    missing = [k for k in REQUIRED_OPTIONS if k not in options]
    if missing:
        raise ChainConfigError(
            f"{path}: [options] must state {', '.join(missing)} (e.g. minimum_column_basis ="
            ' "none"): its default is not neutral, so a chain file says what it means'
        )


def load_chain_config(
    path: Path, inputs_dir: Path | None = None, tables_store: Path | None = None
) -> ChainConfig:
    """`inputs_dir` receives the chart files made from store tables (the chain's own area);
    `tables_store` is the machine's store root (the run config's `[paths] store`)."""
    _require_explicit(path)
    s = load_config(path, ChainSettings)
    t = s.tables
    if t.store is not None:
        raise ChainConfigError(
            f"{path}: [tables] store is a machine setting; remove it and set [paths] store"
            " in the run config"
        )
    start, end = _date(t.date_from), _date(t.date_to)
    if t.dataset is not None and t.directory is None:
        from af.chain.tables import tables_from_store

        if inputs_dir is None or tables_store is None:
            raise ChainConfigError(f"{path}: store tables need an inputs directory and a store")
        ref, tables = tables_from_store(tables_store, t.dataset, inputs_dir, set(t.exclude))
        tables = [
            x
            for x in tables
            if (start is None or x.date >= start) and (end is None or x.date <= end)
        ]
        append, published = None, None
        if t.order == "append-to-published":
            tables, append, published = _append_order(path, t, tables, ref, tables_store)
        policy = _sera_policy(path, s.select)
        removals = (
            _sequence_removals(path, s.select, tables, ref, tables_store, policy)
            if s.select.remove_by_sequence
            else None
        )
        return ChainConfig(
            s.name,
            tables,
            s.options,
            s.seed,
            s.first_map,
            ref.to_json(),
            s.select.remove,
            _selection(t),
            policy,
            s.reference,
            s.adjust_column_bases,
            s.select.drop_cell,
            removals,
            append,
            published,
            list(t.rerun_settings),
        )
    if s.select.remove_by_sequence:
        raise ChainConfigError(
            f"{path}: select.remove_by_sequence needs tables from the store (their rows say which"
            " sequence each point is)"
        )
    if t.order != "date":
        raise ChainConfigError(f"{path}: [tables] order = {t.order!r} needs tables from the store")
    if t.directory is not None and t.group is not None and t.dataset is None:
        tables = tables_from_directory(t.directory, t.group, start, end, set(t.exclude))
        return ChainConfig(
            s.name,
            tables,
            s.options,
            s.seed,
            s.first_map,
            None,
            s.select.remove,
            _selection(t),
            _sera_policy(path, s.select),
            s.reference,
            s.adjust_column_bases,
            s.select.drop_cell,
        )
    raise ChainConfigError(f"{path}: [tables] needs either dataset or directory + group")


def _data_root(chain_path: Path) -> Path | None:
    """The data repo a chain file lives in: the directory holding chains/."""
    return next((d.parent for d in chain_path.resolve().parents if d.name == "chains"), None)


def _sequence_removals(
    chain_path: Path,
    select: Selection,
    tables: list[TableRef],
    tables_ref: Any,
    store_root: Path,
    policy: SeraPolicy,
) -> SequenceRemovals:
    """Decide `[[select.remove_by_sequence]]` on the chain's own tables (af.chain.sequence_select):
    the tables store's rows, the matcher's tables from the chain's data repo, and the pinned
    sequence versions, every one of which must be a published version of its dataset."""
    import json

    from af.chain.sequence_select import MatchRowSequencer, decide, store_inputs
    from af.chart.sera import non_ferret
    from af.seq.matching_rules import matching_rules
    from af.store.ref import StoreRef
    from af.store.store import Store
    from af.tables.model import Table
    from af.util.subtypes import subtypes

    store = Store.open(store_root)
    datasets = sorted(subtypes().keys())
    if sorted(select.sequences) != datasets:
        raise ChainConfigError(
            f"{chain_path}: [select.sequences] must pin every sequence dataset {datasets}, not"
            f" {sorted(select.sequences)}: the matcher reads them all"
        )
    pins = {}
    for dataset, version in select.sequences.items():
        manifest = store.dataset_dir("sequences", dataset) / "versions" / version / "MANIFEST.json"
        if not manifest.is_file():
            raise ChainConfigError(
                f"{chain_path}: sequences/{dataset}@{version} is not in the store"
            )
        pins[dataset] = StoreRef("sequences", dataset, version, sha256_path(manifest))
        store.resolve(pins[dataset])  # the version id is its manifest's hash: checked here
    root = _data_root(chain_path)
    if root is None:
        raise ChainConfigError(
            f"{chain_path}: no chains/ directory above it for the matcher's rules"
        )
    rules = matching_rules(root)
    tables_dir = store.resolve(tables_ref) / "tables"

    def model(table_id: str) -> Table:
        text = (tables_dir / f"{table_id}.json").read_text(encoding="utf-8")
        return Table.from_json(json.loads(text))

    def kept(chart: Any) -> set[int]:
        if policy.mode == "keep":
            return set(range(chart.n_sera))
        gone = {j for j, _, _ in non_ferret(chart, list(policy.markers)).non_ferret}
        return {j for j in range(chart.n_sera) if j not in gone}

    inputs = store_inputs(tables, model, kept)
    decisions = decide(select.remove_by_sequence, inputs, MatchRowSequencer(store, rules, pins))
    return SequenceRemovals(
        tuple(select.remove_by_sequence),
        decisions,
        {d: r.to_json() for d, r in sorted(pins.items())},
        tuple(i.to_json() for i in rules.inputs),
    )


def _append_order(
    chain_path: Path, t: TableSelection, tables: list[TableRef], ref: Any, store_root: Path
) -> tuple[list[TableRef], dict[str, Any], Any]:
    """`[tables] order = "append-to-published"` (af.chain.append): this chain dataset's published
    CURRENT version's order kept, today's new tables after it. Returns the tables, the record,
    and the published version (for `check_settings` once the steps exist)."""
    import json

    from af.chain.__main__ import dataset_from_path
    from af.chain.append import Published, append_order, table_changes
    from af.store.ref import StoreError, StoreRef
    from af.store.store import Store
    from af.tables.model import Table

    dataset = dataset_from_path(chain_path)
    store = Store.open(store_root)
    try:
        current = store.current("chains", dataset)
    except StoreError:
        current = None  # the first version
    today_dir = store.resolve(ref)
    today_doc = json.loads((today_dir / "index.json").read_text())
    today_sha = {x.table_id: sha256_path(x.path) for x in tables}  # what keys each step
    published, old_dir = None, None
    if current is not None:
        vdir = store.resolve(current)
        doc = json.loads((vdir / "chain.json").read_text())
        old_dir = store.resolve(StoreRef.from_json(doc["config"]["tables_source"]))
        published = Published(
            current.version,
            [(s["table_id"], _step_table_sha(vdir / s["directory"])) for s in doc["steps"]],
            vdir,
        )

    def read(directory: Path, table_id: str) -> Table:
        text = (directory / "tables" / f"{table_id}.json").read_text(encoding="utf-8")
        return Table.from_json(json.loads(text))

    def changes(table_id: str) -> dict[str, Any]:
        assert old_dir is not None
        got = table_changes(read(old_dir, table_id), read(today_dir, table_id))
        return got or {"chart (map content the same: an af chart-format change)": 1}

    def why_dropped(table_id: str) -> tuple[str, str]:
        if table_id in today_doc["tables"]:
            if table_id in t.exclude:
                return "excluded", "excluded by the chain file ([tables] exclude)"
            return "excluded", "outside the chain file's date_from/date_to"
        retired = today_doc.get("retired", {}).get(table_id)
        if retired is None:
            return "retired", "no longer in the tables store"
        heirs = [
            tid for tid, e in today_doc["tables"].items()
            if e.get("source_key") == retired.get("source_key")
        ]  # fmt: skip
        if heirs:
            return "retired", f"retired; its source ({retired.get('source_key')}) is now {heirs[0]}"
        return "retired", (
            f"retired with no successor in this dataset (source {retired.get('source_key')}): gone"
            " from the source, or moved to another group"
        )

    order: Order = append_order(
        dataset, published, tables, today_sha, t.rerun_from, changes, why_dropped
    )
    record = order.to_json(published)
    if t.rerun_settings:
        record["rerun_settings"] = [
            {"keys": r.keys, "decided": r.decided} for r in t.rerun_settings
        ]
    return order.tables, record, published


def _step_table_sha(step: Path) -> str:
    """The table chart a published step was built from: its step.json `table_sha256`, or for a
    first step without one, its build's one input."""
    import json

    record = json.loads((step / "step.json").read_text())
    if record.get("table_sha256"):
        return str(record["table_sha256"])
    inputs = json.loads((step / f"{CHOSEN_PROVENANCE}").read_text())["inputs"]
    return str(inputs[-1]["sha256"])


def _sera_policy(chain_path: Path, select: Selection) -> SeraPolicy:
    """The ferret-only policy with its markers file: `[select] non_ferret_markers`, or else
    rules/sera/non_ferret_markers.tsv in the data repo the chain file lives in (the directory
    holding chains/). The file is required either way."""
    path = select.non_ferret_markers
    if path is None:
        root = _data_root(chain_path)
        if root is None:
            raise ChainConfigError(
                f"{chain_path}: no [select] non_ferret_markers and no chains/ directory above it"
            )
        path = root / "rules" / "sera" / "non_ferret_markers.tsv"
    markers = read_markers(path)  # missing or malformed: an error
    return SeraPolicy(
        select.non_ferret_sera,
        select.non_ferret_reason,
        tuple(markers),
        str(path),
        sha256_path(path),
    )


def _selection(t: TableSelection) -> dict[str, Any]:
    """The chain file's [tables] as written (store excluded: refused above)."""
    out = {k: v for k, v in asdict(t).items() if k != "store" and v not in (None, [])}
    if out.get("order") == "date":  # the default, written only when chosen otherwise
        del out["order"]
    return {k: str(v) if isinstance(v, Path) else v for k, v in out.items()}


def _date(text: str | None) -> datetime.date | None:
    return None if text is None else datetime.date.fromisoformat(text)


_TABLE_NAME = re.compile(r"^(?P<group>.+)-(?P<date>\d{8})(?:\.(?P<suffix>\d+))?\.ace$")


def tables_from_directory(
    directory: Path,
    group: str,
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
    exclude: set[str] = frozenset(),  # type: ignore[assignment]
) -> list[TableRef]:
    """A group's `.ace` tables, ordered by parsed (date, suffix); never by string or index.

    Every id in `exclude` must name a table of the group: an exclusion that matches
    nothing is a mistake in the config (design rule 1).
    """
    everything: list[TableRef] = []
    for path in Path(directory).glob(f"{group}-*.ace"):
        m = _TABLE_NAME.match(path.name)
        if not m or m["group"] != group:
            continue
        date = datetime.datetime.strptime(m["date"], "%Y%m%d").date()
        everything.append(TableRef(path.name[: -len(".ace")], path, date, int(m["suffix"] or 1)))
    unmatched = set(exclude) - {t.table_id for t in everything}
    if unmatched:
        raise ChainConfigError(f"exclusions match no table: {sorted(unmatched)}")
    chosen = [
        t
        for t in everything
        if t.table_id not in exclude
        and (date_from is None or t.date >= date_from)
        and (date_to is None or t.date <= date_to)
    ]
    if not chosen:
        raise ChainConfigError(f"no tables for {group} in {directory} ({date_from}..{date_to})")
    return sorted(chosen, key=lambda t: (t.date, t.suffix))


def config_to_json(cfg: ChainConfig) -> dict[str, Any]:
    return {
        "name": cfg.name,
        "seed": cfg.seed,
        "first_map": None if cfg.first_map is None else str(cfg.first_map),
        "tables_source": cfg.tables_source,
        "options": option_parameters(cfg.options),
        **({"select_remove": [r.to_json() for r in cfg.remove]} if cfg.remove else {}),
        **({"select_drop_cell": [r.to_json() for r in cfg.drop_cells]} if cfg.drop_cells else {}),
        **(
            {"select_remove_by_sequence": cfg.sequence_removals.to_json()}
            if cfg.sequence_removals is not None
            else {}
        ),
        **({"selection": cfg.selection} if cfg.selection else {}),
        **({"append": cfg.append} if cfg.append is not None else {}),
        "non_ferret_sera": cfg.sera_policy.to_json(),
        **(
            {"column_basis_adjustments": [a.to_json() for a in cfg.column_basis_adjustments]}
            if cfg.column_basis_adjustments
            else {}
        ),
        **(
            {
                "references": [
                    {"label": r.label, "chart": r.chart, "starts": r.starts} for r in cfg.references
                ]
            }
            if cfg.references
            else {}
        ),
        "tables": [
            {
                "table_id": t.table_id,
                "path": str(t.path),
                "date": t.date.isoformat(),
                "suffix": t.suffix,
                "warnings": list(t.warnings),
                **({"repeat_drops": list(t.repeat_drops)} if t.repeat_drops else {}),
                **({"distinct_marked": list(t.distinct_marked)} if t.distinct_marked else {}),
            }
            for t in cfg.tables
        ],
    }
