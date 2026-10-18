"""Which stored sequences go into a tree: the W1 export rules, written down and counted.

``select(store, subtype, rules)`` is the one implementation (agreed with WS5); the tree export
only writes the FASTA and the provenance. The rules come from config in acmacs-f-data, each with
its reason (``notes/sequences/RULES-DRAFT.md`` R1–R11, R4''). The order they apply in is fixed
here, not in config, so no config can change what beats what:

1. **filters**, in config order: ``host`` (R1), ``date_floor`` (R4), ``aa_deletion`` (R5),
   ``cut`` (R4'', the VCM cut node). A host rule may name an override ``file`` of records whose
   GISAID host is wrong (EPI_ISL, accession, reason); they pass it and are counted as added;
2. ``include_list`` (R7) adds listed records back that a filter removed;
3. ``qc`` (R3) applies to everything, included records too;
4. ``exclude_list`` (R6) removes listed records; exclude beats include;
5. ``low_divergence`` removes records far too close to the outgroup for their collection year
   (:func:`_low_divergence`). It runs last because "expected for the year" is measured on
   what the other rules left;
6. the **outgroup** (R8) is exempt from all of it, must be in the store, and comes first.

Every rule reports what it removed or added. A rule that changes nothing is an error unless it
is marked optional (design rule 1): a rule that has silently stopped matching is how today's
exports drifted. Nothing is deduplicated (R9): identical sequences are separate records.
"""

from __future__ import annotations

import csv
import datetime
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import duckdb
import numpy as np

from af.seq.nextclade import Aligned, QcThresholds, qc_failures
from af.store import Store
from af.store.ref import StoreRef

Key = tuple[str, str]  # (epi_isl, accession)

FILTERS = ("host", "date_floor", "aa_deletion", "cut")
LATER = ("include_list", "qc", "exclude_list", "low_divergence")  # in the order applied
KINDS = (*FILTERS, *LATER)

#: A record is compared with the outgroup only where both have a called base, and only when
#: that is most of the gene: a distance over a short stretch says little about the whole.
MIN_COMPARABLE = 0.9


class SelectionError(RuntimeError):
    """Rules that cannot be applied as written, or that changed nothing and are not optional."""


class Cut(Protocol):
    """What selection needs from a cut pin (af.tree.cut.CutPin, CUT-NODE.md §3e)."""

    @property
    def below(self) -> Collection[Key]: ...

    @property
    def chosen_from(self) -> StoreRef: ...


@dataclass(frozen=True)
class Rule:
    id: str
    kind: str
    reason: str
    optional: bool = False
    allow: list[str] = field(default_factory=list)  # host
    floor: str = ""  # date_floor, ISO date
    position: int = 0  # aa_deletion, 1-based mature HA
    file: Path | None = None  # include_list / exclude_list; host: overrides
    max_unknown_aa: int = -1  # qc
    max_deleted_aa: int = -1  # qc
    min_ratio: float = 0.0  # low_divergence: distance / the year's median, below this is out
    min_year_records: int = 0  # low_divergence: a year with fewer has no median to trust


@dataclass(frozen=True)
class Outgroup:
    epi_isl: str
    accession: str
    reason: str

    @property
    def key(self) -> Key:
        return (self.epi_isl, self.accession)


@dataclass(frozen=True)
class SubtypeRules:
    dataset: str  # sequences/<dataset>
    outgroup: Outgroup
    rules: list[Rule]


@dataclass(frozen=True)
class RuleCount:
    rule: str
    kind: str
    removed: int
    added: int
    remaining: int  # records still in after this rule, outgroup not counted (tree = remaining + 1)
    reason: str
    source: str  # the list file for list rules, else ""
    #: Removed records by the pull that brought them, for rules that say so (low_divergence).
    by_pull: dict[str, int] = field(default_factory=dict)
    #: Records the rule could not judge and so kept (low_divergence): no date, too little
    #: sequence to compare, or a year with too few records for a median.
    untested: int = 0


@dataclass
class Selection:
    keys: list[Key]  # outgroup first, then by collection date, epi_isl, accession
    outgroup: Key
    counts: list[RuleCount]
    store: StoreRef
    total: int  # records in the store version before any rule


@dataclass(frozen=True)
class _Record:
    key: Key
    host: str
    first: datetime.date | None
    aa: str | None
    aligned: Aligned
    pull: str


def select(
    store: Store,
    config: SubtypeRules,
    *,
    cut: Cut | None = None,
    sequences: StoreRef | None = None,
) -> Selection:
    """The selection from ``sequences``, or from the dataset's CURRENT when it is not given.

    A named version is what lets an export be re-made for the version a tree actually read, which
    may no longer be CURRENT (af.tree.export_record.back_fill).
    """
    ref = store.current("sequences", config.dataset) if sequences is None else sequences
    if (ref.kind, ref.dataset) != ("sequences", config.dataset):
        raise SelectionError(f"{config.dataset}: rules for sequences/{config.dataset}, given {ref}")
    store.resolve(ref)
    records = _load(store, ref)
    _check_rules(config.rules, cut)
    if config.outgroup.key not in records:
        raise SelectionError(
            f"{config.dataset}: outgroup {config.outgroup.key} is not in {ref}; a tree cannot "
            "be rooted without it"
        )

    known = _keys(store, cut.chosen_from) if cut is not None else frozenset()
    # The outgroup never enters the rules: it is above every cut and older than every floor,
    # and counting it as removed would misstate what the rules did.
    kept = set(records) - {config.outgroup.key}
    counts: list[RuleCount] = []
    by_kind = {kind: [r for r in config.rules if r.kind == kind] for kind in KINDS}
    for rule in (r for r in config.rules if r.kind in FILTERS):
        caught = {k for k in kept if _filtered(rule, records[k], cut, known)}
        saved = caught & _host_overrides(rule, records)
        kept -= caught - saved
        counts.append(_count(rule, len(caught - saved), len(saved), len(kept)))
    for rule in by_kind["include_list"]:
        listed = _listed(rule, records) - {config.outgroup.key}
        added = listed - kept
        kept |= added
        counts.append(_count(rule, 0, len(added), len(kept)))
    for rule in by_kind["qc"]:
        limits = QcThresholds(rule.max_unknown_aa, rule.max_deleted_aa)
        removed = {k for k in kept if qc_failures(records[k].aligned, limits)}
        kept -= removed
        counts.append(_count(rule, len(removed), 0, len(kept)))
    for rule in by_kind["exclude_list"]:
        removed = _listed(rule, records) & kept
        kept -= removed
        counts.append(_count(rule, len(removed), 0, len(kept)))
    for rule in by_kind["low_divergence"]:
        removed, untested = _low_divergence(rule, kept, records, config.outgroup.key)
        kept -= removed
        by_pull = Counter(records[k].pull for k in removed)
        count = _count(rule, len(removed), 0, len(kept))
        counts.append(replace(count, by_pull=dict(sorted(by_pull.items())), untested=untested))

    idle = [c.rule for c, r in zip(counts, _in_order(config.rules), strict=True)
            if not (c.removed or c.added) and not r.optional]  # fmt: skip
    if idle:
        raise SelectionError(
            f"{config.dataset}: rule(s) changed nothing and are not optional: {idle}"
        )
    ordered = sorted(kept, key=lambda k: (records[k].first or datetime.date.max, k))
    return Selection(
        [config.outgroup.key, *ordered], config.outgroup.key, counts, ref, len(records)
    )


def _in_order(rules: Sequence[Rule]) -> list[Rule]:
    """Rules in the order select() applies them, to pair with the counts."""
    order = [r for r in rules if r.kind in FILTERS]
    for kind in LATER:
        order += [r for r in rules if r.kind == kind]
    return order


def _check_rules(rules: Sequence[Rule], cut: Cut | None) -> None:
    problems = []
    for rule in rules:
        if rule.kind not in KINDS:
            problems.append(f"{rule.id}: unknown kind {rule.kind!r}")
        if not rule.reason.strip():
            problems.append(f"{rule.id}: no reason")  # every rule carries its reason
        if rule.kind in ("include_list", "exclude_list") and (
            rule.file is None or not rule.file.is_file()
        ):
            problems.append(f"{rule.id}: list file {rule.file} missing")
        if rule.kind == "host" and rule.file is not None and not rule.file.is_file():
            problems.append(f"{rule.id}: override file {rule.file} missing")
        if rule.kind == "qc" and min(rule.max_unknown_aa, rule.max_deleted_aa) < 0:
            problems.append(f"{rule.id}: qc needs max_unknown_aa and max_deleted_aa")
        if rule.kind == "aa_deletion" and rule.position < 1:
            problems.append(f"{rule.id}: aa_deletion needs a 1-based position")
        if rule.kind == "date_floor":
            try:
                datetime.date.fromisoformat(rule.floor)
            except ValueError:
                problems.append(f"{rule.id}: floor {rule.floor!r} is not an ISO date")
        if rule.kind == "low_divergence" and not (
            0 < rule.min_ratio < 1 and rule.min_year_records >= 1
        ):
            problems.append(
                f"{rule.id}: low_divergence needs 0 < min_ratio < 1 and min_year_records >= 1"
            )
        if rule.kind == "cut" and cut is None:
            problems.append(f"{rule.id}: a cut rule needs the cycle's cut pin")
    if len({r.id for r in rules}) != len(rules):
        problems.append("rule ids repeat")
    if problems:
        raise SelectionError("; ".join(problems))


def _filtered(rule: Rule, record: _Record, cut: Cut | None, known: frozenset[Key]) -> bool:
    """True when ``rule`` removes ``record``."""
    if rule.kind == "host":
        return record.host not in rule.allow
    if rule.kind == "date_floor":
        # The whole interval must be on or after the floor (Q3): a year-only 2018 does not pass
        # a 2018-03 floor. No date at all is removed and counted here.
        return record.first is None or record.first < datetime.date.fromisoformat(rule.floor)
    if rule.kind == "aa_deletion":
        return (
            record.aa is not None
            and len(record.aa) >= rule.position
            and record.aa[rule.position - 1] == "-"
        )
    if rule.kind == "cut":
        assert cut is not None
        # CUT-NODE.md §3c: keep what is below the cut or was not known when it was chosen.
        return record.key not in cut.below and record.key in known
    raise SelectionError(f"{rule.id}: not a filter")


def _low_divergence(
    rule: Rule, kept: Collection[Key], records: dict[Key, _Record], outgroup: Key
) -> tuple[set[Key], int]:
    """``(removed, untested)``: records much closer to the outgroup than their year's are.

    A virus collected years after the outgroup has had those years to diverge from it. One
    that is nearly identical to it is a mislabelled old virus, a laboratory strain or a
    contaminated sequence, and a group of them pulls the root of the tree in among the
    newest viruses. The test is relative, ``distance / median distance of the same collection
    year < min_ratio``, so one threshold serves every year and subtype without knowing any
    rate of evolution.

    A record that cannot be judged is kept and counted as untested, never removed: no
    collection date, too few comparable sites (:data:`MIN_COMPARABLE`), or a year with fewer
    than ``min_year_records`` measured records, where a median means nothing.
    """
    reference = records[outgroup].aligned.nucleotides
    if reference is None:
        raise SelectionError(f"{rule.id}: the outgroup has no aligned sequence to measure from")
    distance = _distances(reference, {k: records[k].aligned.nucleotides for k in kept})
    by_year: dict[int, list[float]] = {}
    year: dict[Key, int] = {}
    for key, value in distance.items():
        first = records[key].first
        if first is not None:
            year[key] = first.year
            by_year.setdefault(first.year, []).append(value)
    median = {
        y: float(np.median(values))
        for y, values in by_year.items()
        if len(values) >= rule.min_year_records
    }
    tested = {k for k, y in year.items() if median.get(y, 0.0) > 0}
    removed = {k for k in tested if distance[k] / median[year[k]] < rule.min_ratio}
    return removed, len(kept) - len(tested)


def _distances(reference: str, sequences: dict[Key, str | None]) -> dict[Key, float]:
    """Nucleotide p-distance to ``reference`` over the sites where both have A, C, G or T.

    Only records aligned to the reference's length with enough comparable sites are measured.
    """
    called = np.frombuffer(b"ACGT", dtype=np.uint8)
    ref = np.frombuffer(reference.upper().encode(), dtype=np.uint8)
    ref_called = np.isin(ref, called)
    out: dict[Key, float] = {}
    for key, nucleotides in sequences.items():
        if nucleotides is None or len(nucleotides) != len(ref):
            continue
        query = np.frombuffer(nucleotides.upper().encode(), dtype=np.uint8)
        both = ref_called & np.isin(query, called)
        sites = int(both.sum())
        if sites and sites >= MIN_COMPARABLE * len(ref):
            out[key] = float((query[both] != ref[both]).sum()) / sites
    return out


def _keys(store: Store, ref: StoreRef) -> frozenset[Key]:
    """Every key of one store version: what was known when a cut was chosen from it."""
    path = str(store.resolve(ref) / "isolates" / "*" / "*.parquet")
    rows = duckdb.execute(
        "select epi_isl, accession from read_parquet(?, union_by_name=true)", [path]
    ).fetchall()
    return frozenset((e, a) for e, a in rows)


def _listed(rule: Rule, records: dict[Key, _Record]) -> set[Key]:
    """Keys of the list file that are in this store version; the rest match nothing yet."""
    assert rule.file is not None
    want = rule.kind.removesuffix("_list")
    lines = [line for line in rule.file.read_text().splitlines() if not line.startswith("#")]
    keys = {
        (row["epi_isl"], row["accession"])
        for row in csv.DictReader(lines, delimiter="\t")
        if row.get("list", want) == want and row["epi_isl"]
    }
    return keys & records.keys()


def _host_overrides(rule: Rule, records: dict[Key, _Record]) -> set[Key]:
    """Keys a host rule lets through although their GISAID host is not allowed (R1, Q1).

    Every row needs a reason. A row that overrides nothing -- its record is not in this store
    version, or its host is already allowed -- is an error unless the rule is optional: an
    override that has stopped applying is either stale or waiting for a pull, and either way
    someone should look.
    """
    if rule.kind != "host" or rule.file is None:
        return set()
    lines = [line for line in rule.file.read_text().splitlines() if not line.startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    if unexplained := [r["epi_isl"] for r in rows if not (r.get("reason") or "").strip()]:
        raise SelectionError(f"{rule.id}: override rows without a reason: {unexplained}")
    keys = {(r["epi_isl"], r["accession"]) for r in rows}
    idle = sorted(k for k in keys if k not in records or records[k].host in rule.allow)
    if idle and not rule.optional:
        raise SelectionError(f"{rule.id}: override rows that change nothing: {idle}")
    return keys - set(idle)


def _count(rule: Rule, removed: int, added: int, remaining: int) -> RuleCount:
    source = str(rule.file) if rule.file else ""
    return RuleCount(rule.id, rule.kind, removed, added, remaining, rule.reason, source)


def _load(store: Store, ref: StoreRef) -> dict[Key, _Record]:
    """Every record of one store version, with what the rules need.

    ``union_by_name`` because a version's partitions need not share a schema: publish_pull
    rewrites only the pulls it re-stores and hard-links the rest, so a column added to the
    sequences table (aa_partial) exists in some partitions and not others until every pull has
    been re-stored. Without it duckdb refuses the whole glob ("schema mismatch in glob") and
    selection fails for that subtype rather than degrading.
    """
    version = store.resolve(ref)
    sequences = str(version / "sequences" / "*" / "*.parquet")
    rows = duckdb.execute(
        "select i.epi_isl, i.accession, i.host, i.collection_date_first, i.source_pull,"
        " s.align_error,"
        " s.alignment_start, s.alignment_end, s.covers_mature, s.nuc_aligned, s.aa_aligned,"
        " s.failed_cds, s.frameshifts, s.deleted_aa, s.inserted_aa, s.unknown_aa,"
        " s.premature_stop, s.nextclade_qc_status, " + _partial(sequences) +
        " from read_parquet(?, union_by_name=true) i"
        " join read_parquet(?, union_by_name=true) s using (epi_isl, accession)",
        [str(version / "isolates" / "*" / "*.parquet"), sequences],
    ).fetchall()  # fmt: skip
    out = {}
    for row in rows:
        epi, acc, host, first, pull, error, start, end, covers, nuc, aa, failed, *rest = row
        frameshifts, deleted, inserted, unknown, stop, status, partial = rest
        aligned = Aligned(f"{epi}.{acc}", error, start, end, bool(covers), nuc, aa,
                          tuple(failed or ()), frameshifts or 0, deleted or 0, inserted or 0,
                          unknown or 0, bool(stop), status or "", bool(partial))  # fmt: skip
        out[(epi, acc)] = _Record((epi, acc), host, first, aa, aligned, pull)
    return out


def _partial(sequences: str) -> str:
    """The aa_partial column, or false where a version's pulls were all built before it existed.

    Builds before the column could not store a partial protein (a failed CDS discarded the whole
    translation), so false is what those rows mean, not a guess. Where only some partitions have
    the column, union_by_name gives the others NULL, which reads as false for the same reason.
    """
    described = duckdb.execute(
        "describe select * from read_parquet(?, union_by_name=true)", [sequences]
    ).fetchall()
    columns = {c for (c, *_) in described}
    return "s.aa_partial" if "aa_partial" in columns else "false as aa_partial"


def parse_rules(data: dict[str, Any], base_dir: Path) -> SubtypeRules:
    """One subtype's table of the selection config (see acmacs-f-data config/selection.toml)."""
    out = data["outgroup"]
    rules = []
    for item in data.get("rule", []):
        fields = dict(item)
        if "file" in fields:
            fields["file"] = (base_dir / fields["file"]).resolve()
        rules.append(Rule(**fields))
    return SubtypeRules(
        data["dataset"], Outgroup(out["epi_isl"], out["accession"], out["reason"]), rules
    )
