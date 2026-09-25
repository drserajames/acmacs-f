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
5. the **outgroup** (R8) is exempt from all of it, must be in the store, and comes first.

Every rule reports what it removed or added. A rule that changes nothing is an error unless it
is marked optional (design rule 1): a rule that has silently stopped matching is how today's
exports drifted. Nothing is deduplicated (R9): identical sequences are separate records.
"""

from __future__ import annotations

import csv
import datetime
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import duckdb

from af.seq.nextclade import Aligned, QcThresholds, qc_failures
from af.store import Store
from af.store.ref import StoreRef

Key = tuple[str, str]  # (epi_isl, accession)

FILTERS = ("host", "date_floor", "aa_deletion", "cut")
KINDS = (*FILTERS, "include_list", "qc", "exclude_list")


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


def select(store: Store, config: SubtypeRules, *, cut: Cut | None = None) -> Selection:
    ref = store.current("sequences", config.dataset)
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
    for kind in ("include_list", "qc", "exclude_list"):
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


def _keys(store: Store, ref: StoreRef) -> frozenset[Key]:
    """Every key of one store version: what was known when a cut was chosen from it."""
    path = str(store.resolve(ref) / "isolates" / "*" / "*.parquet")
    rows = duckdb.execute("select epi_isl, accession from read_parquet(?)", [path]).fetchall()
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
    version = store.resolve(ref)
    rows = duckdb.execute(
        "select i.epi_isl, i.accession, i.host, i.collection_date_first, s.align_error,"
        " s.alignment_start, s.alignment_end, s.covers_mature, s.nuc_aligned, s.aa_aligned,"
        " s.failed_cds, s.frameshifts, s.deleted_aa, s.inserted_aa, s.unknown_aa,"
        " s.premature_stop, s.nextclade_qc_status"
        " from read_parquet(?) i join read_parquet(?) s using (epi_isl, accession)",
        [str(version / "isolates" / "*" / "*.parquet"),
         str(version / "sequences" / "*" / "*.parquet")],
    ).fetchall()  # fmt: skip
    out = {}
    for row in rows:
        epi, acc, host, first, error, start, end, covers, nuc, aa, failed, *rest = row
        frameshifts, deleted, inserted, unknown, stop, status = rest
        aligned = Aligned(f"{epi}.{acc}", error, start, end, bool(covers), nuc, aa,
                          tuple(failed or ()), frameshifts or 0, deleted or 0, inserted or 0,
                          unknown or 0, bool(stop), status or "")  # fmt: skip
        out[(epi, acc)] = _Record((epi, acc), host, first, aa, aligned)
    return out


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
