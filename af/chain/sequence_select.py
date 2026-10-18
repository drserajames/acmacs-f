"""Points a chain leaves out by their SEQUENCE: `[[select.remove_by_sequence]]` (Q166, Q176).

Some rounds curate their oldest tables by what each virus's HA is, not by its name or passage:
the vidrl B/Vic round removes, among the antigens and sera seen only in tables up to a date, every
one whose HA1 lacks the deletion at 162-164 (the round's `remove_non_3del_curated`). Here that is
data in the chain file, with the sequences it was decided on pinned beside it:

    [select.sequences]                   # every sequence dataset, pinned (the matcher reads all)
    bvic = "<version>"
    ...
    [[select.remove_by_sequence]]
    what = "both"                        # or "antigens", "sera"
    tables_until = "2021-12-09"          # only points seen ONLY in tables up to this date
    deletion = [162, 163, 164]           # 1-based mature-HA positions that must all be deleted
    unsequenced = "remove"               # or "keep": a point with no sequence (Sarah, Q176)
    ties = "by-candidates"               # or "closest-passage", "as-unsequenced" (see `keeps`)
    reason = "..."

Which sequence a point is: `af.serology.joins.match_row` on the point's own row in its table, as
the tables store holds it (name, passage, passage class, lineage, EPI_ISL, reassortant, lab). That
is "what the matcher says for this table's row", not what the maps colour a preparation by
(`preparation_sequences` also weighs other labs' rows of the same preparation; 10, 18 Oct 2026).
On the vidrl B/Vic candidates the two routes keep and remove the same 63 antigens
(notes: chain-work/logs/non3del-route-compare-20261018.txt), and this route needs no serology.

The deletion is tested on the aligned protein as `af.seq.select`'s `aa_deletion` rule (R5) tests
it: `aa[position - 1] == "-"`, here at every listed position.

Decided once, when the chain is loaded, and applied to every table before the merge, beside
`af.chain.select`. The decisions (what is removed, and why) are step parameters, so a chain is
remade when they change and not when only a sequence pin moves without changing them; the pins
are inputs of the published version. A rule with no candidates is an error unless optional.
"""

from __future__ import annotations

import datetime
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from af.chart.model import Antigen, Chart, Serum

WHAT = ("antigens", "sera", "both")
UNSEQUENCED = ("keep", "remove")
TIES = ("by-candidates", "closest-passage", "as-unsequenced")

# A point's verdict. "deleted" keeps it, "not-deleted" removes it; the rest follow the rule.
DELETED, NOT_DELETED = "deleted", "not-deleted"
UNSEQUENCED_VERDICT = "unsequenced"  # no sequence matched (or no dataset covers it)
NO_PROTEIN = "no-protein"  # a sequence matched, but it has no aligned protein to test
TIE_ALL, TIE_NONE, TIE_SPLIT = "tie-all-deleted", "tie-none-deleted", "tie-split"

Key = tuple[str, str, tuple[str, ...], str]  # name, reassortant, annotations, passage or serum id


class SequenceSelectError(ValueError):
    pass


@dataclass(frozen=True)
class SequenceRule:
    what: str
    tables_until: str  # ISO date
    deletion: list[int]
    unsequenced: str
    ties: str
    reason: str
    optional: bool = False

    def __post_init__(self) -> None:
        name = f"select.remove_by_sequence ({self.reason or 'no reason'})"
        if self.what not in WHAT:
            raise SequenceSelectError(f"{name}: what must be one of {WHAT}, not {self.what!r}")
        if not self.reason.strip():
            raise SequenceSelectError("select.remove_by_sequence: every rule needs a reason")
        try:
            datetime.date.fromisoformat(self.tables_until)
        except ValueError as e:
            raise SequenceSelectError(f"{name}: tables_until must be YYYY-MM-DD") from e
        if not self.deletion or any(p < 1 for p in self.deletion):
            raise SequenceSelectError(f"{name}: deletion lists 1-based positions")
        if self.unsequenced not in UNSEQUENCED:
            raise SequenceSelectError(f"{name}: unsequenced must be one of {UNSEQUENCED}")
        if self.ties not in TIES:
            raise SequenceSelectError(f"{name}: ties must be one of {TIES}")

    @property
    def until(self) -> datetime.date:
        return datetime.date.fromisoformat(self.tables_until)

    def covers(self, kind: str) -> bool:
        return self.what == "both" or self.what == kind

    def keeps(self, verdict: str, ranked_deleted: bool | None = None) -> bool:
        """Whether a point with `verdict` stays. A name tie (the matcher found several sequences
        and chose none, after its exact-passage step):
        - "by-candidates": kept when every candidate has the deletion, removed when none has; a
          split (or a candidate with no protein) is no answer;
        - "closest-passage": as the candidate ae's rank takes (best passage rank, then lowest
          EPI_ISL; `af.seq.matching.ae_ranked`), whose protein is `ranked_deleted`; the same
          pick Sarah's Q217 colours a tie by. No ranked candidate, or no protein, is no answer;
        - "as-unsequenced": every tie is no answer.
        No answer follows `unsequenced`."""
        if verdict == DELETED:
            return True
        if verdict == NOT_DELETED:
            return False
        if verdict in (TIE_ALL, TIE_NONE, TIE_SPLIT):
            if self.ties == "by-candidates" and verdict != TIE_SPLIT:
                return verdict == TIE_ALL
            if self.ties == "closest-passage" and ranked_deleted is not None:
                return ranked_deleted
        return self.unsequenced == "keep"

    def to_json(self) -> dict[str, Any]:
        return {
            "what": self.what,
            "tables_until": self.tables_until,
            "deletion": list(self.deletion),
            "unsequenced": self.unsequenced,
            "ties": self.ties,
            "reason": self.reason,
            **({"optional": True} if self.optional else {}),
        }


class Sequencer(Protocol):
    """Where verdicts come from: the matcher's answer for a row, and aligned proteins by
    (EPI_ISL, accession). `MatchRowSequencer` in production; a stand-in in tests."""

    def match(self, row: Mapping[str, Any]) -> Any | None: ...

    def proteins(self, keys: Sequence[tuple[str, str]]) -> dict[tuple[str, str], str | None]: ...


@dataclass(frozen=True)
class Decision:
    what: str  # "antigen" or "serum"
    key: Key
    designation: str
    table_id: str  # the first table it is in
    verdict: str
    removed: bool
    sequences: tuple[tuple[str, str], ...] = ()  # the chosen sequence, or the tied candidates
    # a tie: the candidate ae's rank takes, and whether its protein has the deletion
    ranked: tuple[str, str] | None = None
    ranked_deleted: bool | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "what": self.what,
            "designation": self.designation,
            "table_id": self.table_id,
            "verdict": self.verdict,
            "removed": self.removed,
            "sequences": [list(s) for s in self.sequences],
            **(
                {"ranked": list(self.ranked), "ranked_deleted": self.ranked_deleted}
                if self.ranked is not None
                else {}
            ),
        }


@dataclass(frozen=True)
class SequenceRemovals:
    """The rules, what they decided, and what they were decided on."""

    rules: tuple[SequenceRule, ...]
    decisions: tuple[Decision, ...]
    sequences: Mapping[str, dict] = field(default_factory=dict)  # dataset -> store ref (JSON)
    matching_rules: tuple[dict, ...] = ()  # the matcher's tables read, with hashes

    def removed(self, what: str) -> set[Key]:
        return {d.key for d in self.decisions if d.removed and d.what == what}

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for what in ("antigen", "serum"):
            mine = [d for d in self.decisions if d.what == what]
            out[what] = {
                **dict(sorted(Counter(d.verdict for d in mine).items())),
                "removed": sum(d.removed for d in mine),
            }
        return out

    def to_parameters(self) -> dict[str, Any]:
        """What a step is keyed by: the rules and what they remove, not the pins (a pin that
        moves without changing a decision remakes nothing)."""
        removed = sorted(
            (d.what, d.designation, d.verdict) for d in self.decisions if d.removed
        )  # fmt: skip
        return {"rules": [r.to_json() for r in self.rules], "removed": [list(r) for r in removed]}

    def to_json(self) -> dict[str, Any]:
        return {
            **self.to_parameters(),
            "counts": self.counts(),
            "decisions": [d.to_json() for d in self.decisions],
            "sequences": dict(self.sequences),
            "matching_rules": list(self.matching_rules),
        }


def antigen_key(a: Antigen) -> Key:
    return (a.name, a.reassortant, tuple(a.annotations), a.passage)


def serum_key(s: Serum) -> Key:
    return (s.name, s.reassortant, tuple(s.annotations), s.serum_id)


@dataclass(frozen=True)
class TableInput:
    """One of the chain's tables as the rule sees it: its chart (as the chain reads it, before
    any rule), the tables store's own rows for the same points in the same order, and which of
    its sera the ferret-only policy keeps (a serum the map never has is no candidate)."""

    table_id: str
    date: datetime.date
    chart: Chart
    antigen_rows: Sequence[Mapping[str, Any]]
    serum_rows: Sequence[Mapping[str, Any]]
    kept_sera: frozenset[int]


def decide(
    rules: Sequence[SequenceRule], tables: Sequence[TableInput], sequencer: Sequencer
) -> tuple[Decision, ...]:
    """Every candidate's verdict and whether a rule removes it (see the module docstring)."""
    out: dict[tuple[str, Key], Decision] = {}
    for rule in rules:
        found = _candidates(rule, tables)
        if not found and not rule.optional:
            raise SequenceSelectError(
                f"select.remove_by_sequence ({rule.reason}): no point is seen only in tables up"
                f" to {rule.tables_until}; a rule that matches nothing is an error"
            )
        matches = {c: sequencer.match(row) for c, row in found.items()}
        wanted = sorted(
            {k for m in matches.values() for k in [*_sequence_keys(m), _ranked_key(m)] if k}
        )
        proteins = sequencer.proteins(wanted)
        for (what, key, table_id, designation), m in matches.items():
            verdict, sequences = _verdict(m, proteins, rule.deletion)
            ranked = _ranked_key(m)
            ranked_deleted = (
                None if ranked is None else deleted(proteins.get(ranked), rule.deletion)
            )
            removed = not rule.keeps(verdict, ranked_deleted)
            earlier = out.get((what, key))
            if earlier is not None and (earlier.removed or not removed):
                continue  # a point removed by any rule is removed; the first verdict is kept
            out[(what, key)] = Decision(
                what, key, designation, table_id, verdict, removed, sequences, ranked,
                ranked_deleted,
            )  # fmt: skip
    return tuple(sorted(out.values(), key=lambda d: (d.what, d.designation)))


def _candidates(
    rule: SequenceRule, tables: Sequence[TableInput]
) -> dict[tuple[str, Key, str, str], Mapping[str, Any]]:
    """Points seen only in tables dated up to the rule's date, each with the row of the first
    table it is in."""
    later: set[tuple[str, Key]] = set()
    for t in tables:
        if t.date > rule.until:
            later |= {("antigen", antigen_key(a)) for a in t.chart.antigens}
            later |= {("serum", serum_key(s)) for s in t.chart.sera}
    found: dict[tuple[str, Key, str, str], Mapping[str, Any]] = {}
    seen: set[tuple[str, Key]] = set()
    for t in tables:
        if t.date > rule.until:
            continue
        points: list[tuple[str, Key, str, Mapping[str, Any]]] = []
        if rule.covers("antigens"):
            points += [
                ("antigen", antigen_key(a), a.designation(), t.antigen_rows[i])
                for i, a in enumerate(t.chart.antigens)
            ]
        if rule.covers("sera"):
            points += [
                ("serum", serum_key(s), s.designation(), t.serum_rows[j])
                for j, s in enumerate(t.chart.sera)
                if j in t.kept_sera
            ]
        for what, key, designation, row in points:
            if (what, key) in later or (what, key) in seen:
                continue
            seen.add((what, key))
            found[(what, key, t.table_id, designation)] = row
    return found


def _sequence_keys(m: Any) -> list[tuple[str, str]]:
    if m is None:
        return []
    if m.chosen is not None:
        return [(m.chosen.epi_isl, m.chosen.accession)]
    return [(c.epi_isl, c.accession) for c in m.tied]


def _ranked_key(m: Any) -> tuple[str, str] | None:
    """A tie's ae-ranked candidate (Match.ranked); None for anything that is not a tie."""
    r = None if m is None or m.chosen is not None else getattr(m, "ranked", None)
    return None if r is None else (r.epi_isl, r.accession)


def deleted(aa: str | None, positions: Sequence[int]) -> bool | None:
    """Whether the protein has a gap at every position (1-based mature HA, as R5); None when
    there is no protein to test."""
    if aa is None:
        return None
    return all(len(aa) >= p and aa[p - 1] == "-" for p in positions)


def _verdict(
    m: Any, proteins: Mapping[tuple[str, str], str | None], positions: Sequence[int]
) -> tuple[str, tuple[tuple[str, str], ...]]:
    keys = tuple(_sequence_keys(m))
    if not keys:
        return UNSEQUENCED_VERDICT, ()
    tested = [deleted(proteins.get(k), positions) for k in keys]
    known = [t for t in tested if t is not None]
    if m.chosen is not None:
        if not known:
            return NO_PROTEIN, keys
        return (DELETED if known[0] else NOT_DELETED), keys
    if not known:
        return NO_PROTEIN, keys
    if len(known) < len(tested):
        return TIE_SPLIT, keys  # a candidate with no protein: the tie cannot be read whole
    return (TIE_ALL if all(known) else TIE_NONE if not any(known) else TIE_SPLIT), keys


def apply(removals: SequenceRemovals | None, chart: Chart) -> tuple[Chart, list[str]]:
    """`chart` (one table) without the points the rules removed, and their designations."""
    if removals is None:
        return chart, []
    ag, sr = removals.removed("antigen"), removals.removed("serum")
    gone_ag = [i for i, a in enumerate(chart.antigens) if antigen_key(a) in ag]
    gone_sr = [j for j, s in enumerate(chart.sera) if serum_key(s) in sr]
    if not gone_ag and not gone_sr:
        return chart, []
    report = [chart.antigens[i].designation() for i in gone_ag]
    report += [chart.sera[j].designation() for j in gone_sr]
    keep_ag = [i for i in range(chart.n_antigens) if i not in set(gone_ag)]
    keep_sr = [j for j in range(chart.n_sera) if j not in set(gone_sr)]
    return chart.select(keep_ag, keep_sr), report


# ---- production: the tables store's rows, and the matcher over pinned sequences ----------------


def row_of(table: Any, kind: str, index: int) -> dict[str, Any]:
    """A point's row as `match_row` takes it, from the tables store's own table model: the same
    fields the serology store keeps for that row (its passage as the lab wrote it, without ae's
    harvest date; the store's passage class; the lab's EPI_ISL)."""
    p = (table.antigens if kind == "antigen" else table.sera)[index]
    return {
        "subtype": table.subtype,
        "lineage": p.lineage or table.lineage or "",
        "name": p.name,
        "passage": p.passage,
        "passage_class": p.passage_class,
        "reassortant": p.reassortant,
        "epi_isl": getattr(p, "epi_isl", "") or "",
        "table_lab": table.lab,
    }


class MatchRowSequencer:
    """`match_row` over `indexes_from_store` at the pinned sequences, and their proteins."""

    def __init__(self, store: Any, rules: Any, sequences: Mapping[str, Any]) -> None:
        from af.serology.joins import indexes_from_store

        self.store = store
        self.refs = dict(sequences)
        self.indexes, _ = indexes_from_store(store, rules, self.refs)

    def match(self, row: Mapping[str, Any]) -> Any | None:
        from af.serology.joins import match_row, passage_class_column

        return match_row(row, self.indexes, passage_class_column)

    def proteins(self, keys: Sequence[tuple[str, str]]) -> dict[tuple[str, str], str | None]:
        import duckdb

        if not keys:
            return {}
        paths = [
            str(p)
            for ref in self.refs.values()
            for p in sorted((self.store.resolve(ref) / "sequences").glob("*/*.parquet"))
        ]
        con = duckdb.connect()
        con.execute("CREATE TEMP TABLE wanted (epi_isl VARCHAR, accession VARCHAR)")
        con.executemany("INSERT INTO wanted VALUES (?, ?)", list(keys))
        rows = con.execute(
            "SELECT s.epi_isl, s.accession, s.aa_aligned FROM read_parquet(?, union_by_name=true) s"
            " JOIN wanted USING (epi_isl, accession)",
            [paths],
        ).fetchall()
        got = {(e, a): aa for e, a, aa in rows}
        return {k: got.get(k) for k in keys}


def store_inputs(
    tables: Sequence[Any], store_tables: Callable[[str], Any], kept_sera: Callable[[Any], set[int]]
) -> list[TableInput]:
    """The rule's view of each chain table: `tables` are the chain's TableRefs, `store_tables`
    gives the tables store's model of a table id, `kept_sera` the serum indexes the ferret-only
    policy keeps in a chart."""
    from af.chart.ace import read_chart

    out = []
    for t in tables:
        chart = read_chart(t.path)
        model = store_tables(t.table_id)
        if (len(model.antigens), len(model.sera)) != (chart.n_antigens, chart.n_sera):
            raise SequenceSelectError(f"{t.table_id}: the store's table and its chart disagree")
        out.append(
            TableInput(
                t.table_id,
                t.date,
                chart,
                [row_of(model, "antigen", i) for i in range(chart.n_antigens)],
                [row_of(model, "serum", j) for j in range(chart.n_sera)],
                frozenset(kept_sera(chart)),
            )
        )
    return out
