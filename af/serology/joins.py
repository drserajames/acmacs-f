"""Join serology antigens to their sequences, clades and places.

Which sequence belongs with a table antigen is decided by the sequence workstream's
matcher (:mod:`af.seq.matching`), the one copy of that rule: the lab's EPI_ISL when it
gave one, otherwise the strain name within the antigen's subtype, the antigen's passage
class choosing among a name's sequences. This module runs it for every antigen row, keeps
what it said, and joins the chosen sequence's place and clade.

Every antigen row gets exactly one status, and each is counted, so nothing is dropped
silently:

- ``matched``: the matcher chose a sequence and raised no doubt;
- ``doubtful``: it chose one but flagged it (the EPI_ISL's name differs, an egg antigen
  with no egg sequence, a reassortant, ...). ``af.seq.matching.DOUBTFUL`` says which
  flags; a doubtful match is kept for review and never used unreviewed (it colours no
  geo dot);
- ``unmatched``: no sequence, or a tie between different sequences it would not break.

Every flag is counted too, and the lab's own pairing (``exact`` or ``proxy``) is kept.

Places come from the sequence store's isolates (I3), clades from the clade store's
assignments (I4), both on the chosen ``(epi_isl, accession)``. The clade store writes a null
clade where the nomenclature names none; here that becomes ``""``, so that ``None`` has one
meaning downstream: the sequence has no assignment row at all (a separate count, because
it means the clade store is behind the sequence store, or cannot align the sequence).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pyarrow as pa

from af.chart.model import Chart
from af.clades.coverage import clades_behind_in_content
from af.clades.store import CladeStoreError
from af.seq.matching import Match, SequenceIndex
from af.seq.matching_rules import MatchingRules
from af.seq.names import normalise
from af.seq.passage_match import PassageMatcher
from af.serology.store import StoreError
from af.store import StoreRef
from af.util.subtypes import SubtypeError, subtypes

STATUSES = ("matched", "doubtful", "unmatched")
SEVERAL_DATASETS = "serology.matched-in-several-datasets"
NO_DATASET = "serology.no-sequence-dataset"
# 04-clades' evidence for a tree-derived clade (Sarah, 8 Oct): whether the sequence observes
# its clade's defining loci ("supported" / "unobserved" / "contradicted"), the deepest clade
# its residues support when contradicted, and the loci. Null on fallback rows.
EVIDENCE_COLUMNS = ("clade_evidence", "supported_clade", "clade_evidence_reason")
SUPPORTED = "supported"
# a chain's mark on a repeated sample (af.chart.identity): the chart's, not the preparation's
DISTINCT = "DISTINCT"
# an antigen matched by its name as the sequence store spells it (_match_name)
NAME_NORMALISED = "serology.name-normalised"

ClassOf = Callable[[Mapping[str, Any]], str]


def passage_class_column(row: Mapping[str, Any]) -> str:
    """The tables store's passage class; "unknown" for tables read before it existed."""
    return str(row.get("passage_class") or "unknown")


@dataclass
class LinkCounts:
    clades_joined: bool = True
    by_status: dict[str, int] = field(default_factory=dict)
    by_method: dict[str, int] = field(default_factory=dict)
    by_flag: dict[str, int] = field(default_factory=dict)
    by_status_and_pairing: dict[tuple[str, str], int] = field(default_factory=dict)
    matched_without_clade_row: int = 0
    matched_with_empty_clade: int = 0
    # matched rows by their clade's evidence (tree-derived clades only; fallback rows have none)
    matched_by_evidence: dict[str, int] = field(default_factory=dict)
    # clade datasets read without the evidence columns (older, fallback-only versions; a tree
    # row without them is refused)
    clades_without_evidence_columns: list[str] = field(default_factory=list)
    # clade dataset -> (sequences version it labelled, or None if its provenance names none
    # or several, sequences version the join read), for every clade table behind in content:
    # one whose calls would change if it were relabelled from the sequences read
    clades_behind: dict[str, tuple[str | None, str]] = field(default_factory=dict)
    # the same pair for clade tables labelled from another sequences version whose clade
    # calls' inputs are identical (af.clades.coverage): not behind, the versions only differ
    clades_same_content: dict[str, tuple[str, str]] = field(default_factory=dict)
    # the store versions the join read (link_from_store): {"sequences": {dataset: ref json},
    # "clades": [ref json, ...]}, so a result can be reproduced from them (design rule 5)
    refs: dict[str, Any] = field(default_factory=dict)


def link_sequences(
    con: Any,
    indexes: Mapping[str, SequenceIndex],
    isolates: Sequence[Path],
    clades: Sequence[Path] | None,
    class_of: ClassOf = passage_class_column,
) -> LinkCounts:
    """Match every antigen row and define the view ``antigen_sequences``; return counts.

    ``indexes`` maps a sequence dataset key (h1, h3, bvic, byam) to its matcher index;
    ``isolates`` are the same datasets' isolate Parquet files, for places. ``clades=None``
    joins sequences and places only, deliberately; an empty list is an error.
    """
    if not isolates:
        raise StoreError("no sequence isolates given: cannot join antigens to places")
    if clades is not None and not clades:
        raise StoreError("no clade assignments given: cannot join antigens to clades")
    counts = LinkCounts(clades_joined=clades is not None)
    _match_rows(con, indexes, class_of, counts)
    con.execute(f"CREATE OR REPLACE VIEW isolates AS SELECT * FROM {_parquet(isolates)}")
    if clades is None:
        con.execute(
            "CREATE OR REPLACE VIEW clade_rows AS SELECT NULL::VARCHAR AS epi_isl, "
            "NULL::VARCHAR AS accession, NULL::VARCHAR AS clade, NULL::VARCHAR AS method, "
            + ", ".join(f"NULL::VARCHAR AS {c}" for c in EVIDENCE_COLUMNS)
            + " WHERE false"
        )
    else:
        con.execute(f"CREATE OR REPLACE VIEW clade_rows AS {_clade_rows(clades)}")
    _refuse_duplicates(con, "isolates", "sequence isolates")
    _refuse_duplicates(con, "clade_rows", "clade assignments")
    con.execute(
        """
        CREATE OR REPLACE VIEW antigen_sequences AS
        SELECT m.table_id, m.position, m.method, m.flags, m.doubtful, m.epi_isl, m.accession,
               m.dataset, coalesce(a.sequence_pairing, '') AS pairing,
               m.tied, m.ranked_epi_isl, m.ranked_accession,
               CASE WHEN m.accession IS NULL THEN 'unmatched'
                    WHEN m.doubtful THEN 'doubtful' ELSE 'matched' END AS status,
               i.country, i.region, i.place,
               i.collection_date AS sequence_collection_date,
               i.passage AS sequence_passage,
               -- the clade store writes NULL where the nomenclature names no clade; here that
               -- is '' so NULL keeps one meaning downstream: no assignment row at all
               CASE WHEN k.epi_isl IS NOT NULL THEN coalesce(k.clade, '') END AS clade,
               k.method AS clade_method,
               k.epi_isl IS NOT NULL AS has_clade_row,
               k.clade_evidence, k.supported_clade, k.clade_evidence_reason
        FROM antigen_matches m
        JOIN antigens a ON a.table_id = m.table_id AND a.position = m.position
        LEFT JOIN isolates i ON i.epi_isl = m.epi_isl AND i.accession = m.accession
        LEFT JOIN clade_rows k ON k.epi_isl = m.epi_isl AND k.accession = m.accession
        """
    )
    _count(con, counts)
    return counts


def _check_submitter_labs(con: Any, lab_submitters: Mapping[str, frozenset[str]]) -> None:
    """The submitters table must be keyed by the tables' own lab codes, exactly.

    A table keyed by another spelling of the labs (``cdc`` for ``CDC``) matches no antigen,
    and the own-lab rule would silently never apply (design rule 1).
    """
    labs = {lab for (lab,) in con.execute("SELECT DISTINCT lab FROM tables").fetchall()}
    if labs and not labs & set(lab_submitters):
        raise StoreError(
            "lab submitters name none of the store's table labs: "
            f"table labs {sorted(labs)}, submitters keyed by {sorted(lab_submitters)}"
        )


def _match_rows(
    con: Any, indexes: Mapping[str, SequenceIndex], class_of: ClassOf, counts: LinkCounts
) -> None:
    """Run the matcher on every antigen row; store the answers as table ``antigen_matches``."""
    cursor = con.execute(
        "SELECT a.*, t.subtype, t.lab AS table_lab FROM antigens a JOIN tables t USING (table_id) "
        "ORDER BY a.table_id, a.position"
    )
    columns = [d[0] for d in cursor.description]
    out: dict[str, list[Any]] = {k: [] for k in _MATCH_COLUMNS}
    flags_seen: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    for values in cursor.fetchall():
        row = dict(zip(columns, values, strict=True))
        match = _match_row(row, indexes, class_of)
        chosen = match.chosen if match is not None else None
        flags = list(match.flags) if match is not None else [NO_DATASET]
        flags_seen.update(flags)
        methods[(match.method if match is not None else None) or "none"] += 1
        out["table_id"].append(row["table_id"])
        out["position"].append(row["position"])
        out["method"].append(match.method if match is not None else None)
        out["epi_isl"].append(chosen.epi_isl if chosen else None)
        out["accession"].append(chosen.accession if chosen else None)
        out["dataset"].append(chosen.dataset if chosen else None)
        out["flags"].append(flags)
        out["doubtful"].append(
            bool(match is not None and (match.doubtful or SEVERAL_DATASETS in match.flags))
        )
        tied = match.tied if match is not None else ()
        ranked = match.ranked if match is not None else None
        out["tied"].append([{"epi_isl": c.epi_isl, "accession": c.accession} for c in tied])
        out["ranked_epi_isl"].append(ranked.epi_isl if ranked else None)
        out["ranked_accession"].append(ranked.accession if ranked else None)
    table = pa.table({k: pa.array(v, type=_MATCH_COLUMNS[k]) for k, v in out.items()})
    con.register("antigen_matches_arrow", table)
    con.execute("CREATE OR REPLACE TABLE antigen_matches AS SELECT * FROM antigen_matches_arrow")
    con.unregister("antigen_matches_arrow")
    counts.by_flag = dict(sorted(flags_seen.items()))
    counts.by_method = dict(sorted(methods.items()))


def _match_name(raw: str) -> str:
    """The antigen's name as the sequence store spells it, for the matcher's key.

    Sequence names are stored through :func:`af.seq.names.normalise`; table names are not, and
    keep what it strips, a zero-padded part of a joined isolate ("ABC-01" is "ABC-1" in the
    store). Keyed raw, such an antigen finds no sequence although its sequence is there. Only
    a name normalise parses cleanly is used: one it reports a problem with is matched as
    written, so the join never guesses at a name's shape.
    """
    normalised = normalise(raw)
    return normalised.name if normalised.ok else raw


def _match_row(
    row: Mapping[str, Any], indexes: Mapping[str, SequenceIndex], class_of: ClassOf
) -> Match | None:
    """The matcher's answer for one antigen row; None if no dataset covers its subtype."""
    # the sequence datasets an antigen is matched within, from the subtype table
    # (af/subtypes.toml), in table order: a B antigen of unknown lineage is matched across both
    # B datasets, B/Vic first. A subtype or lineage the table does not list is counted
    # (NO_DATASET), not fatal for the whole join.
    try:
        rows = subtypes().for_table(row["subtype"], row.get("lineage") or "")
    except SubtypeError:
        return None
    datasets = tuple(r.key for r in rows)
    if any(d not in indexes for d in datasets):
        return None
    name = _match_name(row["name"])
    results = [
        indexes[d].match(
            name,
            class_of(row),
            epi_isl=row.get("epi_isl") or "",
            reassortant=row.get("reassortant") or "",
            lab=row.get("table_lab") or "",
            passage=row.get("passage") or "",
        )
        for d in datasets
    ]
    found = [r for r in results if r.chosen is not None]
    if len(found) <= 1:
        result = found[0] if found else results[0]
    else:
        # a B antigen of unknown lineage matched in both B datasets: keep the first, flagged;
        # _match_rows makes it doubtful, since which lineage it is decides everything downstream
        result = replace(found[0], flags=(*found[0].flags, SEVERAL_DATASETS))
    if name != row["name"]:  # counted, so the rows keyed by a respelt name stay visible
        result = replace(result, flags=(*result.flags, NAME_NORMALISED))
    return result


_MATCH_COLUMNS: dict[str, pa.DataType] = {
    "table_id": pa.string(),
    "position": pa.int32(),
    "method": pa.string(),
    "epi_isl": pa.string(),
    "accession": pa.string(),
    "dataset": pa.string(),
    "flags": pa.list_(pa.string()),
    "doubtful": pa.bool_(),
    # a refused name tie: the candidates it was among and ae's rank-first (Q81); else empty
    "tied": pa.list_(pa.struct([("epi_isl", pa.string()), ("accession", pa.string())])),
    "ranked_epi_isl": pa.string(),
    "ranked_accession": pa.string(),
}


def _refuse_duplicates(con: Any, view: str, what: str) -> None:
    """Two rows for one sequence would double antigens in every count downstream."""
    row = con.execute(
        f"SELECT count(*) FROM (SELECT epi_isl, accession FROM {view} "
        "GROUP BY ALL HAVING count(*) > 1)"
    ).fetchone()
    if row and row[0]:
        raise StoreError(f"{what} have {row[0]} sequences with more than one row")


def _count(con: Any, counts: LinkCounts) -> None:
    counts.by_status = {status: 0 for status in STATUSES}
    for status, pairing, n in con.execute(
        "SELECT status, pairing, count(*) FROM antigen_sequences GROUP BY ALL ORDER BY ALL"
    ).fetchall():
        counts.by_status[status] += n
        counts.by_status_and_pairing[status, pairing] = n
    row = con.execute(
        """
        SELECT count(*) FILTER (WHERE NOT has_clade_row),
               count(*) FILTER (WHERE has_clade_row AND coalesce(clade, '') = '')
        FROM antigen_sequences WHERE status = 'matched'
        """
    ).fetchone()
    assert row is not None
    counts.matched_without_clade_row, counts.matched_with_empty_clade = int(row[0]), int(row[1])
    counts.matched_by_evidence = dict(
        con.execute(
            "SELECT clade_evidence, count(*) FROM antigen_sequences "
            "WHERE status = 'matched' AND clade_evidence IS NOT NULL GROUP BY ALL ORDER BY ALL"
        ).fetchall()
    )


def has_evidence_columns(path: Path) -> bool:
    """Whether a clades file records 04-clades' evidence (versions before 8 Oct 2026 do not)."""
    import duckdb

    columns = duckdb.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()
    return {name for name, *_ in columns} >= set(EVIDENCE_COLUMNS)


def _clade_rows(paths: Sequence[Path]) -> str:
    """Every clades file, with the evidence columns. A file written before them has none to
    give: its fallback rows read as having no evidence (none applies to them), but a tree row
    there would be a tree call nobody checked, read as if it had been. That is refused, not
    defaulted (design rule 4); no such file has been published.
    """
    import duckdb

    parts = []
    for path in paths:
        source = f"read_parquet('{Path(path).as_posix()}')"
        if has_evidence_columns(path):
            parts.append(f"SELECT * FROM {source}")
            continue
        (tree_rows,) = duckdb.execute(
            f"SELECT count(*) FROM {source} WHERE method = 'tree'"
        ).fetchone() or (0,)
        if tree_rows:
            raise StoreError(
                f"{path}: {tree_rows} tree-derived clade rows without clade_evidence: their "
                "calls were never checked against the sequence; relabel the clades table"
            )
        nulls = ", ".join(f"NULL::VARCHAR AS {c}" for c in EVIDENCE_COLUMNS)
        parts.append(f"SELECT *, {nulls} FROM {source}")
    return " UNION ALL BY NAME ".join(parts)


def _parquet(paths: Sequence[Path]) -> str:
    listing = ", ".join(f"'{Path(p).as_posix()}'" for p in paths)
    return f"read_parquet([{listing}], union_by_name = true)"


@dataclass(frozen=True)
class TiedSequence:
    """One candidate of a refused name tie, with its clade as the clade store gives it."""

    epi_isl: str
    accession: str
    clade: str | None  # None: no clade row; "": the nomenclature names none
    # 04-clades' evidence for a tree-derived clade (EVIDENCE_COLUMNS); None on fallback rows
    clade_evidence: str | None = None
    supported_clade: str | None = None
    clade_evidence_reason: str | None = None


@dataclass(frozen=True)
class PreparationSequence:
    """The one sequence a preparation's table rows point at, or why there is none."""

    epi_isl: str | None
    accession: str | None
    # the chosen sequence's clade: None = no clade row (unknown), "" = the nomenclature names
    # none (nothing to know). For a preparation's one clade, use preparation_clade().
    clade: str | None
    pairing: str  # "exact" if any row is an exact pairing, else "proxy" or ""
    conflict: bool  # its rows point at different sequences: none is chosen
    # No row matched, and its rows refused a name tie (Sarah, Q81): every candidate, and the
    # one ae's rank takes. Colouring uses them only when the candidates agree, or ranks.
    tied: tuple[TiedSequence, ...] = ()
    ranked: TiedSequence | None = None
    # The doubts (af.seq.matching.USABLE_DOUBTS) of the rows this came from; empty when a
    # clean match decided it. Colouring counts each (Sarah, Q81 D).
    doubts: tuple[str, ...] = ()
    # conflict: the records its rows name, and how one was chosen (Sarah, Q81, 30 Sep):
    # ROWS_PASSAGE_MATCHED (epi_isl/accession/clade are that record), or ROWS_NONE_MATCH /
    # ROWS_SEVERAL_MATCH (none chosen; colouring uses them only when they agree).
    alternatives: tuple[TiedSequence, ...] = ()
    resolution: str = ""
    # the sequence datasets (subtype rows: "bvic", ...) its matched rows found sequences in;
    # tells a B preparation of unknown lineage which lineage it is. Empty for ties.
    datasets: frozenset[str] = frozenset()
    # the chosen sequence's clade evidence (EVIDENCE_COLUMNS: recorded, not acted on here);
    # for ties and unresolved conflicts, each candidate carries its own
    clade_evidence: str | None = None
    supported_clade: str | None = None
    clade_evidence_reason: str | None = None


#: How a preparation whose rows name different records was resolved (Sarah, Q81, 30 Sep:
#: "Passage-matched record"). Passage stays part of antigen identity: these choose only which
#: of the records a preparation's own rows name colours it.
ROWS_PASSAGE_MATCHED = "rows.passage-matched"
ROWS_NONE_MATCH = "rows.agree.no-record-matches-passage"
ROWS_SEVERAL_MATCH = "rows.agree.several-records-match-passage"


PreparationKey = tuple[
    str, str, str, tuple[str, ...], str
]  # subtype, name, reass., annot., passage


def preparation_clade(found: PreparationSequence | None) -> str | None:
    """A preparation's one clade, decided as colouring decides it. Three answers, kept apart:

    - ``None``: **unknown**. No usable sequence; a sequence with no clade row; or candidates
      (a refused tie, rows naming two unresolved records) whose clades disagree, with no
      ranked pick;
    - ``""``: **known to have no clade**: the nomenclature names none for its sequence;
    - a clade name.

    A consumer must not collapse the first two: that reads a missing sequence as a virus
    outside every clade. With no chosen sequence, the candidates' common clade is the answer,
    else (a tie that splits) the clade of ae's ranked pick. Candidates are compared by clade,
    stricter than colouring's comparison by colour (two clades can share a colour row).
    ``af.clades`` turns the answer into a lineage (None stays None, "" becomes ()).
    """
    if found is None:
        return None
    if found.epi_isl is not None:
        return found.clade
    clades = {c.clade for c in found.tied or found.alternatives}
    if len(clades) == 1:
        return clades.pop()
    return found.ranked.clade if found.ranked is not None else None


def preparation_key(chart: Chart, kind: str, index: int) -> PreparationKey:
    """The serology preparation a chart point is: the key :func:`preparation_sequences` uses.

    The one copy of that key (Q46: maps and geo colour through the same join). A chart's
    passage ("P") is already the identity passage serology keys by: the lab's passage with its
    harvest date (:meth:`af.tables.model.Antigen.ae_passage`), so it is used as written. The
    subtype is the chart's own ("V"), the table subtype. Only antigens are preparations: a
    serum is not, and asking for one is an error.

    A chain merge marks a repeated sample ``DISTINCT`` so the chart keeps it apart
    (:mod:`af.chart.identity`); that is the chart's bookkeeping, not part of the preparation,
    and no serology row carries it. It is dropped, so the repeat is coloured as the
    preparation it is (the same virus, the same sequence) instead of finding no sequence.
    """
    if kind != "antigen":
        raise ValueError(f"preparation_key: {kind!r} points are not preparations (antigen only)")
    subtype = chart.info.get("V")
    if not subtype:
        raise ValueError("preparation_key: the chart has no subtype (info 'V')")
    if not 0 <= index < len(chart.antigens):
        raise IndexError(f"preparation_key: antigen {index} of {len(chart.antigens)}")
    a = chart.antigens[index]
    annotations = tuple(x for x in a.annotations if x != DISTINCT)
    return (str(subtype), a.name, a.reassortant, annotations, a.passage)


_PREP = "t.subtype, a.name, a.reassortant, a.annotations, a.identity_passage"
_ROWS = """FROM antigen_sequences s
        JOIN antigens a ON a.table_id = s.table_id AND a.position = s.position
        JOIN tables t ON t.table_id = s.table_id"""


def preparation_sequences(
    con: Any, passages: PassageMatcher
) -> dict[PreparationKey, PreparationSequence]:
    """For every preparation (as :func:`af.serology.query.preparations` groups them) that a
    sequence can colour, where it comes from. Needs the ``antigen_sequences`` view.

    In order, the first that applies:

    1. its ``matched`` rows (no doubt);
    2. its ``doubtful`` rows whose every doubt is one colouring accepts
       (:data:`af.seq.matching.USABLE_DOUBTS`: ae uses these; Sarah, Q81 D), with the doubts
       kept in ``doubts`` so they are counted. Any other doubt leaves the row out;
    3. its rows that refused a name tie (and have no other doubt, or only usable ones): no
       sequence, the ``tied`` candidates (the union over its rows), and as ``ranked`` the
       lowest by EPI_ISL number of its rows' ranked picks (Sarah, Q81).

    A preparation appears in many tables; normally every row names the same isolate. When
    rows name different GISAID records (two deposits of one virus, e.g. its S1 isolate and its
    S2 passage) the preparation is marked ``conflict``, and takes the record whose passage
    matches its own (``passages``, :class:`af.seq.passage_match.PassageMatcher`; Sarah, Q81):
    the one scoring highest, if it is the only one; otherwise none, and colouring falls back to
    agreement among them. Never one picked by table order. Passage stays part of the
    preparation's identity: this only chooses among the records its own rows name.
    """
    from af.seq.matching import AMBIGUOUS, DOUBTFUL, USABLE_DOUBTS

    refused = _sql_list(DOUBTFUL - USABLE_DOUBTS - {AMBIGUOUS})
    usable = _sql_list(USABLE_DOUBTS)
    no_refused_doubt = f"len(list_filter(s.flags, f -> f IN ({refused}))) = 0"
    out = _single_sequences(con, passages, "s.status = 'matched'")
    doubtful = _single_sequences(
        con,
        passages,
        f"s.status = 'doubtful' AND {no_refused_doubt} "
        f"AND NOT list_contains(s.flags, '{AMBIGUOUS}')",
        usable,
    )
    for key, found in doubtful.items():
        out.setdefault(key, found)
    tie_rows = (
        f"s.status = 'unmatched' AND list_contains(s.flags, '{AMBIGUOUS}') AND {no_refused_doubt}"
    )
    for key, tied in _tied_preparations(con, tie_rows, usable).items():
        out.setdefault(key, tied)
    return out


def _sql_list(flags: Collection[str]) -> str:
    return ", ".join(f"'{flag}'" for flag in sorted(flags)) or "NULL"


def _single_sequences(
    con: Any, passages: PassageMatcher, where: str, usable: str | None = None
) -> dict[PreparationKey, PreparationSequence]:
    """One sequence per preparation from the rows ``where`` selects, or a conflict."""
    doubts = (
        f"list_sort(list_distinct(flatten(list(list_filter(s.flags, f -> f IN ({usable}))))))"
        if usable
        else "[]::VARCHAR[]"
    )
    rows = con.execute(
        f"""
        SELECT {_PREP},
               list(DISTINCT struct_pack(epi := s.epi_isl, acc := s.accession, clade := s.clade,
                                         passage := coalesce(s.sequence_passage, ''),
                                         ev := s.clade_evidence, sup := s.supported_clade,
                                         why := s.clade_evidence_reason)) AS sequences,
               any_value(s.epi_isl), any_value(s.accession), any_value(s.clade),
               bool_or(s.pairing = 'exact'), bool_or(s.pairing = 'proxy'), {doubts},
               list(DISTINCT s.dataset) FILTER (WHERE s.dataset IS NOT NULL)
        {_ROWS}
        WHERE {where}
        GROUP BY {_PREP}
        """
    ).fetchall()
    out: dict[PreparationKey, PreparationSequence] = {}
    for subtype, name, reassortant, annots, passage, seqs, epi, acc, clade, ex, px, dts, ds in rows:
        key = (subtype, name, reassortant, tuple(annots), passage)
        pairing = "exact" if ex else "proxy" if px else ""
        datasets = frozenset(ds or ())
        if len({(r["epi"], r["acc"]) for r in seqs}) > 1:
            found = _resolve(key[4], seqs, passages, pairing, tuple(dts))
            out[key] = replace(found, datasets=datasets)
        else:
            only = seqs[0]  # one record (listed once per passage it was deposited with)
            out[key] = PreparationSequence(
                epi, acc, clade, pairing, conflict=False, doubts=tuple(dts), datasets=datasets,
                **_evidence(only["ev"], only["sup"], only["why"]),
            )  # fmt: skip
    return out


def _resolve(
    passage: str,
    records: list[dict[str, Any]],
    passages: PassageMatcher,
    pairing: str,
    doubts: tuple[str, ...],
) -> PreparationSequence:
    """A preparation whose rows name several records: the one whose passage matches, alone."""
    from af.seq.matching import epi_order

    alternatives = tuple(
        TiedSequence(r["epi"], r["acc"], r["clade"], **_evidence(r["ev"], r["sup"], r["why"]))
        for r in sorted(records, key=lambda r: (epi_order(r["epi"]), r["acc"]))
    )
    scores = {(r["epi"], r["acc"]): passages.score(passage, r["passage"]) for r in records}
    best = max(scores.values())
    top = [a for a in alternatives if scores[(a.epi_isl, a.accession)] == best]
    if best > 0 and len(top) == 1:
        chosen = top[0]
        return PreparationSequence(
            chosen.epi_isl, chosen.accession, chosen.clade, pairing, conflict=True,
            doubts=doubts, alternatives=alternatives, resolution=ROWS_PASSAGE_MATCHED,
            clade_evidence=chosen.clade_evidence, supported_clade=chosen.supported_clade,
            clade_evidence_reason=chosen.clade_evidence_reason,
        )  # fmt: skip
    return PreparationSequence(
        None, None, None, pairing, conflict=True, doubts=doubts, alternatives=alternatives,
        resolution=ROWS_NONE_MATCH if best == 0 else ROWS_SEVERAL_MATCH,
    )  # fmt: skip


def _tied_preparations(
    con: Any, where: str, usable: str
) -> dict[PreparationKey, PreparationSequence]:
    """Preparations whose rows (``where``) refused a name tie."""
    from af.seq.matching import epi_order

    rows = con.execute(
        f"""
        SELECT {_PREP}, s.tied, s.ranked_epi_isl, s.ranked_accession, s.pairing,
               list_filter(s.flags, f -> f IN ({usable}))
        {_ROWS}
        WHERE {where}
        """
    ).fetchall()
    clade_of = {
        (epi, acc): (clade, _evidence(ev, sup, why))
        for epi, acc, clade, ev, sup, why in con.execute(
            f"""
            SELECT c.epi_isl, c.accession,
                   CASE WHEN k.epi_isl IS NOT NULL THEN coalesce(k.clade, '') END,
                   k.clade_evidence, k.supported_clade, k.clade_evidence_reason
            FROM (SELECT DISTINCT unnest(s.tied, recursive := true)
                  FROM antigen_sequences s WHERE {where}) c
            LEFT JOIN clade_rows k ON k.epi_isl = c.epi_isl AND k.accession = c.accession
            """
        ).fetchall()
    }
    tied: dict[PreparationKey, set[tuple[str, str]]] = {}
    ranked: dict[PreparationKey, set[tuple[str, str]]] = {}
    pairings: dict[PreparationKey, set[str]] = {}
    doubts: dict[PreparationKey, set[str]] = {}
    for subtype, name, reassortant, annots, passage, cands, epi, acc, pairing, dts in rows:
        key = (subtype, name, reassortant, tuple(annots), passage)
        tied.setdefault(key, set()).update((c["epi_isl"], c["accession"]) for c in cands)
        if epi is not None:
            ranked.setdefault(key, set()).add((epi, acc))
        pairings.setdefault(key, set()).add(pairing)
        doubts.setdefault(key, set()).update(dts)
    out: dict[PreparationKey, PreparationSequence] = {}
    for key, pairs in tied.items():
        if not pairs:  # a refused tie must name its candidates; none to colour from otherwise
            continue
        order = sorted(pairs, key=lambda ea: (epi_order(ea[0]), ea[1]))
        seqs = tuple(
            TiedSequence(epi, acc, *_tied_clade(clade_of.get((epi, acc)))) for epi, acc in order
        )
        pick = min(ranked.get(key, ()), key=lambda ea: (epi_order(ea[0]), ea[1]), default=None)
        pairing = next((p for p in ("exact", "proxy") if p in pairings[key]), "")
        out[key] = PreparationSequence(
            None, None, None, pairing, conflict=False, tied=seqs,
            ranked=next((t for t in seqs if (t.epi_isl, t.accession) == pick), None),
            doubts=tuple(sorted(doubts[key])),
        )  # fmt: skip
    return out


def _evidence(evidence: str | None, supported: str | None, reason: str | None) -> dict[str, Any]:
    return {
        "clade_evidence": evidence,
        "supported_clade": supported,
        "clade_evidence_reason": reason,
    }


def _tied_clade(found: tuple[str | None, dict[str, Any]] | None) -> tuple[Any, ...]:
    """A tie candidate's clade and evidence; (None, no evidence) without a clade row."""
    if found is None:
        return (None,)
    clade, evidence = found
    return (clade, *evidence.values())


def link_from_store(
    con: Any,
    store: Any,
    rules: MatchingRules,
    *,
    with_clades: bool,
    class_of: ClassOf = passage_class_column,
    sequences: Mapping[str, StoreRef] | None = None,
    clades: Sequence[StoreRef] | None = None,
) -> LinkCounts:
    """:func:`link_sequences` over the ``sequences/*`` (and ``clades/*``) datasets.

    By default the CURRENT versions; ``sequences`` (dataset -> ref, one for every dataset the
    join reads) and ``clades`` (the clade dataset refs) pin them instead, so a result can be
    reproduced after CURRENT moves (design rule 5). Either way the versions read are returned
    in ``LinkCounts.refs``.

    ``with_clades=False`` is the deliberate sequences-only join; with ``True`` a missing
    clade dataset is an error, not an empty join. ``rules`` are the matcher's tables
    (:func:`af.seq.matching_rules.matching_rules`), already checked against the tables' lab
    codes; the checks here are the ones that need the store. Every submitter named must
    submit something in the sequence store, and the submitters must be keyed by labs the
    serology store holds, or the own-lab tie-break would silently never apply. Every
    location equivalent's GISAID spelling must be held by a stored sequence unless the row is
    marked optional (Sarah, Q44).
    """
    from af.seq.matching import (
        check_equivalents,
        check_lab_submitters,
        equivalents_table,
        index_from_store,
        sequences_ref,
    )

    datasets = sorted(subtypes().keys())
    if sequences is None:
        present = {ref.dataset for ref in store.list_datasets("sequences")}
        missing = [d for d in datasets if d not in present]
        if missing:
            raise StoreError(f"sequence datasets missing from the store: {', '.join(missing)}")
    read = {d: sequences_ref(store, d, sequences) for d in datasets}
    if rules.submitters:
        # a submitter name that no longer appears in the store would silently stop breaking ties
        check_lab_submitters(store, datasets, dict(rules.submitters), read)
        _check_submitter_labs(con, rules.submitters)
    if rules.equivalents:
        # a GISAID spelling no stored sequence has would silently do nothing
        check_equivalents(store, datasets, list(rules.equivalents), read)
    indexes = {d: index_from_store(store, [d], rules.passage, read) for d in datasets}
    for index in indexes.values():
        index.submitters = dict(rules.submitters)
        index.number_rules = dict(rules.number_rules)
        index.equivalents = equivalents_table(rules.equivalents)
    isolates = [
        path
        for d in datasets
        for path in sorted((store.resolve(read[d]) / "isolates").glob("*/*.parquet"))
    ]
    clade_paths = None
    clade_refs: list[StoreRef] = []
    behind: dict[str, tuple[str | None, str]] = {}
    same: dict[str, tuple[str, str]] = {}
    if with_clades:
        clade_refs = list(clades) if clades is not None else list(store.list_datasets("clades"))
        if not clade_refs:
            raise StoreError("no clades datasets given or in the store")
        clade_paths = [
            p for ref in clade_refs for p in sorted(store.resolve(ref).glob("*.parquet"))
        ]
        behind, same = _clades_behind(store, clade_refs, read)
    elif clades is not None:
        raise StoreError("clades pinned for a sequences-only join (with_clades=False)")
    counts = link_sequences(con, indexes, isolates, clade_paths, class_of)
    counts.clades_without_evidence_columns = sorted(
        ref.dataset
        for ref in clade_refs
        if not all(has_evidence_columns(p) for p in store.resolve(ref).glob("*.parquet"))
    )
    counts.clades_behind = behind
    counts.clades_same_content = same
    counts.refs = {
        "sequences": {d: ref.to_json() for d, ref in read.items()},
        "clades": [ref.to_json() for ref in clade_refs],
    }
    return counts


def _clades_behind(
    store: Any, refs: Sequence[Any], read: Mapping[str, StoreRef]
) -> tuple[dict[str, tuple[str | None, str]], dict[str, tuple[str, str]]]:
    """Clade tables behind the sequences the join read, and those only labelled from another
    version.

    A clade table covers the sequences its provenance names; sequences added since have no row
    until it is refreshed, and show up as "matched without clade row". This says why, per
    dataset, instead of leaving the count to be puzzled over. It compares content
    (:func:`af.clades.coverage.clades_behind_in_content`), not version labels: a sequences
    republish that changes nothing the clade calls read (a name fixed, a protein padded) leaves
    the table current, and a relabel would return the same table. Such tables are returned
    separately, with their version pair, so the difference stays visible.
    """
    behind: dict[str, tuple[str | None, str]] = {}
    same: dict[str, tuple[str, str]] = {}
    for ref in refs:
        sequences = (
            read[ref.dataset] if ref.dataset in read else store.current("sequences", ref.dataset)
        )
        try:
            content = clades_behind_in_content(store, ref, sequences)
        except CladeStoreError:
            # provenance names no sequences version of its own (or several): cannot be current
            behind[ref.dataset] = (None, sequences.version)
            continue
        pair = (content.labelled.version, sequences.version)
        if content.behind:
            behind[ref.dataset] = pair
        elif not content.same_version:
            same[ref.dataset] = pair
    return behind, same
