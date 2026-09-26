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

from af.seq.matching import Match, SequenceIndex
from af.serology.store import StoreError

STATUSES = ("matched", "doubtful", "unmatched")
SEVERAL_DATASETS = "serology.matched-in-several-datasets"
NO_DATASET = "serology.no-sequence-dataset"

#: Which sequence datasets an antigen is matched within, by (subtype, lineage). A B antigen
#: of unknown lineage is matched across both B datasets.
DATASETS_FOR: dict[tuple[str, str], tuple[str, ...]] = {
    ("A(H1N1)", ""): ("h1",),
    ("A(H3N2)", ""): ("h3",),
    ("B", "VICTORIA"): ("bvic",),
    ("B", "YAMAGATA"): ("byam",),
    ("B", ""): ("bvic", "byam"),
}

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
            "NULL::VARCHAR AS accession, NULL::VARCHAR AS clade, NULL::VARCHAR AS method "
            "WHERE false"
        )
    else:
        con.execute(f"CREATE OR REPLACE VIEW clade_rows AS SELECT * FROM {_parquet(clades)}")
    _refuse_duplicates(con, "isolates", "sequence isolates")
    _refuse_duplicates(con, "clade_rows", "clade assignments")
    con.execute(
        """
        CREATE OR REPLACE VIEW antigen_sequences AS
        SELECT m.table_id, m.position, m.method, m.flags, m.doubtful, m.epi_isl, m.accession,
               m.dataset, coalesce(a.sequence_pairing, '') AS pairing,
               CASE WHEN m.accession IS NULL THEN 'unmatched'
                    WHEN m.doubtful THEN 'doubtful' ELSE 'matched' END AS status,
               i.country, i.region, i.place,
               i.collection_date AS sequence_collection_date,
               -- the clade store writes NULL where the nomenclature names no clade; here that
               -- is '' so NULL keeps one meaning downstream: no assignment row at all
               CASE WHEN k.epi_isl IS NOT NULL THEN coalesce(k.clade, '') END AS clade,
               k.method AS clade_method,
               k.epi_isl IS NOT NULL AS has_clade_row
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
    table = pa.table({k: pa.array(v, type=_MATCH_COLUMNS[k]) for k, v in out.items()})
    con.register("antigen_matches_arrow", table)
    con.execute("CREATE OR REPLACE TABLE antigen_matches AS SELECT * FROM antigen_matches_arrow")
    con.unregister("antigen_matches_arrow")
    counts.by_flag = dict(sorted(flags_seen.items()))
    counts.by_method = dict(sorted(methods.items()))


def _match_row(
    row: Mapping[str, Any], indexes: Mapping[str, SequenceIndex], class_of: ClassOf
) -> Match | None:
    """The matcher's answer for one antigen row; None if no dataset covers its subtype."""
    datasets = DATASETS_FOR.get((row["subtype"], row.get("lineage") or ""))
    if not datasets or any(d not in indexes for d in datasets):
        return None
    results = [
        indexes[d].match(
            row["name"],
            class_of(row),
            epi_isl=row.get("epi_isl") or "",
            reassortant=row.get("reassortant") or "",
            lab=row.get("table_lab") or "",
        )
        for d in datasets
    ]
    found = [r for r in results if r.chosen is not None]
    if len(found) <= 1:
        return found[0] if found else results[0]
    # a B antigen of unknown lineage matched in both B datasets: keep the first, flagged;
    # _match_rows makes it doubtful, since which lineage it is decides everything downstream
    return replace(found[0], flags=(*found[0].flags, SEVERAL_DATASETS))


_MATCH_COLUMNS: dict[str, pa.DataType] = {
    "table_id": pa.string(),
    "position": pa.int32(),
    "method": pa.string(),
    "epi_isl": pa.string(),
    "accession": pa.string(),
    "dataset": pa.string(),
    "flags": pa.list_(pa.string()),
    "doubtful": pa.bool_(),
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


def _parquet(paths: Sequence[Path]) -> str:
    listing = ", ".join(f"'{Path(p).as_posix()}'" for p in paths)
    return f"read_parquet([{listing}], union_by_name = true)"


@dataclass(frozen=True)
class PreparationSequence:
    """The one sequence a preparation's table rows point at, or why there is none."""

    epi_isl: str | None
    accession: str | None
    clade: str | None
    pairing: str  # "exact" if any row is an exact pairing, else "proxy" or ""
    conflict: bool  # its rows point at different sequences: none is chosen


PreparationKey = tuple[
    str, str, str, tuple[str, ...], str
]  # subtype, name, reass., annot., passage


def preparation_sequences(con: Any) -> dict[PreparationKey, PreparationSequence]:
    """For every preparation (as :func:`af.serology.query.preparations` groups them) with at
    least one ``matched`` row, its sequence. Doubtful rows are left out: unreviewed, they
    must not decide anything. Needs the ``antigen_sequences`` view.

    A preparation appears in many tables; normally every row names the same isolate. When
    rows name different sequences the preparation is marked ``conflict`` and gets none,
    rather than one picked by table order.
    """
    rows = con.execute(
        """
        SELECT t.subtype, a.name, a.reassortant, a.annotations, a.identity_passage,
               list(DISTINCT s.epi_isl || '|' || s.accession) AS sequences,
               any_value(s.epi_isl), any_value(s.accession), any_value(s.clade),
               bool_or(s.pairing = 'exact'), bool_or(s.pairing = 'proxy')
        FROM antigen_sequences s
        JOIN antigens a ON a.table_id = s.table_id AND a.position = s.position
        JOIN tables t ON t.table_id = s.table_id
        WHERE s.status = 'matched'
        GROUP BY t.subtype, a.name, a.reassortant, a.annotations, a.identity_passage
        """
    ).fetchall()
    out: dict[PreparationKey, PreparationSequence] = {}
    for subtype, name, reassortant, annots, passage, seqs, epi, acc, clade, ex, px in rows:
        key = (subtype, name, reassortant, tuple(annots), passage)
        pairing = "exact" if ex else "proxy" if px else ""
        if len(seqs) > 1:
            out[key] = PreparationSequence(None, None, None, pairing, conflict=True)
        else:
            out[key] = PreparationSequence(epi, acc, clade, pairing, conflict=False)
    return out


def link_from_store(
    con: Any,
    store: Any,
    passage_rules: Sequence[Any],
    *,
    with_clades: bool,
    lab_submitters: Mapping[str, frozenset[str]] | None = None,
    number_rules: Mapping[str, Any] | None = None,
    lab_codes: Collection[str] | None = None,
    class_of: ClassOf = passage_class_column,
) -> LinkCounts:
    """:func:`link_sequences` over the CURRENT ``sequences/*`` (and ``clades/*``) datasets.

    ``with_clades=False`` is the deliberate sequences-only join; with ``True`` a missing
    clade dataset is an error, not an empty join. ``lab_submitters`` (lab -> the exact
    GISAID submitting-lab names, :func:`af.seq.matching.read_lab_submitters`) lets the
    matcher settle a name tie by the antigen's own lab; without it that rule never applies.
    ``number_rules`` (lab -> :class:`af.seq.matching.NumberRule`, from config) let a lab's
    antigens match its own deposits by isolate number when the name does not. Both tables
    are keyed by the tables' own lab codes, exactly, checked against ``lab_codes`` (every lab
    code the table readers use, from config) with :func:`af.seq.matching.check_lab_codes`.
    ``lab_codes`` is required with either table: the labs merely present in the store would
    refuse a lab whose tables have not arrived yet.
    """
    from af.seq.matching import check_lab_codes, check_lab_submitters, index_from_store

    # configuration first, before the store is read: a bad rule table fails fast
    if (lab_submitters is not None or number_rules is not None) and lab_codes is None:
        raise StoreError("lab_codes (from config) are needed to check lab_submitters/number_rules")
    if lab_submitters is not None:
        assert lab_codes is not None
        check_lab_codes(dict(lab_submitters), lab_codes, "lab_submitters")
    if number_rules is not None:
        assert lab_codes is not None
        check_lab_codes(dict(number_rules), lab_codes, "number_rules")
    datasets = sorted({d for group in DATASETS_FOR.values() for d in group})
    present = {ref.dataset for ref in store.list_datasets("sequences")}
    missing = [d for d in datasets if d not in present]
    if missing:
        raise StoreError(f"sequence datasets missing from the store: {', '.join(missing)}")
    if lab_submitters is not None:
        # a submitter name that no longer appears in the store would silently stop breaking ties
        check_lab_submitters(store, datasets, dict(lab_submitters))
        _check_submitter_labs(con, lab_submitters)
    indexes = {d: index_from_store(store, [d], passage_rules) for d in datasets}
    for index in indexes.values():
        index.submitters = dict(lab_submitters or {})
        index.number_rules = dict(number_rules or {})
    isolates = [
        path
        for d in datasets
        for path in sorted(
            (store.resolve(store.current("sequences", d)) / "isolates").glob("*/*.parquet")
        )
    ]
    clades = None
    if with_clades:
        refs = store.list_datasets("clades")
        if not refs:
            raise StoreError("no clades datasets in the store")
        clades = [p for ref in refs for p in sorted(store.resolve(ref).glob("*.parquet"))]
    return link_sequences(con, indexes, isolates, clades, class_of)
