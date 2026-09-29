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

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pyarrow as pa

from af.seq.matching import Match, SequenceIndex
from af.seq.matching_rules import MatchingRules
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
    # clade dataset -> (sequences version it labelled, or None if its provenance names none,
    # current sequences version), for every clade table behind the sequence store
    clades_behind: dict[str, tuple[str | None, str]] = field(default_factory=dict)


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
               m.tied, m.ranked_epi_isl, m.ranked_accession,
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
            passage=row.get("passage") or "",
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


def _parquet(paths: Sequence[Path]) -> str:
    listing = ", ".join(f"'{Path(p).as_posix()}'" for p in paths)
    return f"read_parquet([{listing}], union_by_name = true)"


@dataclass(frozen=True)
class TiedSequence:
    """One candidate of a refused name tie, with its clade as the clade store gives it."""

    epi_isl: str
    accession: str
    clade: str | None  # None: no clade row; "": the nomenclature names none


@dataclass(frozen=True)
class PreparationSequence:
    """The one sequence a preparation's table rows point at, or why there is none."""

    epi_isl: str | None
    accession: str | None
    clade: str | None
    pairing: str  # "exact" if any row is an exact pairing, else "proxy" or ""
    conflict: bool  # its rows point at different sequences: none is chosen
    # No row matched, and its rows refused a name tie (Sarah, Q81): every candidate, and the
    # one ae's rank takes. Colouring uses them only when the candidates agree, or ranks.
    tied: tuple[TiedSequence, ...] = ()
    ranked: TiedSequence | None = None


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

    A preparation none of whose rows matched, but whose rows refused a name tie and have no
    other doubt, is included with no sequence and its ``tied`` candidates (Sarah, Q81): the
    union over its rows, and as ``ranked`` the lowest by EPI_ISL number of its rows' ranked
    picks (rows of one preparation normally rank the same candidate).
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
    for key, tied in _tied_preparations(con).items():
        out.setdefault(key, tied)
    return out


def _tied_preparations(con: Any) -> dict[PreparationKey, PreparationSequence]:
    """Preparations whose rows refused a name tie and carry no other doubt."""
    from af.seq.matching import AMBIGUOUS, DOUBTFUL, epi_order

    others = ", ".join(f"'{flag}'" for flag in sorted(DOUBTFUL - {AMBIGUOUS}))
    where = f"""s.status = 'unmatched' AND list_contains(s.flags, '{AMBIGUOUS}')
                AND len(list_filter(s.flags, f -> f IN ({others}))) = 0"""
    rows = con.execute(
        f"""
        SELECT t.subtype, a.name, a.reassortant, a.annotations, a.identity_passage,
               s.tied, s.ranked_epi_isl, s.ranked_accession, s.pairing
        FROM antigen_sequences s
        JOIN antigens a ON a.table_id = s.table_id AND a.position = s.position
        JOIN tables t ON t.table_id = s.table_id
        WHERE {where}
        """
    ).fetchall()
    clade_of = {
        (epi, acc): clade
        for epi, acc, clade in con.execute(
            f"""
            SELECT c.epi_isl, c.accession,
                   CASE WHEN k.epi_isl IS NOT NULL THEN coalesce(k.clade, '') END
            FROM (SELECT DISTINCT unnest(s.tied, recursive := true)
                  FROM antigen_sequences s WHERE {where}) c
            LEFT JOIN clade_rows k ON k.epi_isl = c.epi_isl AND k.accession = c.accession
            """
        ).fetchall()
    }
    tied: dict[PreparationKey, set[tuple[str, str]]] = {}
    ranked: dict[PreparationKey, set[tuple[str, str]]] = {}
    pairings: dict[PreparationKey, set[str]] = {}
    for subtype, name, reassortant, annots, passage, cands, epi, acc, pairing in rows:
        key = (subtype, name, reassortant, tuple(annots), passage)
        tied.setdefault(key, set()).update((c["epi_isl"], c["accession"]) for c in cands)
        if epi is not None:
            ranked.setdefault(key, set()).add((epi, acc))
        pairings.setdefault(key, set()).add(pairing)
    out: dict[PreparationKey, PreparationSequence] = {}
    for key, pairs in tied.items():
        order = sorted(pairs, key=lambda ea: (epi_order(ea[0]), ea[1]))
        seqs = tuple(TiedSequence(epi, acc, clade_of.get((epi, acc))) for epi, acc in order)
        pick = min(ranked.get(key, ()), key=lambda ea: (epi_order(ea[0]), ea[1]), default=None)
        pairing = next((p for p in ("exact", "proxy") if p in pairings[key]), "")
        out[key] = PreparationSequence(
            None, None, None, pairing, conflict=False, tied=seqs,
            ranked=next((t for t in seqs if (t.epi_isl, t.accession) == pick), None),
        )  # fmt: skip
    return out


def link_from_store(
    con: Any,
    store: Any,
    rules: MatchingRules,
    *,
    with_clades: bool,
    class_of: ClassOf = passage_class_column,
) -> LinkCounts:
    """:func:`link_sequences` over the CURRENT ``sequences/*`` (and ``clades/*``) datasets.

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
    )

    datasets = sorted({d for group in DATASETS_FOR.values() for d in group})
    present = {ref.dataset for ref in store.list_datasets("sequences")}
    missing = [d for d in datasets if d not in present]
    if missing:
        raise StoreError(f"sequence datasets missing from the store: {', '.join(missing)}")
    if rules.submitters:
        # a submitter name that no longer appears in the store would silently stop breaking ties
        check_lab_submitters(store, datasets, dict(rules.submitters))
        _check_submitter_labs(con, rules.submitters)
    if rules.equivalents:
        # a GISAID spelling no stored sequence has would silently do nothing
        check_equivalents(store, datasets, list(rules.equivalents))
    indexes = {d: index_from_store(store, [d], rules.passage) for d in datasets}
    for index in indexes.values():
        index.submitters = dict(rules.submitters)
        index.number_rules = dict(rules.number_rules)
        index.equivalents = equivalents_table(rules.equivalents)
    isolates = [
        path
        for d in datasets
        for path in sorted(
            (store.resolve(store.current("sequences", d)) / "isolates").glob("*/*.parquet")
        )
    ]
    clades = None
    behind: dict[str, tuple[str | None, str]] = {}
    if with_clades:
        refs = store.list_datasets("clades")
        if not refs:
            raise StoreError("no clades datasets in the store")
        clades = [p for ref in refs for p in sorted(store.resolve(ref).glob("*.parquet"))]
        behind = _clades_behind(store, refs)
    counts = link_sequences(con, indexes, isolates, clades, class_of)
    counts.clades_behind = behind
    return counts


def _clades_behind(store: Any, refs: Sequence[Any]) -> dict[str, tuple[str | None, str]]:
    """Clade tables labelled from an older sequences version than the current one.

    A clade table covers the sequences version its provenance names; sequences added since
    have no row until the clade table is refreshed, and show up as "matched without clade
    row". This says why, per dataset, instead of leaving the count to be puzzled over.
    """
    out: dict[str, tuple[str | None, str]] = {}
    for ref in refs:
        provenance = json.loads((store.resolve(ref) / "PROVENANCE.json").read_text())
        labelled = [
            item["store"]["version"]
            for item in provenance.get("inputs", [])
            if "store" in item and item["store"].get("kind") == "sequences"
            and item["store"].get("dataset") == ref.dataset
        ]  # fmt: skip
        current = store.current("sequences", ref.dataset).version
        version = labelled[0] if len(labelled) == 1 else None
        if version != current:
            out[ref.dataset] = (version, current)
    return out
