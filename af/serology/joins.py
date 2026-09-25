"""Join serology antigens to their sequences, clades and locations.

Where a lab states which sequenced isolate an antigen is (CDC gives an EPI_ISL for 99% of
its antigens), the join is on that id, never on the name: a name match can pick the wrong
passage or a different virus with a similar name. Name matching for labs that give no id
belongs to the sequence workstream and plugs in as further rows of ``antigen_links``.

EPI_ISL alone is not unique in GISAID (one isolate can have several HA accessions), so an
antigen is matched only when its EPI_ISL has exactly one HA accession in the sequence
store. Every antigen gets exactly one status, and each status is counted, so nothing is
dropped silently:

- ``matched``: one accession; the clade and location come with it;
- ``ambiguous``: several accessions for the EPI_ISL; none is chosen;
- ``not_in_store``: the EPI_ISL is not in the sequence store (for example a submission
  older than the pulls);
- ``no_link``: the lab gave no EPI_ISL.

The lab's own pairing is carried through: ``proxy`` means the lab paired the antigen with
a related isolate's sequence (a different passage or harvest), which makes the clade a
likely but not certain property of the tested virus. Consumers decide whether to use
proxies; the counts say how many there are.

Inputs, as agreed between workstreams:

- ``antigen_links`` (a view the caller defines): ``table_id, position, epi_isl, pairing``;
- the sequence store's ``isolates.parquet`` (I3): ``epi_isl, accession, country, region,
  place, collection_date``, and more;
- the clade store's ``assignments.parquet`` (I4): ``epi_isl, accession, clade, method``.
  An empty ``clade`` means the nomenclature names no clade there. A matched sequence with
  no assignment row at all is a separate count, because it means the clade store is behind
  the sequence store.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.serology.store import StoreError

STATUSES = ("matched", "ambiguous", "not_in_store", "no_link")


@dataclass
class LinkCounts:
    by_status: dict[str, int] = field(default_factory=dict)
    by_status_and_pairing: dict[tuple[str, str], int] = field(default_factory=dict)
    matched_without_clade_row: int = 0
    matched_with_empty_clade: int = 0


def link_sequences(con: Any, isolates: Sequence[Path], clades: Sequence[Path]) -> LinkCounts:
    """Define the view ``antigen_sequences`` (one row per antigen link) and count it."""
    if not isolates:
        raise StoreError("no sequence isolates given: cannot join antigens to sequences")
    if not clades:
        raise StoreError("no clade assignments given: cannot join antigens to clades")
    con.execute(f"CREATE OR REPLACE VIEW isolates AS SELECT * FROM {_parquet(isolates)}")
    con.execute(f"CREATE OR REPLACE VIEW clade_rows AS SELECT * FROM {_parquet(clades)}")
    _refuse_duplicate_clade_rows(con)
    con.execute(
        """
        CREATE OR REPLACE VIEW antigen_sequences AS
        WITH candidates AS (
            SELECT l.table_id, l.position, count(i.accession) AS n
            FROM antigen_links l LEFT JOIN isolates i ON i.epi_isl = l.epi_isl
            GROUP BY l.table_id, l.position
        )
        SELECT l.table_id, l.position, l.epi_isl, coalesce(l.pairing, '') AS pairing,
               CASE WHEN coalesce(l.epi_isl, '') = '' THEN 'no_link'
                    WHEN c.n = 0 THEN 'not_in_store'
                    WHEN c.n > 1 THEN 'ambiguous'
                    ELSE 'matched' END AS status,
               i.accession, i.country, i.region, i.place,
               i.collection_date AS sequence_collection_date,
               k.clade, k.method AS clade_method,
               k.epi_isl IS NOT NULL AS has_clade_row
        FROM antigen_links l
        JOIN candidates c USING (table_id, position)
        LEFT JOIN isolates i ON c.n = 1 AND i.epi_isl = l.epi_isl
        LEFT JOIN clade_rows k ON k.epi_isl = i.epi_isl AND k.accession = i.accession
        """
    )
    return _counts(con)


def _refuse_duplicate_clade_rows(con: Any) -> None:
    """Two clade rows for one sequence would double antigens in every count downstream."""
    row = con.execute(
        "SELECT count(*) FROM (SELECT epi_isl, accession FROM clade_rows "
        "GROUP BY ALL HAVING count(*) > 1)"
    ).fetchone()
    if row and row[0]:
        raise StoreError(f"clade assignments have {row[0]} sequences with more than one row")


def _counts(con: Any) -> LinkCounts:
    counts = LinkCounts(by_status={status: 0 for status in STATUSES})
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
    return counts


def _parquet(paths: Sequence[Path]) -> str:
    listing = ", ".join(f"'{Path(p).as_posix()}'" for p in paths)
    return f"read_parquet([{listing}], union_by_name = true)"
