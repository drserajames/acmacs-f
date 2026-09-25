"""Joins to invented sequence (I3) and clade (I4) Parquet files."""

from pathlib import Path
from typing import Any

import duckdb
import pytest

from af.serology import query
from af.serology.joins import link_sequences, preparation_sequences
from af.serology.store import StoreError, build


def _parquet(con: duckdb.DuckDBPyConnection, path: Path, select: str) -> Path:
    con.execute(f"COPY ({select}) TO '{path.as_posix()}' (FORMAT parquet)")
    return path


@pytest.fixture
def con(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        """
        CREATE VIEW antigen_links AS SELECT * FROM (VALUES
            ('t1', 0, 'EPI_ISL_1', 'exact'),
            ('t1', 1, 'EPI_ISL_2', 'proxy'),
            ('t2', 0, 'EPI_ISL_1', 'proxy'),
            ('t2', 1, 'EPI_ISL_3', 'exact'),
            ('t2', 2, 'EPI_ISL_9', 'exact'),
            ('t2', 3, '', ''),
            ('t2', 4, 'EPI_ISL_4', 'exact')
        ) AS v(table_id, position, epi_isl, pairing)
        """
    )
    return con


def _inputs(con: duckdb.DuckDBPyConnection, tmp_path: Path) -> tuple[Path, Path]:
    isolates = _parquet(
        con,
        tmp_path / "isolates.parquet",
        """SELECT * FROM (VALUES
            ('EPI_ISL_1', 'ACC1', 'COUNTRY-A', 'REGION-1', 'PLACE-A', DATE '2021-01-02'),
            ('EPI_ISL_2', 'ACC2', 'COUNTRY-B', 'REGION-2', 'PLACE-B', DATE '2021-02-03'),
            ('EPI_ISL_3', 'ACC3a', 'COUNTRY-A', 'REGION-1', 'PLACE-C', DATE '2021-03-04'),
            ('EPI_ISL_3', 'ACC3b', 'COUNTRY-A', 'REGION-1', 'PLACE-C', DATE '2021-03-04'),
            ('EPI_ISL_4', 'ACC4', 'COUNTRY-C', 'REGION-3', 'PLACE-D', DATE '2021-04-05')
        ) AS v(epi_isl, accession, country, region, place, collection_date)""",
    )
    clades = _parquet(
        con,
        tmp_path / "assignments.parquet",
        """SELECT * FROM (VALUES
            ('EPI_ISL_1', 'ACC1', 'CLADE-X', 'tree'),
            ('EPI_ISL_2', 'ACC2', '', 'tree')
        ) AS v(epi_isl, accession, clade, method)""",
    )
    return isolates, clades


def test_every_antigen_gets_one_status(con: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
    isolates, clades = _inputs(con, tmp_path)
    counts = link_sequences(con, [isolates], [clades])
    assert counts.by_status == {"matched": 4, "ambiguous": 1, "not_in_store": 1, "no_link": 1}
    assert counts.by_status_and_pairing[("matched", "proxy")] == 2
    # EPI_ISL_4 has a sequence but no clade row; EPI_ISL_2's clade is empty (unnamed)
    assert counts.matched_without_clade_row == 1
    assert counts.matched_with_empty_clade == 1
    rows = {
        (t, p): (status, acc, clade, place)
        for t, p, status, acc, clade, place in con.execute(
            "SELECT table_id, position, status, accession, clade, place FROM antigen_sequences"
        ).fetchall()
    }
    assert rows["t1", 0] == ("matched", "ACC1", "CLADE-X", "PLACE-A")
    assert rows["t2", 1] == ("ambiguous", None, None, None)  # two accessions: none chosen
    assert rows["t2", 2][0] == "not_in_store" and rows["t2", 3][0] == "no_link"
    assert len(rows) == 7  # one row per antigen link, never duplicated by the joins


def test_missing_inputs_and_duplicate_clade_rows_are_refused(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    isolates, clades = _inputs(con, tmp_path)
    with pytest.raises(StoreError, match="no sequence isolates"):
        link_sequences(con, [], [clades])
    with pytest.raises(StoreError, match="no clade assignments"):
        link_sequences(con, [isolates], [])
    doubled = _parquet(
        con,
        tmp_path / "doubled.parquet",
        f"SELECT * FROM read_parquet('{clades.as_posix()}') UNION ALL "
        f"SELECT * FROM read_parquet('{clades.as_posix()}')",
    )
    with pytest.raises(StoreError, match="more than one row"):
        link_sequences(con, [isolates], [doubled])


def test_preparation_sequences_agree_or_conflict(tmp_path: Path, syn: Any) -> None:
    """One preparation in two tables naming one sequence gets it; naming two gets none."""
    same = {"name": syn.virus("Somewhere", 1), "passage": "MDCK1", "date": "2021-01-05"}
    split = {"name": syn.virus("Somewhere", 2), "passage": "SIAT1", "date": "2021-01-06"}
    serum = {"name": syn.virus("Elsewhere", 3), "serum_id": "S-1"}
    tables = [
        syn.table("t1", [same, split], [serum], [[["80"]], [["40"]]]),
        syn.table("t2", [same, split], [serum], [[["160"]], [["40"]]], date="2021-03-05"),
    ]
    build(tables, tmp_path / "v1", syn.rules)
    con = query.connect(tmp_path / "v1")
    con.execute(
        """CREATE VIEW antigen_links AS SELECT * FROM (VALUES
            ('t1', 0, 'EPI_ISL_1', 'proxy'), ('t2', 0, 'EPI_ISL_1', 'exact'),
            ('t1', 1, 'EPI_ISL_2', 'exact'), ('t2', 1, 'EPI_ISL_4', 'exact')
        ) AS v(table_id, position, epi_isl, pairing)"""
    )
    isolates, clades = _inputs(con, tmp_path)
    link_sequences(con, [isolates], [clades])
    got = {key[1]: value for key, value in preparation_sequences(con).items()}
    agreed = got[syn.virus("Somewhere", 1)]
    assert (agreed.accession, agreed.clade, agreed.pairing) == ("ACC1", "CLADE-X", "exact")
    assert not agreed.conflict
    assert got[syn.virus("Somewhere", 2)].conflict
