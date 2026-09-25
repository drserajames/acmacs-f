"""Joins through af.seq.matching to invented sequences (I3) and clades (I4)."""

from pathlib import Path
from typing import Any

import duckdb
import pytest

from af.seq.matching import Candidate, SequenceIndex
from af.serology import query
from af.serology.joins import SEVERAL_DATASETS, link_sequences, preparation_sequences
from af.serology.store import StoreError, build


def _parquet(path: Path, select: str) -> Path:
    duckdb.execute(f"COPY ({select}) TO '{path.as_posix()}' (FORMAT parquet)")
    return path


def _candidate(
    epi: str, acc: str, name: str, kind: str, seq: str, dataset: str = "h3"
) -> Candidate:
    return Candidate(epi, acc, dataset, name, "P", kind, seq)


def gisaid(prefix: str, place: str, number: int) -> str:
    """An invented GISAID-style name (bare type prefix), assembled so no strain-shaped text
    is committed."""
    return "/".join([prefix, place, str(number), "2021"])


def _isolates_and_clades(tmp_path: Path) -> tuple[Path, Path]:
    isolates = _parquet(
        tmp_path / "isolates.parquet",
        """SELECT * FROM (VALUES
            ('EPI_ISL_1', 'ACC1', 'COUNTRY-A', 'REGION-1', 'PLACE-A', '2021-01-02'),
            ('EPI_ISL_2', 'ACC2', 'COUNTRY-B', 'REGION-2', 'PLACE-B', '2021-02-03'),
            ('EPI_ISL_3', 'ACC3', 'COUNTRY-A', 'REGION-1', 'PLACE-C', '2021-03-04')
        ) AS v(epi_isl, accession, country, region, place, collection_date)""",
    )
    clades = _parquet(
        tmp_path / "assignments.parquet",
        """SELECT * FROM (VALUES
            ('EPI_ISL_1', 'ACC1', 'CLADE-X', 'tree'),
            ('EPI_ISL_3', 'ACC3', NULL, 'tree')
        ) AS v(epi_isl, accession, clade, method)""",
    )
    return isolates, clades


def _store(tmp_path: Path, syn: Any) -> Any:
    def antigen(place: str, number: int, epi: str = "") -> dict[str, Any]:
        return {"name": syn.virus(place, number), "passage": "MDCK1", "date": "2021-01-05",
                "epi_isl": epi, "sequence_pairing": "exact" if epi else ""}  # fmt: skip

    antigens = [
        antigen("SOMEWHERE", 1, "EPI_ISL_1"),  # the lab's EPI_ISL: matched
        antigen("SOMEWHERE", 2, "EPI_ISL_2"),  # EPI_ISL whose name differs: doubtful
        antigen("ELSEWHERE", 3),  # by name, cell antigen, one cell sequence: matched
        antigen("ELSEWHERE", 4),  # by name, two different cell sequences: unmatched (tie)
        antigen("NOWHERE", 5),  # no sequence at all: unmatched
    ]
    serum = {"name": syn.virus("SOMEWHERE", 9), "serum_id": "S-1"}
    tables = [syn.table("t1", antigens, [serum], [[["40"]] for _ in antigens])]
    build(tables, tmp_path / "v1", syn.rules)
    return query.connect(tmp_path / "v1")


def _index() -> SequenceIndex:
    index = SequenceIndex()
    index.add(_candidate("EPI_ISL_1", "ACC1", gisaid("A", "SOMEWHERE", 1), "cell", "s1"))
    index.add(_candidate("EPI_ISL_2", "ACC2", gisaid("A", "OTHERPLACE", 2), "cell", "s2"))
    index.add(_candidate("EPI_ISL_3", "ACC3", gisaid("A", "ELSEWHERE", 3), "cell", "s3"))
    index.add(_candidate("EPI_ISL_4", "ACC4", gisaid("A", "ELSEWHERE", 4), "cell", "s4a"))
    index.add(_candidate("EPI_ISL_5", "ACC5", gisaid("A", "ELSEWHERE", 4), "cell", "s4b"))
    return index


def cell(row: Any) -> str:
    return "cell"


def test_every_antigen_gets_one_status(tmp_path: Path, syn: Any) -> None:
    con = _store(tmp_path, syn)
    isolates, clades = _isolates_and_clades(tmp_path)
    counts = link_sequences(con, {"h3": _index()}, [isolates], [clades], class_of=cell)
    assert counts.by_status == {"matched": 2, "doubtful": 1, "unmatched": 2}
    assert counts.by_method == {"epi_isl": 2, "name": 2, "none": 1}  # NOWHERE: no candidate
    assert counts.by_flag["match.epi-name-differs"] == 1
    assert counts.by_flag["match.ambiguous"] == 1
    # ELSEWHERE/3: the clade store's NULL (the nomenclature names none) becomes ''
    assert counts.matched_with_empty_clade == 1
    assert counts.matched_without_clade_row == 0
    rows = {
        position: (status, accession, clade, place)
        for position, status, accession, clade, place in con.execute(
            "SELECT position, status, accession, clade, place FROM antigen_sequences"
        ).fetchall()
    }
    assert rows[0] == ("matched", "ACC1", "CLADE-X", "PLACE-A")
    assert rows[1][0] == "doubtful"
    assert rows[2] == ("matched", "ACC3", "", "PLACE-C")
    assert rows[3] == ("unmatched", None, None, None)
    assert rows[4] == ("unmatched", None, None, None)
    assert len(rows) == 5  # one row per antigen, never duplicated by the joins
    # doubtful rows never decide a preparation's sequence
    preps = {key[1]: value for key, value in preparation_sequences(con).items()}
    assert set(preps) == {syn.virus("SOMEWHERE", 1), syn.virus("ELSEWHERE", 3)}


def test_b_antigen_of_unknown_lineage_found_in_both_datasets_is_doubtful(
    tmp_path: Path, syn: Any
) -> None:
    antigen = {"name": syn.virus("SOMEWHERE", 1, prefix="B"), "passage": "MDCK1"}
    serum = {"name": syn.virus("SOMEWHERE", 9, prefix="B"), "serum_id": "S-1"}
    table = syn.table("b1", [antigen], [serum], [[["40"]]], subtype="B")
    build([table], tmp_path / "v", syn.rules)
    con = query.connect(tmp_path / "v")
    indexes = {"bvic": SequenceIndex(), "byam": SequenceIndex()}
    indexes["bvic"].add(
        _candidate("EPI_ISL_7", "ACC7", gisaid("B", "SOMEWHERE", 1), "cell", "v", "bvic")
    )
    indexes["byam"].add(
        _candidate("EPI_ISL_8", "ACC8", gisaid("B", "SOMEWHERE", 1), "cell", "y", "byam")
    )
    isolates, _ = _isolates_and_clades(tmp_path)
    counts = link_sequences(con, indexes, [isolates], None, class_of=cell)
    assert counts.by_status["doubtful"] == 1
    assert counts.by_flag[SEVERAL_DATASETS] == 1
    assert not counts.clades_joined


def test_missing_inputs_and_duplicate_rows_are_refused(tmp_path: Path, syn: Any) -> None:
    con = _store(tmp_path, syn)
    isolates, clades = _isolates_and_clades(tmp_path)
    with pytest.raises(StoreError, match="no sequence isolates"):
        link_sequences(con, {"h3": _index()}, [], [clades], class_of=cell)
    with pytest.raises(StoreError, match="no clade assignments"):
        link_sequences(con, {"h3": _index()}, [isolates], [], class_of=cell)
    doubled = _parquet(
        tmp_path / "doubled.parquet",
        f"SELECT * FROM read_parquet('{clades.as_posix()}') UNION ALL "
        f"SELECT * FROM read_parquet('{clades.as_posix()}')",
    )
    with pytest.raises(StoreError, match="more than one row"):
        link_sequences(con, {"h3": _index()}, [isolates], [doubled], class_of=cell)
