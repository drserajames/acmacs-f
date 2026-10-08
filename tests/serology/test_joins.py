"""Joins through af.seq.matching to invented sequences (I3) and clades (I4)."""

from pathlib import Path
from typing import Any

import duckdb
import pytest

from af.seq.matching import Candidate, SequenceIndex
from af.serology import query
from af.serology.joins import (
    SEVERAL_DATASETS,
    link_sequences,
    passage_class_column,
    preparation_sequences,
)
from af.serology.store import StoreError, build
from tests.seq.test_matching_rules import passage_matcher


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
            ('EPI_ISL_1', 'ACC1', 'COUNTRY-A', 'REGION-1', 'PLACE-A', '2021-01-02', 'MDCK1'),
            ('EPI_ISL_2', 'ACC2', 'COUNTRY-B', 'REGION-2', 'PLACE-B', '2021-02-03', 'MDCK1'),
            ('EPI_ISL_3', 'ACC3', 'COUNTRY-A', 'REGION-1', 'PLACE-C', '2021-03-04', 'MDCK1')
        ) AS v(epi_isl, accession, country, region, place, collection_date, passage)""",
    )
    clades = _parquet(
        tmp_path / "assignments.parquet",
        """SELECT * FROM (VALUES
            ('EPI_ISL_1', 'ACC1', 'CLADE-X', 'fallback'),
            ('EPI_ISL_3', 'ACC3', NULL, 'fallback')
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
    preps = {
        key[1]: value
        for key, value in preparation_sequences(con, passage_matcher(tmp_path)).items()
    }
    with_sequence = {name for name, p in preps.items() if p.epi_isl is not None}
    assert with_sequence == {
        syn.virus("SOMEWHERE", 1), syn.virus("SOMEWHERE", 2), syn.virus("ELSEWHERE", 3)
    }  # fmt: skip
    # a clean match carries no doubt; the doubtful one ae would use carries its flag (Q81 D)
    assert preps[syn.virus("SOMEWHERE", 1)].doubts == ()
    assert preps[syn.virus("SOMEWHERE", 2)].doubts == ("match.epi-name-differs",)
    # the refused tie (ELSEWHERE/4) chooses nothing, but keeps its candidates for colouring
    # (Q81); both rank alike on passage, so ae's rank falls to the lower EPI_ISL number
    tie = preps[syn.virus("ELSEWHERE", 4)]
    assert tie.epi_isl is None and not tie.conflict
    assert [(t.epi_isl, t.accession) for t in tie.tied] == [
        ("EPI_ISL_4", "ACC4"),
        ("EPI_ISL_5", "ACC5"),
    ]
    assert tie.ranked is not None and tie.ranked.epi_isl == "EPI_ISL_4"
    assert set(preps) == with_sequence | {syn.virus("ELSEWHERE", 4)}  # NOWHERE/5: nothing


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


def test_a_name_tie_is_settled_by_the_antigens_own_lab(tmp_path: Path, syn: Any) -> None:
    """ELSEWHERE/4 has two different cell sequences; only one was submitted by LABX."""
    con = _store(tmp_path, syn)
    isolates, clades = _isolates_and_clades(tmp_path)
    index = SequenceIndex()
    for epi, acc, seq, submitter in (
        ("EPI_ISL_4", "ACC4", "s4a", "Lab X Institute"),
        ("EPI_ISL_5", "ACC5", "s4b", "Another Institute"),
    ):
        index.add(Candidate(epi, acc, "h3", gisaid("A", "ELSEWHERE", 4), "P", "cell", seq,
                            submitter))  # fmt: skip
    index.submitters = {"LABX": frozenset({"Lab X Institute"})}
    counts = link_sequences(con, {"h3": index}, [isolates], [clades], class_of=cell)
    assert counts.by_flag["match.own-lab"] == 1
    row = con.execute(
        "SELECT status, accession FROM antigen_sequences WHERE position = 3"
    ).fetchone()
    assert row == ("matched", "ACC4")


def test_submitters_keyed_by_another_spelling_of_the_labs_are_refused(
    tmp_path: Path, syn: Any
) -> None:
    from af.serology.joins import _check_submitter_labs

    con = _store(tmp_path, syn)  # its table's lab is LABX
    _check_submitter_labs(con, {"LABX": frozenset({"Lab X Institute"})})
    with pytest.raises(StoreError, match="name none of the store's table labs"):
        _check_submitter_labs(con, {"labx": frozenset({"Lab X Institute"})})


def test_doubtful_matches_ae_uses_are_kept_with_their_flag_and_others_left_out(
    tmp_path: Path, syn: Any
) -> None:
    """Sarah, Q81 D: colour through the doubtful matches ae uses (egg antigen with only a cell
    sequence, reassortant, EPI_ISL whose name differs), flagged; never other doubts, and a
    clean row of the same preparation always wins."""

    def antigen(place: str, number: int, passage: str, **extra: Any) -> dict[str, Any]:
        klass = "egg" if passage.startswith("E") else "cell"
        return {"name": syn.virus(place, number), "passage": passage, "passage_class": klass,
                **extra}  # fmt: skip

    serum = {"name": syn.virus("SOMEWHERE", 9), "serum_id": "S-1"}
    first = [
        antigen("EGGTOWN", 1, "E3"),  # only a cell sequence: doubtful, usable
        antigen("REASTOWN", 2, "MDCK1", reassortant="NIB-1"),  # reassortant: doubtful, usable
        antigen("SPLITTOWN", 3, "MDCK1", epi_isl="EPI_ISL_30"),  # several accessions: refused
        antigen("BOTHTOWN", 4, "MDCK1", epi_isl="EPI_ISL_41"),  # doubtful here (name differs)
    ]
    later = [antigen("BOTHTOWN", 4, "MDCK1")]  # ...and clean by name in a later table
    tables = [
        syn.table("t1", first, [serum], [[["40"]] for _ in first]),
        syn.table("t2", later, [serum], [[["40"]]], date="2021-04-01"),
    ]
    build(tables, tmp_path / "v", syn.rules)
    con = query.connect(tmp_path / "v")
    index = SequenceIndex()
    for c in (
        _candidate("EPI_ISL_10", "ACC10", gisaid("A", "EGGTOWN", 1), "cell", "e"),
        _candidate("EPI_ISL_20", "ACC20", gisaid("A", "REASTOWN", 2), "cell", "r"),
        _candidate("EPI_ISL_30", "ACC30", gisaid("A", "SPLITTOWN", 3), "cell", "s1"),
        _candidate("EPI_ISL_30", "ACC31", gisaid("A", "SPLITTOWN", 3), "cell", "s2"),
        _candidate("EPI_ISL_41", "ACC41", gisaid("A", "ELSEWHERE", 99), "cell", "o"),
        _candidate("EPI_ISL_40", "ACC40", gisaid("A", "BOTHTOWN", 4), "cell", "b"),
    ):
        index.add(c)
    isolates, clades = _isolates_and_clades(tmp_path)
    link_sequences(con, {"h3": index}, [isolates], [clades], class_of=passage_class_column)
    preps = {
        key[1]: value
        for key, value in preparation_sequences(con, passage_matcher(tmp_path)).items()
    }
    egg = preps[syn.virus("EGGTOWN", 1)]
    assert egg.accession == "ACC10" and egg.doubts == ("match.egg-antigen-non-egg-sequence",)
    reassortant = preps[syn.virus("REASTOWN", 2)]
    assert reassortant.accession == "ACC20" and reassortant.doubts == ("match.reassortant",)
    assert syn.virus("SPLITTOWN", 3) not in preps  # a doubt ae has no rule for: never used
    both = preps[syn.virus("BOTHTOWN", 4)]
    assert both.accession == "ACC40" and both.doubts == ()  # the clean row wins


def test_rows_naming_two_records_take_the_one_whose_passage_matches(
    tmp_path: Path, syn: Any
) -> None:
    """Sarah, Q81 (30 Sep): "Passage-matched record". Each preparation below is titrated in two
    tables whose EPI_ISL pairings name two GISAID records of the same virus. Passage stays part
    of the preparation's identity; this chooses only among the records its own rows name."""
    from af.serology.joins import ROWS_NONE_MATCH, ROWS_PASSAGE_MATCHED, ROWS_SEVERAL_MATCH

    def antigen(place: str, passage: str, date: str | None, epi: str) -> dict[str, Any]:
        return {"name": syn.virus(place, 1), "passage": passage, "passage_date": date,
                "passage_class": "egg" if passage.startswith("E") else "cell",
                "epi_isl": epi, "sequence_pairing": "exact"}  # fmt: skip

    serum = {"name": syn.virus("SOMEWHERE", 9), "serum_id": "S-1"}
    first = [antigen("ONETOWN", "SIAT2", "2021-03-01", "EPI_ISL_50"),
             antigen("BOTHTOWN", "MDCK2", None, "EPI_ISL_60"),
             antigen("NONETOWN", "E3", None, "EPI_ISL_70")]  # fmt: skip
    second = [antigen("ONETOWN", "SIAT2", "2021-03-01", "EPI_ISL_51"),
              antigen("BOTHTOWN", "MDCK2", None, "EPI_ISL_61"),
              antigen("NONETOWN", "E3", None, "EPI_ISL_71")]  # fmt: skip
    tables = [
        syn.table("t1", first, [serum], [[["40"]] for _ in first]),
        syn.table("t2", second, [serum], [[["40"]] for _ in second], date="2021-04-01"),
    ]
    build(tables, tmp_path / "v", syn.rules)
    con = query.connect(tmp_path / "v")
    records = {  # EPI number: (place, GISAID's passage for that record)
        50: ("ONETOWN", "S1"), 51: ("ONETOWN", "S2 (2021-03-01)"),
        60: ("BOTHTOWN", "C2"), 61: ("BOTHTOWN", "MDCK2"),
        70: ("NONETOWN", "OR"), 71: ("NONETOWN", "C1"),
    }  # fmt: skip
    index = SequenceIndex()
    for n, (place, _) in records.items():
        index.add(_candidate(f"EPI_ISL_{n}", f"ACC{n}", gisaid("A", place, 1), "cell", f"s{n}"))
    rows = ", ".join(f"('EPI_ISL_{n}', 'ACC{n}', 'C', 'R', '{p}', '2021-01-01', '{passage}')"
                     for n, (p, passage) in records.items())  # fmt: skip
    isolates = _parquet(
        tmp_path / "records.parquet",
        f"SELECT * FROM (VALUES {rows}) AS v(epi_isl, accession, country, region, place, "
        "collection_date, passage)",
    )
    clade_rows = ", ".join(f"('EPI_ISL_{n}', 'ACC{n}', 'CLADE-X', 'fallback')" for n in records)
    clades = _parquet(
        tmp_path / "record-clades.parquet",
        f"SELECT * FROM (VALUES {clade_rows}) AS v(epi_isl, accession, clade, method)",
    )
    link_sequences(con, {"h3": index}, [isolates], [clades], class_of=passage_class_column)
    preps = {
        key[1]: value
        for key, value in preparation_sequences(con, passage_matcher(tmp_path)).items()
    }
    one = preps[syn.virus("ONETOWN", 1)]
    assert one.conflict and one.resolution == ROWS_PASSAGE_MATCHED and one.accession == "ACC51"
    assert [a.accession for a in one.alternatives] == ["ACC50", "ACC51"]
    both = preps[syn.virus("BOTHTOWN", 1)]  # "C2" and "MDCK2" are both MDCK2
    assert both.resolution == ROWS_SEVERAL_MATCH and both.accession is None
    none = preps[syn.virus("NONETOWN", 1)]  # an egg preparation; neither record is E3
    assert none.resolution == ROWS_NONE_MATCH and none.accession is None


def test_a_chart_antigen_has_the_same_preparation_key_as_serology(tmp_path: Path, syn: Any) -> None:
    """preparation_key is the one copy of the key: a chart's passage is already the identity
    passage (the lab's passage with its harvest date), so it equals serology's own key."""
    from af.chart.model import Antigen, Chart, Serum, Titres
    from af.chart.titre import Titre
    from af.serology.joins import preparation_key

    a = {"name": syn.virus("ONETOWN", 1), "passage": "SIAT2", "passage_date": "2021-03-01",
         "reassortant": "", "annotations": ["CLONE-1"]}  # fmt: skip
    serum = {"name": syn.virus("SOMEWHERE", 9), "serum_id": "S-1"}
    build([syn.table("t1", [a], [serum], [[["40"]]])], tmp_path / "v", syn.rules)
    (prep,) = query.preparations(query.connect(tmp_path / "v"))
    chart = Chart(
        {"V": "A(H3N2)"},
        [Antigen(a["name"], passage="SIAT2 (2021-03-01)", annotations=("CLONE-1",))],
        [Serum(serum["name"], serum_id="S-1")],
        Titres([[Titre.parse("40")]]),
    )
    assert preparation_key(chart, "antigen", 0) == prep.key()
    with pytest.raises(ValueError, match="not preparations"):
        preparation_key(chart, "serum", 0)
    with pytest.raises(IndexError):
        preparation_key(chart, "antigen", 1)
    with pytest.raises(ValueError, match="no subtype"):
        preparation_key(Chart({}, chart.antigens, chart.sera, chart.titres), "antigen", 0)


def test_a_preparations_one_clade_keeps_unknown_apart_from_no_clade() -> None:
    """None is unknown; "" is known to have no clade; collapsing them would read a missing
    sequence as a virus outside every clade."""
    from af.serology.joins import PreparationSequence, TiedSequence, preparation_clade

    def chosen(clade: str | None) -> PreparationSequence:
        return PreparationSequence("EPI_ISL_1", "ACC1", clade, "", conflict=False)

    def unchosen(
        *clades: str | None, ranked: str | None = None, tie: bool = True
    ) -> PreparationSequence:
        seqs = tuple(TiedSequence(f"EPI_ISL_{i}", f"ACC{i}", c) for i, c in enumerate(clades))
        pick = TiedSequence("EPI_ISL_9", "ACC9", ranked) if ranked is not None else None
        if tie:
            return PreparationSequence(None, None, None, "", conflict=False, tied=seqs, ranked=pick)
        return PreparationSequence(None, None, None, "", conflict=True, alternatives=seqs)

    assert preparation_clade(None) is None  # no usable sequence
    assert preparation_clade(chosen(None)) is None  # a sequence with no clade row
    assert preparation_clade(chosen("")) == ""  # the nomenclature names none
    assert preparation_clade(chosen("P.1")) == "P.1"
    assert preparation_clade(unchosen("P.1", "P.1")) == "P.1"  # a tie that agrees
    assert preparation_clade(unchosen("P.1", "P.2", ranked="P.2")) == "P.2"  # split: ae's pick
    assert preparation_clade(unchosen("P.1", "P.2", tie=False)) is None  # conflict, disagreeing
    assert preparation_clade(unchosen("P.1", "P.1", tie=False)) == "P.1"  # conflict, agreeing
    assert preparation_clade(unchosen("", "")) == ""  # agreeing on no clade


def test_a_padded_joined_isolate_matches_the_stores_spelling(tmp_path: Path, syn: Any) -> None:
    """Sequence names are stored normalised ("ABC-1"); a table keeps the lab's "ABC-01". The
    join keys the antigen as the store spells it, and counts the rows it respelt."""
    from af.serology.joins import NAME_NORMALISED

    padded = "/".join(("A(H3N2)", "SOMEWHERE", "-".join(("ABC", "01")), "2021"))
    stored = "/".join(("A", "SOMEWHERE", "-".join(("ABC", "1")), "2021"))
    a = {"name": padded, "passage": "MDCK1", "date": "2021-01-05"}
    serum = {"name": syn.virus("SOMEWHERE", 9), "serum_id": "S-1"}
    build([syn.table("t1", [a], [serum], [[["40"]]])], tmp_path / "v", syn.rules)
    con = query.connect(tmp_path / "v")
    isolates, clades = _isolates_and_clades(tmp_path)
    index = SequenceIndex()
    index.add(_candidate("EPI_ISL_1", "ACC1", stored, "cell", "s1"))
    counts = link_sequences(con, {"h3": index}, [isolates], [clades], class_of=cell)
    assert counts.by_flag[NAME_NORMALISED] == 1
    row = con.execute("SELECT status, epi_isl FROM antigen_sequences").fetchone()
    assert row == ("matched", "EPI_ISL_1")


def test_a_distinct_repeat_is_keyed_as_its_preparation() -> None:
    """A chain marks a repeated sample DISTINCT to keep it apart on the chart; serology has no
    such annotation, so the mark must not stop the repeat finding its preparation."""
    from af.chart.model import Antigen, Chart, Serum, Titres
    from af.chart.titre import Titre
    from af.serology.joins import preparation_key

    name = "/".join(("A(H3N2)", "ONETOWN", "1", "2021"))
    chart = Chart(
        {"V": "A(H3N2)"},
        [
            Antigen(name, passage="SIAT2", annotations=("CLONE-1",)),
            Antigen(name, passage="SIAT2", annotations=("CLONE-1", "DISTINCT")),
        ],
        [Serum("/".join(("A(H3N2)", "SOMEWHERE", "9", "2021")), serum_id="S-1")],
        Titres([[Titre.parse("40")], [Titre.parse("80")]]),
    )
    first, repeat = (preparation_key(chart, "antigen", i) for i in (0, 1))
    assert repeat == first and first[3] == ("CLONE-1",)


def test_clade_evidence_reaches_each_preparation_and_candidate(tmp_path: Path, syn: Any) -> None:
    """04-clades' evidence for a tree call reaches the chosen record and every tie candidate,
    recorded only. A clades file written before the evidence existed reads as having none on
    fallback rows; a tree row there was never checked, so it is refused."""
    con = _store(tmp_path, syn)
    isolates, _ = _isolates_and_clades(tmp_path)
    new = _parquet(
        tmp_path / "new.parquet",
        """SELECT * FROM (VALUES
            ('EPI_ISL_1', 'ACC1', 'CLADE-X', 'tree', 'contradicted', 'CLADE', '145'),
            ('EPI_ISL_4', 'ACC4', 'CLADE-Z', 'tree', 'unobserved', NULL, '158'),
            ('EPI_ISL_5', 'ACC5', 'CLADE-Z', 'tree', 'supported', NULL, NULL)
        ) AS v(epi_isl, accession, clade, method, clade_evidence, supported_clade,
               clade_evidence_reason)""",
    )
    old = _parquet(
        tmp_path / "old.parquet",
        """SELECT * FROM (VALUES ('EPI_ISL_3', 'ACC3', 'CLADE-Y', 'fallback'))
           AS v(epi_isl, accession, clade, method)""",
    )
    counts = link_sequences(con, {"h3": _index()}, [isolates], [new, old], class_of=cell)
    assert counts.matched_by_evidence == {"contradicted": 1}
    preps = {k[1]: v for k, v in preparation_sequences(con, passage_matcher(tmp_path)).items()}
    one = preps[gisaid("A(H3N2)", "SOMEWHERE", 1)]
    assert (one.clade, one.clade_evidence, one.supported_clade, one.clade_evidence_reason) == (
        "CLADE-X", "contradicted", "CLADE", "145",
    )  # fmt: skip
    assert preps[gisaid("A(H3N2)", "ELSEWHERE", 3)].clade_evidence is None  # a fallback row
    tie = preps[gisaid("A(H3N2)", "ELSEWHERE", 4)]
    assert {(t.epi_isl, t.clade_evidence) for t in tie.tied} == {
        ("EPI_ISL_4", "unobserved"),
        ("EPI_ISL_5", "supported"),
    }

    older = _parquet(
        tmp_path / "older.parquet",
        """SELECT * FROM (VALUES ('EPI_ISL_3', 'ACC3', 'CLADE-Y', 'fallback'))
           AS v(epi_isl, accession, clade, method)""",
    )
    counts = link_sequences(con, {"h3": _index()}, [isolates], [older], class_of=cell)
    assert counts.matched_by_evidence == {}
    unchecked = _parquet(
        tmp_path / "unchecked.parquet",
        """SELECT * FROM (VALUES ('EPI_ISL_1', 'ACC1', 'CLADE-X', 'tree'))
           AS v(epi_isl, accession, clade, method)""",
    )
    with pytest.raises(StoreError, match="tree-derived clade rows without clade_evidence"):
        link_sequences(con, {"h3": _index()}, [isolates], [unchecked], class_of=cell)
