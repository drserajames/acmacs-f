"""Antigen <-> sequence matching. Every name, id and passage here is invented."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from af.seq import matching as M

RULES = [
    M.PassageRule(re.compile(r"(^|[^A-Z])E\s*[0-9]", re.I), M.EGG, "egg step"),
    M.PassageRule(re.compile(r"MDCK|SIAT|(^|[^A-Z])C\s*[0-9]", re.I), M.CELL, "cell step"),
    M.PassageRule(re.compile(r"ORIGINAL|CLINICAL", re.I), M.ORIGINAL, "specimen"),
]


def candidate(
    n: int, passage: str, seq: str = "h1", name: str = "A/EXAMPLETOWN/7/2024"
) -> M.Candidate:
    return M.Candidate(f"EPI_ISL_{n}", f"EPI{n}", "h3", name, passage,
                       M.passage_class(passage, RULES), seq)  # fmt: skip


def index(*candidates: M.Candidate) -> M.SequenceIndex:
    idx = M.SequenceIndex()
    for c in candidates:
        idx.add(c)
    return idx


ANTIGEN = "A(H3N2)/EXAMPLETOWN/7/2024"


class TestPassageClass:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("E3", M.EGG), ("MDCK2", M.CELL), ("C1+C1", M.CELL), ("Clinical specimen", M.ORIGINAL),
         ("E2/MDCK1", M.MIXED), ("", M.UNKNOWN), ("gSingle", M.UNKNOWN)],
    )  # fmt: skip
    def test_classes(self, text: str, expected: str) -> None:
        assert M.passage_class(text, RULES) == expected


class TestByEpi:
    def test_the_labs_pairing_wins(self) -> None:
        idx = index(candidate(1, "E3", "egg"), candidate(2, "MDCK1", "cell"))
        match = idx.match(ANTIGEN, M.EGG, epi_isl="EPI_ISL_2")
        assert (match.method, match.chosen and match.chosen.epi_isl, match.flags) == (
            "epi_isl", "EPI_ISL_2", (),
        )  # fmt: skip

    def test_a_name_that_disagrees_is_flagged_doubtful(self) -> None:
        idx = index(candidate(1, "MDCK1", name="A/EXAMPLEVILLE/1/2024"))
        match = idx.match(ANTIGEN, M.CELL, epi_isl="EPI_ISL_1")
        assert match.flags == (M.EPI_NAME_DIFFERS,) and match.doubtful

    def test_an_unknown_epi_falls_back_to_the_name_and_says_so(self) -> None:
        match = index(candidate(1, "MDCK1")).match(ANTIGEN, M.CELL, epi_isl="EPI_ISL_9")
        assert (match.method, match.flags) == ("name", (M.EPI_NOT_IN_STORE,))


class TestByName:
    def test_the_type_prefix_does_not_matter(self) -> None:
        match = index(candidate(1, "MDCK1")).match(ANTIGEN, M.CELL)
        assert match.chosen is not None and match.chosen.epi_isl == "EPI_ISL_1"

    def test_egg_antigen_takes_the_egg_sequence(self) -> None:
        idx = index(candidate(1, "MDCK1", "cell"), candidate(2, "E3", "egg"),
                    candidate(3, "Original", "orig"))  # fmt: skip
        match = idx.match(ANTIGEN, M.EGG)
        assert (match.chosen and match.chosen.epi_isl, match.flags) == ("EPI_ISL_2", ())

    def test_egg_antigen_with_only_cell_sequences_is_matched_but_doubtful(self) -> None:
        """T31: attaching a cell sequence to an egg antigen is never silent."""
        match = index(candidate(1, "MDCK1")).match(ANTIGEN, M.EGG)
        assert match.chosen is not None
        assert match.flags == (M.EGG_WITHOUT_EGG_SEQUENCE,) and match.doubtful

    def test_cell_antigen_prefers_cell_then_original(self) -> None:
        match = index(candidate(1, "Original")).match(ANTIGEN, M.CELL)
        assert match.flags == (M.CELL_FROM_ORIGINAL,) and not match.doubtful

    def test_identical_duplicates_choose_the_lowest_key(self) -> None:
        match = index(candidate(2, "MDCK1", "same"), candidate(1, "SIAT1", "same")).match(
            ANTIGEN, M.CELL
        )
        assert (match.chosen and match.chosen.epi_isl, match.flags) == (
            "EPI_ISL_1", (M.IDENTICAL_DUPLICATES,),
        )  # fmt: skip

    def test_different_sequences_in_one_tier_are_not_chosen(self) -> None:
        match = index(candidate(1, "MDCK1", "a"), candidate(2, "SIAT1", "b")).match(ANTIGEN, M.CELL)
        assert match.chosen is None and match.flags == (M.AMBIGUOUS,) and match.doubtful
        assert len(match.candidates) == 2

    def test_no_candidate_is_no_match(self) -> None:
        match = index(candidate(1, "MDCK1")).match("A(H3N2)/EXAMPLEVILLE/1/2024", M.CELL)
        assert (match.method, match.chosen, match.flags) == (None, None, (M.NO_MATCH,))

    def test_a_reassortant_is_doubtful(self) -> None:
        match = index(candidate(1, "E3")).match(ANTIGEN, M.EGG, reassortant="EXAMPLE-REASSORTANT-1")
        assert M.REASSORTANT in match.flags and match.doubtful

    def test_every_match_is_counted(self) -> None:
        idx = index(candidate(1, "MDCK1"))
        idx.match(ANTIGEN, M.CELL)
        idx.match("A(H3N2)/EXAMPLEVILLE/1/2024", M.CELL)
        assert idx.counts == {"match.clean": 1, M.NO_MATCH: 1}


def test_passage_rules_file_with_an_unknown_class_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "rules.tsv"
    path.write_text("# comment\npattern\tclass\treason\nX\tbanana\twhy\n")
    with pytest.raises(ValueError, match="banana"):
        M.read_passage_rules(path)
