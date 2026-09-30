"""Which of a preparation's named GISAID records has its passage (Sarah, Q81, 30 Sep)."""

from pathlib import Path

import pytest

from tests.seq.test_matching_rules import passage_matcher


@pytest.mark.parametrize(
    ("record", "canonical"),
    [
        ("S2 (2024-03-19)", ("SIAT2", "2024-03-19")),  # GISAID's abbreviation, ISO date
        ("S1(12/11/2024)Lot # 4", ("SIAT1", "2024-12-11")),  # M/D/YYYY, a note after it
        ("S2/S2(102121)", ("SIAT2/SIAT2", "2021-10-21")),  # MMDDYY
        ("C2, MDCK2", ("MDCK2/MDCK2", "")),  # comma-separated segments
        ("E3", ("E3", "")),
        ("OR_IR", (None, "")),  # does not read: never a match
        ("", (None, "")),
    ],
)
def test_record_passages_are_read_canonically(
    tmp_path: Path, record: str, canonical: tuple[str | None, str]
) -> None:
    assert passage_matcher(tmp_path).canonical(record) == canonical


@pytest.mark.parametrize(
    ("preparation", "record", "score"),
    [
        ("SIAT3 (2022-11-23)", "S3 (2022-11-23)", 2),  # steps and harvest date
        ("SIAT1 (2024-03-12)", "S1", 1),  # steps; the record gives no date
        ("MDCK2/MDCK2", "C2, MDCK2", 1),  # steps; neither gives a date
        ("SIAT2 (2023-12-17)", "S2 (2023-12-15)", 0),  # another harvest
        ("SIAT3 (2022-11-23)", "S1", 0),  # another passage
        ("E3 (2019-04-16)", "OR", 0),  # the original specimen is not the egg isolate
        ("SIAT1", "OR_IR", 0),  # unreadable
    ],
)
def test_scores(tmp_path: Path, preparation: str, record: str, score: int) -> None:
    assert passage_matcher(tmp_path).score(preparation, record) == score
