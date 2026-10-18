"""Every name here is synthetic, in the gate's reserved EXAMPLE… namespace.

The shapes are real: each case is a shape that occurs in submitted sequence metadata,
and several are shapes the old pipeline got wrong silently. The places are not — a
plausible-looking invented city would be indistinguishable from a real one.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from af.seq.names import (
    EXTRA_SLASH_IN_PAREN,
    FIELDS,
    SHAPE,
    TYPE_MISMATCH,
    TYPE_UNKNOWN,
    YEAR,
    expand_year,
    normalise,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A/EXAMPLETOWN/7/2021", "A/EXAMPLETOWN/7/2021"),
        ("  a/exampletown/7/2021  ", "A/EXAMPLETOWN/7/2021"),  # case and outer spaces
        ("A / EXAMPLETOWN / 7 / 2021", "A/EXAMPLETOWN/7/2021"),  # spaces around slashes
        ("A/EXAMPLETOWN/007/2021", "A/EXAMPLETOWN/7/2021"),  # padding stripped
        ("A/EXAMPLETOWN-EXAMPLEISLES/7/2021", "A/EXAMPLETOWN EXAMPLEISLES/7/2021"),
        ("A(H3N2)/EXAMPLETOWN/7/2021", "A(H3N2)/EXAMPLETOWN/7/2021"),  # subtype kept
        ("B/EXAMPLETOWN/7/2021", "B/EXAMPLETOWN/7/2021"),
    ],
)
def test_well_formed_names(raw: str, expected: str) -> None:
    result = normalise(raw)
    assert result.name == expected
    assert result.ok, result.problems


def test_isolate_of_only_zeros_is_not_emptied() -> None:
    assert normalise("A/EXAMPLETOWN/0/2021").name == "A/EXAMPLETOWN/0/2021"


def test_location_spelling_is_left_alone() -> None:
    """af keeps the submitter's spelling; ae rewrote it and produced two names."""
    for spelling in ("EXAMPLETOWN", "EXAMPLEBURG", "EXAMPLETOWNE"):
        assert normalise(f"A/{spelling}/7/2021").name == f"A/{spelling}/7/2021"


class TestExtra:
    def test_text_after_the_year_is_split_off_not_dropped(self) -> None:
        result = normalise("A/EXAMPLETOWN/7/2021 XX-99")
        assert (result.name, result.extra) == ("A/EXAMPLETOWN/7/2021", "XX-99")
        assert result.ok

    def test_hyphen_separated_extra(self) -> None:
        result = normalise("A/EXAMPLETOWN/7/2021-EXAMPLELAB-V9")
        assert (result.name, result.extra) == ("A/EXAMPLETOWN/7/2021", "EXAMPLELAB-V9")

    def test_parenthesised_extra(self) -> None:
        result = normalise("A/EXAMPLETOWN/7/2021 (99)")
        assert (result.name, result.extra) == ("A/EXAMPLETOWN/7/2021", "(99)")
        assert result.ok

    def test_parenthesised_extra_containing_a_slash(self) -> None:
        """The bug this module was written to fix.

        Splitting on "/" first makes the year part of the isolate, and the name comes
        out wrong while merely being *reported*. The tail comes off first instead.
        """
        result = normalise("B/EXAMPLETOWN/1234567/2021 (99/228)")
        assert result.name == "B/EXAMPLETOWN/1234567/2021"
        assert result.extra == "(99/228)"
        assert result.problems == [EXTRA_SLASH_IN_PAREN]

    def test_two_trailing_groups_are_reported_not_guessed(self) -> None:
        result = normalise("A/EXAMPLETOWN/1148/2021 EXAMPLELAB-2301 (99/214)")
        assert result.problems == [EXTRA_SLASH_IN_PAREN]
        assert result.name == "A/EXAMPLETOWN/1148/2021"


class TestProblems:
    def test_unqualified_name_is_kept_never_invented(self) -> None:
        result = normalise("EXAMPLEID_2300085")
        assert result.problems == [SHAPE]
        assert result.name == "EXAMPLEID_2300085"

    def test_lab_code_between_location_and_isolate(self) -> None:
        result = normalise("B/EXAMPLETOWN/EXAMPLELAB/23/2021")
        assert result.name == "B/EXAMPLETOWN/EXAMPLELAB-23/2021"
        assert result.problems == [FIELDS]

    def test_several_extra_fields(self) -> None:
        result = normalise("A/EXAMPLETOWN/EXAMPLELAB/EXAMPLEUNIT/01/2025")
        assert result.name == "A/EXAMPLETOWN/EXAMPLELAB-EXAMPLEUNIT-1/2025"
        assert result.problems == [FIELDS]

    def test_mistyped_year_is_reported_not_corrected(self) -> None:
        result = normalise("B/EXAMPLETOWN/247/20223")
        assert result.problems == [YEAR]
        assert result.name.endswith("/20223")

    def test_type_field_holding_something_else(self) -> None:
        result = normalise("EXAMPLELIN/EXAMPLETOWN/7/2021")
        assert TYPE_UNKNOWN in result.problems

    def test_name_type_wins_over_the_sources_type(self) -> None:
        """Relabelling to match the file loses the evidence that something is wrong."""
        result = normalise("A(H1N1)/EXAMPLETOWN/7/2021", subtype="A(H3N2)")
        assert result.name.startswith("A(H1N1)/")
        assert result.problems == [TYPE_MISMATCH]

    def test_bare_a_matches_any_a_subtype_source(self) -> None:
        assert normalise("A/EXAMPLETOWN/7/2021", subtype="A(H3N2)").ok

    def test_no_source_subtype_means_no_mismatch(self) -> None:
        assert normalise("A(H1N1)/EXAMPLETOWN/7/2021").ok


def test_padding_is_stripped_inside_a_joined_isolate() -> None:
    """``02`` and ``2`` are one virus wherever the padding sits, so both give one name."""
    joined = "A/EXAMPLETOWN/EXAMPLELAB-7/2025"
    assert normalise("A/EXAMPLETOWN/EXAMPLELAB/007/2025").name == joined
    assert normalise("A/EXAMPLETOWN/EXAMPLELAB/7/2025").name == joined


def test_zeros_in_a_non_numeric_part_are_left_alone() -> None:
    result = normalise("A/EXAMPLETOWN/0EXAMPLELAB/7/2025")
    assert result.name == "A/EXAMPLETOWN/0EXAMPLELAB-7/2025"


def test_problem_codes_are_stable_strings() -> None:
    """They end up in counts and provenance, so they must not drift."""
    assert (SHAPE, TYPE_UNKNOWN, TYPE_MISMATCH, YEAR, FIELDS, EXTRA_SLASH_IN_PAREN) == (
        "name.shape",
        "name.type",
        "name.type_mismatch",
        "name.year",
        "name.fields",
        "name.extra_slash_in_paren",
    )


@pytest.mark.parametrize(
    "raw",
    [
        "a/exampletown-exampleisles/01/2024",
        "A/ EXAMPLETOWN -EXAMPLEISLES- /7/2024",
        "A/EXAMPLETOWN/EXAMPLELAB/7/2024",
        "B/EXAMPLETOWN/7/2024 (x/y)",
        "A(H3N2)/EXAMPLETOWN/0007/2024-EXTRA",
        "A/EXAMPLETOWN/7",
    ],
)
def test_normalise_is_idempotent(raw: str) -> None:
    once = normalise(raw).name
    assert normalise(once).name == once


def test_normalise_is_idempotent_on_random_names() -> None:
    # covers "X - Y" and "X--Y" locations; master's normaliser fails 63 of these
    rng = random.Random(3)
    alphabet = "AB- _/()0123456789"
    for _ in range(5000):
        raw = "A/" + "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 24))) + "/2024"
        once = normalise(raw).name
        assert normalise(once).name == once, raw


def test_store_names_are_fixed_points(af_data: Path) -> None:
    """Every name-location the store holds normalises to itself (a real-data list)."""
    path = af_data / "fixtures" / "sequences" / "name-locations.txt"
    if not path.is_file():
        pytest.skip(f"{path} not present")
    locations = [line for line in path.read_text().splitlines() if line and line[0] != "#"]
    moved = [loc for loc in locations if normalise(f"A/{loc}/1/2000").name != f"A/{loc}/1/2000"]
    assert not moved, f"{len(moved)} of {len(locations)} are not fixed points: {moved[:5]}"


class TestTwoDigitYear:
    """A two-digit name year gets its century from the collection year, or is left alone."""

    @pytest.mark.parametrize(
        ("raw", "collected", "expected"),
        [
            ("A/EXAMPLETOWN/12/26", 2026, "A/EXAMPLETOWN/12/2026"),
            ("A/EXAMPLETOWN/12/98", 1998, "A/EXAMPLETOWN/12/1998"),  # not 2098, as ae wrote
            ("A/EXAMPLETOWN/12/26", 2025, "A/EXAMPLETOWN/12/2026"),  # the season convention
            ("A/EXAMPLETOWN/12/25", 2026, "A/EXAMPLETOWN/12/2025"),  # named at isolation
            ("A/EXAMPLETOWN/12/00", 1999, "A/EXAMPLETOWN/12/2000"),  # across the century
            ("A/EXAMPLETOWN/12/99", 2000, "A/EXAMPLETOWN/12/1999"),
        ],
    )
    def test_the_century_that_fits_the_collection_year_is_written(
        self, raw: str, collected: int, expected: str
    ) -> None:
        result = normalise(raw, collected=collected)
        assert result.name == expected
        assert result.ok, result.problems
        assert result.two_digit_year == raw[-2:]

    def test_digits_no_century_fits_are_not_a_year_and_stay_as_written(self) -> None:
        # a code or serial where the year belongs: ae would have written 2090
        result = normalise("A/EXAMPLETOWN/EXAMPLECODE/90", collected=2019)
        assert result.name == "A/EXAMPLETOWN/EXAMPLECODE/90"
        assert result.problems == [YEAR]
        assert result.two_digit_year == ""

    def test_without_a_collection_year_nothing_is_guessed(self) -> None:
        result = normalise("A/EXAMPLETOWN/12/26")
        assert (result.name, result.problems, result.two_digit_year) == (
            "A/EXAMPLETOWN/12/26", [YEAR], ""
        )  # fmt: skip

    def test_text_after_a_two_digit_year_is_split_off(self) -> None:
        result = normalise("A/EXAMPLETOWN/12/26-EXAMPLEX", collected=2026)
        assert (result.name, result.extra) == ("A/EXAMPLETOWN/12/2026", "EXAMPLEX")

    @pytest.mark.parametrize("last", ["7", "026", "2X"])
    def test_other_short_years_are_still_a_problem(self, last: str) -> None:
        assert normalise(f"A/EXAMPLETOWN/12/{last}", collected=2026).problems == [YEAR]

    def test_four_digit_years_are_not_touched_by_the_collection_year(self) -> None:
        # the name year and the collection year disagreeing is af.seq.gisaid's report
        result = normalise("A/EXAMPLETOWN/12/2021", collected=1998)
        assert (result.name, result.two_digit_year) == ("A/EXAMPLETOWN/12/2021", "")

    def test_an_expanded_name_normalises_to_itself(self) -> None:
        once = normalise("A/EXAMPLETOWN/012/26", collected=2026).name
        assert normalise(once).name == once == "A/EXAMPLETOWN/12/2026"

    def test_at_most_one_century_fits(self) -> None:
        assert expand_year("26", 2026) == 2026
        assert expand_year("26", 1950) is None
        assert all(expand_year(f"{yy:02d}", year) is not None
                   for yy in range(100) for year in (1900 + yy, 2000 + yy))  # fmt: skip
