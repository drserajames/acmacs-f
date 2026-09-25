"""Every name here is synthetic, in the gate's reserved EXAMPLE… namespace.

The shapes are real: each case is a shape that occurs in submitted sequence metadata,
and several are shapes the old pipeline got wrong silently. The places are not — a
plausible-looking invented city would be indistinguishable from a real one.
"""

from __future__ import annotations

import pytest

from af.seq.names import (
    EXTRA_SLASH_IN_PAREN,
    FIELDS,
    SHAPE,
    TYPE_MISMATCH,
    TYPE_UNKNOWN,
    YEAR,
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
