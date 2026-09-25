"""The conversions every clade rule depends on, including the ones that cost time before."""

from __future__ import annotations

import pytest

from af.clades.coordinates import (
    COORDINATES,
    Position,
    Unexpressible,
    UnknownLocusError,
    convert,
    coordinates_for,
)


def test_ha1_position_is_unchanged() -> None:
    assert convert("HA1", 145, "N", COORDINATES["A(H3N2)"]) == Position("aa", 145, "N")


@pytest.mark.parametrize(
    ("subtype", "expected"),
    [("A(H1N1)", 327 + 151), ("A(H3N2)", 329 + 151), ("B/Vic", 347 + 151)],
)
def test_ha2_position_adds_the_ha1_length(subtype: str, expected: int) -> None:
    assert convert("HA2", 151, "K", COORDINATES[subtype]) == Position("aa", expected, "K")


def test_bvic_ha2_offset_is_347() -> None:
    """Measured, not assumed: earlier notes said 346, which put every B/Vic HA2 rule one
    residue out. See notes/clades/ENGINE-COMPARISON.md."""
    assert COORDINATES["B/Vic"].ha1_length == 347


def test_nucleotide_position_subtracts_the_offset() -> None:
    assert convert("nuc", 100, "A", COORDINATES["A(H3N2)"]) == Position("nuc", 35, "A")


def test_signal_peptide_is_unexpressible_not_an_error() -> None:
    result = convert("SigPep", 3, "T", COORDINATES["A(H3N2)"])
    assert isinstance(result, Unexpressible)
    assert "signal peptide" in result.reason


def test_nucleotide_before_the_mature_ha_is_unexpressible() -> None:
    result = convert("nuc", 10, "T", COORDINATES["A(H3N2)"])
    assert isinstance(result, Unexpressible)
    assert "precedes" in result.reason


def test_first_nucleotide_of_the_mature_ha_is_position_one() -> None:
    coordinates = COORDINATES["A(H3N2)"]
    assert convert("nuc", coordinates.nuc_offset + 1, "C", coordinates) == Position("nuc", 1, "C")


def test_unknown_locus_is_fatal() -> None:
    with pytest.raises(UnknownLocusError, match="unknown locus"):
        convert("HA3", 1, "A", COORDINATES["A(H3N2)"])


def test_unknown_subtype_is_fatal() -> None:
    with pytest.raises(KeyError, match="no clade coordinates"):
        coordinates_for("A(H5N1)")
