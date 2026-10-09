"""The conversions every clade rule depends on, including the ones that cost time before."""

from __future__ import annotations

import pytest

from af.clades.coordinates import (
    Position,
    Unexpressible,
    UnknownLocusError,
    convert,
)
from af.clades.subtypes import CladeSubtypeError, coordinates_for


def test_ha1_position_is_unchanged() -> None:
    assert convert("HA1", 145, "N", coordinates_for("A(H3N2)")) == Position("aa", 145, "N")


@pytest.mark.parametrize(
    ("subtype", "expected"),
    [("A(H1N1)", 327 + 151), ("A(H3N2)", 329 + 151), ("B/Vic", 347 + 151)],
)
def test_ha2_position_adds_the_ha1_length(subtype: str, expected: int) -> None:
    assert convert("HA2", 151, "K", coordinates_for(subtype)) == Position("aa", expected, "K")


def test_bvic_ha2_offset_is_347() -> None:
    """Measured, not assumed: earlier notes said 346, which put every B/Vic HA2 rule one
    residue out. See notes/clades/ENGINE-COMPARISON.md."""
    assert coordinates_for("B/Vic").ha1_length == 347


def test_nucleotide_position_subtracts_the_offset() -> None:
    assert convert("nuc", 100, "A", coordinates_for("A(H3N2)")) == Position("nuc", 35, "A")


def test_signal_peptide_is_unexpressible_not_an_error() -> None:
    result = convert("SigPep", 3, "T", coordinates_for("A(H3N2)"))
    assert isinstance(result, Unexpressible)
    assert "signal peptide" in result.reason


def test_nucleotide_before_the_mature_ha_is_unexpressible() -> None:
    result = convert("nuc", 10, "T", coordinates_for("A(H3N2)"))
    assert isinstance(result, Unexpressible)
    assert "precedes" in result.reason


def test_nucleotide_after_the_mature_ha_is_unexpressible() -> None:
    """B/Vic A's nuc 1843 lies in the 3' untranslated region. As a position it could never be
    observed, so every A virus would count as having that locus unobservable."""
    result = convert("nuc", 1843, "T", coordinates_for("B/Vic"))
    assert isinstance(result, Unexpressible)
    assert "follows" in result.reason


def test_last_nucleotide_of_the_mature_ha_is_expressible() -> None:
    coordinates = coordinates_for("B/Vic")
    last = coordinates.nuc_offset + coordinates.mature_nt
    assert convert("nuc", last, "A", coordinates) == Position("nuc", coordinates.mature_nt, "A")
    assert isinstance(convert("nuc", last + 1, "A", coordinates), Unexpressible)


def test_residue_after_the_mature_ha_is_unexpressible() -> None:
    coordinates = coordinates_for("A(H3N2)")
    last_ha2 = coordinates.mature_nt // 3 - coordinates.ha1_length
    assert convert("HA2", last_ha2, "K", coordinates) == Position("aa", 550, "K")
    assert isinstance(convert("HA2", last_ha2 + 1, "K", coordinates), Unexpressible)


def test_first_nucleotide_of_the_mature_ha_is_position_one() -> None:
    coordinates = coordinates_for("A(H3N2)")
    assert convert("nuc", coordinates.nuc_offset + 1, "C", coordinates) == Position("nuc", 1, "C")


def test_unknown_locus_is_fatal() -> None:
    with pytest.raises(UnknownLocusError, match="unknown locus"):
        convert("HA3", 1, "A", coordinates_for("A(H3N2)"))


def test_unknown_subtype_is_fatal() -> None:
    with pytest.raises(CladeSubtypeError, match="no subtype 'A\\(H5N1\\)'"):
        coordinates_for("A(H5N1)")
