"""Unknown, gap and mismatch: the three outcomes a locus test can have."""

from __future__ import annotations

import pytest

from af.clades.sequence import AlignedSequence, Evidence, GapSupport, translate


def test_asparagine_is_not_unknown() -> None:
    """N is unknown in DNA and asparagine in protein. Treating it as unknown in an amino
    acid string mislabelled 2,179 H3 leaves (notes/clades/ENGINE-COMPARISON.md)."""
    sequence = AlignedSequence(amino_acids="ANK")
    assert sequence.evidence("aa", 2, "N") is Evidence.MATCHES
    assert sequence.evidence("aa", 2, "K") is Evidence.CONTRADICTS


def test_x_is_unknown_in_amino_acids() -> None:
    assert AlignedSequence(amino_acids="AXK").evidence("aa", 2, "N") is Evidence.UNOBSERVABLE


def test_n_is_unknown_in_nucleotides() -> None:
    sequence = AlignedSequence(amino_acids="A", nucleotides="ANG")
    assert sequence.evidence("nuc", 2, "C") is Evidence.UNOBSERVABLE


def test_position_past_the_end_is_unobservable_not_a_mismatch() -> None:
    """A short sequence carries no evidence; ae read past the end as a space, so every
    rule beyond the end silently failed."""
    assert AlignedSequence(amino_acids="AK").evidence("aa", 40, "N") is Evidence.UNOBSERVABLE


def test_missing_nucleotides_are_unobservable() -> None:
    assert AlignedSequence(amino_acids="AK").evidence("nuc", 1, "C") is Evidence.UNOBSERVABLE


def test_observed_gap_is_evidence_of_a_deletion() -> None:
    sequence = AlignedSequence(amino_acids="A-K", gaps=GapSupport.OBSERVED)
    assert sequence.evidence("aa", 2, "-") is Evidence.MATCHES
    assert sequence.evidence("aa", 2, "N") is Evidence.CONTRADICTS


def test_gap_blind_reconstruction_cannot_speak_to_a_deletion() -> None:
    """raxml-ng and IQ-TREE never emit a gap, so a deletion locus is unobservable there;
    treating their residue as a contradiction cost 9% agreement on B/Vic."""
    sequence = AlignedSequence(amino_acids="ANK", gaps=GapSupport.GAP_BLIND)
    assert sequence.evidence("aa", 2, "-") is Evidence.UNOBSERVABLE
    assert sequence.evidence("aa", 2, "N") is Evidence.MATCHES


@pytest.mark.parametrize(
    ("codons", "expected"),
    [("AAA", "K"), ("AAAAAT", "KN"), ("---", "-"), ("AAA---AAT", "K-N"), ("AANAAT", "XN")],
)
def test_translation_keeps_deletions_and_marks_partial_codons_unknown(
    codons: str, expected: str
) -> None:
    """'---' must become '-', not 'X': Bio.Seq.translate turns it into X and destroys the
    signal the B/Vic C-lineage clades are defined by."""
    assert translate(codons) == expected


def test_from_nucleotides_translates_and_keeps_both_strings() -> None:
    sequence = AlignedSequence.from_nucleotides("AAAAAT")
    assert sequence.amino_acids == "KN"
    assert sequence.nucleotides == "AAAAAT"
