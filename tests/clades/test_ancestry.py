"""Tree calls resting on ancestry alone are marked, never hidden (af.clades.ancestry).

Synthetic nomenclature (``synthetic.py``): 5 K makes a virus P; 9 T and 331 W make it P.1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from af.clades import ancestry
from af.clades.sequence import AlignedSequence
from af.clades.store import CladeRow

from .synthetic import build_clone, load_synthetic
from .test_assign import protein

SUBTYPE = "A(H3N2)"


def sequence(**states: str) -> AlignedSequence:
    return AlignedSequence(amino_acids=protein(**states), nucleotides="A" * 1200)


def row(index: int, clade: str | None, method: str = "tree") -> CladeRow:
    return CladeRow(f"EPI_ISL_{index}", f"EPI{index}", SUBTYPE, clade, method)


def key(index: int) -> tuple[str, str]:
    return (f"EPI_ISL_{index}", f"EPI{index}")


def test_own_loci_split_unobservable_from_contradicted(tmp_path: Path) -> None:
    clade_set = load_synthetic(build_clone(tmp_path).parent)
    shown = ancestry.own_loci(sequence(p5="K", p9="T", p331="W"), clade_set, "P.1")
    assert shown.supported
    hidden = ancestry.own_loci(sequence(p5="K", p9="X", p331="W"), clade_set, "P.1")
    assert hidden.unobservable == ("9T",) and not hidden.contradicted
    reverted = ancestry.own_loci(sequence(p5="K", p9="T", p331="R"), clade_set, "P.1")
    assert reverted.contradicted == ("331W",)
    assert reverted.reason() == "own loci contradicted: 331W"


def test_only_tree_calls_the_fallback_does_not_share_and_the_sequence_does_not_show(
    tmp_path: Path,
) -> None:
    clade_set = load_synthetic(build_clone(tmp_path).parent)
    rows = [
        row(1, "P.1"),  # fallback agrees: not judged, whatever the sequence shows
        row(2, "P.1"),  # fallback says P, sequence shows P.1's loci: supported, not marked
        row(3, "P.1"),  # fallback says P, 9 unobservable: marked
        row(4, "P.1"),  # fallback says P, 331 reverted: marked
        row(5, "P.1", "fallback"),  # a fallback row is never marked
    ]
    fallback = {key(1): "P.1", key(2): "P", key(3): "P", key(4): "P", key(5): "P.1"}
    sequences = {
        key(2): sequence(p5="K", p9="T", p331="W"),
        key(3): sequence(p5="K", p9="X", p331="W"),
        key(4): sequence(p5="K", p9="T", p331="R"),
    }
    assert ancestry.candidates(rows, fallback) == {key(2), key(3), key(4)}
    marked = {
        (r.epi_isl, r.accession): r for r in ancestry.mark(rows, fallback, sequences, clade_set)
    }
    assert [marked[key(i)].ancestry_only for i in range(1, 6)] == [False, False, True, True, False]
    assert marked[key(3)].ancestry_reason == "own loci unobservable: 9T"
    assert marked[key(4)].ancestry_reason == "own loci contradicted: 331W"
    assert marked[key(3)].clade == "P.1"  # recorded, not changed
    assert ancestry.summary(list(marked.values())) == {
        "rows": 2,
        "with_contradicted_loci": 1,
        "unobservable_loci_only": 1,
    }


def test_a_candidate_without_a_sequence_is_fatal(tmp_path: Path) -> None:
    clade_set = load_synthetic(build_clone(tmp_path).parent)
    with pytest.raises(ValueError, match="no aligned sequence"):
        ancestry.mark([row(1, "P.1")], {key(1): "P"}, {}, clade_set)
