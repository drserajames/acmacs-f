"""What a tree row's own sequence shows of its clade (af.clades.evidence).

Synthetic nomenclature (``synthetic.py``): 5 K makes a virus P; 9 T and 331 W make it P.1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from af.clades import evidence
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


def test_three_states_and_the_supported_ancestor(tmp_path: Path) -> None:
    clade_set = load_synthetic(build_clone(tmp_path).parent)
    rows = [
        row(1, "P.1"),  # all own loci shown: supported
        row(2, "P.1"),  # 9 unreadable: unobserved (strict: one is enough)
        row(3, "P.1"),  # 331 reverted, P's 5K shown: contradicted, coloured as P
        row(4, "P.1"),  # 331 reverted and P's 5K reverted too: contradicted, nothing supported
        row(5, "P.1", "fallback"),  # fallback rows are never judged
        row(6, None),  # no clade: nothing to judge
    ]
    sequences = {
        key(1): sequence(p5="K", p9="T", p331="W"),
        key(2): sequence(p5="K", p9="X", p331="W"),
        key(3): sequence(p5="K", p9="T", p331="R"),
        key(4): sequence(p5="E", p9="T", p331="R"),
    }
    judged = {(r.epi_isl, r.accession): r for r in evidence.judge(rows, sequences, clade_set)}
    states = [judged[key(i)].clade_evidence for i in range(1, 7)]
    assert states == ["supported", "unobserved", "contradicted", "contradicted", None, None]
    assert [judged[key(i)].supported_clade for i in range(1, 5)] == [None, None, "P", None]
    assert judged[key(2)].clade_evidence_reason == "own loci unobservable: 9T"
    assert judged[key(3)].clade_evidence_reason == "own loci contradicted: 331W"
    assert all(judged[key(i)].clade == "P.1" for i in range(1, 5))  # the label never changes
    report = evidence.summary(list(judged.values()))
    assert report["supported"] == 1 and report["unobserved"] == 1 and report["contradicted"] == 2
    assert report["contradicted_without_supported_clade"] == 1
    assert report["contradicted_colour_from"] == {"P.1 -> P": 1, "P.1 -> none supported": 1}


def test_a_tree_row_without_a_sequence_is_fatal(tmp_path: Path) -> None:
    clade_set = load_synthetic(build_clone(tmp_path).parent)
    with pytest.raises(ValueError, match="no aligned sequence"):
        evidence.judge([row(1, "P.1")], {}, clade_set)
