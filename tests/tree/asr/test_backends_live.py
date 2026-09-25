"""Run each external ASR backend for real, on a small synthetic alignment.

Skipped when the tool is not installed, so the suite still passes on a bare machine; the point is
that when a tool *is* there, af's wiring to it is exercised — the command line, the output parsing
and the clade matching — rather than only mocked.

The alignment is generated here (no real sequences in a public repo) and is deliberately tiny, so
each backend finishes in seconds.
"""

from __future__ import annotations

import random

import pytest

from af.tree import asr
from af.tree.io import fasta, newick
from af.tree.model import Tree

BACKENDS = [
    pytest.param("treetime", "treetime", id="treetime", marks=pytest.mark.tool("treetime")),
    pytest.param("iqtree", "iqtree3", id="iqtree", marks=pytest.mark.tool("iqtree3")),
    pytest.param("raxml", "raxml-ng", id="raxml-ng", marks=pytest.mark.tool("raxml-ng")),
]


def synthetic(n_leaves: int = 24, length: int = 300, seed: int = 7) -> tuple[Tree, dict[str, str]]:
    """A balanced tree whose sequences evolve down it, so ancestors are recoverable."""
    rng = random.Random(seed)
    root_sequence = "".join(rng.choice("ACGT") for _ in range(length))

    names = [f"leaf{index:03d}" for index in range(n_leaves)]

    def build(labels: list[str]) -> str:
        if len(labels) == 1:
            return f"{labels[0]}:0.01"
        middle = len(labels) // 2
        return f"({build(labels[:middle])},{build(labels[middle:])}):0.01"

    # A three-way root, as a real tree has after rooting on the outgroup (the delivered H3 tree's
    # root has 198 children). A two-child root is the degenerate case: an unrooting backend then
    # has no node for the root *or* for one of its children, because they are the same split.
    third = len(names) // 3
    tree = newick.loads(
        f"({build(names[:third])},{build(names[third : 2 * third])},{build(names[2 * third :])});"
    )
    tree.assign_ids()

    sequences: dict[str, str] = {}
    current: dict[int, str] = {id(tree.root): root_sequence}
    for node in tree.preorder():
        if node.parent is None:
            continue
        parent_sequence = list(current[id(node.parent)])
        for _ in range(rng.randint(1, 4)):
            site = rng.randrange(length)
            parent_sequence[site] = rng.choice("ACGT")
        mine = "".join(parent_sequence)
        current[id(node)] = mine
        if node.is_leaf:
            sequences[node.name or ""] = mine
    return tree, sequences


@pytest.mark.parametrize(("backend_name", "executable"), BACKENDS)
def test_backend_reconstructs_every_internal_node(backend_name, executable, tmp_path) -> None:
    tree, sequences = synthetic()
    alignment = tmp_path / "aln.fasta"
    fasta.write_alignment(alignment, sequences)

    backend = asr.get_backend(backend_name)
    states = backend.reconstruct(tree, alignment, tmp_path / backend_name, threads=2)

    assert states.backend == backend_name
    assert states.backend_version.startswith(("treetime/", "iqtree/", "raxml-ng/"))
    # Every internal node got a state, except that an unrooting backend has none for the root:
    # in an unrooted tree the root is a point on a branch, not a node.
    assert set(asr.unmatched_ids(tree, states.nucleotides)) <= {tree.root.node_id}
    lengths = {len(sequence) for sequence in states.nucleotides.values()}
    assert lengths == {len(next(iter(sequences.values())))}


@pytest.mark.parametrize(("backend_name", "executable"), BACKENDS)
def test_backend_leaves_the_branch_lengths_alone(backend_name, executable, tmp_path) -> None:
    """af never lets a backend re-optimise lengths: they come from CMAPLE."""
    tree, sequences = synthetic()
    before = [node.branch_length for node in tree.preorder()]
    alignment = tmp_path / "aln.fasta"
    fasta.write_alignment(alignment, sequences)
    asr.get_backend(backend_name).reconstruct(tree, alignment, tmp_path / backend_name)
    assert [node.branch_length for node in tree.preorder()] == before


@pytest.mark.tool("treetime")
def test_treetime_reconstructs_a_deletion(tmp_path) -> None:
    """The property B/Vic's clades depend on, and the reason TreeTime is the default."""
    tree, sequences = synthetic(n_leaves=16, length=120)
    # Give half the leaves a clean 3-codon deletion, as B/Vic's C lineage has.
    deleted = sorted(sequences)[:8]
    for name in deleted:
        sequences[name] = sequences[name][:30] + "-" * 9 + sequences[name][39:]
    alignment = tmp_path / "aln.fasta"
    fasta.write_alignment(alignment, sequences)

    states = asr.get_backend("treetime").reconstruct(tree, alignment, tmp_path)
    with_gaps = [sequence for sequence in states.nucleotides.values() if "-" in sequence[30:39]]
    assert with_gaps, "treetime should place the deletion at the ancestors of the deleted clade"


def test_a_backend_that_is_not_installed_says_so(tmp_path) -> None:
    tree, sequences = synthetic(n_leaves=4, length=30)
    alignment = tmp_path / "aln.fasta"
    fasta.write_alignment(alignment, sequences)
    backend = asr.get_backend("iqtree", executable="definitely-not-a-real-binary")
    with pytest.raises(asr.BackendUnavailable):
        backend.reconstruct(tree, alignment, tmp_path)
