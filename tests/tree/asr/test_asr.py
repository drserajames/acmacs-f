"""The ASR interface: translation, substitutions, clade matching, and the parsimony backend.

The external backends (TreeTime, IQ-TREE, raxml-ng) are exercised in test_backends_live.py, which
skips when the tool is not installed. What is tested here is everything that must hold whichever
backend ran — and the clade matching that is what makes them interchangeable.
"""

from __future__ import annotations

import pytest

from af.tree import asr
from af.tree.asr.matching import MatchError, match_states_by_clade
from af.tree.io import fasta, newick
from af.tree.model import Tree


def tree_of(text: str) -> Tree:
    tree = newick.loads(text)
    tree.assign_ids()
    return tree


# ---- translation -------------------------------------------------------------------


def test_translate_keeps_a_deleted_codon() -> None:
    """Biopython turns '---' into X and loses the deletion; B/Vic's clades are defined by it."""
    assert asr.translate("AAA---TTT") == "K-F"


def test_translate_marks_a_part_gapped_codon_unknown() -> None:
    assert asr.translate("AAA-TTTTT") == "KXF"


def test_translate_handles_ambiguity_without_raising() -> None:
    assert asr.translate("AAANNNTTT") == "KXF"


# ---- substitutions -----------------------------------------------------------------


def test_substitutions_are_reported_per_branch() -> None:
    tree = tree_of("((a:1,b:1)inner:1,c:1);")
    inner = next(node for node in tree.internal() if node is not tree.root)
    states = asr.AncestralStates(
        nucleotides={tree.root.node_id: "AAA", inner.node_id: "AAC"},
        backend="test",
        backend_version="0",
        seconds=0.0,
    )
    leaves = {"a": "AAC", "b": "AAG", "c": "AAA"}
    subs = states.substitutions(tree, leaves)
    assert [str(s) for s in subs[(tree.root.node_id, inner.node_id)]] == ["K1N"]
    leaf_b = next(leaf for leaf in tree.leaves() if leaf.name == "b")
    assert [str(s) for s in subs[(inner.node_id, leaf_b.node_id)]] == ["N1K"]


def test_an_unknown_state_is_not_a_substitution() -> None:
    tree = tree_of("(a:1,b:1);")
    states = asr.AncestralStates(
        nucleotides={tree.root.node_id: "NNN"}, backend="t", backend_version="0", seconds=0.0
    )
    assert states.substitutions(tree, {"a": "AAA", "b": "AAA"}) == {}


def test_substitutions_can_be_restricted_to_chosen_positions() -> None:
    tree = tree_of("(a:1,b:1);")
    states = asr.AncestralStates(
        nucleotides={tree.root.node_id: "AAAAAA"}, backend="t", backend_version="0", seconds=0.0
    )
    leaves = {"a": "AAAAAC", "b": "AAAAAA"}  # only the second codon differs (K -> N)
    assert states.substitutions(tree, leaves, positions=[1]) == {}
    assert len(states.substitutions(tree, leaves, positions=[2])) == 1


# ---- backend selection -------------------------------------------------------------


def test_the_default_backend_is_treetime() -> None:
    assert asr.DEFAULT_BACKEND == "treetime"
    assert asr.get_backend().name == "treetime"


def test_an_unknown_backend_is_an_error_not_a_fallback() -> None:
    with pytest.raises(ValueError, match="unknown ASR backend"):
        asr.get_backend("magic")


def test_backend_from_config_passes_options_through() -> None:
    backend = asr.backend_from_config({"backend": "raxml", "model": "GTR+G"})
    assert backend.name == "raxml" and backend.model == "GTR+G"  # type: ignore[attr-defined]


def test_no_backend_optimises_branch_lengths() -> None:
    """The lengths come from CMAPLE; re-optimising them departs from the tree that was built."""
    for name in asr.available_backends():
        assert asr.get_backend(name).optimises_branch_lengths is False


def test_a_gap_blind_backend_is_refused_where_deletions_define_clades() -> None:
    """B/Vic has 6 deletion-defined mutations; raxml-ng and IQ-TREE reconstruct no gaps at all."""
    for name in ("raxml", "iqtree"):
        with pytest.raises(asr.GapCapabilityError, match="cannot reconstruct deletions"):
            asr.check_gap_capability(asr.get_backend(name), [163, 164])


def test_gap_capable_backends_pass_the_same_check() -> None:
    for name in ("treetime", "parsimony"):
        asr.check_gap_capability(asr.get_backend(name), [163, 164])


def test_the_check_passes_when_no_clade_is_defined_by_a_deletion() -> None:
    """H3's defining mutations are all ordinary residues, so any backend is allowed."""
    asr.check_gap_capability(asr.get_backend("iqtree"), [])


# ---- clade matching ----------------------------------------------------------------


def test_states_match_even_when_the_tool_renames_and_reroots(tmp_path) -> None:
    """The reason backends are interchangeable: nodes are matched by leaf set, not by name."""
    tree = tree_of("((a:1,b:1):1,(c:1,d:1):1);")
    tool = tmp_path / "tool.nwk"
    # Same tree, rooted on 'a', with the tool's own node names.
    tool.write_text("(a:1,(b:1,(c:1,d:1)N2:1)N1:1);")
    states = match_states_by_clade(tree, tool, {"N1": "AAA", "N2": "CCC"})
    cd = next(
        node for node in tree.internal() if {leaf.name for leaf in _leaves_of(node)} == {"c", "d"}
    )
    assert states[cd.node_id] == "CCC"


def test_a_tool_tree_with_different_leaves_is_an_error(tmp_path) -> None:
    tree = tree_of("((a:1,b:1):1,c:1);")
    tool = tmp_path / "tool.nwk"
    tool.write_text("((a:1,b:1)N1:1,zzz:1);")
    with pytest.raises(MatchError):
        match_states_by_clade(tree, tool, {"N1": "AAA"})


def test_a_clade_and_its_complement_do_not_both_claim_one_tool_node(tmp_path) -> None:
    """Regression: complement matching once gave (c,d) the sequence belonging to (a,b).

    In a rooted tree a clade's complement is also a node, and matching a split in both orientations
    without spending the tool's node made one af node quietly receive another's states.
    """
    tree = tree_of("((a:1,b:1):1,(c:1,d:1):1);")
    tool = tmp_path / "tool.nwk"
    tool.write_text("((a:1,b:1)N1:1,(c:1,d:1):1);")
    states = match_states_by_clade(tree, tool, {"N1": "AAA"})
    assert len(states) == 1
    ab = next(
        node for node in tree.internal() if {leaf.name for leaf in _leaves_of(node)} == {"a", "b"}
    )
    assert set(states) == {ab.node_id}


def test_nexus_output_is_read(tmp_path) -> None:
    """TreeTime writes its annotated tree as NEXUS."""
    tree = tree_of("((a:1,b:1):1,c:1);")
    tool = tmp_path / "tool.nexus"
    tool.write_text("#NEXUS\nBegin Trees;\nTree tree1=((a:1,b:1)NODE_1:1,c:1);\nEnd;\n")
    states = match_states_by_clade(tree, tool, {"NODE_1": "GGG"})
    assert list(states.values()) == ["GGG"]


def test_unmatched_nodes_are_counted_not_hidden(tmp_path) -> None:
    """IQ-TREE returned 13,476 of 13,477 real nodes; name matching would have hidden the gap."""
    tree = tree_of("((a:1,b:1):1,(c:1,d:1):1);")
    tool = tmp_path / "tool.nwk"
    tool.write_text("((a:1,b:1)N1:1,(c:1,d:1):1);")
    states = match_states_by_clade(tree, tool, {"N1": "AAA"})
    assert asr.unmatched_count(tree, states) == 2


# ---- the parsimony backend ---------------------------------------------------------


def test_parsimony_reconstructs_an_obvious_ancestor(tmp_path) -> None:
    tree = tree_of("((a:1,b:1):1,(c:1,d:1):1);")
    alignment = tmp_path / "aln.fasta"
    fasta.write_alignment(alignment, {"a": "AAAT", "b": "AAAT", "c": "AAAT", "d": "AAAT"})
    states = asr.get_backend("parsimony").reconstruct(tree, alignment, tmp_path)
    assert set(states.nucleotides.values()) == {"AAAT"}
    assert asr.unmatched_count(tree, states.nucleotides) == 0


def test_parsimony_can_place_a_deletion_at_an_internal_node(tmp_path) -> None:
    """Unlike raxml-ng and IQ-TREE, which have no gap state at all."""
    tree = tree_of("((a:1,b:1):1,(c:1,d:1):1);")
    alignment = tmp_path / "aln.fasta"
    fasta.write_alignment(alignment, {"a": "AAA---", "b": "AAA---", "c": "AAACCC", "d": "AAACCC"})
    states = asr.get_backend("parsimony").reconstruct(tree, alignment, tmp_path)
    ab = next(
        node for node in tree.internal() if {leaf.name for leaf in _leaves_of(node)} == {"a", "b"}
    )
    assert states.nucleotides[ab.node_id] == "AAA---"
    assert asr.translate(states.nucleotides[ab.node_id]) == "K-"


def test_parsimony_is_deterministic(tmp_path) -> None:
    tree = tree_of("((a:1,b:1):1,(c:1,d:1):1);")
    alignment = tmp_path / "aln.fasta"
    fasta.write_alignment(alignment, {"a": "AAA", "b": "CCC", "c": "GGG", "d": "TTT"})
    backend = asr.get_backend("parsimony")
    first = backend.reconstruct(tree, alignment, tmp_path).nucleotides
    second = backend.reconstruct(tree, alignment, tmp_path).nucleotides
    assert first == second


def test_a_leaf_with_no_sequence_is_an_error(tmp_path) -> None:
    tree = tree_of("((a:1,b:1):1,c:1);")
    alignment = tmp_path / "aln.fasta"
    fasta.write_alignment(alignment, {"a": "AAA", "b": "AAA"})
    with pytest.raises(Exception, match="no sequence"):
        asr.get_backend("parsimony").reconstruct(tree, alignment, tmp_path)


def _leaves_of(node):  # noqa: ANN001, ANN202 - test helper
    stack = [node]
    while stack:
        current = stack.pop()
        if current.is_leaf:
            yield current
        stack.extend(current.children)
