"""Building and finishing a tree.

The CMAPLE tests run the real thing when it is installed and skip otherwise. The finishing tests
(root, collapse, ladderize) need no tool and always run, because that is where the counts a
provenance record relies on are produced.
"""

from __future__ import annotations

import random
import shutil

import pytest

from af.run import JobFailed
from af.tree import build
from af.tree.build import CmapleSettings
from af.tree.io import fasta, newick


def alignment_of(tmp_path, n_leaves: int = 20, length: int = 400, seed: int = 3):
    """A small alignment with an outgroup that really is an outgroup."""
    rng = random.Random(seed)
    root = [rng.choice("ACGT") for _ in range(length)]
    sequences = {}
    # The outgroup is deliberately far from the rest.
    outgroup = root.copy()
    for _ in range(length // 8):
        outgroup[rng.randrange(length)] = rng.choice("ACGT")
    sequences["outgroup"] = "".join(outgroup)
    for index in range(n_leaves):
        mine = root.copy()
        for _ in range(rng.randint(1, 6)):
            mine[rng.randrange(length)] = rng.choice("ACGT")
        sequences[f"leaf{index:03d}"] = "".join(mine)
    path = tmp_path / "aln.fasta"
    fasta.write_alignment(path, sequences)
    return path, sequences


# ---- finishing (no external tool) --------------------------------------------------


def test_finishing_roots_collapses_and_counts() -> None:
    tree = newick.loads("(((a:1,b:1):0,(c:1,d:1):1):1,outgroup:1);")
    result = build.finish_tree(tree, outgroup="outgroup")
    assert "outgroup" in {child.name for child in result.tree.root.children}
    assert result.counts["branches_collapsed"] == 1
    assert result.counts["leaves"] == 5
    assert result.counts["collapse_tolerance"] == 0.0


def test_finishing_never_loses_a_leaf() -> None:
    tree = newick.loads("(((a:1,b:1):0,(c:1,d:1):0):0,outgroup:1);")
    result = build.finish_tree(tree, outgroup="outgroup")
    assert sorted(leaf.name or "" for leaf in result.tree.leaves()) == [
        "a",
        "b",
        "c",
        "d",
        "outgroup",
    ]
    assert result.counts["leaves"] == 5


def test_finishing_assigns_ids_that_survive_the_ladderizing() -> None:
    tree = newick.loads("(((a:1,b:1):1,(c:1,d:1):1):1,outgroup:1);")
    result = build.finish_tree(tree, outgroup="outgroup")
    before = sorted(node.node_id for node in result.tree.internal())
    result.tree.ladderize(smallest_first=False)
    result.tree.assign_ids()
    assert sorted(node.node_id for node in result.tree.internal()) == before


def test_the_collapse_tolerance_is_reported_even_when_it_does_nothing() -> None:
    """An inert threshold should be visible: 5e-5 never fired on any delivered tree."""
    tree = newick.loads("(((a:1,b:1):0.1,(c:1,d:1):0.1):0.1,outgroup:1);")
    result = build.finish_tree(tree, outgroup="outgroup", collapse_tolerance=5e-5)
    assert result.counts["branches_collapsed"] == 0
    assert result.counts["collapse_tolerance"] == 5e-5


# ---- settings ----------------------------------------------------------------------


def test_an_unknown_search_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="search must be one of"):
        CmapleSettings(search="THOROUGH")


def test_a_non_positive_minimum_branch_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        CmapleSettings(min_branch_length=0.0)


# ---- CMAPLE itself -----------------------------------------------------------------


@pytest.mark.skipif(shutil.which("cmaple") is None, reason="cmaple is not installed")
def test_cmaple_builds_a_tree_with_every_sequence(tmp_path) -> None:
    alignment, sequences = alignment_of(tmp_path)
    result = build.build(
        alignment,
        outgroup="outgroup",
        out_dir=tmp_path / "out",
        settings=CmapleSettings(threads=2),
    )
    assert sorted(leaf.name or "" for leaf in result.tree.leaves()) == sorted(sequences)
    assert result.counts["builder"].startswith("cmaple/")  # type: ignore[union-attr]
    assert result.counts["from_scratch"] == "yes"
    assert "outgroup" in {child.name for child in result.tree.root.children}


@pytest.mark.skipif(shutil.which("cmaple") is None, reason="cmaple is not installed")
def test_identical_sequences_are_all_kept(tmp_path) -> None:
    """Sarah's requirement: no deduplication. Every EPI id has to reach the tree."""
    _, sequences = alignment_of(tmp_path, n_leaves=6)
    shared = sequences["leaf000"]
    for name in ("leaf001", "leaf002", "leaf003"):
        sequences[name] = shared
    alignment = tmp_path / "dup.fasta"
    fasta.write_alignment(alignment, sequences)
    result = build.build(
        alignment,
        outgroup="outgroup",
        out_dir=tmp_path / "out",
        settings=CmapleSettings(threads=2),
    )
    assert sorted(leaf.name or "" for leaf in result.tree.leaves()) == sorted(sequences)


@pytest.mark.skipif(shutil.which("cmaple") is None, reason="cmaple is not installed")
def test_an_incremental_build_starts_from_the_previous_tree(tmp_path) -> None:
    alignment, sequences = alignment_of(tmp_path, n_leaves=12)
    first = build.build(
        alignment,
        outgroup="outgroup",
        out_dir=tmp_path / "one",
        settings=CmapleSettings(threads=2),
    )
    # Drop two leaves, as a week's selection does, and reuse the tree.
    keep = [name for name in sequences if name not in {"leaf000", "leaf001"}]
    smaller = tmp_path / "smaller.fasta"
    fasta.write_alignment(smaller, {name: sequences[name] for name in keep})
    start, removed = build.prune_starting_tree(first.tree, keep, tmp_path / "start.nwk")
    assert removed == 2

    second = build.build(
        smaller,
        outgroup="outgroup",
        out_dir=tmp_path / "two",
        settings=CmapleSettings(threads=2),
        starting_tree=start,
    )
    assert sorted(leaf.name or "" for leaf in second.tree.leaves()) == sorted(keep)
    assert second.counts["from_scratch"] == "no"


def test_a_failed_build_raises_rather_than_reporting_success(tmp_path) -> None:
    """Production's make-cmaple mails "completed" and exits 0 when CMAPLE failed."""
    alignment, _ = alignment_of(tmp_path, n_leaves=4, length=60)
    settings = CmapleSettings(executable="definitely-not-a-real-binary")
    with pytest.raises((JobFailed, FileNotFoundError, OSError)):
        build.build(alignment, outgroup="outgroup", out_dir=tmp_path / "out", settings=settings)
