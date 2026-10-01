"""The vanished-clade diagnostic (af.tree.clade_coverage). Synthetic data only.

It answers the question that nearly slipped through on the real H3 tree: which clades do the leaves'
own calls say are there, that no node of the tree carries? There, 3,697 leaves were called G.2.2 and
no node was labelled it, while every other check passed (notes/trees/G2-SPLIT.md).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from af.tree.clade_coverage import find_vanished_clades, read_expected
from af.tree.io import i6
from af.tree.populate import CladeCall, CladeResult, populate

from .tree_fixtures import KEYS, built, records, states_for


def version(tmp_path: Path, labels: dict[str, str]) -> Path:
    """An I6 version whose nodes carry `labels` (by letter from tree_fixtures), others nothing."""
    tree, ids = built()
    # The engine is handed CladeInput, whose node_id is the stable hex id, not the integer.
    by_id = {f"{ids[letter]:016x}": clade for letter, clade in labels.items()}

    def assign(nodes):
        # The engine must answer for every node; None means "no clade here".
        calls = {node.node_id: CladeCall(by_id.get(node.node_id)) for node in nodes}
        return CladeResult("synthetic@1", calls)

    result = populate(tree, "h3", records(), states_for(ids), assign_clades=assign)
    directory = tmp_path / "version"
    directory.mkdir()
    i6.write(result, directory, "weekly")
    return directory


def test_a_clade_the_leaves_claim_and_no_node_carries_is_reported(tmp_path: Path) -> None:
    """The tree labels the clade's own node 'P'; the calls say its leaves are 'Q'."""
    directory = version(tmp_path, {"root": "P", "x": "P", "z": "P"})
    expected = {KEYS["c"]: "Q", KEYS["d"]: "Q", KEYS["a"]: "P", KEYS["b"]: "P"}
    found = find_vanished_clades(directory, expected, min_leaves=2)
    assert [item.clade for item in found] == ["Q"]
    vanished = found[0]
    assert vanished.expected_leaves == 2
    assert vanished.holder_label == "P"  # what the tree says instead
    assert vanished.holder_from_this_clade == 2
    assert vanished.purity == 1.0
    assert "no node labelled it" in vanished.describe()
    assert vanished.siblings  # the holder's parent has other children, which explain it
    assert vanished.to_json()["holder_purity"] == 1.0


def test_a_clade_the_tree_does_carry_is_not_reported(tmp_path: Path) -> None:
    directory = version(tmp_path, {"root": "P", "z": "Q"})
    expected = {KEYS["c"]: "Q", KEYS["d"]: "Q", KEYS["a"]: "P"}
    assert find_vanished_clades(directory, expected, min_leaves=2) == []


def test_a_clade_below_the_threshold_is_not_reported(tmp_path: Path) -> None:
    directory = version(tmp_path, {"root": "P"})
    expected = {KEYS["c"]: "Q", KEYS["d"]: "Q"}
    assert find_vanished_clades(directory, expected, min_leaves=3) == []
    assert [i.clade for i in find_vanished_clades(directory, expected, min_leaves=2)] == ["Q"]


def test_leaves_with_no_call_and_leaves_not_in_the_tree_are_ignored(tmp_path: Path) -> None:
    directory = version(tmp_path, {"root": "P"})
    expected = {KEYS["c"]: "Q", KEYS["d"]: "Q", "EPI_ISL_999|EPI999": "Q"}
    found = find_vanished_clades(directory, expected, min_leaves=2)
    assert found[0].expected_leaves == 2  # the key that is not a leaf of this tree is not counted


def test_expected_calls_read_from_a_tsv_with_or_without_a_header(tmp_path: Path) -> None:
    with_header = tmp_path / "a.tsv"
    with_header.write_text("leaf_id\tclade\nEPI_ISL_1|EPI1\tG.2.2\nEPI_ISL_2|EPI2\t\n")
    assert read_expected(with_header) == {"EPI_ISL_1|EPI1": "G.2.2"}
    without = tmp_path / "b.tsv"
    without.write_text("EPI_ISL_1|EPI1\tG.2.2\n")
    assert read_expected(without) == {"EPI_ISL_1|EPI1": "G.2.2"}
    empty = tmp_path / "empty.tsv"
    empty.write_text("")
    with pytest.raises(ValueError, match="no rows"):
        read_expected(empty)
