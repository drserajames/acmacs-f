"""i6.read: a published version back as the PopulatedTree that wrote it. Synthetic only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from af.tree.io import i6
from af.tree.populate import CladeCall, CladeResult, populate
from af.tree.report_filter import report_tree

from .test_report_filter import CUTOFF, index
from .tree_fixtures import KEYS, LEAF_SEQ, built, records, states_for

ALIGNMENT = {KEYS[n]: LEAF_SEQ[n] for n in "oabcd"}


def labelled(scale: str = "mutations"):
    tree, ids = built()

    def engine(nodes):
        calls = {n.node_id: CladeCall("J" if n.is_leaf else "J.2", support=2) for n in nodes}
        return CladeResult("synthetic@1", calls, {"J": None, "J.2": "J"})

    return populate(
        tree,
        "h3",
        records(),
        states_for(ids),
        branch_scale=scale,  # type: ignore[arg-type]
        assign_clades=engine,
        continent_of=lambda record: "EUROPE" if record.region == "Europe" else None,
        outgroup=KEYS["o"],
    )


def round_trip(populated, tmp_path: Path) -> tuple[Path, Path]:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    i6.write(populated, first, "weekly")
    i6.write(i6.read(first, ALIGNMENT), second, "weekly")
    return first, second


@pytest.mark.parametrize("scale", ["mutations", "ml"])
def test_a_read_version_writes_back_identically(tmp_path: Path, scale: str) -> None:
    first, second = round_trip(labelled(scale), tmp_path)
    for name in (i6.NODES_FILE, i6.TREE_FILE, i6.ANCESTRAL_FILE):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    assert json.loads((first / i6.META_FILE).read_text()) == json.loads(
        (second / i6.META_FILE).read_text()
    )


def test_a_report_tree_with_titres_round_trips(tmp_path: Path) -> None:
    report = report_tree(labelled("ml"), CUTOFF, index())
    first, second = round_trip(report, tmp_path)
    assert (first / i6.NODES_FILE).read_bytes() == (second / i6.NODES_FILE).read_bytes()


def test_a_report_tree_from_a_read_version_equals_one_from_the_original(tmp_path: Path) -> None:
    """What the report stage relies on: derive from the published version, not the in-memory one."""
    original = labelled("ml")
    (tmp_path / "w").mkdir()
    i6.write(original, tmp_path / "w", "weekly")
    from_store = report_tree(i6.read(tmp_path / "w", ALIGNMENT), CUTOFF, index())
    in_memory = report_tree(original, CUTOFF, index())
    for directory, populated in (("a", from_store), ("b", in_memory)):
        (tmp_path / directory).mkdir()
        i6.write(populated, tmp_path / directory, "report")
    assert (tmp_path / "a" / i6.NODES_FILE).read_bytes() == (
        tmp_path / "b" / i6.NODES_FILE
    ).read_bytes()


def test_a_leaf_missing_from_the_alignment_is_an_error(tmp_path: Path) -> None:
    i6.write(labelled(), tmp_path, "weekly")
    with pytest.raises(i6.I6Error, match="no sequence"):
        i6.read(tmp_path, {k: v for k, v in ALIGNMENT.items() if k != KEYS["a"]})


def test_an_internal_label_that_is_not_its_id_is_an_error(tmp_path: Path) -> None:
    i6.write(labelled(), tmp_path, "weekly")
    path = tmp_path / i6.TREE_FILE
    node = next(n for n in i6.read(tmp_path, ALIGNMENT).tree.internal() if n.parent is not None)
    path.write_text(path.read_text().replace(node.id_hex, "0" * 16, 1))
    with pytest.raises(i6.I6Error, match="is not its id"):
        i6.read(tmp_path, ALIGNMENT)


def test_the_build_record_survives_reading_and_the_report_filter(tmp_path: Path) -> None:
    populated = labelled("ml")
    populated.build = {"backend": "cmaple", "version": "cmaple/9.9", "parameters": {}}
    (tmp_path / "w").mkdir()
    i6.write(populated, tmp_path / "w", "weekly")
    reread = i6.read(tmp_path / "w", ALIGNMENT)
    assert reread.build == populated.build
    assert report_tree(reread, CUTOFF, index()).build == populated.build


def test_no_build_record_writes_no_build_key(tmp_path: Path) -> None:
    i6.write(labelled(), tmp_path, "weekly")
    assert "build" not in i6.read_metadata(tmp_path)
