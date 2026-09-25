"""The clade engine against real WHO CC trees, through the private data repo.

The synthetic tests say the engine follows its own rules. This one says those rules give
the right answer on real influenza trees, which is the claim the engine was chosen on
(``notes/clades/ENGINE-COMPARISON.md``): agreement with Nextclade of 99.6% on H3, 99.2%
on H1 and 99.8% on B/Vic, against 93.0/88.8/94.9% for the labels ae produces today.

The fixture is a subsample of the September 2026 round's trees, in acmacs-f-data
(``fixtures/clades/``), so nothing real is in this public repo; these tests skip when that
repo is absent, but **fail** when it is present and no longer matches the nomenclature the
clones are on, because that is precisely when the engine needs re-measuring.

Because af labels a node from its own sequence and its parent's label, a leaf's label
depends only on the path from the root, so the subsample reproduces the full tree's labels
exactly — hence per-leaf equality below rather than a tolerance.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from af.clades.assign import Node, TreeAssignment, assign_tree
from af.clades.nomenclature import load_clade_set
from af.clades.sequence import AlignedSequence, GapSupport

CLONES = Path.home() / "AC/eu/influenza-clade-nomenclature"
#: how far the fixture's agreement may sit from the whole tree's, in percentage points
TOLERANCE = 0.5


def fixture_directory(af_data: Path) -> Path:
    directory = af_data / "fixtures" / "clades"
    if not (directory / "expected.json").is_file():
        pytest.skip(f"clade fixture not found at {directory}")
    return directory


def expectations(af_data: Path) -> dict:
    with (fixture_directory(af_data) / "expected.json").open() as stream:
        return json.load(stream)


def subtype_keys(af_data: Path) -> list[str]:
    return sorted(expectations(af_data)["subtypes"])


def build_nodes(directory: Path, key: str, gaps: GapSupport) -> list[Node]:
    """The fixture's induced subtree: internal nodes at the reconstruction's gap support,
    leaves always observed."""
    nodes: list[Node] = []
    with (directory / f"{key}.nodes.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            nodes.append(
                Node(
                    row["label"],
                    row["parent"] or None,
                    AlignedSequence.from_nucleotides(row["nucleotides"], gaps=gaps),
                )
            )
    with (directory / f"{key}.leaves.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            nodes.append(
                Node(
                    row["label"],
                    row["parent"],
                    AlignedSequence.from_nucleotides(row["nucleotides"]),
                )
            )
    return nodes


def leaf_rows(directory: Path, key: str) -> dict[str, dict[str, str]]:
    with (directory / f"{key}.leaves.tsv").open() as stream:
        return {row["label"]: row for row in csv.DictReader(stream, delimiter="\t")}


def run(af_data: Path, key: str) -> tuple[TreeAssignment, dict, dict[str, dict[str, str]]]:
    directory = fixture_directory(af_data)
    expected = expectations(af_data)["subtypes"][key]
    if not CLONES.is_dir():
        pytest.skip(f"nomenclature clones not found at {CLONES}")
    clade_set = load_clade_set(expected["subtype"], CLONES)
    if clade_set.version != expected["clade_set_version"]:
        # Not a skip. The day the pin moves is the day this test is most worth running,
        # and a skipped test reads as a passing one: the fixture would sit unregenerated
        # and unnoticed. Fail, and say exactly how to put it right.
        raise AssertionError(
            f"the clade fixture and the nomenclature clones disagree for {key}.\n"
            f"  fixture built at: {expected['clade_set_version']}\n"
            f"  clones now at:    {clade_set.version}\n"
            "Regenerate the fixture and commit it to acmacs-f-data, recording in the commit "
            "message what moved and why any number changed:\n"
            "  python3 $AF_DATA/fixtures/clades/make_fixture.py\n"
            "If the clones moved by accident, check out the pinned commit instead."
        )
    gaps = GapSupport(expected["gap_support"])
    result = assign_tree(build_nodes(directory, key, gaps), clade_set, require_gap_support=False)
    return result, expected, leaf_rows(directory, key)


@pytest.mark.parametrize("key", ["h3", "h1", "bvic"])
def test_labels_match_the_recorded_assignment(af_data: Path, key: str) -> None:
    """Every leaf keeps the clade recorded when the fixture was built: an exact
    regression test, so any change in the engine's behaviour names the viruses it moved."""
    result, expected, _ = run(af_data, key)
    moved = {
        leaf: (recorded, result.clade(leaf))
        for leaf, recorded in expected["labels"].items()
        if result.clade(leaf) != recorded
    }
    assert not moved, f"{len(moved)} leaves changed clade, e.g. {dict(list(moved.items())[:5])}"


@pytest.mark.parametrize("key", ["h3", "h1", "bvic"])
def test_agreement_with_nextclade_holds(af_data: Path, key: str) -> None:
    """The measurement the engine was chosen on, and the reason the fixture exists."""
    result, expected, rows = run(af_data, key)
    agreeing = sum(1 for leaf, row in rows.items() if result.clade(leaf) == row["nextclade"])
    percent = 100 * agreeing / len(rows)
    assert agreeing == expected["agreement_on_fixture"]
    assert abs(percent - expected["whole_tree_agreement_percent"]) <= TOLERANCE


@pytest.mark.parametrize("key", ["h3", "h1", "bvic"])
def test_every_assigned_clade_is_within_its_parents_clade(af_data: Path, key: str) -> None:
    """Descent must hold on real data too: a node's clade is its parent's or below it.

    This is the property that makes a single stored label safe. If it broke, a selector
    asking for a clade would silently miss viruses sitting under it.
    """
    directory = fixture_directory(af_data)
    result, expected, _ = run(af_data, key)
    clade_set = load_clade_set(expected["subtype"], CLONES)
    parents: dict[str, str | None] = {}
    for suffix in ("nodes.tsv", "leaves.tsv"):
        with (directory / f"{key}.{suffix}").open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                parents[row["label"]] = row["parent"] or None
    checked = 0
    for node, assignment in result.assignments.items():
        parent = parents.get(node)
        if assignment.clade is None or parent is None:
            continue
        parent_clade = result.assignments[parent].clade
        if parent_clade is None:
            continue
        assert clade_set.is_within(assignment.clade, parent_clade), (
            f"{node} is {assignment.clade}, which is not within its parent {parent}'s "
            f"{parent_clade}"
        )
        checked += 1
    assert checked > len(expected["labels"]) // 2, "too few parent-child pairs checked"


def test_bvic_uses_gap_aware_states(af_data: Path) -> None:
    """B/Vic defines clades by a deletion, so its fixture must carry states from a
    reconstruction that can represent one; otherwise the figures above mean nothing."""
    expected = expectations(af_data)["subtypes"]["bvic"]
    assert expected["gap_support"] == GapSupport.OBSERVED.value
    assert expected["asr_states"].startswith("treetime")
