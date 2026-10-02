"""Cutting the tree at a node (af.tree.cut). Synthetic data only; every name is invented.

The synthetic tree is the real situation in miniature (``notes/trees/CUT-NODE.md``): a modern clade
of three children that holds nearly all the recently collected leaves, an old lineage that holds
none of them -- except one recent newcomer, which is exactly what made tolerance 0 useless on the
real H3 and H1 trees -- and an outgroup, which is above every cut by design.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from af.tree import cut
from af.tree.build import finish_tree
from af.tree.io import i6, newick
from af.tree.populate import LeafRecord, leaf_key, populate

OUTGROUP = 0
OLD = (1, 2, 3)
NEWCOMER = 4  # recent, but up in the old lineage: the reason a tolerance is needed at all
BIG, MIDDLE, SMALL = range(5, 11), range(11, 15), range(15, 19)
CYCLE = "2026-09-NH"

DATES = {
    OUTGROUP: datetime.date(2009, 6, 10),
    **{n: datetime.date(2019, 3 + n, 10) for n in OLD},
    NEWCOMER: datetime.date(2026, 4, 10),
    **{n: datetime.date(2026, 1, 10) for n in BIG},
    **{n: datetime.date(2025, 6, 10) for n in MIDDLE},
    **{n: datetime.date(2025, 9, 10) for n in SMALL},
}


def key(number: int) -> str:
    return leaf_key(f"EPI_ISL_7{number:04d}", f"EPI7{number:04d}")


def pair(number: int) -> tuple[str, str]:
    return (f"EPI_ISL_7{number:04d}", f"EPI7{number:04d}")


def clade(numbers: Iterable[int], length: float = 0.02) -> str:
    return "(" + ",".join(f"{key(n)}:{length}" for n in numbers) + f"):{length}"


def shape(*, newcomer_under_the_cut: bool = False) -> str:
    """The default tree, or one with the newcomer moved down into the modern clade."""
    old = clade(OLD) if newcomer_under_the_cut else f"({clade(OLD)},{key(NEWCOMER)}:0.03):0.05"
    modern = [clade(BIG), clade(MIDDLE), clade(SMALL)]
    if newcomer_under_the_cut:
        modern.append(f"{key(NEWCOMER)}:0.03")
    return f"({key(OUTGROUP)}:0.3,{old},({','.join(modern)}):0.06);"


def version(directory: Path, *, drop: tuple[int, ...] = (), text: str | None = None) -> Path:
    """An I6 version of the synthetic tree, without the leaves in ``drop``."""
    tree = finish_tree(newick.loads(text or shape()), outgroup=key(OUTGROUP)).tree
    if drop:
        # As a rebuild would: the leaves are gone, unary nodes go with them, and the ids are
        # recomputed -- so a node's id legitimately changes, which is what the pin has to survive.
        gone = {key(number) for number in drop}
        tree.prune([leaf.name or "" for leaf in tree.leaves() if leaf.name not in gone])
        tree.remove_unary()
        tree.ladderize()
        tree.assign_ids()
    leaves = {
        key(number): LeafRecord(
            epi_isl=pair(number)[0],
            accession=pair(number)[1],
            name=f"A/EXAMPLETOWN/{number}/{day.year}",
            nucleotides="ACGTACGTAC",
            collection_date=day,
            date_precision="day",
            collection_date_first=day,
            collection_date_last=day,
            country="EXAMPLELAND",
            region="Europe",
        )
        for number, day in DATES.items()
    }
    # No ASR: a cut is a question about topology and dates, so the tests do not invent states.
    populated = populate(tree, "h3", leaves, None, outgroup=key(OUTGROUP))
    directory.mkdir(parents=True, exist_ok=True)
    i6.write(populated, directory, "test-cut", source={"outgroup": key(OUTGROUP)})
    return directory


def leaf_node_id(directory: Path, number: int) -> str:
    """A leaf's stable node id, which is not its key: the cut CLI is given node ids."""
    rows = i6.read_nodes(directory, columns=["node_id", "leaf_id"]).to_pylist()
    return next(row["node_id"] for row in rows if row["leaf_id"] == key(number))


def cut_node(directory: Path, tolerance: float = 0.1) -> cut.CutProposal:
    (proposal,) = cut.propose(directory, tolerances=(tolerance,))
    return proposal


def write_pin(
    directory: Path,
    tree: Path,
    *,
    anchors: tuple[tuple[str, str], ...],
    node_id: str,
    below: frozenset[tuple[str, str]],
    quorum: int = 2,
    expected_leaves: int | None = None,
    expected_children: int = 3,
    reason: str = "the synthetic cut, for a test",
    subtype: str = "h3",
    cycle: str = CYCLE,
) -> Path:
    """A cuts file as acmacs-f-data would hold it, plus its below-list."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "below.tsv").write_text(
        "leaf_id\n" + "".join(f"{epi}|{acc}\n" for epi, acc in sorted(below))
    )
    reference = i6.read_metadata(tree)
    assert reference["outgroup"] == key(OUTGROUP)  # the version really does record it
    lines = [
        "[[cut]]",
        f'subtype = "{subtype}"',
        f'cycle = "{cycle}"',
        f'node_id = "{node_id}"',
        "chosen_on = 2026-09-30",
        'chosen_from = { kind = "sequences", dataset = "h3", version = "0123456789abcdef", '
        'manifest_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" }',
        f"quorum = {quorum}",
        f"expected_leaves = {len(below) if expected_leaves is None else expected_leaves}",
        f"expected_children = {expected_children}",
        'below_file = "below.tsv"',
        f'reason = "{reason}"',
        "anchors = [",
        *[f'    "{epi}|{acc}",' for epi, acc in anchors],
        "]",
    ]
    path = directory / "cuts.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def pinned(tmp_path: Path, *, quorum: int = 2) -> tuple[Path, cut.CutPin]:
    """The default tree, with a pin taken from its own cut node."""
    tree = version(tmp_path / "v1")
    node = cut_node(tree).node_id
    path = write_pin(
        tmp_path,
        tree,
        anchors=cut.pick_anchors(tree, node, count=6),
        node_id=node,
        below=cut.below_keys(tree, node),
        quorum=quorum,
    )
    return tree, cut.load_cut_pin(path, CYCLE, "h3")


# ---------------------------------------------------------------------------------------------
# Proposing


def test_the_cut_is_the_deepest_node_holding_all_but_a_tolerated_fraction(tmp_path: Path) -> None:
    tree = version(tmp_path / "v1")
    proposal = cut_node(tree, 0.1)
    assert not proposal.is_root
    assert (proposal.leaves_below, proposal.leaves_total) == (14, 19)
    assert (proposal.recent_below, proposal.recent_total) == (14, 15)
    assert proposal.children == 3
    assert proposal.newest_above == DATES[NEWCOMER]  # the leaf the tolerance allowed above the cut
    assert "keeps 14 of 19" in proposal.describe()
    assert proposal.to_json()["leaves_above"] == 5


def test_one_recent_leaf_in_an_old_lineage_keeps_a_zero_tolerance_cut_at_the_root(
    tmp_path: Path,
) -> None:
    """Why tolerance 0 is in the table and not usable: it returned the root on H3 and H1 too."""
    tree = version(tmp_path / "v1")
    at_zero, at_tenth = cut.propose(tree, tolerances=(0.0, 0.1))
    assert at_zero.is_root and at_zero.leaves_below == 19
    assert at_zero.newest_above is None  # nothing is above the root
    assert not at_tenth.is_root
    # With that one leaf moved down into the modern clade, no tolerance is needed at all.
    moved = version(tmp_path / "v2", text=shape(newcomer_under_the_cut=True))
    assert not cut_node(moved, 0.0).is_root


def test_the_window_is_counted_from_the_newest_date_in_the_tree_not_today(tmp_path: Path) -> None:
    """Design rule 7. The newest leaf here is from 2026, so a 2026 run and a 2030 run agree."""
    tree = version(tmp_path / "v1")
    proposal = cut_node(tree)
    assert proposal.window_to == DATES[NEWCOMER]
    assert proposal.window_from == datetime.date(2024, 4, 10)
    assert cut.propose(tree, window_months=1, tolerances=(0.1,))[0].recent_total == 1


def test_a_tolerance_outside_zero_to_one_is_refused(tmp_path: Path) -> None:
    tree = version(tmp_path / "v1")
    with pytest.raises(cut.CutError, match="fractions"):
        cut.propose(tree, tolerances=(1.5,))


# ---------------------------------------------------------------------------------------------
# Anchors and the below-list


def test_anchors_are_spread_over_the_children_and_reproducible(tmp_path: Path) -> None:
    """The MRCA is the cut node only if the anchors span two children, so spread is the point."""
    tree = version(tmp_path / "v1")
    node = cut_node(tree).node_id
    anchors = cut.pick_anchors(tree, node, count=6)
    assert len(set(anchors)) == 6
    assert anchors == cut.pick_anchors(tree, node, count=6)  # deterministic (design rule 8)
    groups = {frozenset(BIG), frozenset(MIDDLE), frozenset(SMALL)}
    numbers = {int(epi.removeprefix("EPI_ISL_7")) for epi, _ in anchors}
    assert all(numbers & group for group in groups)  # every child is represented
    assert numbers <= set(BIG) | set(MIDDLE) | set(SMALL)  # all below the cut


def test_more_anchors_than_leaves_or_fewer_than_two_are_refused(tmp_path: Path) -> None:
    tree = version(tmp_path / "v1")
    node = cut_node(tree).node_id
    with pytest.raises(cut.CutError, match="at least 2"):
        cut.pick_anchors(tree, node, count=1)
    with pytest.raises(cut.CutError, match="anchors asked for"):
        cut.pick_anchors(tree, node, count=99)
    with pytest.raises(cut.CutError, match="is a leaf"):
        cut.pick_anchors(tree, leaf_node_id(tree, OUTGROUP), count=6)
    with pytest.raises(cut.CutError, match="no node"):
        cut.pick_anchors(tree, "no-such-node", count=6)


def test_the_below_list_is_every_leaf_under_the_node(tmp_path: Path) -> None:
    tree = version(tmp_path / "v1")
    below = cut.below_keys(tree, cut_node(tree).node_id)
    assert below == {pair(n) for n in (*BIG, *MIDDLE, *SMALL)}
    assert pair(OUTGROUP) not in below and pair(NEWCOMER) not in below


# ---------------------------------------------------------------------------------------------
# The pin as a file


def test_a_pin_round_trips_from_a_cuts_file(tmp_path: Path) -> None:
    tree, pin = pinned(tmp_path)
    assert (pin.subtype, pin.cycle, pin.quorum) == ("h3", CYCLE, 2)
    assert pin.chosen_on == datetime.date(2026, 9, 30)
    assert pin.chosen_from.version == "0123456789abcdef"
    assert len(pin.below) == 14 and len(pin.anchors) == 6
    assert pin.node_id == cut_node(tree).node_id


def test_a_cycle_or_subtype_with_no_row_is_an_error_not_a_default(tmp_path: Path) -> None:
    tree = version(tmp_path / "v1")
    node = cut_node(tree).node_id
    path = write_pin(
        tmp_path,
        tree,
        anchors=cut.pick_anchors(tree, node, count=6),
        node_id=node,
        below=cut.below_keys(tree, node),
    )
    with pytest.raises(cut.CutError, match="no cut for cycle"):
        cut.load_cut_pin(path, "2027-02-SH", "h3")
    with pytest.raises(cut.CutError, match="no cut for cycle"):
        cut.load_cut_pin(path, CYCLE, "h1")


def test_a_pin_that_contradicts_itself_is_refused(tmp_path: Path) -> None:
    tree = version(tmp_path / "v1")
    node = cut_node(tree).node_id
    anchors = cut.pick_anchors(tree, node, count=6)
    below = cut.below_keys(tree, node)
    with pytest.raises(cut.CutError, match="but below.tsv lists"):
        cut.load_cut_pin(
            write_pin(tmp_path / "a", tree, anchors=anchors, node_id=node, below=below,
                      expected_leaves=99),
            CYCLE, "h3",
        )  # fmt: skip
    with pytest.raises(cut.CutError, match="not in the below-list"):
        cut.load_cut_pin(
            write_pin(tmp_path / "b", tree, anchors=(*anchors[:5], pair(OUTGROUP)), node_id=node,
                      below=below),
            CYCLE, "h3",
        )  # fmt: skip
    with pytest.raises(cut.CutError, match="no reason"):
        cut.load_cut_pin(
            write_pin(tmp_path / "c", tree, anchors=anchors, node_id=node, below=below, reason=" "),
            CYCLE, "h3",
        )  # fmt: skip
    with pytest.raises(cut.CutError, match="quorum 1"):
        cut.load_cut_pin(
            write_pin(tmp_path / "d", tree, anchors=anchors, node_id=node, below=below, quorum=1),
            CYCLE, "h3",
        )  # fmt: skip


def test_selection_keeps_by_membership_not_by_date(tmp_path: Path) -> None:
    """A virus collected before the cut and submitted after it must not be dropped."""
    _, pin = pinned(tmp_path)
    below = next(iter(sorted(pin.below)))
    assert pin.keeps(below, known_at_the_cut=True)
    assert pin.keeps(pair(NEWCOMER), known_at_the_cut=False)  # the cut never saw it
    assert not pin.keeps(pair(NEWCOMER), known_at_the_cut=True)  # known, and above the cut


# ---------------------------------------------------------------------------------------------
# Recovering it


def test_the_mrca_of_the_anchors_is_the_cut_node(tmp_path: Path) -> None:
    tree, pin = pinned(tmp_path)
    recovered = cut.recover(tree, pin)
    assert recovered.node_id == pin.node_id and not recovered.node_id_changed
    assert (recovered.leaves_below, recovered.leaf_drift) == (14, 0.0)
    assert (recovered.anchors_found, recovered.anchors_missing) == (6, ())
    assert recovered.children_spanned == 3
    assert "all anchors found" in recovered.describe()


def test_anchors_lost_within_the_quorum_recover_and_are_named(tmp_path: Path) -> None:
    """GISAID records are withdrawn and renamed, so some loss is normal -- but never silent."""
    tree, pin = pinned(tmp_path)
    gone = sorted(pin.anchors)[:2]  # both from the largest child, so two children still have theirs
    thinned = version(
        tmp_path / "v2", drop=tuple(int(epi.removeprefix("EPI_ISL_7")) for epi, _ in gone)
    )
    # Two withdrawn leaves are 14% of this 14-leaf cut and 0.002% of the real H1 one, so the
    # drift limit is loosened to the scale of the toy tree rather than to excuse real drift.
    recovered = cut.recover(thinned, pin, max_leaf_drift=0.2)
    assert recovered.anchors_found == 4
    assert recovered.below_under == recovered.below_present == 12
    assert sorted(recovered.anchors_missing) == gone
    assert recovered.children_spanned >= 2
    assert "2 anchor(s) missing" in recovered.describe()


def test_fewer_anchors_than_the_quorum_stops_the_run(tmp_path: Path) -> None:
    tree, pin = pinned(tmp_path, quorum=6)
    thinned = version(tmp_path / "v2", drop=(next(iter(BIG)),))
    with pytest.raises(cut.CutError, match="under the quorum of 6"):
        cut.recover(thinned, pin)


def test_an_mrca_that_descended_into_one_child_is_caught_by_the_below_list(
    tmp_path: Path,
) -> None:
    """The failure the design note described, caught by composition rather than by child count.

    With only one child's anchors left, the MRCA is that child. It is a perfectly ordinary node
    spanning two of *its* children, so counting children cannot see the fault; what sees it is that
    the viruses the cut kept last time are no longer under it.
    """
    tree, pin = pinned(tmp_path)
    away = tuple(
        number
        for epi, _ in pin.anchors
        if (number := int(epi.removeprefix("EPI_ISL_7"))) not in set(BIG)
    )
    thinned = version(tmp_path / "v2", drop=away)
    # The leaf-count check is relaxed, so the below-list is the only thing that can refuse this.
    with pytest.raises(cut.CutError, match="no longer below the recovered cut"):
        cut.recover(thinned, pin, max_leaf_drift=0.9)
    loose = cut.recover(thinned, pin, max_leaf_drift=0.9, max_below_outside=0.9)
    assert loose.children_spanned == 2  # reported, and no help at all here
    assert (loose.below_present, loose.below_under) == (10, 6)
    assert loose.below_outside == 4 and round(loose.below_shortfall, 2) == 0.4


def test_a_cut_whose_leaf_count_has_drifted_is_refused(tmp_path: Path) -> None:
    tree, pin = pinned(tmp_path)
    thinned = version(tmp_path / "v2", drop=tuple(MIDDLE))
    with pytest.raises(cut.CutError, match="has moved"):
        cut.recover(thinned, pin)
    loose = cut.recover(thinned, pin, max_leaf_drift=0.9)  # the same tree, under a looser limit
    assert loose.leaves_below == 10 and round(loose.leaf_drift, 2) == 0.29
    assert loose.node_id_changed  # a pruned tree renumbers: the id alone is not the test


# ---------------------------------------------------------------------------------------------
# Has circulation above the cut resumed?


def test_resumption_counts_the_leaves_the_cut_never_saw_and_exempts_the_outgroup(
    tmp_path: Path,
) -> None:
    tree, pin = pinned(tmp_path)
    found = cut.check_resumption(tree, pin, pin.node_id, limit=0.5)
    assert found.above_cut == 4  # 3 old + the newcomer; the outgroup is exempt by design
    assert found.resumed == 4 and found.moved_above == 0
    assert found.newest_resumed == DATES[NEWCOMER]
    assert round(found.fraction, 3) == round(4 / 19, 3)
    assert "kept because the cut never saw them" in found.describe()


def test_resumption_above_the_limit_stops_the_run(tmp_path: Path) -> None:
    tree, pin = pinned(tmp_path)
    with pytest.raises(cut.CutError, match="circulation above the cut has resumed"):
        cut.check_resumption(tree, pin, pin.node_id, limit=0.05)


def test_pinned_leaves_that_now_sit_above_the_cut_are_counted_separately(tmp_path: Path) -> None:
    """A below-list leaf above the cut is the topology having moved, not a new virus."""
    tree, pin = pinned(tmp_path)
    inner = cut.propose(tree, tolerances=(0.6,))[0]  # a node inside the cut: 10 of its leaves above
    assert inner.node_id != pin.node_id
    found = cut.check_resumption(tree, pin, inner.node_id, limit=0.9)
    assert found.moved_above == 8 and found.resumed == 4


# ---------------------------------------------------------------------------------------------
# The command line, which is how a person uses this at the end of a VCM


def test_the_cli_proposes_then_pins_then_recovers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tree = version(tmp_path / "v1")
    assert cut.main(["propose", str(tree), "--json"]) == 0
    proposed = json.loads(capsys.readouterr().out)
    assert len(proposed) == len(cut.TOLERANCES)
    # Every tolerance in the default sweep returns the root here: one recent leaf of 15 above the
    # cut is 6.7%, above the widest default of 5%. On the real trees the band is far finer.
    assert all(row["is_root"] for row in proposed)
    node = cut_node(tree, 0.1).node_id
    assert (
        cut.main(
            ["pin", str(tree), node, "--anchors", "6", "--below-out", str(tmp_path / "below.tsv")]
        )
        == 0
    )
    emitted = capsys.readouterr().out
    assert emitted.startswith("[[cut]]") and "below-list: 14 keys" in emitted
    # The emitted block is a real pin once a person adds the fields it says are theirs to add.
    (tmp_path / "cuts.toml").write_text(
        emitted.replace("# subtype, cycle", "# subtype, cycle".replace("#", "#"))
        + f'subtype = "h3"\ncycle = "{CYCLE}"\nbelow_file = "below.tsv"\n'
        + 'reason = "the synthetic cut, from the CLI"\n'
        + 'chosen_from = { kind = "sequences", dataset = "h3", version = "0123456789abcdef", '
        + 'manifest_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" }\n'
    )
    assert cut.main(["recover", str(tree), str(tmp_path / "cuts.toml"), CYCLE, "h3",
                     "--max-resumed", "0.5", "--json"]) == 0  # fmt: skip
    found = json.loads(capsys.readouterr().out)
    assert found["recovered"]["node_id"] == node
    assert found["recovered"]["below_under"] == 14
    assert found["resumption"]["resumed"] == 4
    assert found["recovered"]["anchors_found"] == 6  # quorum emitted as 5 of 6, not 15 of 6
