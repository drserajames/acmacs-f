"""af.tree.trim and the trimmed report tree. Synthetic only (tree_fixtures' invented tree).

The fixture's branch x->y carries K2E with two tips under it (a, b); the rule ``2E`` with
``min_tips`` 2 picks y.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from af.tree.io import i6
from af.tree.report_filter import report_tree
from af.tree.trim import (
    TrimError,
    TrimRule,
    find_trim_node,
    load_trim_rules,
    parse_substitution,
    rule_for,
)

from .test_report_filter import CUTOFF, full, index
from .tree_fixtures import KEYS

EARLY = datetime.date(2000, 1, 1)  # nothing is old enough for the date rule
HEADER = "subtype\tsubstitution\tmin_tips\treason\n"


def rule(substitution: str = "2E", min_tips: int = 2) -> TrimRule:
    position, to = parse_substitution(substitution)
    return TrimRule("h3", position, to, min_tips, "invented")


def test_the_one_branch_carrying_the_substitution_is_picked() -> None:
    populated, ids = full(assign=True)
    match = find_trim_node(populated, rule())
    assert match.node_id == f"{ids['y']:016x}"
    assert match.tips == 2 and match.clade == "J.2"
    assert match.keys == {KEYS["a"], KEYS["b"]}


def test_the_trimmed_tree_is_rooted_at_the_cut_and_has_no_outgroup() -> None:
    populated, ids = full()
    result = report_tree(populated, EARLY, index(), trim=find_trim_node(populated, rule()))
    assert {leaf.name for leaf in result.tree.leaves()} == {KEYS["a"], KEYS["b"]}
    assert result.tree.root.node_id == ids["y"] and result.tree.root.branch_length == 0.0
    assert "outgroup" not in result.counts
    # the cut node's own ancestral state is the root's, and the lengths inside are unchanged
    assert result.states is not None
    assert result.states.nucleotides[ids["y"]] == populated.states.nucleotides[ids["y"]]
    a = next(leaf for leaf in result.tree.leaves() if leaf.name == KEYS["a"])
    assert a.branch_length == pytest.approx(0.2)
    assert result.counts["removed_by_trim"] == 2  # c and d; the outgroup is recorded apart
    assert result.counts["removed_by_trim_titrated"] == 1  # c, by EPI_ISL
    assert result.counts["removed_by_trim_otherwise_kept"] == 2
    assert result.counts["removed_by_trim_by_year"] == {"2023": 1, "2024": 1}
    assert result.counts["trim"]["outgroup_removed"] == KEYS["o"]
    assert result.counts["trim"]["substitution"] == "2E"


def test_the_date_rule_runs_inside_the_cut_and_counts_apart() -> None:
    """A cut at x (a substitution put on x->root) keeps a, b, c, d; b then goes by date."""
    populated, ids = full()
    populated.aa_subs[ids["x"]] = ["P3S"]
    trimmed = report_tree(populated, CUTOFF, index(), trim=find_trim_node(populated, rule("3S", 4)))
    untrimmed = report_tree(populated, CUTOFF, index())
    assert {leaf.name for leaf in trimmed.tree.leaves()} == {KEYS["a"], KEYS["c"], KEYS["d"]}
    # x is the root; b went, so its id moved, and a hangs from it directly (y spliced out)
    assert sorted(len(c.children) for c in trimmed.tree.root.children) == [0, 2]
    assert trimmed.counts["removed_before_cutoff"] == 1  # b: year-only 2021, untitrated
    assert trimmed.counts["kept_titrated_before_cutoff"] == 2  # a and c
    for key in ("removed_before_cutoff", "kept_titrated_before_cutoff", "kept_undated"):
        assert trimmed.counts[key] == untrimmed.counts[key]
    assert trimmed.counts["removed_by_trim"] == 0
    assert trimmed.counts["removed_by_trim_by_year"] == {}


def test_a_cut_the_date_rule_leaves_one_leaf_in_is_refused() -> None:
    populated, _ = full()
    with pytest.raises(ValueError, match="1 leaf left"):
        report_tree(populated, CUTOFF, index(), trim=find_trim_node(populated, rule()))


def test_without_a_trim_the_report_tree_is_unchanged(tmp_path: Path) -> None:
    populated, _ = full(assign=True)
    first = report_tree(populated, CUTOFF, index())
    second = report_tree(populated, CUTOFF, index(), trim=None)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    i6.write(first, tmp_path / "a", "report-x", {})
    i6.write(second, tmp_path / "b", "report-x", {})
    for path in sorted((tmp_path / "a").iterdir()):
        assert path.read_bytes() == (tmp_path / "b" / path.name).read_bytes(), path.name
    assert not any(key.startswith("removed_by_trim") for key in first.counts)


def test_a_trimmed_tree_round_trips_through_i6_with_no_outgroup(tmp_path: Path) -> None:
    populated, _ = full(assign=True)
    result = report_tree(populated, EARLY, index(), trim=find_trim_node(populated, rule()))
    i6.write(result, tmp_path, "report-x", {})
    assert i6.read_metadata(tmp_path)["outgroup"] is None
    alignment = {key: record.nucleotides for key, record in populated.leaves.items()}
    back = i6.read(tmp_path, alignment)
    assert {leaf.name for leaf in back.tree.leaves()} == {KEYS["a"], KEYS["b"]}
    assert back.counts["trim"]["node"] == result.counts["trim"]["node"]


def test_no_branch_over_the_threshold_is_refused_with_the_candidates() -> None:
    populated, ids = full()
    with pytest.raises(TrimError, match=f"0 branches carry 2E.*{ids['y']:016x} \\(2 tips"):
        find_trim_node(populated, rule(min_tips=3))


def test_two_branches_over_the_threshold_are_refused() -> None:
    populated, ids = full()
    populated.aa_subs[ids["z"]] = ["Q2E"]  # another from-state: still 2E
    with pytest.raises(TrimError, match="2 branches carry 2E"):
        find_trim_node(populated, rule())


@pytest.mark.parametrize("other", ["J2J", "J12O", "J22O"])
def test_position_and_residue_are_exact(other: str) -> None:
    """Rule 2O (J/O: invented residues); only a substitution ending in exactly 2O matches."""
    populated, ids = full()
    populated.aa_subs[ids["y"]] = [other]
    with pytest.raises(TrimError, match="0 branches"):
        find_trim_node(populated, rule("2O"))
    populated.aa_subs[ids["y"]] = ["J2O"]
    assert find_trim_node(populated, rule("2O")).tips == 2


def test_a_rule_for_another_subtype_is_refused() -> None:
    populated, _ = full()
    with pytest.raises(TrimError, match="rule for h1 applied to a h3 tree"):
        find_trim_node(populated, TrimRule("h1", 2, "E", 2, "invented"))


def test_the_rules_load_with_their_reason(tmp_path: Path) -> None:
    path = tmp_path / "trim.tsv"
    path.write_text("# where the rows came from\n" + HEADER + "h3\t223V\t15000\tinvented\n")
    rules = load_trim_rules(path)
    assert rules["h3"] == TrimRule("h3", 223, "V", 15000, "invented")
    assert rule_for(rules, "h3").substitution == "223V"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("h3\t223V\t15000\t\n", "no reason"),
        ("h3\t223V\t15000\tx\nh3\t223V\t15000\ty\n", "appears twice"),
        ("h3\tI223V\t15000\tx\n", "expected <position><residue>"),
        ("h3\t223v\t15000\tx\n", "expected <position><residue>"),
        ("h3\t0V\t15000\tx\n", "expected <position><residue>"),
        ("h3\t223V\tmany\tx\n", "min_tips"),
        ("h3\t223V\t1\tx\n", "at least 2"),
        ("\t223V\t15000\tx\n", "no subtype"),
    ],
)
def test_malformed_rules_are_refused(tmp_path: Path, rows: str, message: str) -> None:
    path = tmp_path / "trim.tsv"
    path.write_text(HEADER + rows)
    with pytest.raises(TrimError, match=message):
        load_trim_rules(path)


def test_a_missing_column_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "trim.tsv"
    path.write_text("subtype\tsubstitution\treason\nh3\t223V\tx\n")
    with pytest.raises(TrimError, match="missing column"):
        load_trim_rules(path)


def test_a_subtype_with_no_row_is_an_error_not_a_silent_no_trim(tmp_path: Path) -> None:
    path = tmp_path / "trim.tsv"
    path.write_text(HEADER + "h3\t223V\t15000\tx\n")
    with pytest.raises(TrimError, match="no trim rule for 'h1'"):
        rule_for(load_trim_rules(path), "h1")
