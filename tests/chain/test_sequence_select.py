"""`[[select.remove_by_sequence]]`: points seen only in a chain's early tables, kept or removed by
whether their HA has a deletion, decided once and applied to every table before the merge."""

import dataclasses
from types import SimpleNamespace

import pytest

from af.chain.backend import StubOptimiser
from af.chain.config import ChainConfigError, config_to_json
from af.chain.engine import run_chain
from af.chain.select import SelectError, Selection
from af.chain.sequence_select import (
    DELETED,
    NO_PROTEIN,
    NOT_DELETED,
    TIE_ALL,
    TIE_NONE,
    TIE_SPLIT,
    UNSEQUENCED_VERDICT,
    SequenceRemovals,
    SequenceRule,
    SequenceSelectError,
    TableInput,
    apply,
    decide,
    deleted,
)
from af.chart.ace import read_chart

from .test_engine import config, reused, run, tables  # noqa: F401  (tables is a fixture)

RULE = dict(what="both", tables_until="2021-01-31", deletion=[3, 4], unsequenced="remove",
            ties="by-candidates", reason="test")  # fmt: skip
GAP, NO_GAP = "MK--LV", "MKTILV"  # gaps at 1-based 3 and 4, and none


def rule(**changes):
    return SequenceRule(**{**RULE, **changes})


# ---- the rule itself ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"what": "points"}, "what must be"),
        ({"reason": " "}, "needs a reason"),
        ({"tables_until": "31 Jan 2021"}, "YYYY-MM-DD"),
        ({"deletion": []}, "1-based positions"),
        ({"deletion": [0]}, "1-based positions"),
        ({"unsequenced": "drop"}, "unsequenced must be"),
        ({"ties": "keep"}, "ties must be"),
    ],
)
def test_a_rule_says_everything_it_means(changes, message):
    with pytest.raises(SequenceSelectError, match=message):
        rule(**changes)


def test_the_deletion_is_read_as_r5_reads_it_and_never_guessed():
    assert deleted(GAP, [3, 4]) is True and deleted(NO_GAP, [3, 4]) is False
    assert deleted(None, [3]) is None
    # unobservable is not "not deleted": past the end, or unread ("X", af's padding)
    assert deleted("MK-", [3, 4]) is None
    assert deleted("MK-XLV", [3, 4]) is None
    assert deleted("MKXXLV", [3, 4]) is None
    # a residue read at one position is evidence enough that the deletion is not there
    assert deleted("MKTXLV", [3, 4]) is False


@pytest.mark.parametrize(
    "ties, unsequenced, kept",
    [
        # deleted, not-deleted, unsequenced, no-protein, tie-all, tie-none, tie-split
        ("by-candidates", "remove", [True, False, False, False, True, False, False]),
        ("by-candidates", "keep", [True, False, True, True, True, False, True]),
        ("as-unsequenced", "remove", [True, False, False, False, False, False, False]),
        ("as-unsequenced", "keep", [True, False, True, True, True, True, True]),
        # closest-passage with no ranked candidate: every tie is no answer
        ("closest-passage", "remove", [True, False, False, False, False, False, False]),
    ],
)
def test_what_each_verdict_keeps(ties, unsequenced, kept):
    r = rule(ties=ties, unsequenced=unsequenced)
    verdicts = [DELETED, NOT_DELETED, UNSEQUENCED_VERDICT, NO_PROTEIN, TIE_ALL, TIE_NONE, TIE_SPLIT]
    assert [r.keeps(v) for v in verdicts] == kept


@pytest.mark.parametrize("tie", [TIE_ALL, TIE_NONE, TIE_SPLIT])
def test_closest_passage_decides_a_tie_by_the_ranked_candidate(tie):
    r = rule(ties="closest-passage")
    assert r.keeps(tie, ranked_deleted=True) is True
    assert r.keeps(tie, ranked_deleted=False) is False
    assert r.keeps(tie, ranked_deleted=None) is False  # no ranked protein: as unsequenced
    assert r.keeps(DELETED, ranked_deleted=False) is True  # a chosen sequence is not a tie
    # by-candidates leaves a split tie to `unsequenced`
    assert rule(ties="by-candidates").keeps(TIE_SPLIT) is False
    assert rule(ties="by-candidates", unsequenced="keep").keeps(TIE_SPLIT) is True


# ---- deciding, with a stand-in for the matcher ---------------------------------------------------


def _seq(epi):
    return SimpleNamespace(epi_isl=epi, accession="A1")


class Sequencer:
    """By name: a chosen sequence, a tie, or nothing; proteins by EPI_ISL."""

    def __init__(self, answers, proteins):
        self.answers, self._proteins, self.asked = answers, proteins, []

    def match(self, row):
        self.asked.append(row["name"])
        a = self.answers.get(row["name"])
        if a is None:
            return SimpleNamespace(chosen=None, tied=())
        if isinstance(a, list):  # a tie; ae's ranked pick is the first listed
            tied = tuple(_seq(e) for e in a)
            return SimpleNamespace(chosen=None, tied=tied, ranked=tied[0])
        return SimpleNamespace(chosen=_seq(a), tied=())

    def proteins(self, keys):
        return {k: self._proteins.get(k[0]) for k in keys}


def _inputs(directory, kept_sera=None):
    """The synthetic chain tables as the rule sees them: rows carry just the name."""
    cfg = config(directory)
    out = []
    for t in cfg.tables:
        c = read_chart(t.path)
        kept = frozenset(range(c.n_sera)) if kept_sera is None else kept_sera
        out.append(TableInput(t.table_id, t.date, c, [{"name": a.name} for a in c.antigens],
                              [{"name": s.name} for s in c.sera], kept))  # fmt: skip
    return cfg, out


def test_only_points_seen_only_in_the_early_tables_are_candidates(tables):  # noqa: F811
    cfg, inputs = _inputs(tables)
    until = cfg.tables[1].date.isoformat()  # the first two tables are early
    seq = Sequencer({}, {})
    decisions = decide([rule(tables_until=until)], inputs, seq)
    early = {a.name for t in inputs[:2] for a in t.chart.antigens}
    later = {a.name for t in inputs[2:] for a in t.chart.antigens}
    assert {d.designation.split()[0] for d in decisions} == early - later  # refs are in every table
    assert all(d.what == "antigen" for d in decisions)  # every serum is in a later table too
    assert all(d.verdict == UNSEQUENCED_VERDICT and d.removed for d in decisions)
    assert sorted(seq.asked) == sorted(early - later)  # each candidate asked about once


def test_verdicts_come_from_the_proteins_of_the_matched_sequences(tables):  # noqa: F811
    cfg, inputs = _inputs(tables)
    until = cfg.tables[0].date.isoformat()
    names = [a.name for a in inputs[0].chart.antigens if a.name.startswith("TEST-10")]
    answers = {names[0]: "E1", names[1]: "E2", names[2]: "E3", names[3]: ["E1", "E4"],
               names[4]: ["E2", "E5"], names[5]: ["E1", "E2"]}  # fmt: skip
    proteins = {"E1": GAP, "E2": NO_GAP, "E4": GAP, "E5": NO_GAP}  # E3: matched, no protein
    got = {d.designation.split()[0]: (d.verdict, d.removed) for d in
           decide([rule(tables_until=until)], inputs, Sequencer(answers, proteins))}  # fmt: skip
    assert got[names[0]] == (DELETED, False)
    assert got[names[1]] == (NOT_DELETED, True)
    assert got[names[2]] == (NO_PROTEIN, True)
    assert got[names[3]] == (TIE_ALL, False)
    assert got[names[4]] == (TIE_NONE, True)
    assert got[names[5]] == (TIE_SPLIT, True)
    assert got[names[6]] == (UNSEQUENCED_VERDICT, True)


def test_a_serum_the_map_never_has_is_no_candidate(tables):  # noqa: F811
    cfg, inputs = _inputs(tables, kept_sera=frozenset())
    decisions = decide([rule(what="sera", tables_until="2100-01-01", optional=True)], inputs,
                       Sequencer({}, {}))  # fmt: skip
    assert decisions == ()


def test_a_rule_with_no_candidates_is_an_error_unless_optional(tables):  # noqa: F811
    _, inputs = _inputs(tables)
    with pytest.raises(SequenceSelectError, match="matches nothing"):
        decide([rule(tables_until="2000-01-01")], inputs, Sequencer({}, {}))
    assert decide([rule(tables_until="2000-01-01", optional=True)], inputs, Sequencer({}, {})) == ()


# ---- applying, and what a chain records ----------------------------------------------------------


def _removals(directory, until_index=0):
    cfg, inputs = _inputs(directory)
    r = rule(tables_until=cfg.tables[until_index].date.isoformat())
    decisions = decide([r], inputs, Sequencer({}, {}))  # every candidate unsequenced: removed
    return cfg, SequenceRemovals((r,), decisions, {"bvic": {"version": "x"}}, ())


def test_apply_takes_the_removed_points_out_of_a_table(tables):  # noqa: F811
    cfg, removals = _removals(tables)
    chart = read_chart(cfg.tables[0].path)
    out, report = apply(removals, chart)
    gone = {d.designation for d in removals.decisions if d.removed}
    assert set(report) == gone and len(report) == len(gone) > 0
    assert out.n_antigens == chart.n_antigens - len(gone) and out.n_sera == chart.n_sera
    assert apply(None, chart) == (chart, [])
    later, report = apply(removals, read_chart(cfg.tables[1].path))
    assert report == []  # the candidates are in the early table only


def test_a_chain_leaves_them_out_and_is_keyed_by_what_is_removed(tables, tmp_path):  # noqa: F811
    cfg, removals = _removals(tables)
    plain = run(cfg, tmp_path / "s")
    assert reused(plain) == [False] * 4
    assert "select_remove_by_sequence" not in config_to_json(cfg)  # no rule: nothing written
    on = dataclasses.replace(cfg, sequence_removals=removals)
    results = run(on, tmp_path / "s")
    assert reused(results) == [False] * 4  # a selection changed: every step remade
    assert sorted(results[0].record["removed_by_sequence"]) == sorted(
        d.designation for d in removals.decisions if d.removed
    )
    merged = read_chart(results[-1].directory / "merge.ace")
    names = {a.designation() for a in merged.antigens}
    assert not names & {d.designation for d in removals.decisions if d.removed}
    doc = config_to_json(on)["select_remove_by_sequence"]
    assert doc["counts"]["antigen"]["removed"] == len(results[0].record["removed_by_sequence"])
    assert doc["sequences"] == {"bvic": {"version": "x"}}
    # a pin that moves without changing a decision remakes nothing
    moved = dataclasses.replace(removals, sequences={"bvic": {"version": "y"}})
    assert (
        reused(run(dataclasses.replace(cfg, sequence_removals=moved), tmp_path / "s")) == [True] * 4
    )


def test_merge_all_records_what_each_table_lost(tables, tmp_path):  # noqa: F811
    cfg, removals = _removals(tables)
    opts = dataclasses.replace(cfg.options, merge_all=True)
    on = dataclasses.replace(cfg, options=opts, sequence_removals=removals)
    [result] = run_chain(on, tmp_path / "s", optimiser=StubOptimiser())
    lost = result.record["removed_by_sequence"]
    assert list(lost) == [cfg.tables[0].table_id]


# ---- the chain file ------------------------------------------------------------------------------


def test_pins_and_rules_come_together():
    with pytest.raises(SelectError, match="needs \\[select.sequences\\]"):
        Selection(remove_by_sequence=[rule()])
    with pytest.raises(SelectError, match="for no remove_by_sequence rule"):
        Selection(sequences={"bvic": "0" * 16})


def test_a_chain_of_directory_tables_cannot_use_it(tables, tmp_path):  # noqa: F811
    from af.chain.config import load_chain_config

    from .sera_markers import write_markers

    chain = tmp_path / "chains" / "c.toml"
    chain.parent.mkdir()
    write_markers(tmp_path / "rules" / "sera" / "non_ferret_markers.tsv")
    chain.write_text(
        f'name = "x"\nseed = 1\n[tables]\ndirectory = "{tables}"\ngroup = "h9-hi-test-lab"\n'
        '[options]\nminimum_column_basis = "none"\n[select.sequences]\nbvic = "0000000000000000"\n'
        '[[select.remove_by_sequence]]\nwhat = "both"\ntables_until = "2021-01-31"\n'
        'deletion = [162, 163, 164]\nunsequenced = "remove"\nties = "by-candidates"\nreason = "t"\n'
    )
    with pytest.raises(ChainConfigError, match="needs tables from the store"):
        load_chain_config(chain)


def test_closest_passage_follows_the_ranked_candidates_protein(tables):  # noqa: F811
    cfg, inputs = _inputs(tables)
    until = cfg.tables[0].date.isoformat()
    names = [a.name for a in inputs[0].chart.antigens if a.name.startswith("TEST-10")]
    answers = {names[0]: ["E1", "E2"], names[1]: ["E2", "E1"], names[2]: ["E3", "E1"]}
    proteins = {"E1": GAP, "E2": NO_GAP}  # E3: no protein
    r = rule(tables_until=until, ties="closest-passage", optional=True)
    got = {d.designation.split()[0]: d for d in decide([r], inputs, Sequencer(answers, proteins))}
    assert got[names[0]].verdict == TIE_SPLIT and not got[names[0]].removed  # ranked E1: gap
    assert got[names[1]].verdict == TIE_SPLIT and got[names[1]].removed  # ranked E2: none
    assert got[names[2]].removed and got[names[2]].ranked_deleted is None  # no protein
    assert got[names[0]].to_json()["ranked"] == ["E1", "A1"]


def test_an_unread_position_never_removes_a_point_as_not_deleted(tables):  # noqa: F811
    cfg, inputs = _inputs(tables)
    until = cfg.tables[0].date.isoformat()
    name = next(a.name for a in inputs[0].chart.antigens if a.name.startswith("TEST-10"))
    seq = Sequencer({name: "E1"}, {"E1": "MK-XLV"})
    keep = rule(tables_until=until, unsequenced="keep", optional=True)
    [d] = [d for d in decide([keep], inputs, seq) if d.designation.startswith(name + " ")]
    assert d.verdict == NO_PROTEIN and not d.removed


def test_an_unread_ranked_candidate_follows_unsequenced(tables):  # noqa: F811
    """closest-passage on a tie whose ranked pick has "X" at a deletion position: no answer,
    so `unsequenced` decides, never "not deleted"."""
    cfg, inputs = _inputs(tables)
    until = cfg.tables[0].date.isoformat()
    name = next(a.name for a in inputs[0].chart.antigens if a.name.startswith("TEST-10"))
    seq = Sequencer({name: ["E1", "E2"]}, {"E1": "MK-XLV", "E2": GAP})
    for unsequenced, removed in (("keep", False), ("remove", True)):
        r = rule(tables_until=until, ties="closest-passage", unsequenced=unsequenced, optional=True)
        [d] = [d for d in decide([r], inputs, seq) if d.designation.startswith(name + " ")]
        assert d.ranked == ("E1", "A1") and d.ranked_deleted is None and d.removed is removed
