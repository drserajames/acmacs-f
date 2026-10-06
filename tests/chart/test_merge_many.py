"""merge_many: the chart a fold of merge() makes, with each cell merged once.

The same check on every real chain is notes/chains/tools/merge-many-equality.py (the store is
not reachable from tests); here, synthetic tables with every rule the fold applies.
"""

import copy
import json
import random

import numpy as np
import pytest

from af.chart.merge import MergeError, MergeOptions, MergeStep, MergeType, merge, merge_many

from .test_merge import AG, SR, table

SIMPLE = MergeOptions(merge_type=MergeType.SIMPLE, combine_cheating_assays=True)
TITRES = ["<10", "10", "20", "40", "80", "160", "320", "640", "1280", "2560", ">2560", ">5120"]


def random_tables(seed: int, n_tables: int = 12) -> list:
    """Overlapping antigens and sera, some passage-less (never matched), '<' and '>' readings,
    spread readings (SD drops), and copies of the reference block (combined assays)."""
    rnd = random.Random(seed)
    sera = [SR(n, f"F{n}") for n in range(1, 7)]
    refs = [AG(n) for n in range(1, 7)]  # same names as the sera: reference antigens
    ref_cells = {(i, j): rnd.choice(TITRES) for i in range(6) for j in range(6)}
    out = []
    for t in range(n_tables):
        use_sera = sorted(rnd.sample(range(6), rnd.randint(3, 6)))
        passages = ["E3", "SIAT1", "MDCK2"]
        tests = [AG(100 + rnd.randint(0, 25), p=rnd.choice(passages)) for _ in range(5)]
        tests = list({(a.name, a.passage): a for a in tests}.values())
        if t % 4 == 3:
            tests.append(AG(500 + t, p=""))  # passage-less: a new point every time
        if t % 5 == 4:  # the reference block copied from the first table's titres
            use_sera = list(range(6))
            ags = refs + tests
            cells = dict(ref_cells)
        else:
            ags = rnd.sample(refs, 3) + tests
            cells = {}
            for i, a in enumerate(ags):
                for j in range(len(use_sera)):
                    if a in refs and t == 0:
                        continue
                    if rnd.random() < 0.7:
                        cells[(i, j)] = rnd.choice(TITRES)
        if t == 0:
            ags, use_sera, cells = refs, list(range(6)), dict(ref_cells)
        out.append(table(ags, [sera[j] for j in use_sera], cells, date=f"2021{t + 1:02d}01"[:8]))
    return out


def fold(charts, options):
    m, reports = charts[0], []
    for c in charts[1:]:
        m, r = merge(m, c, options)
        reports.append(r)
    return m, reports


def assert_same(a, b, last, rb, reports):
    assert [vars(x) for x in a.antigens] == [vars(x) for x in b.antigens]
    assert [vars(x) for x in a.sera] == [vars(x) for x in b.sera]
    assert a.titres.layers == b.titres.layers
    assert a.titres.table == b.titres.table
    if a.forced_column_bases is None:
        assert b.forced_column_bases is None
    else:
        assert np.array_equal(a.forced_column_bases, b.forced_column_bases)
    assert a.info == b.info
    for f in (
        "common_antigens",
        "common_sera",
        "new_antigens",
        "new_sera",
        "cheating_assay",
        "skipped_reference_antigens",
        "outcomes",
        "dropped",
        "column_basis_slack",
    ):
        assert getattr(last, f) == getattr(rb, f), f
    assert rb.steps == [MergeStep.of(r) for r in reports]


@pytest.mark.parametrize("seed", range(20))
def test_same_chart_and_report_as_the_fold(seed):
    charts = random_tables(seed)
    m, reports = fold(charts, SIMPLE)
    mm, rm = merge_many(charts, SIMPLE)
    assert_same(m, mm, reports[-1], rm, reports)


def test_the_synthetic_tables_exercise_every_rule():
    seen = {"cheating": 0, "dropped": 0, "slack": 0}
    for seed in range(20):
        _, rm = merge_many(random_tables(seed), SIMPLE)
        seen["cheating"] += sum(s.cheating_assay for s in rm.steps)
        seen["dropped"] += len(rm.dropped)
        seen["slack"] += len(rm.column_basis_slack)
    assert all(seen.values()), seen


def test_inputs_untouched_and_nothing_mutable_shared():
    charts = random_tables(3)
    charts[0].antigens[0].extra["note"] = ["x"]
    before = copy.deepcopy(charts)
    m, _ = merge_many(charts, SIMPLE)
    assert [vars(a) for c in charts for a in c.antigens] == [
        vars(a) for c in before for a in c.antigens
    ]
    assert [c.titres.layers for c in charts] == [c.titres.layers for c in before]
    m.antigens[0].extra["note"].append("y")
    m.titres.layers[0][(0, 0)] = m.titres.layers[0][(0, 1)]
    m.info["S"].append({})
    assert charts[0].antigens[0].extra == {"note": ["x"]}
    assert charts[0].titres.layers == before[0].titres.layers
    assert charts[0].info == before[0].info


def test_a_duplicate_the_fold_rejects_is_rejected():
    a = table([AG(1), AG(2, p="")], [SR(1, "F1")], {(0, 0): "40", (1, 0): "20"})
    b = table([AG(1), AG(2, p="")], [SR(1, "F1")], {(0, 0): "80", (1, 0): "40"}, date="20210201")
    with pytest.raises(MergeError) as fold_error:
        fold([a, b], SIMPLE)
    with pytest.raises(MergeError) as many_error:
        merge_many([a, b], SIMPLE)
    assert str(many_error.value) == str(fold_error.value)


def test_one_chart_is_a_copy_and_no_merge():
    a = table([AG(1)], [SR(1, "F1")], {(0, 0): "40"})
    m, r = merge_many([a], SIMPLE)
    assert m is not a and m.titres.table == a.titres.table and r.steps == [] and not r.outcomes


def test_simple_merges_only_and_at_least_one_chart():
    a = table([AG(1)], [SR(1, "F1")], {(0, 0): "40"})
    with pytest.raises(MergeError, match="simple"):
        merge_many([a, a], MergeOptions(merge_type=MergeType.INCREMENTAL))
    with pytest.raises(MergeError, match="at least one"):
        merge_many([], SIMPLE)


def test_report_to_dict_is_json_with_stable_keys():
    _, rm = merge_many(random_tables(5), SIMPLE)
    d = rm.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert list(d) == [
        "common_antigens",
        "common_sera",
        "new_antigens",
        "new_sera",
        "cheating_assay",
        "skipped_reference_antigens",
        "outcomes",
        "dropped",
        "column_basis_slack",
        "steps",
    ]
    assert len(d["steps"]) == 11 and d["outcomes"] == dict(sorted(d["outcomes"].items()))
    assert sum(d["outcomes"].values()) == len(
        {k for layer in merge_many(random_tables(5), SIMPLE)[0].titres.layers for k in layer}
    )
