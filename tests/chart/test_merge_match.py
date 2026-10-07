"""Point matching: STRICT (ae, the chains), STRICT_THEN_NAME (Sarah, 1 Oct 2026), caller pairs.

The same rules against pyacmapcheck's conservative_merge, on real tables and on synthetic ones
rich in name matches, are notes/chains/tools/pyacmapcheck-equality/ (outside the repo).
"""

import random

import pytest

from af.chart.merge import (
    Match,
    MergeError,
    MergeOptions,
    MergeStep,
    MergeType,
    find_duplicates,
    merge,
    merge_many,
)

from .test_merge import AG, SR, table

NAME = Match.STRICT_THEN_NAME
SIMPLE = MergeOptions(merge_type=MergeType.SIMPLE)


def names(chart):
    return [(a.name, a.passage) for a in chart.antigens]


def test_strict_never_pairs_a_point_without_its_passage():
    a = table([AG(1), AG(2)], [SR(1, "F1")], {(0, 0): "40", (1, 0): "80"})
    b = table([AG(2, p="")], [SR(1, "F1")], {(0, 0): "160"}, date="20210201")
    m, rep = merge(a, b, SIMPLE)
    assert m.n_antigens == 3 and rep.matching["strict"] == {"antigens": 0, "sera": 1}


def test_name_pairs_a_point_without_a_passage_and_the_merge_takes_the_partners():
    a = table([AG(1), AG(2, p="")], [SR(1, "F1")], {(0, 0): "40", (1, 0): "80"})
    b = table([AG(2, p="SIAT1")], [SR(1, "F1")], {(0, 0): "160"}, date="20210201")
    m, rep = merge(a, b, SIMPLE, match=NAME)
    assert names(m) == [("TEST-1", "E3"), ("TEST-2", "SIAT1")]
    assert rep.matching["name_fallback"] == {"antigens": 1, "sera": 0}
    assert str(m.titres.table[1][0]) == "113"  # 80 and 160 merged
    assert rep.filled_antigens == [1]


def test_the_taken_passage_is_kept_for_later_tables():
    # Sarah, Q131 R2: kept, so a later table's point matches it strictly, and one with another
    # passage stays a point of its own
    a = table([AG(2, p="")], [SR(1, "F1")], {(0, 0): "80"})
    b = table([AG(2, p="SIAT1")], [SR(1, "F1")], {(0, 0): "160"}, date="20210201")
    c = table(
        [AG(2, p="SIAT1"), AG(2, p="E3")], [SR(1, "F1")], {(0, 0): "40", (1, 0): "20"}, "20210301"
    )
    m, reports = a, []
    for t in (b, c):
        m, r = merge(m, t, SIMPLE, match=NAME)
        reports.append(r)
    assert names(m) == [("TEST-2", "SIAT1"), ("TEST-2", "E3")]
    assert reports[1].matching["strict"]["antigens"] == 1
    assert reports[1].matching["name_fallback"]["antigens"] == 0


def test_two_points_without_a_passage_pair_and_stay_without_one():
    a = table([AG(2, p="")], [SR(1, "")], {(0, 0): "80"})
    b = table([AG(2, p="")], [SR(1, "")], {(0, 0): "160"}, date="20210201")
    m, rep = merge(a, b, SIMPLE, match=NAME)
    assert names(m) == [("TEST-2", "")] and m.sera[0].serum_id == ""
    assert rep.matching["name_fallback"] == {"antigens": 1, "sera": 1}


def test_an_ambiguous_name_pairs_nothing_and_is_counted():
    a = table([AG(2, p="E3"), AG(2, p="SIAT1")], [SR(1, "F1")], {(0, 0): "80", (1, 0): "40"})
    b = table([AG(2, p="")], [SR(1, "F1")], {(0, 0): "160"}, date="20210201")
    m, rep = merge(a, b, SIMPLE, match=NAME)
    assert m.n_antigens == 3 and rep.matching["ambiguous"]["antigens"] == 1
    assert rep.matching["name_fallback"]["antigens"] == 0


def test_a_strict_pair_is_never_reused_by_the_name_rule():
    a = table([AG(2, p="E3")], [SR(1, "F1")], {(0, 0): "80"})
    b = table(
        [AG(2, p="E3"), AG(2, p="")], [SR(1, "F1")], {(0, 0): "160", (1, 0): "20"}, "20210201"
    )
    m, rep = merge(a, b, SIMPLE, match=NAME)
    assert names(m) == [("TEST-2", "E3"), ("TEST-2", "")]
    assert (
        rep.matching["strict"]["antigens"] == 1 and rep.matching["name_fallback"]["antigens"] == 0
    )


def test_distinct_points_never_pair_by_name():
    a = table([AG(2, p="", annotations=("DISTINCT",))], [SR(1, "F1")], {(0, 0): "80"})
    b = table([AG(2, p="")], [SR(1, "F1")], {(0, 0): "160"}, date="20210201")
    m, rep = merge(a, b, SIMPLE, match=NAME)
    assert m.n_antigens == 2 and rep.matching["name_fallback"]["antigens"] == 0


def test_an_empty_field_is_a_duplicate_only_under_strict():
    c = table([AG(2, p=""), AG(2, p="")], [SR(1, ""), SR(1, "")], {})
    assert len(find_duplicates(c)) == 2
    assert find_duplicates(c, NAME) == []
    with pytest.raises(MergeError, match="duplicates"):
        merge(c, c, SIMPLE)
    merge(c, c, SIMPLE, match=NAME)  # all four ambiguous, none merged


def test_copied_references_are_judged_on_strict_pairs_only():
    # Sarah, Q131 R1 (c): the second table repeats the reference titres, but one reference
    # lacks its passage, so it pairs only by name and the block is NOT taken as copied
    ref, sera = [AG(1), AG(2)], [SR(1, "F1"), SR(2, "F2")]
    cells = {(0, 0): "640", (0, 1): "80", (1, 0): "40", (1, 1): "320"}
    a = table(ref + [AG(5)], sera, {**cells, (2, 0): "20"})
    b = table([AG(1), AG(2, p=""), AG(6)], sera, {**cells, (2, 1): "160"}, date="20210201")
    options = MergeOptions(merge_type=MergeType.SIMPLE, combine_copied_references=True)
    m, rep = merge(a, b, options, match=NAME)
    assert not rep.copied_references and rep.matching["name_fallback"]["antigens"] == 1
    assert any(ag == 1 for ag, _ in m.titres.layers[1])  # the reference's readings are merged
    strict_b = table(ref + [AG(6)], sera, {**cells, (2, 1): "160"}, date="20210201")
    _, strict = merge(a, strict_b, options, match=NAME)
    assert strict.copied_references  # all strict: copied, as before


def test_caller_pairs_replace_matching_and_are_checked():
    a = table([AG(1), AG(2)], [SR(1, "F1")], {(0, 0): "40", (1, 0): "80"})
    b = table([AG(7, p="X"), AG(8)], [SR(9, "F9")], {(0, 0): "160"}, date="20210201")
    m, rep = merge(a, b, SIMPLE, ag_pairs={0: 1}, sr_pairs={0: 0})
    assert m.n_antigens == 3 and m.n_sera == 1 and str(m.titres.table[1][0]) == "113"
    assert rep.matching["caller"] == {"antigens": 1, "sera": 1}
    with pytest.raises(MergeError, match="out of range"):
        merge(a, b, SIMPLE, ag_pairs={5: 0})
    with pytest.raises(MergeError, match="one to one"):
        merge(a, b, SIMPLE, ag_pairs={0: 0, 1: 0})
    d = table([AG(1, annotations=("DISTINCT",))], [SR(1, "F1")], {})
    with pytest.raises(MergeError, match="DISTINCT"):
        merge(d, b, SIMPLE, ag_pairs={0: 0})


def random_tables(seed: int) -> list:
    rnd = random.Random(seed)
    out = []
    for t in range(rnd.randint(4, 10)):
        ags, seen = [], set()
        for _ in range(rnd.randint(4, 10)):
            a = AG(rnd.randint(1, 10), p=rnd.choice(["E3", "SIAT1", "", ""]))
            if a.passage and (a.name, a.passage) in seen:
                continue
            seen.add((a.name, a.passage))
            ags.append(a)
        srs, seen = [], set()
        for _ in range(rnd.randint(2, 5)):
            s = SR(rnd.randint(1, 5), rnd.choice(["F1", "F2", "", ""]))
            if s.serum_id and (s.name, s.serum_id) in seen:
                continue
            seen.add((s.name, s.serum_id))
            srs.append(s)
        cells = {
            (i, j): rnd.choice(["<10", "20", "80", "320", "1280", ">1280"])
            for i in range(len(ags))
            for j in range(len(srs))
            if rnd.random() < 0.8
        }
        out.append(table(ags, srs, cells, date=f"2021{t + 1:02d}01"))
    return out


@pytest.mark.parametrize("seed", range(30))
def test_merge_many_with_the_name_rule_is_the_fold(seed):
    charts = random_tables(seed)
    fold, reports = charts[0], []
    for c in charts[1:]:
        fold, r = merge(fold, c, SIMPLE, match=NAME)
        reports.append(r)
    many, rm = merge_many(charts, SIMPLE, match=NAME)
    assert [vars(a) for a in many.antigens] == [vars(a) for a in fold.antigens]
    assert [vars(s) for s in many.sera] == [vars(s) for s in fold.sera]
    assert many.titres.layers == fold.titres.layers and many.titres.table == fold.titres.table
    assert rm.outcomes == reports[-1].outcomes and rm.dropped == reports[-1].dropped
    assert rm.steps == [MergeStep.of(r) for r in reports]


def test_the_random_tables_exercise_the_name_rule():
    fallback = ambiguous = 0
    for seed in range(30):
        _, rm = merge_many(random_tables(seed), SIMPLE, match=NAME)
        fallback += sum(sum(s.matching["name_fallback"].values()) for s in rm.steps)
        ambiguous += sum(sum(s.matching["ambiguous"].values()) for s in rm.steps)
    assert fallback > 20 and ambiguous > 20, (fallback, ambiguous)


def test_report_matching_is_in_to_dict():
    _, rm = merge_many(random_tables(3), SIMPLE, match=NAME)
    d = rm.to_dict()
    assert list(d["matching"]) == ["strict", "name_fallback", "ambiguous", "caller"]
    assert all(set(v) == {"antigens", "sera"} for v in d["matching"].values())
    assert all("matching" in s for s in d["steps"])
