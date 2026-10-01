"""Forced column bases per stage (af.chart.column_bases), on a synthetic chart."""

import dataclasses
import json

import numpy as np
import pytest

from af.chart.ace import read_chart, write_chart
from af.chart.column_bases import (
    ColumnBaseOverride,
    ColumnBaseOverrideError,
    apply_column_base_override,
)
from af.chart.model import Projection, Serum

from .test_ace import small_chart


def bases(projection: Projection) -> np.ndarray:
    assert projection.forced_column_bases is not None
    return projection.forced_column_bases


def rule(*sera: str, value: float = 5.0) -> ColumnBaseOverride:
    return ColumnBaseOverride("lower", "a discarded titre set the base", "test", sera, value)


def test_named_serum_gets_the_forced_base_and_the_stage_its_own_bases():
    chart = small_chart()
    stage = chart.projections[0]
    before = chart.column_bases(projection=stage)
    result = apply_column_base_override(chart, stage, rule("TEST-5 F002"))
    after = result.projection.forced_column_bases
    assert after is not None
    assert before[1] != 5.0 and after[1] == 5.0 and after[0] == before[0]
    assert result.projection.stress is None  # the old stress used the old bases
    assert stage.forced_column_bases is None and stage.stress == 1.5  # input untouched
    assert chart.forced_column_bases is None  # the chart's bases are not the stage's
    assert np.array_equal(
        chart.optimiser_arrays(projection=result.projection)["column_bases"], after
    )
    assert np.array_equal(chart.optimiser_arrays(projection=stage)["column_bases"], before)
    report = result.report(rule("TEST-5 F002"))
    assert report["sera"] == 1 and report["changed"] == 1
    assert report["changes"] == [{"designation": "TEST-5 F002", "before": before[1], "after": 5.0}]


def test_report_is_plain_json_for_provenance():
    chart = small_chart()
    r = rule("TEST-5 F002")
    report = apply_column_base_override(chart, chart.projections[0], r).report(r)
    back = json.loads(json.dumps(report, allow_nan=False))
    assert back == report and back["override"] == "lower" and back["decided"] == "test"
    changes = back["changes"]
    assert all(type(c[k]) is float for c in changes for k in ("before", "after"))


def test_idempotent():
    chart = small_chart()
    once = apply_column_base_override(chart, chart.projections[0], rule("TEST-5 F002"))
    twice = apply_column_base_override(chart, once.projection, rule("TEST-5 F002"))
    assert np.array_equal(bases(once.projection), bases(twice.projection))
    assert twice.report(rule("TEST-5 F002"))["changed"] == 0


@pytest.mark.parametrize(
    "sera, message",
    [
        (("NO-SUCH F009",), "matches 0 sera"),
        (("TEST-5 F002", "TEST-5 F002"), "named twice"),
        ((), "no sera named"),
    ],
)
def test_every_name_must_match_exactly_one_serum(sera, message):
    chart = small_chart()
    with pytest.raises(ColumnBaseOverrideError, match=message):
        apply_column_base_override(chart, chart.projections[0], rule(*sera))


def test_a_designation_shared_by_two_sera_is_refused():
    chart = small_chart()
    chart.sera[0] = Serum("TEST-5", serum_id="F002")
    with pytest.raises(ColumnBaseOverrideError, match="matches 2 sera"):
        apply_column_base_override(chart, chart.projections[0], rule("TEST-5 F002"))


def test_each_stage_keeps_its_bases_through_ace(tmp_path):
    chart = small_chart()
    first = chart.projections[0]
    second = apply_column_base_override(chart, first, rule("TEST-5 F002")).projection
    chart.projections.append(dataclasses.replace(second, stress=2.0))
    write_chart(chart, tmp_path / "stages.ace")
    back = read_chart(tmp_path / "stages.ace")
    assert back.projections[0].forced_column_bases is None
    assert np.array_equal(bases(back.projections[1]), bases(second))
    assert np.array_equal(back.column_bases(projection=back.projections[1]), bases(second))


def _merged_with_a_greater_than():
    """Two synthetic tables, one with a `>` cell: the merge forces every serum's basis."""
    from af.chart.merge import MergeOptions, MergeType, merge
    from af.chart.model import Antigen, Chart, Serum, Titres, empty_table
    from af.chart.titre import Titre

    def table(cells, date):
        t = empty_table(2, 2)
        for (i, j), v in cells.items():
            t[i][j] = Titre.parse(v)
        return Chart(
            info={"D": date},
            antigens=[Antigen("TEST-A0", passage="MDCK1"), Antigen("TEST-A1", passage="MDCK1")],
            sera=[Serum("TEST-S0", serum_id="S0"), Serum("TEST-S1", serum_id="S1")],
            titres=Titres(t),
        )

    a = table({(0, 0): "40", (0, 1): "80", (1, 0): "20", (1, 1): "40"}, "20210115")
    b = table({(0, 0): ">1280", (0, 1): "80", (1, 0): "20", (1, 1): "40"}, "20210215")
    merged, _ = merge(a, b, MergeOptions(merge_type=MergeType.SIMPLE))
    assert merged.forced_column_bases is not None
    return merged


def test_minimum_column_basis_raises_forced_chart_bases_as_ae():
    # ae: Chart::column_bases applies mcb to the forced basis too (max(forced, mcb))
    merged = _merged_with_a_greater_than()
    forced = np.asarray(merged.forced_column_bases, dtype=float)
    assert np.array_equal(merged.column_bases("none"), forced)
    raised = merged.column_bases("2560")  # log2(2560/10) = 8
    assert np.array_equal(raised, np.maximum(forced, 8.0))
    assert not np.array_equal(raised, merged.column_bases("none"))


def test_projection_forced_bases_are_used_verbatim_as_ae():
    from af.chart.model import Projection

    merged = _merged_with_a_greater_than()
    own = np.array([1.0, 2.0])  # below any minimum
    p = Projection(layout=np.zeros((merged.n_points, 2)), forced_column_bases=own)
    assert np.array_equal(merged.column_bases("2560", projection=p), own)
