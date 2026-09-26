"""Measurement only (Sarah, Q64): merge drops and control-chart flags are reported, not applied."""

import dataclasses

from af.chain.diagnostics import control_flags, flags
from af.chain.tables import repeat_drops
from af.chart.merge import MergeOptions, MergeReport, merge_layers
from af.chart.model import Antigen, Chart, Serum, Titres
from af.chart.titre import Titre

from .test_tables import make_table


def test_repeat_readings_the_merge_drops_are_listed():
    table = make_table(0)
    assert repeat_drops(table) == []  # 40 and 80: merged, nothing dropped
    table = dataclasses.replace(table, titres=[list(r) for r in table.titres])
    table.titres[1][2] = ["20", "1280"]  # SD of the logs 3 > 1.0
    [line] = repeat_drops(table)
    assert "TEST-1" in line and "LOT2" in line and "20 1280 → * (sd-too-big)" in line


def test_layer_merge_lists_the_cells_it_drops():
    layers = [{(0, 0): Titre.parse("20"), (0, 1): Titre.parse("40")}, {(0, 0): Titre.parse("1280")}]
    report = MergeReport()
    merged, _ = merge_layers(layers, 1, 2, MergeOptions(), report)
    assert str(merged[0][0]) == "*" and report.dropped == [(0, 0, "sd-too-big")]
    assert report.outcomes["sd-too-big"] == 1


def chart_with_series(series: list[str]) -> Chart:
    layers = [{(0, 0): Titre.parse(t)} for t in series]
    return Chart(
        info={},
        antigens=[Antigen("TEST-A", passage="P1")],
        sera=[Serum("TEST-S", serum_id="S1")],
        titres=Titres([[Titre.parse(series[-1])]], layers),
    )


def test_a_flag_is_reported_at_the_table_that_completes_it():
    before = ["40", "40", "40", "40", "40"]
    assert control_flags(chart_with_series(before)) == []
    # 640 is 4 log2 units above the median: rule 1 (a point >= 3 out) and diff of 3
    rules = {f["rule"] for f in control_flags(chart_with_series(before + ["640"]))}
    assert rules == {"1 >=3", "diff of 3"}
    # one table later the outlier is old news: rule 1 is not reported again
    later = {f["rule"] for f in control_flags(chart_with_series(before + ["640", "40"]))}
    assert "1 >=3" not in later
    # a second outlier in the same cell is news again
    again = {f["rule"] for f in control_flags(chart_with_series(before + ["640", "40", "640"]))}
    assert "1 >=3" in again


def test_trend_is_not_tested_on_continuous_titres():
    rising = ["21", "43", "87", "175", "351"]  # off the two-fold series (e.g. HINT)
    assert all(f["rule"] != "trend 4" for f in control_flags(chart_with_series(rising)))
    two_fold = ["20", "40", "80", "160", "320"]
    assert "trend 4" in {f["rule"] for f in control_flags(chart_with_series(two_fold))}


def test_new_control_flags_make_one_line():
    d = {"control_flags": [{"rule": "1 >=3"}, {"rule": "diff of 3"}]}
    assert "2 new control-chart flags" in flags(d)
