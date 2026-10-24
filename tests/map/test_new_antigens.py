"""af.map.new_antigens: the interim reports' outlines for antigens newly titrated since cut-offs."""

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from af.chart.model import Antigen, Chart, Serum, Titres
from af.chart.titre import Titre
from af.map.new_antigens import NewAntigenTier, first_titrated, new_antigen_styles

D = dt.date
VCM = NewAntigenTier("since the VCM", D(2026, 9, 21), 3.0, "Sarah, M2: thin, since the VCM")
INTERIM = NewAntigenTier("since the last interim", D(2026, 10, 10), 6.0, "Sarah, M2: thick")


def chart() -> Chart:
    """Four antigens first titrated in 2025 (old), September (just before the VCM cut-off),
    October 5th (between the cut-offs, the interim's being October 10th) and October 20th
    (after both); the old one is titrated again in October, which does not make it new."""
    t = Titre.parse
    antigens = [Antigen(f"A(H3N2)/EXAMPLETOWN/{i}/2025") for i in range(4)]
    table = [[t("40")] for _ in antigens]
    layers = [
        {(0, 0): t("40")},
        {(1, 0): t("40")},
        {(2, 0): t("80"), (0, 0): t("40")},
        {(3, 0): t("160"), (2, 0): t("*")},
    ]
    sources = [{"D": "20250105"}, {"D": "20260915"}, {"D": "20261005-002"}, {"D": "20261020"}]
    return Chart({"S": sources}, antigens, [Serum("A(H3N2)/EXAMPLETOWN/9/2025", "F9")],
                 Titres(table, layers))  # fmt: skip


def points(c: Chart) -> list[Any]:
    pytest.importorskip("matplotlib")
    from af.map.figure import chart_points

    return chart_points(c, np.zeros((c.n_points, 2)))


def test_first_titrated_is_the_earliest_table_with_a_titre() -> None:
    assert first_titrated(chart()) == [
        D(2025, 1, 5),
        D(2026, 9, 15),
        D(2026, 10, 5),
        D(2026, 10, 20),
    ]
    single = Chart({"D": "20240301"}, chart().antigens, chart().sera, Titres(chart().titres.table))
    assert first_titrated(single) == [D(2024, 3, 1)] * 4


def test_each_new_antigen_takes_the_latest_cut_off_it_passes(tmp_path: Path) -> None:
    c = chart()
    styles, record = new_antigen_styles(c, points(c), (VCM, INTERIM), "lab-a-hi")
    assert set(styles) == {"ag2", "ag3"}  # ag0 old, ag1 before the VCM cut-off
    assert styles["ag2"].outline_width == 3.0 and styles["ag3"].outline_width == 6.0
    assert styles["ag3"].outline == "black" and styles["ag3"].raise_rank == 1
    assert [(r["first_titrated_after"], r["outlined"]) for r in record] == [(2, 1), (1, 1)]
    assert record[0]["ae_rule_not_in_previous_chart"] is None  # no round to compare with
    # the first interim: both cut-offs are the VCM's, and the thick tier (configured last) wins
    same = NewAntigenTier("since the last interim", VCM.since, 6.0, "first interim")
    styles, _ = new_antigen_styles(c, points(c), (VCM, same), "lab-a-hi")
    assert {k: s.outline_width for k, s in styles.items()} == {"ag2": 6.0, "ag3": 6.0}
    # a vaccine mark keeps its own outline; a quiet month is counted, not refused
    styles, _ = new_antigen_styles(c, points(c), (VCM,), "lab-a-hi", frozenset({"ag3"}))
    assert set(styles) == {"ag2"}
    quiet = NewAntigenTier("since later", D(2026, 12, 1), 6.0, "nothing yet")
    styles, record = new_antigen_styles(c, points(c), (quiet,), "lab-a-hi")
    assert styles == {} and record[0]["first_titrated_after"] == 0
    missing = NewAntigenTier("compared", VCM.since, 3.0, "r", compare_round=tmp_path)
    _, record = new_antigen_styles(c, points(c), (missing,), "lab-a-hi")
    assert record[0]["ae_rule_not_in_previous_chart"] is None  # that round has no such chart


def test_a_raised_new_antigen_is_drawn_over_antigens_and_under_the_marks() -> None:
    import matplotlib.pyplot as plt

    from af.map.render import DEFAULT_LOOK, PointStyle, draw_scene
    from af.map.style import Window, style_points
    from af.map.viewport import Frame

    from .test_finish import SCHEME, synthetic_points

    scene = style_points(synthetic_points(), SCHEME, Window("all", None), title="t",
                         vaccines={"ag100": "Te/25-cell"})  # fmt: skip
    fig = plt.figure(figsize=(4, 4), dpi=72)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    styles = {"ag101": PointStyle(outline="black", outline_width=6.0, raise_rank=1)}
    draw_scene(ax, scene, Frame(-7.0, 7.0, 14.0), {}, DEFAULT_LOOK, styles=styles)
    z = sorted({p.get_zorder() for p in ax.patches})
    thick = [p for p in ax.patches if p.get_linewidth() == 6.0]
    assert len(thick) == 1 and 4 < thick[0].get_zorder() < 5  # over plain (4), under marks (5)
    assert 5 in z
    plt.close(fig)
