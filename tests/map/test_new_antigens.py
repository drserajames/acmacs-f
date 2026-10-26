"""af.map.new_antigens: interim reports' outlines for antigens new since the previous reports."""

import datetime as dt
from typing import Any

import numpy as np
import pytest

from af.chart.model import Antigen, Chart, Serum, Titres
from af.chart.titre import Titre
from af.map import new_antigens
from af.map.new_antigens import NewAntigenTier, first_titrated, new_antigen_styles, outlines_on_page

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


def previous(c: Chart, n: int) -> Chart:
    """The previous report's map: the first n antigens of ``c``."""
    return Chart(c.info, c.antigens[:n], c.sera, Titres(c.titres.table[:n]))


def test_an_antigen_is_new_when_the_previous_reports_map_lacks_it() -> None:
    """Q264 b: new to the reader, against the previous reports' maps; the first-table date is
    recorded beside (ag1's table is dated before the VCM cut-off but the VCM map lacks it)."""
    c = chart()
    vcm_map, interim_map = previous(c, 1), previous(c, 3)  # the interim map has ag0..ag2
    styles, record = new_antigen_styles(c, points(c), (VCM, INTERIM), (vcm_map, interim_map))
    assert set(styles) == {"ag1", "ag2", "ag3"}
    assert {k: s.outline_width for k, s in styles.items()} == {"ag1": 3.0, "ag2": 3.0, "ag3": 6.0}
    assert styles["ag3"].outline == "black" and styles["ag3"].raise_rank == 1
    assert [
        (r["new_since_previous_map"], r["drawn_new"], r["first_titrated_after"]) for r in record
    ] == [
        (3, 2, 2),  # ag1 is new to the reader though titrated before the cut-off
        (1, 1, 1),
    ]
    # the first interim: both tiers compare with the VCM's map; the thick one (listed last) wins
    same = NewAntigenTier("since the last interim", VCM.since, 6.0, "first interim")
    styles, _ = new_antigen_styles(c, points(c), (VCM, same), (vcm_map, vcm_map))
    assert {k: s.outline_width for k, s in styles.items()} == {"ag1": 6.0, "ag2": 6.0, "ag3": 6.0}
    # a vaccine mark keeps its own outline; a quiet month is counted, not refused
    styles, _ = new_antigen_styles(c, points(c), (VCM,), (vcm_map,), frozenset({"ag3"}))
    assert set(styles) == {"ag1", "ag2"}
    styles, record = new_antigen_styles(c, points(c), (VCM,), (c,))
    assert styles == {} and record[0]["new_since_previous_map"] == 0
    with pytest.raises(ValueError, match="previous maps for"):
        new_antigen_styles(c, points(c), (VCM, INTERIM), (vcm_map,))


def test_a_tier_names_each_maps_previous_report_by_chain_version() -> None:
    t = NewAntigenTier(
        "since the VCM", VCM.since, 3.0, "r", {"lab-a-hi": "lab/a-hi/merged@0123456789abcdef"}
    )
    assert t.previous_map("lab-a-hi") == ("lab/a-hi/merged", "0123456789abcdef")
    with pytest.raises(ValueError, match="names no previous map for lab-b-hi"):
        t.previous_map("lab-b-hi")
    with pytest.raises(ValueError, match="not dataset@version"):
        NewAntigenTier("x", VCM.since, 3.0, "r", {"lab-a-hi": "lab/a-hi/merged"})


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


def page_points() -> list[Any]:
    """New antigens drawn in colour (ag0), dated before the window (ag1), with no clade colour
    (ag2, unsequenced), undated (ag3), both before the window and uncoloured (ag4), and one of
    the older tier (ag5, coloured)."""
    from af.map.style import PointIn

    def p(i: int, date: dt.date | None, labels: frozenset[str]) -> PointIn:
        return PointIn(f"ag{i}", f"ag{i}", "antigen", (float(i), 0.0), labels, date=date,
                       sequenced=bool(labels))  # fmt: skip

    a = frozenset({"A"})
    return [p(0, D(2026, 10, 1), a), p(1, D(2026, 1, 1), a), p(2, D(2026, 10, 2), frozenset()),
            p(3, None, a), p(4, D(2026, 1, 1), frozenset()), p(5, D(2026, 10, 3), a)]  # fmt: skip


def test_a_page_outlines_only_the_new_antigens_it_draws_in_colour(monkeypatch: Any) -> None:
    """Sarah, 26 Oct 2026: "Only show the bold outline if the antigen would be colored." Grey
    from the window and grey for want of a clade colour are counted apart, and either can be
    switched back on alone."""
    pytest.importorskip("matplotlib")
    from af.map.render import PointStyle
    from af.map.style import Window

    from .test_finish import SCHEME

    thick = PointStyle(outline="black", outline_width=6.0, raise_rank=1)
    styles = {f"ag{i}": thick for i in range(5)}
    styles["ag5"] = PointStyle(outline="black", outline_width=3.0, raise_rank=0)
    record = [{"name": "since the VCM", "drawn_new": 1}, {"name": "since the last interim",
                                                          "drawn_new": 5}]  # fmt: skip
    since_sep = Window("since-sep", D(2026, 9, 22))

    def page(window: Window) -> tuple[set[str], list[tuple[int, int, int]]]:
        kept, rec = outlines_on_page(page_points(), SCHEME, window, {}, styles, record)
        counts = [(r["outlined"], r["suppressed_window_grey"], r["suppressed_unpainted"])
                  for r in rec]  # fmt: skip
        return set(kept), counts

    assert page(since_sep) == ({"ag0", "ag5"}, [(1, 0, 0), (1, 3, 1)])  # ag4 counts as window
    assert page(Window("all", None)) == ({"ag0", "ag1", "ag3", "ag5"}, [(1, 0, 0), (3, 0, 2)])
    kept, rec = outlines_on_page(page_points(), SCHEME, since_sep, {}, styles, record)
    assert (
        kept["ag0"] is thick
        and rec[1]["drawn_new"] == 5
        and record[1].keys() == {"name", "drawn_new"}
    )
    monkeypatch.setattr(new_antigens, "SUPPRESS_UNPAINTED", False)  # the window alone
    assert page(since_sep) == ({"ag0", "ag2", "ag5"}, [(1, 0, 0), (2, 3, 0)])
    monkeypatch.setattr(new_antigens, "SUPPRESS_UNPAINTED", True)
    monkeypatch.setattr(new_antigens, "SUPPRESS_WINDOW_GREY", False)  # no clade colour alone
    assert page(since_sep) == ({"ag0", "ag1", "ag3", "ag5"}, [(1, 0, 0), (3, 0, 2)])
