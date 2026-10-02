"""af.map.figure: a chart drawn the report way on any matplotlib Axes."""

import numpy as np
import pytest

from af.chart.model import Antigen, Chart, Serum, Titres
from af.map.colouring import ChartColours
from af.map.figure import FigureError, chart_points, chart_scene, draw_axes, frame_around
from af.map.render import DEFAULT_LOOK
from af.map.style import ColourRow, ColourScheme

pytest.importorskip("matplotlib")


def chart() -> Chart:
    antigens = [
        Antigen("/".join(("A(H3N2)", "OLDTOWN", str(i + 1), "2021")), passage=p, date="2021-03-01")
        for i, p in enumerate(["SIAT1", "E3", "SIAT2"])
    ]
    sera = [Serum("/".join(("A(H3N2)", "OLDTOWN", "9", "2020")), serum_id="S1", passage="E4")]
    return Chart({"V": "A(H3N2)"}, antigens, sera, Titres([[] for _ in antigens]))


XY = np.array([[0.0, 0.0], [2.0, 1.0], [np.nan, np.nan], [1.0, 3.0]])
SCHEME = ColourScheme("test", (ColourRow("Clade X", "#123456", frozenset({"X"})),))
COLOURS = ChartColours(
    SCHEME, (frozenset({"X"}), frozenset(), frozenset({"X"})), (True, False, True), {}
)


def test_points_follow_the_chart_and_skip_missing_coordinates() -> None:
    pts = chart_points(chart(), XY, labels=COLOURS.labels)
    assert [p.id for p in pts] == ["ag0", "ag1", "ag2", "sr0"]
    assert pts[1].passage_class == "egg" and pts[3].kind == "serum"
    assert pts[2].xy is None  # no coordinates: never drawn, never guessed
    with pytest.raises(FigureError, match="layout has shape"):
        chart_points(chart(), XY[:3])


def test_scene_paints_by_the_given_colours() -> None:
    scene = chart_scene(chart(), XY, COLOURS, title="T")
    colours = {p.id: p.colour for p in scene.points}
    assert colours["ag0"] == "#123456" and colours["ag1"] is None


def test_frame_covers_the_shown_points() -> None:
    f = frame_around(chart_scene(chart(), XY, COLOURS, title="T"), margin=1.0)
    assert f.size == pytest.approx(5.0)  # widest extent 3 + 2 * margin
    assert frame_around(chart_scene(chart(), XY, COLOURS, title="T"), size=10).size == 10
    # Placed by its top-left corner as drawn, y up: the largest y, not the smallest (Q105).
    assert (f.x, f.y) == pytest.approx((-1.5, 4.0))


def test_draw_axes_refuses_a_stretched_panel() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 2))
    scene = chart_scene(chart(), XY, COLOURS, title="T")
    with pytest.raises(FigureError, match="must be square"):
        draw_axes(ax, scene, frame_around(scene))
    plt.close(fig)


def test_a_panel_is_the_page_scaled() -> None:
    """Text is the report page's size times panel side over page side."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4, 4), dpi=72)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    scene = chart_scene(chart(), XY, COLOURS, title="T")
    draw_axes(ax, scene, frame_around(scene))
    title = next(t for t in ax.texts if t.get_text() == "T")
    scale = 4 * 72 / DEFAULT_LOOK.page_points
    assert title.get_fontsize() == pytest.approx(DEFAULT_LOOK.title_size * scale)
    assert len(ax.patches) >= 4  # 3 drawn points + legend box (antigen 2 has no coordinates)
    plt.close(fig)


def test_marker_for_passage_is_the_scenes_rule() -> None:
    from af.map.figure import marker_for_passage

    c = chart()
    scene = chart_scene(c, XY, COLOURS, title="T")
    points = [("antigen", a) for a in c.antigens] + [("serum", s) for s in c.sera]
    got = [marker_for_passage(kind, p.passage, p.reassortant) for kind, p in points]
    assert got == [p.marker for p in scene.points]
    assert set(got) >= {"circle", "egg", "uglyegg"}  # the fixture has cell, egg and an egg serum
    with pytest.raises(FigureError, match="kind"):
        marker_for_passage("titre", "E3")
