"""af.map.figure: a chart drawn the report way on any matplotlib Axes."""

from typing import Any

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


def test_the_public_types_are_re_exported_from_figure() -> None:
    """A caller pins to af.map.figure alone (pyacmapcheck, figure API §6)."""
    import subprocess
    import sys

    import af.map.colouring as colouring
    import af.map.figure as figure
    import af.map.render as render
    import af.map.style as style

    assert figure.Look is render.Look and figure.DEFAULT_LOOK is render.DEFAULT_LOOK
    assert figure.ColourScheme is style.ColourScheme and figure.ColourRow is style.ColourRow
    assert figure.ChartColours is colouring.ChartColours
    assert (figure.egg_path, figure.ugly_egg_path) == (render.egg_path, render.ugly_egg_path)
    assert (figure.GREY, figure.SERUM_OUTLINE) == (render.GREY, render.SERUM_OUTLINE)
    assert figure.Scene is style.Scene and figure.Window is style.Window
    assert all(hasattr(figure, name) for name in figure.__all__)
    # Drawing with your own colours does not load the stores: ChartColours is imported on use.
    probe = "import sys, af.map.figure; print('af.map.colouring' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def _points_drawn(ax: Any) -> dict[tuple[float, float], Any]:
    """The point patches by page position (legend patches sit in the legend box, ignored)."""
    out = {}
    for patch in ax.patches:
        ext = patch.get_path().transformed(patch.get_patch_transform()).get_extents()
        out[(round(float(ext.x0 + ext.x1) / 2, 4), round(float(ext.y0 + ext.y1) / 2, 4))] = patch
    return out


def test_per_point_styles_are_opt_in() -> None:
    """R's figure style (pyacmapcheck): serum outline colours, faded points, a thick outline
    on one antigen, a global opacity. Without them the drawing is the report's."""
    import dataclasses

    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    from af.map.figure import PointStyle

    scene = chart_scene(chart(), XY, COLOURS, title="T")
    frame = frame_around(scene)
    page = frame.page(scene.xy())

    def draw(**kwargs: Any) -> dict[str, Any]:
        fig = plt.figure(figsize=(4, 4), dpi=72)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        draw_axes(ax, scene, frame, legend=False, title=False, **kwargs)
        drawn = _points_drawn(ax)
        lines = {line.get_color() for line in ax.lines}
        plt.close(fig)
        by_id: dict[str, Any] = {}
        for i, p in enumerate(scene.points):
            if p.xy is not None:
                by_id[p.id] = drawn[(round(float(page[i, 0]), 4), round(float(page[i, 1]), 4))]
        by_id["grid"] = lines
        return by_id

    plain = draw()
    styled = draw(
        look=dataclasses.replace(DEFAULT_LOOK, alpha=0.8),
        styles={
            "sr0": PointStyle(outline="#ff0000"),
            "ag0": PointStyle(alpha=0.2),
            "ag1": PointStyle(outline="#000000", outline_width=3.0),
        },
    )
    assert plain["sr0"].get_edgecolor() == to_rgba("#9a9a9a")  # the report's serum outline
    assert plain["ag0"].get_alpha() is None
    assert styled["sr0"].get_edgecolor() == to_rgba("#ff0000", 0.8)  # own colour, global alpha
    assert styled["ag0"].get_alpha() == 0.2  # a point's own opacity beats the global one
    assert styled["ag1"].get_linewidth() == pytest.approx(3 * plain["ag1"].get_linewidth() / 0.8)
