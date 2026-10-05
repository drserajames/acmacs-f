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
    with pytest.raises(FigureError, match="must have its frame.s shape"):
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
    on one antigen, a global opacity and grid colour. Without them the drawing is the report's."""
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
        look=dataclasses.replace(DEFAULT_LOOK, alpha=0.8, grid_colour="#e5e5e5"),
        styles={
            "sr0": PointStyle(outline="#ff0000"),
            "ag0": PointStyle(alpha=0.2),
            "ag1": PointStyle(outline="#000000", outline_width=3.0),
        },
    )
    assert plain["sr0"].get_edgecolor() == to_rgba("#9a9a9a")  # the report's serum outline
    assert plain["ag0"].get_alpha() is None and plain["grid"] == {"#dddddd"}
    assert styled["sr0"].get_edgecolor() == to_rgba("#ff0000", 0.8)  # own colour, global alpha
    assert styled["ag0"].get_alpha() == 0.2  # a point's own opacity beats the global one
    assert styled["ag1"].get_linewidth() == pytest.approx(3 * plain["ag1"].get_linewidth() / 0.8)
    assert styled["grid"] == {"#e5e5e5"}


def test_draw_axes_says_which_points_the_frame_clips() -> None:
    import matplotlib.pyplot as plt

    from af.map.viewport import Frame

    scene = chart_scene(chart(), XY, COLOURS, title="T")
    fig = plt.figure(figsize=(4, 4), dpi=72)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    # a 2.5-unit frame from (-0.5, 2): ag0 (0,0) and ag1 (2,1) are inside, sr0 (1,3) is above it
    clipped = draw_axes(ax, scene, Frame(-0.5, 2.0, 2.5), legend=False, title=False)
    plt.close(fig)
    assert clipped == ("sr0",)


def test_a_marked_point_keeps_its_old_name_for_now() -> None:
    """ScenePoint.mark is the renderer's word; .vaccine stays readable while callers move over."""
    scene = chart_scene(chart(), XY, COLOURS, title="T", vaccines={"ag1": "V1"})
    marked = next(p for p in scene.points if p.id == "ag1")
    assert marked.mark == "V1" and marked.vaccine == "V1"


# ---------------------------------------------------------------- rectangular frames (Sarah, 2 Oct)


def test_a_rectangular_frame_divides_each_axis_by_its_own_side() -> None:
    from af.map.viewport import Frame

    frame = Frame(0.0, 5.0, 10.0, height=5.0)
    assert frame.aspect == 0.5 and frame.tall == 5.0
    assert frame.page(np.array([[10.0, 0.0], [5.0, 2.5]])).tolist() == [[1.0, 1.0], [0.5, 0.5]]
    assert Frame(0.0, 0.0, 4.0).aspect == 1.0  # square: as before
    with pytest.raises(ValueError, match="positive"):
        Frame(0.0, 0.0, 4.0, height=0.0)


def test_frame_around_can_fit_the_points_extent() -> None:
    scene = chart_scene(chart(), XY, COLOURS, title="T")  # shown x 0..2, y 0..3
    f = frame_around(scene, margin=1.0, square=False)
    assert (f.x, f.y, f.size, f.tall) == pytest.approx((-1.0, 4.0, 4.0, 5.0))
    with pytest.raises(FigureError, match="square"):
        frame_around(scene, size=10, square=False)


def test_a_rectangular_panel_keeps_map_units_and_circles_round() -> None:
    import matplotlib.pyplot as plt

    from af.map.viewport import Frame

    scene = chart_scene(chart(), XY, COLOURS, title="T")
    frame = Frame(-1.0, 4.0, 8.0, height=4.0)  # twice as wide as tall
    fig = plt.figure(figsize=(4, 2), dpi=72)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    draw_axes(ax, scene, frame, legend=False, title=False)
    fig.canvas.draw()
    from matplotlib.patches import Circle, Rectangle

    shapes = set()
    for patch in ax.patches:  # each marker keeps its own proportions, not the panel's
        ext = patch.get_window_extent()
        if isinstance(patch, (Circle, Rectangle)):
            assert ext.width == pytest.approx(ext.height, rel=1e-6)
            shapes.add(type(patch).__name__)
        else:  # an egg is 1.191 times as tall as wide (kateri's); an ugly egg as tall as wide
            ratio = ext.height / ext.width
            shape = "egg" if ratio == pytest.approx(1.191, abs=0.01) else "uglyegg"
            assert shape == "egg" or ratio == pytest.approx(1.0, rel=1e-6)
            shapes.add(shape)
    assert shapes == {"Circle", "egg", "uglyegg"}  # the fixture: cell, egg and an egg serum
    # ag0 (0,0) and ag1 (2,1): 2 units across and 1 up are 2:1 on the page too
    page = ax.transData.transform(frame.page(np.array([[0.0, 0.0], [2.0, 1.0]])) * [1, 0.5])
    dx, dy = page[1] - page[0]
    assert dx == pytest.approx(2 * dy)  # display y grows upwards: map up is display up
    assert dx == pytest.approx(2 * 4 * 72 / 8)  # 2 units of an 8-unit frame on a 4 in panel
    plt.close(fig)
    square = plt.figure(figsize=(3, 3), dpi=72)
    with pytest.raises(FigureError, match="8 x 4 map units"):
        draw_axes(square.add_axes((0.0, 0.0, 1.0, 1.0)), scene, frame)
    plt.close(square)


def test_a_rectangular_pdf_has_its_frames_shape_and_its_i7_says_so(tmp_path: Any) -> None:
    import datetime as dt

    from af.map.i7 import i7_document
    from af.map.render import draw_pdf
    from af.map.viewport import Frame

    scene = chart_scene(chart(), XY, COLOURS, title="T")
    frame = Frame(-1.0, 4.0, 8.0, height=4.0)
    pdf = tmp_path / "rect.pdf"
    draw_pdf(scene, frame, {}, pdf)
    box = pdf.read_bytes().split(b"/MediaBox")[1].split(b"]")[0]
    width, height = (float(v) for v in box.strip(b" [").split()[2:4])
    assert height == pytest.approx(width / 2)
    doc = i7_document(
        scene, frame, {}, chart="c", pdf=pdf,
        created=dt.datetime(2026, 10, 2, tzinfo=dt.UTC),
        provenance={"inputs": {"x": {"sha256": "0" * 64}}},
    )  # fmt: skip
    assert doc["map"]["viewport"] == [-1.0, 4.0, 8.0, 4.0]


def test_a_point_can_be_drawn_larger() -> None:
    """R draws removed points larger (sera at 7 against 5): PointStyle.scale multiplies one
    point's size and nothing else."""
    import matplotlib.pyplot as plt

    from af.map.figure import PointStyle

    scene = chart_scene(chart(), XY, COLOURS, title="T")
    frame = frame_around(scene)

    def sizes(**kwargs: Any) -> dict[tuple[float, float], float]:
        """Marker width by marker centre (a scale changes the size, never the centre)."""
        fig = plt.figure(figsize=(4, 4), dpi=72)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        draw_axes(ax, scene, frame, legend=False, title=False, **kwargs)
        fig.canvas.draw()
        out = {}
        for patch in ax.patches:
            e = patch.get_window_extent()
            out[(round(float(e.x0 + e.x1) / 2, 2), round(float(e.y0 + e.y1) / 2, 2))] = float(
                e.width
            )
        plt.close(fig)
        return out

    plain = sizes()
    bigger = sizes(styles={"sr0": PointStyle(scale=1.4), "ag1": PointStyle(scale=2.0)})
    assert plain.keys() == bigger.keys() and len(plain) == 3  # ag0, ag1, sr0 (ag2: no coordinates)
    grown = sorted(round(bigger[c] / plain[c], 6) for c in plain)
    assert grown == [1.0, 1.4, 2.0]  # ag0 as it was; sr0 and ag1 by their own scales
    with pytest.raises(ValueError, match="positive"):
        PointStyle(scale=0.0)
    assert PointStyle(scale=1.4).recorded() == {"scale": 1.4}  # recorded in the I7 like the rest


def test_i7_points_carry_the_full_passage(tmp_path: Any) -> None:
    """Two preparations of one virus with one date and one passage class differ only by their
    passage string; the I7 records it so a reader can tell them apart (11-reports)."""
    import datetime as dt

    from af.map.i7 import i7_document
    from af.map.render import draw_pdf
    from af.map.viewport import Frame

    name = "/".join(("A(H3N2)", "OLDTOWN", "1", "2021"))
    antigens = [Antigen(name, passage=p, date="2021-03-01") for p in ("SIAT1", "SIAT2", "")]
    sera = [Serum("/".join(("A(H3N2)", "OLDTOWN", "9", "2020")), serum_id="S1", passage="E4")]
    two = Chart({"V": "A(H3N2)"}, antigens, sera, Titres([[] for _ in antigens]))
    scene = chart_scene(
        two,
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        ChartColours(SCHEME, (frozenset(),) * 3, (False,) * 3, {}),
        title="T",
    )
    pdf = tmp_path / "p.pdf"
    frame = Frame(-1.0, 2.0, 3.0)
    draw_pdf(scene, frame, {}, pdf)
    doc = i7_document(
        scene, frame, {}, chart="c", pdf=pdf,
        created=dt.datetime(2026, 10, 5, tzinfo=dt.UTC),
        provenance={"inputs": {"x": {"sha256": "0" * 64}}},
    )  # fmt: skip
    ags = doc["map"]["antigens"]
    assert [a["passage_class"] for a in ags[:2]] == ["cell", "cell"]  # the class cannot tell
    assert [a["passage"] for a in ags] == ["SIAT1", "SIAT2", ""]  # the passage can; none is ""
    assert doc["map"]["sera"][0]["passage"] == "E4"
