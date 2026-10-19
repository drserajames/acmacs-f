"""Map titles and vaccine labels drawn as the shipped report draws them (style step 6)."""

import matplotlib
import numpy as np
import pytest

from af.chart.model import Antigen, Chart, Titres
from af.map.build import BuildError, map_title
from af.map.config import TitlesConfig
from af.map.labels import Placed, place_labels
from af.map.render import DEFAULT_LOOK, draw_scene, title_box
from af.map.style import Scene, Window
from af.map.viewport import Frame

matplotlib.use("pdf")

TITLES = TitlesConfig(
    labs={"LABX": "Labx"}, assays={"HI": "HI", "NEUT": "Neut"}, assay_subtypes=("A(H3N2)",)
)


def chart(virus: str, assay: str) -> Chart:
    antigens = [Antigen("/".join(("A", "OLDTOWN", "1", "2021")))]  # invented, joined for the gate
    return Chart({"V": virus, "l": "LABX", "A": assay}, antigens, [], Titres([[]]))


def test_titles_name_the_lab_and_the_assay_as_configured() -> None:
    assert map_title(chart("A(H3N2)", "HI"), None, TITLES) == "Labx A(H3N2) HI by clade"
    assert map_title(chart("A(H3N2)", "NEUT"), None, TITLES) == "Labx A(H3N2) Neut by clade"
    assert map_title(chart("A(H1N1)", "HI"), None, TITLES) == "Labx A(H1N1) by clade"
    assert map_title(chart("A(H3N2)", "HI"), "Own title", TITLES) == "Own title"
    # without a round's [titles]: lab codes as they are, no assay words
    assert map_title(chart("A(H3N2)", "HI"), None) == "LABX A(H3N2) by clade"


def test_an_assay_with_no_title_word_is_refused() -> None:
    with pytest.raises(BuildError, match="'OTHER' .* no word"):
        map_title(chart("A(H3N2)", "OTHER"), None, TITLES)


def test_labels_sit_next_to_their_mark_even_over_points() -> None:
    """Nearest first: a label stays at the first radius although points lie under it there."""
    anchor = np.array([[0.5, 0.5]])
    ring = np.array(
        [[0.5 + 0.06 * np.cos(a), 0.5 + 0.06 * np.sin(a)] for a in np.linspace(0, 6.2, 24)]
    )
    (placed,) = place_labels(anchor, ["Ex/21-cell"], ring, [], sizes=[(0.1, 0.035)])
    left, top, right, bottom = placed.box
    gap = np.hypot(max(left - 0.5, 0, 0.5 - right), max(top - 0.5, 0, 0.5 - bottom))
    assert gap == pytest.approx(0.0275, abs=1e-6)  # just clear of the 0.025 mark
    assert placed.covered_points > 0


def scene(title: str) -> Scene:
    return Scene(title=title, window=Window("all", None), scheme="none", points=[], legend=[])


def test_the_title_box_is_the_drawn_title_measured() -> None:
    """Same length, different letters: an estimate by character count could not tell them apart."""
    wide, narrow = (
        title_box(scene("W" * 20), DEFAULT_LOOK),
        title_box(scene("i" * 20), DEFAULT_LOOK),
    )
    assert wide.right > 2 * narrow.right


def test_labels_carry_a_white_halo() -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4, 4), dpi=72)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    from af.map.figure import chart_scene
    from tests.map.test_figure import COLOURS, XY
    from tests.map.test_figure import chart as invented_chart

    label = Placed("Ex/21-cell", (0.5, 0.5), (0.5, 0.5, 0.6, 0.53), 0, 0)
    drawn = chart_scene(invented_chart(), XY, COLOURS, title="T")
    draw_scene(ax, drawn, Frame(-5.0, 5.0, 10.0), {"x": label}, DEFAULT_LOOK, legend=False)
    text = next(t for t in ax.texts if t.get_text() == "Ex/21-cell")
    (halo,) = text.get_path_effects()
    assert halo._gc["foreground"] == "white"  # type: ignore[attr-defined]
    assert text.get_fontsize() == DEFAULT_LOOK.label_size
    plt.close(fig)


def test_a_label_keeps_off_another_vaccines_mark() -> None:
    """Two vaccines side by side: neither label may sit on the other's mark (no halo mends that)."""
    from af.map.labels import _on_circle

    anchors = np.array([[0.5, 0.5], [0.56, 0.5]])
    placed = place_labels(
        anchors,
        ["Ex/21-cell", "Ex/22-cell"],
        anchors,
        [],
        sizes=[(0.1, 0.035)] * 2,
        mark_radius=0.025,
    )
    for p in placed:
        assert not any(_on_circle(p.box, a, 0.025) for a in anchors)
    unguarded = place_labels(
        anchors, ["Ex/21-cell", "Ex/22-cell"], anchors, [], sizes=[(0.1, 0.035)] * 2
    )
    assert any(
        _on_circle(p.box, a, 0.025) for p in unguarded for a in anchors
    )  # the guard mattered


def test_furniture_outranks_a_neighbouring_mark() -> None:
    """The legend is drawn over labels: a label rather touches another vaccine's mark than hides
    under the legend."""
    from af.map.viewport import Box

    legend = Box("legend", 0.0, 0.0, 0.47, 1.0)  # everything left of the vaccine
    anchors = np.array([[0.5, 0.5], [0.56, 0.5], [0.5, 0.56], [0.5, 0.44]])  # boxed in on 3 sides
    texts = ["Ex/21-cell", "Ex/22-cell", "Ex/23-cell", "Ex/24-cell"]
    (first, *_) = place_labels(
        anchors, texts, anchors, [legend], sizes=[(0.1, 0.035)] * 4, mark_radius=0.025
    )
    assert first.box[0] >= legend.right


@pytest.mark.parametrize("text", ["Ex/21-cell", "Ex/21-egg"])
def test_a_label_is_drawn_inside_its_placed_box(text: str) -> None:
    """The placer keeps boxes apart, so the glyphs must stay in theirs. Aligned "bottom",
    matplotlib kept room for a descender "-cell" does not have and drew it about 6 points above
    its box, onto the label above (R5 review, 19 Oct 2026)."""
    import matplotlib.pyplot as plt

    from af.map.render import text_size

    look = DEFAULT_LOOK
    w, h = (v / look.page_points for v in text_size(text, look.label_size))
    box = (0.3, 0.5, 0.3 + w, 0.5 + h)
    fig = plt.figure(figsize=(look.page_points / 72,) * 2, dpi=72 * 4)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    label = Placed(text, (0.2, 0.5), box, 0, 0)
    from af.map.figure import chart_scene
    from tests.map.test_figure import COLOURS, XY
    from tests.map.test_figure import chart as invented_chart

    drawn = chart_scene(invented_chart(), XY, COLOURS, title="")
    far = Frame(100.0, 100.0, 10.0)  # every point off the page: only the label is drawn dark
    draw_scene(ax, drawn, far, {"x": label}, look, legend=False, title=False)
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    rows, cols = np.nonzero(rgb.max(axis=2) < 128)  # the glyphs (black; grid and halo light)
    px = rgb.shape[0]
    top, bottom = rows.min() / px, (rows.max() + 1) / px
    assert box[1] - 1 / px <= top and bottom <= box[3] + 1 / px
    assert box[0] - 1 / px <= cols.min() / px and (cols.max() + 1) / px <= box[2] + 1 / px


def test_a_crowd_of_vaccines_leaves_no_label_on_another() -> None:
    """Eight marks in a tight crowd: every near place touches a label or a mark. A label on
    another label must lose to one further out, not tie with a label on a mark (R5 review)."""
    from itertools import combinations

    from af.map.labels import _grown, _overlap

    rng = np.random.default_rng(3)
    anchors = 0.5 + rng.normal(size=(8, 2)) * 0.03
    gap = 2.0 / 800
    placed = place_labels(
        anchors,
        [f"Ex/2{i}-cell" for i in range(8)],
        anchors,
        [],
        sizes=[(0.18, 0.03)] * 8,
        mark_radius=0.025,
        gap=gap,
    )
    assert not [
        (a.text, b.text) for a, b in combinations(placed, 2) if _overlap(_grown(a.box, gap), b.box)
    ]
