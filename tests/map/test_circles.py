"""af.map.circles and the renderer's circles: serum circles drawn as the addenda drew them."""

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from af.chart.serum_circles import Circle, Coverage, HomologousAntigen, SerumCircle
from af.map.circles import NO_COLOUR, CircleError, CircleLook, serum_circle_marks
from af.map.render import CircleMark

pytest.importorskip("matplotlib")

from af.map.finish import finish_map  # noqa: E402
from af.map.style import Window  # noqa: E402

from .test_finish import SCHEME, SINCE, synthetic_points  # noqa: E402

NO_COVERAGE = Coverage("best-match", None, None, None, (), (), 0, "no_homologous_titre")


def record(
    *,
    theoretical: float | None = 2.5,
    antigen: int | None = 2,
    xy: tuple[float, float] | None = (1.0, 1.0),
) -> SerumCircle:
    """A serum with two homologous preparations: the first in chart order (ag1) gives the larger
    radius, the second (ag2) the drawn one."""
    homologous = (
        HomologousAntigen(1, "EX/1/2020 E3", "640", 3.0, None, "good"),
        HomologousAntigen(2, "EX/1/2020 SIAT1", "1280", theoretical, None, "good"),
    )
    status = "good" if theoretical is not None else "no_radius"
    circle = Circle(
        rule="best-match", empirical=None, empirical_antigen=None, titre="1280",
        theoretical=theoretical, theoretical_antigen=antigen,
        theoretical_titre="1280" if antigen == 2 else None, floored=False, status=status,
    )  # fmt: skip
    return SerumCircle(
        0, "EX/1/2020", "EX/1/2020 S1", "S1", "SIAT1", "", (), "cell", xy, 2.0, 9.0, "none",
        "computed", homologous, circle, NO_COVERAGE, circle, NO_COVERAGE,
    )  # fmt: skip


PAINTED = {"ag1": "#111111", "ag2": "#222222"}


def test_a_circle_takes_the_colour_of_the_antigen_its_radius_comes_from() -> None:
    marks, styles = serum_circle_marks([record()], PAINTED)
    assert marks == [CircleMark("sr0", 2.5, "#222222")]  # not ag1, first in chart order
    style = styles["sr0"]
    assert (style.fill, style.outline, style.outline_width) == ("black", "#222222", 3.0)
    assert style.on_top and style.scale == pytest.approx(1.3)  # ae's 26 points over af's 20


def test_an_unpainted_antigen_and_a_missing_radius_draw_as_ae_did() -> None:
    (unpainted,), _ = serum_circle_marks([record()], {"ag1": "#111111"})
    assert unpainted.fill == NO_COLOUR
    (none,), _ = serum_circle_marks([record(theoretical=None, antigen=None)], PAINTED)
    assert none.dashed and none.radius == 2.0 and none.fill == NO_COLOUR  # radius fold
    with pytest.raises(CircleError, match="no coordinates"):
        serum_circle_marks([record(xy=None)], PAINTED)
    with pytest.raises(CircleError, match="not one of"):
        CircleLook(radius="widest")


def test_circles_are_drawn_over_the_points_and_recorded(tmp_path: Path) -> None:
    common: dict[str, Any] = dict(
        chart="synthetic", scheme=SCHEME, window=Window("all", None), title="Synthetic",
        frame_size=14.0, must_show_since=SINCE, vaccine_labels={},
        created=dt.datetime(2026, 10, 21, tzinfo=dt.UTC),
        provenance={"inputs": {"chain": {"name": "synthetic", "sha256": "0" * 64}}},
    )  # fmt: skip
    marks = [CircleMark("sr1", 2.0, "#123456"), CircleMark("sr2", 3.0, "#654321", dashed=True)]
    result = finish_map(synthetic_points(), out_pdf=tmp_path / "c.pdf", circles=marks, **common)
    doc = json.loads(result.i7.read_text())["map"]
    assert doc["circles"] == [m.recorded() for m in marks]
    plain = finish_map(synthetic_points(), out_pdf=tmp_path / "p.pdf", **common)
    assert "circles" not in json.loads(plain.i7.read_text())["map"]
    with pytest.raises(ValueError, match="no drawn point"):
        finish_map(
            synthetic_points(), out_pdf=tmp_path / "x.pdf",
            circles=[CircleMark("ag-noxy", 2.0, "#123456")], **common,
        )  # fmt: skip


def test_a_styled_point_on_top_is_drawn_over_every_other_point() -> None:
    import matplotlib.pyplot as plt

    from af.map.render import DEFAULT_LOOK, PointStyle, draw_scene
    from af.map.style import style_points
    from af.map.viewport import Frame

    scene = style_points(synthetic_points(), SCHEME, Window("all", None), title="t")
    fig = plt.figure(figsize=(4, 4), dpi=72)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    styles = {"sr1": PointStyle(fill="black", on_top=True)}
    draw_scene(ax, scene, Frame(-7.0, 7.0, 14.0), {}, DEFAULT_LOOK, styles=styles,
               circles=[CircleMark("sr1", 2.0, "#123456")])  # fmt: skip
    points = [p for p in ax.patches if p.get_zorder() < 7]
    top = max(p.get_zorder() for p in points)
    (styled,) = [p for p in points if p.get_zorder() == top]
    assert tuple(styled.get_facecolor()) == (0.0, 0.0, 0.0, 1.0)  # the black serum, alone on top
    circles = [p for p in ax.patches if p.get_zorder() == 7]
    assert len(circles) == 2  # the translucent fill, then the outline
    assert circles[0].get_alpha() == pytest.approx(0.2)
    plt.close(fig)
