"""The map legend drawn as the shipped report draws it (style step 7)."""

import dataclasses
from pathlib import Path

import pytest

from af.map.config import LegendConfig
from af.map.render import DEFAULT_LOOK, legend_box
from af.map.roundconfig import ConfigError, load_maps_config
from af.map.style import Scene, Window

from .test_roundconfig import GOOD, write


def scene(rows: list[tuple[str, int]]) -> Scene:
    legend = [(text, "#123456", n) for text, n in rows]
    return Scene(title="T", window=Window("all", None), scheme="x", points=[], legend=legend)


def test_the_box_is_its_rows_and_its_measured_text() -> None:
    look, pts = DEFAULT_LOOK, DEFAULT_LOOK.page_points
    box = legend_box(scene([("A", 1), ("B", 2)]), look)
    assert (box.bottom - box.top) * pts == pytest.approx(2 * 24.5 + 2.26)
    assert box.left * pts == pytest.approx(10.0) and box.bottom * pts == pytest.approx(790.0)
    wide = legend_box(scene([("WWWW", 1)]), look)
    narrow = legend_box(scene([("iiii", 1)]), look)
    assert wide.right > narrow.right  # measured in the font, not counted in characters
    bigger = dataclasses.replace(look, legend_text_size=20.41, legend_row_pitch=28.6)
    assert legend_box(scene([("WWWW", 1)]), bigger).right > wide.right


LEGENDS = """
[[legends]]
subtype = "B"
text_size = 20.41
row_pitch = 28.6
text_left = 34
padding = 9.32
"""


def test_a_subtype_sets_its_own_legend(tmp_path: Path) -> None:
    config, _ = load_maps_config(write(tmp_path, GOOD + LEGENDS))
    assert config.legend("B", "HI") == LegendConfig("B", 20.41, 28.6, 34.0, 9.32)
    assert config.legend("A(H3N2)", "HI") is None  # the default legend


def test_a_legend_rule_names_every_size(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="missing padding"):
        load_maps_config(write(tmp_path, GOOD + LEGENDS.replace("padding = 9.32\n", "")))
    with pytest.raises(ConfigError, match="unknown key"):
        load_maps_config(write(tmp_path, GOOD + LEGENDS + "font = 1\n"))
