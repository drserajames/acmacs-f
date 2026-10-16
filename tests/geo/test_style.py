"""Per-slot geo looks from the round's figure-style file (style step 10)."""

from pathlib import Path

import pytest

from af.geo.render import Look
from af.geo.style import GeoStyleError, look_record, read_geo_looks

SHIPPED = """
[fonts]  # 08-maps' table: not geo's to read
sans = "x.otf"

[geo.h3]
packing = "ae-rings"
coloured_edge = "none"
coloured_edge_width = 0
uncoloured_edge_width = 0.5

[geo.bvic]
packing = "honeycomb"
coloured_edge = "#808080"
coloured_edge_width = 1.0
uncoloured_edge_width = 0.3
"""


def _file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "figures.toml"
    path.write_text(text)
    return path


def test_each_slot_gets_its_look(tmp_path: Path) -> None:
    looks = read_geo_looks(_file(tmp_path, SHIPPED), {"h3", "bvic"})
    h3, bvic = looks["h3"], looks["bvic"]
    assert (h3.coloured_edge, h3.coloured_edge_width, h3.uncoloured_edge_width) == (
        "none", 0.0, 0.5
    )  # fmt: skip
    assert (bvic.packing, bvic.coloured_edge, bvic.coloured_edge_width) == (
        "honeycomb", "#808080", 1.0
    )  # fmt: skip
    assert h3.dot_diameter == Look().dot_diameter  # the rest is ae's, unchanged
    record = look_record(bvic)
    assert record["uncoloured_edge_width"] == 0.3 and record["central_longitude"] == 11.762094


@pytest.mark.parametrize(
    ("change", "slots", "message"),
    [
        (("", ""), {"h3", "bvic", "h1"}, r"no \[geo.<slot>\] table for drawn slots \['h1'\]"),
        (("", ""), {"h3"}, r"tables for slots not drawn: \['bvic'\]"),
        (('packing = "ae-rings"', 'packing = "rings"'), {"h3", "bvic"}, "packing 'rings'"),
        (("uncoloured_edge_width = 0.5\n", ""), {"h3", "bvic"}, "missing keys"),
        (
            ("uncoloured_edge_width = 0.5", "uncoloured_edge_width = 0.5\noutline = 1"),
            {"h3", "bvic"},
            r"unknown keys \['outline'\]",
        ),
        (('coloured_edge = "none"', 'coloured_edge = "grey"'), {"h3", "bvic"}, "'#rrggbb'"),
        (
            ("uncoloured_edge_width = 0.5", "uncoloured_edge_width = 0"),
            {"h3", "bvic"},
            "would hide uncoloured dots",
        ),
        (
            ("coloured_edge_width = 0\n", "coloured_edge_width = 0.5\n"),
            {"h3", "bvic"},
            "coloured_edge 'none' with width 0.5",
        ),
    ],
)
def test_a_wrong_or_missing_look_is_an_error(
    tmp_path: Path, change: tuple[str, str], slots: set[str], message: str
) -> None:
    text = SHIPPED.replace(*change, 1) if change[0] else SHIPPED
    with pytest.raises(GeoStyleError, match=message):
        read_geo_looks(_file(tmp_path, text), slots)


def test_no_file_or_no_geo_tables_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(GeoStyleError, match="not found"):
        read_geo_looks(tmp_path / "missing.toml", {"h3"})
    with pytest.raises(GeoStyleError, match="no"):
        read_geo_looks(_file(tmp_path, "[fonts]\nsans = 'x'\n"), {"h3"})
