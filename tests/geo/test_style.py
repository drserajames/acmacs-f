"""Per-slot geo looks from the round's figure-style file (style step 10)."""

from pathlib import Path

import pytest

from af.geo.placement import LATTICE, PILES, Placement
from af.geo.render import Look
from af.geo.style import GeoStyleError, look_record, read_geo_looks, read_geo_placements

SHIPPED = """
[fonts]  # 08-maps' table: not geo's to read
sans = "x.otf"

[geo.h3]
packing = "ae-rings"
coloured_edge = "none"
coloured_edge_width = 0
uncoloured_edge_width = 0.5
clade_not_defined_fill = "#ffffff"
clade_not_defined_edge = "#606060"
clade_not_defined_edge_width = 0.6

[geo.bvic]
packing = "honeycomb"
coloured_edge = "#808080"
coloured_edge_width = 1.0
uncoloured_edge_width = 0.3
clade_not_defined_fill = "none"
clade_not_defined_edge = "#404040"
clade_not_defined_edge_width = 0.8
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
    # Q215: a sequenced dot whose clade is not defined has its own look, per slot
    assert (h3.clade_not_defined_fill, h3.clade_not_defined_edge) == ("#ffffff", "#606060")
    assert (bvic.clade_not_defined_fill, bvic.clade_not_defined_edge_width) == ("none", 0.8)


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
        (
            ('clade_not_defined_fill = "#ffffff"', 'clade_not_defined_fill = "white"'),
            {"h3", "bvic"},
            "clade_not_defined_fill 'white'",
        ),
        (
            ('clade_not_defined_edge = "#606060"', 'clade_not_defined_edge = "none"'),
            {"h3", "bvic"},
            "clade_not_defined_edge 'none'",
        ),
        (  # the two kinds of uncoloured dot are keyed apart, so they must not draw alike
            (
                'clade_not_defined_fill = "#ffffff"\nclade_not_defined_edge = "#606060"\n'
                "clade_not_defined_edge_width = 0.6",
                'clade_not_defined_fill = "none"\nclade_not_defined_edge = "#A0A0A0"\n'
                "clade_not_defined_edge_width = 0.5",
            ),
            {"h3", "bvic"},
            "would draw as a not-sequenced one",
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


LATTICE_KEYS = """placement = "lattice"
lattice_pitch = 6.4
lattice_order = "nearest-offer-first"
lattice_reserve_point = true
"""


def _with(text: str, extra: str, slot: str = "h3") -> str:
    """``text`` with ``extra`` lines added to ``[geo.<slot>]``."""
    return text.replace(f"[geo.{slot}]\n", f"[geo.{slot}]\n{extra}", 1)


def test_a_slot_naming_no_placement_is_drawn_as_before(tmp_path: Path) -> None:
    path = _file(tmp_path, SHIPPED)
    assert read_geo_placements(path, {"h3", "bvic"}) == {"h3": None, "bvic": None}
    # and the look is untouched by the placement keys, so the figure's look record is too
    plain = read_geo_looks(path, {"h3", "bvic"})
    named = read_geo_looks(_file(tmp_path, _with(SHIPPED, LATTICE_KEYS)), {"h3", "bvic"})
    assert plain == named


def test_the_lattice_and_its_parameters_come_from_config(tmp_path: Path) -> None:
    text = _with(_with(SHIPPED, LATTICE_KEYS), "lattice_max_move = 20\n" + LATTICE_KEYS, "bvic")
    got = read_geo_placements(_file(tmp_path, text), {"h3", "bvic"})
    assert got["h3"] == Placement(LATTICE, 6.4, "nearest-offer-first", True, None)
    assert got["bvic"] is not None and got["bvic"].max_move == 20.0
    piles = read_geo_placements(
        _file(tmp_path, _with(SHIPPED, 'placement = "piles"\n')), {"h3", "bvic"}
    )
    assert piles["h3"] == Placement(PILES) and piles["h3"].to_json() == {"rule": "piles"}


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ('placement = "scatter"\n', "placement 'scatter'"),
        ('placement = "piles"\nlattice_pitch = 6.4\n', "given with placement"),
        ("lattice_pitch = 6.4\n", "given without placement"),
        ('placement = "lattice"\n', "needs"),
        (LATTICE_KEYS.replace("6.4", "0"), "lattice_pitch 0"),
        (LATTICE_KEYS.replace("nearest-offer-first", "largest-first"), "lattice_order"),
        (LATTICE_KEYS.replace("true", '"yes"'), "lattice_reserve_point"),
        (LATTICE_KEYS + "lattice_max_move = -1\n", "lattice_max_move -1"),
        (LATTICE_KEYS + "lattice_spacing = 2\n", "unknown keys"),
    ],
)
def test_a_wrong_placement_is_an_error(tmp_path: Path, extra: str, message: str) -> None:
    path = _file(tmp_path, _with(SHIPPED, extra))
    with pytest.raises(GeoStyleError, match=message):
        read_geo_placements(path, {"h3", "bvic"})
        read_geo_looks(path, {"h3", "bvic"})
