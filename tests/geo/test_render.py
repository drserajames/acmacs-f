"""Rendering geo maps from an I7 document, with an invented coastline shapefile."""

from pathlib import Path

import pytest
import shapefile  # pyshp, a cartopy dependency

from af.geo.render import GeoRenderError, packed_offsets, render_geo


def _coastline(tmp_path: Path) -> Path:
    """A one-square 'continent', written as a polygon shapefile."""
    path = tmp_path / "coast"
    with shapefile.Writer(str(path), shapeType=shapefile.POLYGON) as writer:
        writer.field("name", "C")
        writer.poly([[[-10.0, -10.0], [-10.0, 10.0], [10.0, 10.0], [10.0, -10.0], [-10.0, -10.0]]])
        writer.record("square")
    return path.with_suffix(".shp")


DOC = {
    "subtype": "A(H3N2)",
    "periods": [
        {
            "period": "2021-01",
            "title": "January 2021",
            "locations": [
                {"name": "Here", "points": [
                    {"color": "#0000aa", "count": 3, "clade": "Clade P.1"},
                    {"color": "transparent", "count": 2},
                ]},
                {"name": "Nowhere", "points": [{"color": "transparent", "count": 4}]},
            ],
        },
        {"period": "2021-02", "title": "February 2021", "locations": []},
    ],
}  # fmt: skip


def test_one_pdf_per_month_and_unplaced_dots_reported(tmp_path: Path) -> None:
    coords = {"Here": (5.0, 5.0)}
    report = render_geo(DOC, coords.get, _coastline(tmp_path), tmp_path / "out", "h3")
    assert [p.name for p in report.files] == ["h3-2021-01.pdf", "h3-2021-02.pdf"]
    assert all(p.read_bytes().startswith(b"%PDF") for p in report.files)
    assert report.drawn == {"2021-01": 5, "2021-02": 0}  # an empty month is still a map
    assert report.no_coordinates == {"Nowhere": 4}


def test_missing_coastline_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(GeoRenderError, match="not found"):
        render_geo(DOC, lambda name: None, tmp_path / "missing.shp", tmp_path / "out", "h3")


def test_packing_is_centre_then_rings() -> None:
    offsets = packed_offsets(8, 1.0)
    assert offsets[0] == (0.0, 0.0)
    assert len(offsets) == 8 and len(set(offsets)) == 8
    assert all(abs((x * x + y * y) ** 0.5 - 1.0) < 1e-9 for x, y in offsets[1:7])  # first ring
    assert abs((offsets[7][0] ** 2 + offsets[7][1] ** 2) ** 0.5 - 2.0) < 1e-9  # second ring
