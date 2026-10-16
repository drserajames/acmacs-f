"""Rendering geo maps from an I7 document, with an invented coastline shapefile."""

import re
from collections import Counter
from pathlib import Path

import pytest
import shapefile  # pyshp, a cartopy dependency

from af.geo.render import (
    GeoRenderError,
    Look,
    honeycomb_offsets,
    packed_offsets,
    render_geo,
    render_period,
    ring_capacity,
    wrap_longitude,
)


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


def test_packing_follows_ae_rings() -> None:
    """ae's ring r holds floor(2 pi r^2) dots (6, 25, 56): the overlap that makes a disc."""
    assert [ring_capacity(r) for r in (1, 2, 3, 4)] == [6, 25, 56, 100]
    offsets = packed_offsets(1 + 6 + 25 + 2, 6.4)
    radii = Counter(round((x * x + y * y) ** 0.5 / 6.4) for x, y in offsets)
    assert radii == {0: 1, 1: 6, 2: 25, 3: 2}  # the last ring holds what is left


def test_longitudes_wrap_at_ae_seam() -> None:
    centre = Look().central_longitude  # the map spans 168.24 W to 191.76 E
    assert wrap_longitude(centre, centre) == 0.0
    assert wrap_longitude(-170.0, centre) == pytest.approx(178.237906)  # west of the seam: right
    assert wrap_longitude(-168.0, centre) == pytest.approx(-179.762094)  # just east: left


def test_figure_is_ae_size_and_a_cluster_at_the_seam_is_counted(tmp_path: Path) -> None:
    period = {"period": "2021-01", "title": "January 2021", "locations": [
        {"name": "Edge", "points": [{"color": "#0000aa", "count": 7, "clade": "Clade P.1"}]},
        {"name": "Middle", "points": [{"color": "#0000aa", "count": 7, "clade": "Clade P.1"}]},
    ]}  # fmt: skip
    coords = {"Edge": (-169.5, 0.0), "Middle": (20.0, 0.0)}
    pdf, cut = tmp_path / "m.pdf", Counter[str]()
    assert render_period(period, coords.get, _coastline(tmp_path), pdf, clipped=cut) == 14
    # wrapped to 1.3 degrees inside the right edge: its centre stays, its ring is cut, counted
    assert cut == {"Edge": 4}
    box = re.search(rb"/MediaBox \[ *0 +0 +([\d.]+) +([\d.]+)", pdf.read_bytes())
    assert box is not None and (float(box[1]), round(float(box[2]), 2)) == (800.0, 401.23)


def test_honeycomb_packing_is_a_look_choice(tmp_path: Path) -> None:
    """Q208 is a config line: af's earlier packing stays available, and a typo is an error."""
    offsets = honeycomb_offsets(1 + 6 + 12, 1.0)
    radii = Counter(round((x * x + y * y) ** 0.5) for x, y in offsets)
    assert radii == {0: 1, 1: 6, 2: 12}
    period = {"period": "2021-01", "title": "January 2021", "locations": []}
    with pytest.raises(GeoRenderError, match="unknown packing"):
        render_period(period, lambda n: None, _coastline(tmp_path), tmp_path / "m.pdf",
                      Look(packing="hexagons"))  # fmt: skip
