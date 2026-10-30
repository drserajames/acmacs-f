"""Rendering geo maps from an I7 document, with an invented coastline shapefile."""

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import shapefile  # pyshp, a cartopy dependency

from af.geo.render import (
    GeoRenderError,
    Hiding,
    Look,
    honeycomb_offsets,
    inward_shift,
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


def test_figure_is_ae_size_and_a_cluster_at_the_seam_is_moved_inside(tmp_path: Path) -> None:
    period = {"period": "2021-01", "title": "January 2021", "locations": [
        {"name": "Edge", "points": [{"color": "#0000aa", "count": 7, "clade": "Clade P.1"}]},
        {"name": "Middle", "points": [{"color": "#0000aa", "count": 7, "clade": "Clade P.1"}]},
    ]}  # fmt: skip
    coords = {"Edge": (-169.5, 0.0), "Middle": (20.0, 0.0)}
    pdf, moved = tmp_path / "m.pdf", Counter[str]()
    got = render_period(period, coords.get, _coastline(tmp_path), pdf, moved_inside=moved)
    assert got.dots == 14
    # wrapped to 1.3 degrees inside the right edge, its ring would be cut: the whole cluster
    # moves in instead, and its dots are counted; the one in the middle stays where it is
    assert moved == {"Edge": 7}
    box = re.search(rb"/MediaBox \[ *0 +0 +([\d.]+) +([\d.]+)", pdf.read_bytes())
    assert box is not None and (float(box[1]), round(float(box[2]), 2)) == (800.0, 401.23)


def test_a_cluster_moves_only_as_far_as_the_edge_requires() -> None:
    assert inward_shift(-10.0, 10.0, 1.0, 180.0) == 0.0  # inside: stays
    assert inward_shift(170.0, 179.5, 1.0, 180.0) == pytest.approx(-0.5)  # right edge: in by 0.5
    assert inward_shift(-179.5, -170.0, 1.0, 180.0) == pytest.approx(0.5)  # left edge
    assert inward_shift(85.0, 89.8, 1.0, 90.0) == pytest.approx(-0.8)  # top edge
    assert inward_shift(170.0, 179.0, 1.0, 180.0) == 0.0  # touching the edge is not cut


def test_honeycomb_packing_is_a_look_choice(tmp_path: Path) -> None:
    """Q208 is a config line: af's earlier packing stays available, and a typo is an error."""
    offsets = honeycomb_offsets(1 + 6 + 12, 1.0)
    radii = Counter(round((x * x + y * y) ** 0.5) for x, y in offsets)
    assert radii == {0: 1, 1: 6, 2: 12}
    period = {"period": "2021-01", "title": "January 2021", "locations": []}
    with pytest.raises(GeoRenderError, match="unknown packing"):
        render_period(period, lambda n: None, _coastline(tmp_path), tmp_path / "m.pdf",
                      Look(packing="hexagons"))  # fmt: skip


def _two_places(apart: float, top: dict[str, Any] | None = None) -> dict[str, Any]:
    """Two one-dot locations ``apart`` degrees of longitude apart; the second is drawn over."""
    return {"period": "2021-01", "title": "January 2021", "locations": [
        {"name": "Aaa", "points": [{"color": "#0000aa", "count": 1, "clade": "Clade P.1"}]},
        {"name": "Bbb", "points": [top or {"color": "#aa0000", "count": 1, "clade": "Clade P.2"}]},
    ]}  # fmt: skip


def test_dots_a_neighbouring_location_covers_are_counted(tmp_path: Path) -> None:
    """Q280: a dot of one place drawn over by another place's is counted, not silently lost.
    One figure point is 0.45 degrees of longitude, so 0.1 degrees apart is all but concentric."""
    coords = {"Aaa": (0.0, 0.0), "Bbb": (0.1, 0.0)}
    got = render_period(_two_places(0.1), coords.get, _coastline(tmp_path), tmp_path / "a.pdf")
    assert got.dots == 2
    assert (got.hiding.half_covered, got.hiding.mostly_covered) == (1, 1)
    assert got.hiding.invisible_locations == 1  # the one drawn first; the other is whole
    assert got.hiding.worst == pytest.approx(0.96, abs=0.02)
    # 1.26 degrees apart: over half of the dot, and still readable under the other
    coords = {"Aaa": (0.0, 0.0), "Bbb": (1.26, 0.0)}
    got = render_period(_two_places(1.26), coords.get, _coastline(tmp_path), tmp_path / "b.pdf")
    assert (got.hiding.half_covered, got.hiding.mostly_covered) == (1, 0)
    assert got.hiding.invisible_locations == 0
    # far enough apart that the dots do not meet at all
    coords = {"Aaa": (0.0, 0.0), "Bbb": (10.0, 0.0)}
    got = render_period(_two_places(10.0), coords.get, _coastline(tmp_path), tmp_path / "c.pdf")
    assert got.hiding == Hiding(dots=2)


def test_one_location_overlapping_itself_is_not_hiding(tmp_path: Path) -> None:
    """The packing's own overlap is the look (6.4 pt step under an 8 pt dot): not a loss."""
    period = {"period": "2021-01", "title": "January 2021", "locations": [
        {"name": "Aaa", "points": [{"color": "#0000aa", "count": 19, "clade": "Clade P.1"}]},
    ]}  # fmt: skip
    got = render_period(period, lambda n: (0.0, 0.0), _coastline(tmp_path), tmp_path / "m.pdf")
    assert got.dots == 19 and got.hiding == Hiding(dots=19)


def test_a_dot_with_no_fill_hides_nothing_and_can_itself_be_hidden(tmp_path: Path) -> None:
    """An uncoloured dot is a ring: it paints no disc, so what is under it is still read."""
    ring = {"color": "transparent", "count": 1, "uncoloured": "not_sequenced"}
    coords = {"Aaa": (0.0, 0.0), "Bbb": (0.1, 0.0)}
    over = render_period(
        _two_places(0.1, top=ring), coords.get, _coastline(tmp_path), tmp_path / "r.pdf"
    )
    assert over.hiding == Hiding(dots=2)  # the ring on top covers nothing
    under = {"period": "2021-01", "title": "January 2021", "locations": [
        {"name": "Aaa", "points": [ring]},
        {"name": "Bbb", "points": [{"color": "#0000aa", "count": 1, "clade": "Clade P.1"}]},
    ]}  # fmt: skip
    got = render_period(under, coords.get, _coastline(tmp_path), tmp_path / "u.pdf")
    assert got.hiding.mostly_covered == 1 and got.hiding.invisible_locations == 1


def test_an_unknown_uncoloured_class_is_an_error(tmp_path: Path) -> None:
    doc = {"period": "2021-01", "title": "January 2021", "locations": [
        {"name": "Here", "points": [{"color": "transparent", "count": 1, "uncoloured": "grey"}]},
    ]}  # fmt: skip
    with pytest.raises(GeoRenderError, match="unknown uncoloured class 'grey'"):
        render_period(doc, lambda name: (1.0, 2.0), _coastline(tmp_path), tmp_path / "f.pdf")
