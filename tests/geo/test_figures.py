"""Report geo figures: one figure.{pdf,i7.json} per month, and the clade key as data."""

import json
from pathlib import Path

import pytest

from af.geo.figures import clade_key, write_geo_figures
from af.map.style import ColourRow, ColourScheme
from af.report.i7 import validate

from .test_render import _coastline as coastline_file  # the renderer tests' invented shapefile


def _doc() -> dict:
    return {
        "subtype": "A(H3N2)",
        "periods": [
            {
                "period": "2021-01",
                "title": "January 2021",
                "locations": [
                    {
                        "name": "ALPHA",
                        "points": [
                            {"color": "#aa0000", "count": 2, "clade": "Clade R"},
                            {"color": "transparent", "count": 1},
                        ],
                    }
                ],
            },
            {"period": "2021-02", "title": "February 2021", "locations": []},
        ],
        "undated": 0,
        "no_location": [],
    }


def test_every_month_is_its_own_report_figure(tmp_path: Path) -> None:
    coastline = coastline_file(tmp_path)
    written = write_geo_figures(
        _doc(), "h3", lambda name: (10.0, 20.0), coastline, tmp_path / "figures",
        {"serology": "v-test"},
    )  # fmt: skip
    assert written.drawn == {"2021-01": 3, "2021-02": 0}
    for month in ("2021-01", "2021-02"):
        folder = tmp_path / "figures" / "geo" / "h3" / month / "af"
        i7 = json.loads((folder / "figure.i7.json").read_text())
        validate(i7)
        assert i7["kind"] == "geo" and i7["geo"]["month"] == month
        assert i7["provenance"]["serology"] == "v-test"
        assert (folder / "figure.pdf").stat().st_size > 0


def test_the_clade_key_lists_every_scheme_entry_with_its_dots() -> None:
    scheme = ColourScheme(
        "h3 test",
        (
            ColourRow("Clade R", "#aa0000", frozenset({"R"})),
            ColourRow("Clade S", "#00aa00", frozenset({"S"})),
        ),
    )
    key = clade_key(_doc(), [scheme])
    assert key["months"] == ["2021-01", "2021-02"]
    assert [(e["legend"], e["window"]) for e in key["entries"]] == [("Clade R", 2), ("Clade S", 0)]
    assert key["entries"][0]["per_month"] == {"2021-01": 2}
    assert key["uncoloured"] == {"per_month": {"2021-01": 1}, "window": 1}
    with pytest.raises(ValueError, match="not in the scheme"):
        clade_key(_doc(), [ColourScheme("h3 empty", ())])
