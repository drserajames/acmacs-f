"""Report geo figures: one figure.{pdf,i7.json} per month, and the clade key as data."""

import json
import re
from pathlib import Path

import pytest

from af.geo.figures import clade_key, write_geo_figures
from af.geo.render import Look
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
        assert i7["provenance"]["moved_inside_edge"] == {}  # counted every month, even at 0
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


def test_unpainted_dots_get_the_warning_in_the_figure_and_the_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each month's figure carries its unpainted dots and the one warning sentence (geo's unit
    is the virus); the key totals them over the window. A month with none carries nothing."""
    from af.geo.colours import unpainted_warning

    doc = _doc()
    doc["periods"][0]["unpainted"] = [
        {"clade": "Q.1", "tree_clades": {"Q.1.2": 1}, "points": 1, "preparations": 2}
    ]
    doc["periods"][1]["unpainted"] = []
    coastline = coastline_file(tmp_path)
    with pytest.raises(ValueError, match="no scheme"):
        write_geo_figures(doc, "h3", lambda _: (0.0, 0.0), coastline, tmp_path / "f", {})
    write_geo_figures(
        doc, "h3", lambda _: (0.0, 0.0), coastline, tmp_path / "f", {}, scheme="h3 test"
    )  # fmt: skip
    sentence = unpainted_warning(1, "Q.1", {"Q.1.2": 1}, "h3 test", unit=("virus", "viruses"))
    assert sentence.startswith("WARNING: 1 virus is not painted")
    folder = tmp_path / "f" / "geo" / "h3"
    january = json.loads((folder / "2021-01" / "af" / "figure.i7.json").read_text())
    validate(january)
    assert january["provenance"]["decisions"]["unpainted_clades"] == [
        {"clade": "Q.1", "tree_clades": {"Q.1.2": 1}, "points": 1, "preparations": 2,
         "scheme": "h3 test", "warning": sentence}
    ]  # fmt: skip
    february = json.loads((folder / "2021-02" / "af" / "figure.i7.json").read_text())
    assert "decisions" not in february["provenance"]
    assert sentence in capsys.readouterr().out
    scheme = ColourScheme("h3 test", (ColourRow("Clade R", "#aa0000", frozenset({"R"})),))
    assert clade_key(doc, [scheme])["unpainted_clades"] == [
        {"clade": "Q.1", "tree_clades": {"Q.1.2": 1}, "points": 1, "preparations": 2,
         "per_month": {"2021-01": 1}, "scheme": "h3 test", "warning": sentence}
    ]  # fmt: skip
    assert clade_key(_doc(), [scheme])["unpainted_clades"] == []


def test_dots_painted_by_placement_get_their_flag_in_the_figure_and_the_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q183 b: a month whose dots are painted by tree placement alone carries the flag's
    sentence (geo's unit is the virus) in its figure; the key totals them over the window."""
    from af.geo.colours import placement_warning

    doc = _doc()
    doc["periods"][0]["painted_by_placement"] = [
        {"tree_clades": {"R": 2}, "points": 2, "preparations": 3}
    ]
    doc["periods"][1]["painted_by_placement"] = []
    coastline = coastline_file(tmp_path)
    write_geo_figures(doc, "h3", lambda _: (0.0, 0.0), coastline, tmp_path / "f", {})
    sentence = placement_warning(2, {"R": 2}, unit=("virus", "viruses"))
    folder = tmp_path / "f" / "geo" / "h3"
    january = json.loads((folder / "2021-01" / "af" / "figure.i7.json").read_text())
    validate(january)
    assert january["provenance"]["decisions"]["painted_by_placement"] == [
        {"tree_clades": {"R": 2}, "points": 2, "preparations": 3, "warning": sentence}
    ]
    february = json.loads((folder / "2021-02" / "af" / "figure.i7.json").read_text())
    assert "decisions" not in february["provenance"]
    assert sentence in capsys.readouterr().out
    scheme = ColourScheme("h3 test", (ColourRow("Clade R", "#aa0000", frozenset({"R"})),))
    assert clade_key(doc, [scheme])["painted_by_placement"] == [
        {"tree_clades": {"R": 2}, "points": 2, "preparations": 3,
         "per_month": {"2021-01": 2}, "warning": sentence}
    ]  # fmt: skip
    assert clade_key(_doc(), [scheme])["painted_by_placement"] == []


def test_uncoloured_classes_are_drawn_and_keyed_apart(tmp_path: Path) -> None:
    """Sarah, Q215: "mark unsequenced & clade not defined on different lines". Each uncoloured
    point carries its class with the fill and ring it is drawn in, the PDF draws them so, and the
    key lists the classes in print order with the same swatches (11-reports checks one against
    the other); a supported clade the scheme lacks is listed only when drawn."""
    import matplotlib

    doc = _doc()
    doc["periods"][0]["locations"][0]["points"] = [
        {"color": "#aa0000", "count": 2, "clade": "Clade R"},
        {"color": "transparent", "count": 3, "uncoloured": "not_sequenced"},
        {"color": "transparent", "count": 1, "uncoloured": "clade_not_defined"},
    ]
    doc["periods"][0]["clade_not_defined"] = {"not_observable": 1}  # why, from the records
    look = Look(uncoloured_edge_width=0.5, clade_not_defined_edge="#404040")
    with matplotlib.rc_context({"pdf.compression": 0}):
        write_geo_figures(
            doc, "h3", lambda name: (10.0, 20.0), coastline_file(tmp_path), tmp_path / "figures",
            {}, look=look,
        )  # fmt: skip
    folder = tmp_path / "figures" / "geo" / "h3" / "2021-01" / "af"
    points = json.loads((folder / "figure.i7.json").read_text())["geo"]["locations"][0]["points"]
    assert points[1:] == [
        {"color": "transparent", "count": 3, "uncoloured": "not_sequenced",
         "edge": "#a0a0a0", "edge_width": 0.5},
        {"color": "#ffffff", "count": 1, "uncoloured": "clade_not_defined",
         "edge": "#404040", "edge_width": 0.6},
    ]  # fmt: skip
    pdf = (folder / "figure.pdf").read_bytes()
    assert re.search(rb"\s0\.25098\d* G\s", pdf)  # the #404040 ring (a grey: one value)
    assert re.search(rb"\s0\.62745\d* G\s", pdf)  # the #a0a0a0 ring
    scheme = ColourScheme("h3 test", (ColourRow("Clade R", "#aa0000", frozenset({"R"})),))
    key = clade_key(doc, [scheme], look)
    assert key["uncoloured_classes"] == [
        {"class": "not_sequenced", "fill": "transparent", "edge": "#a0a0a0", "edge_width": 0.5,
         "per_month": {"2021-01": 3}, "window": 3},
        {"class": "clade_not_defined", "fill": "#ffffff", "edge": "#404040", "edge_width": 0.6,
         "per_month": {"2021-01": 1}, "window": 1, "states": {"not_observable": 1}},
    ]  # fmt: skip
    assert key["uncoloured"]["window"] == 4  # the total, as before
    doc["periods"][0]["locations"][0]["points"].append(
        {"color": "transparent", "count": 2, "uncoloured": "clade_not_in_scheme"}
    )
    classes = clade_key(doc, [scheme], look)["uncoloured_classes"]
    assert [(c["class"], c["window"]) for c in classes] == [
        ("not_sequenced", 3), ("clade_not_defined", 1), ("clade_not_in_scheme", 2)
    ]  # fmt: skip
    assert classes[2]["fill"] == "#ffffff"  # drawn as a clade not defined, keyed apart
