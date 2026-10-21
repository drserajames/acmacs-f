"""af.geo.build's figures are report figures: af.report.build takes them as they are written."""

from __future__ import annotations

import datetime as dt
import json
import types
from pathlib import Path
from typing import Any

import pytest

from af.geo import build as geo_build
from af.map.colouring import StoreColours
from af.map.style import ColourRow, ColourScheme
from af.report import build as report_build
from af.report.config import load
from af.report.provenance import ProvenanceError
from af.serology.outputs import OutputsReport
from af.store import Provenance, Store, StoreRef, read_manifest
from af.util.artefacts import sha256_path
from af.util.config import ConfigError
from af.util.fonts import load_fonts

from ..util.test_fonts import bundled, fonts_toml  # matplotlib's own fonts stand in for af's
from .test_render import _coastline as coastline_file  # the renderer tests' invented shapefile

T0 = dt.datetime(2026, 9, 1, 12, tzinfo=dt.UTC)
CONFIG = """
[report]
id = "geo-report"
kind = "monthly"
title = "Test"
centre = "Lab X"
period = { first = "2021-01", last = "2021-02" }
data_cutoff = 2021-02-28

[figures]
allow_placeholders = false

[[sections]]
kind = "geo"
title = "Geo"
slots = ["geo/h3"]
grid = [1, 2]
"""


def _publish(
    store: Store, kind: str, dataset: str, inputs: tuple[StoreRef, ...] = (), text: str = "v1"
) -> StoreRef:
    with store.build(kind, dataset) as version:
        (version.path / "data.txt").write_text(f"{kind} {dataset} {text}")
        return version.publish(Provenance("test", inputs, {}, T0, T0))


def _doc() -> dict[str, Any]:
    point = {"color": "#aa0000", "count": 1, "clade": "Clade R"}
    return {
        "subtype": "A(H3N2)",
        "periods": [
            {"period": "2021-01", "title": "January 2021",
             "locations": [{"name": "ALPHA", "points": [point]}], "unpainted": []},
            {"period": "2021-02", "title": "February 2021", "locations": [], "unpainted": []},
        ],
        "undated": 0,
        "no_location": [],
    }  # fmt: skip


H3_LOOK = """
[geo.h3]
packing = "ae-rings"
coloured_edge = "none"
coloured_edge_width = 0
uncoloured_edge_width = 0.5
clade_not_defined_fill = "#ffffff"
clade_not_defined_edge = "#606060"
clade_not_defined_edge_width = 0.6
"""


def _style(tmp_path: Path, tables: str = H3_LOOK, fonts: str | None = None) -> Path:
    """The round's figure-style file: its fonts and the geo looks (an empty [geo] for no
    slots)."""
    path = tmp_path / "figures.toml"
    path.write_text((fonts_toml() if fonts is None else fonts) + (tables or "[geo]\n"))
    return path


def _geo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cutoff: dt.date | None = None,
    allow_empty: tuple[str, ...] = (),
    doc: dict[str, Any] | None = None,
    seen: dict[str, Any] | None = None,
) -> tuple[Store, list[StoreRef], Path]:
    """Run build_geo over a test store, its reads stood in for: serology, h3 sequences and the
    h3 clades labelled from them, as StoreColours' join would have read them."""
    store = Store.create(tmp_path / "store")
    serology = _publish(store, "serology", "all")
    sequences = _publish(store, "sequences", "h3")
    clades = _publish(store, "clades", "h3", (sequences,))
    colours = object.__new__(StoreColours)  # the join's reads only: no clones or user tables
    colours.serology = serology
    colours.store_read = {}
    colours.links = types.SimpleNamespace(  # type: ignore[assignment]  # the refs only
        refs={"sequences": {"h3": sequences.to_json()}, "clades": [clades.to_json()]}
    )
    scheme = ColourScheme("h3 test", (ColourRow("Clade R", "#aa0000", frozenset({"R"})),))
    report = OutputsReport(
        serology=serology, geo_docs={"A(H3N2)": doc or _doc()}, coordinates=lambda _: (10.0, 20.0),
        colour_schemes={"h3": scheme},
        colour_provenance={"h3": {"datasets": ["h3"], "source": "store", "scheme": "h3 test",
                                  "user_tables": {"tables.py": "ab" * 32}, "coloured": {}}},
    )  # fmt: skip
    monkeypatch.setattr(geo_build, "StoreColours", lambda *args, **kwargs: colours)
    monkeypatch.setattr(geo_build, "matching_rules", lambda _: None)

    def make(*args: Any, **kwargs: Any) -> OutputsReport:
        if seen is not None:
            seen.update(kwargs)
        return report

    monkeypatch.setattr(geo_build, "make_geo_and_stat", make)
    monkeypatch.setattr(geo_build, "load_fonts", lambda path: load_fonts(path, bundled))
    figures = tmp_path / "figures"
    geo_build.build_geo(
        store, tmp_path, tmp_path, tmp_path, coastline_file(tmp_path),
        geo_build.Month(2021, 1), geo_build.Month(2021, 2), figures, tmp_path / "work",
        {"h3": "h3 test"}, {"A(H3N2)": "h3"}, _style(tmp_path), created=T0, cutoff=cutoff,
        allow_empty=allow_empty,
    )  # fmt: skip
    return store, [serology, sequences, clades], figures


def test_geo_figures_carry_created_and_the_store_versions_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """11-reports, 11 Oct: every report figure needs provenance.created (time-zoned) and
    provenance.store_refs (StoreRef JSON); without them the report refused every geo month."""
    _, refs, figures = _geo(tmp_path, monkeypatch)
    for month in ("2021-01", "2021-02"):
        i7 = json.loads((figures / "geo" / "h3" / month / "af" / "figure.i7.json").read_text())
        assert i7["provenance"]["created"] == T0.isoformat()
        # drawn with its slot's look from the figure-style file, both recorded (rule 5)
        assert i7["provenance"]["look"]["coloured_edge"] == "none"
        assert i7["provenance"]["look"]["uncoloured_edge_width"] == 0.5
        assert i7["provenance"]["figure_config"]["sha256"] == sha256_path(tmp_path / "figures.toml")
        assert i7["provenance"]["store_refs"] == [r.to_json() for r in refs]
        # the slot's own facts, for its how-made note (af.report.howmade.geo_note)
        assert i7["provenance"]["table_subtype"] == "A(H3N2)"
        assert i7["provenance"]["colour_source"] == {
            "h3": {"source": "store", "scheme": "h3 test", "user_tables": {"tables.py": "ab" * 32}}
        }
        assert i7["provenance"]["undated"] == 0 and i7["provenance"]["no_location"] == 0
        fonts = i7["provenance"]["fonts"]  # the font files it was drawn in, hashed (rule 5)
        assert fonts == load_fonts(tmp_path / "figures.toml", bundled).inputs()
        assert Path(fonts["font_sans"]["path"]).name == "STIXGeneral.ttf"
    with pytest.raises(ValueError, match="time zone"):
        geo_build.build_geo(
            Store.open(tmp_path / "store"), tmp_path, tmp_path, tmp_path, tmp_path,
            geo_build.Month(2021, 1), geo_build.Month(2021, 1), figures, tmp_path / "work",
            {}, {}, _style(tmp_path, ""), created=dt.datetime(2026, 9, 1),  # noqa: DTZ001
        )  # fmt: skip


def test_a_cut_off_inside_the_last_month_is_in_its_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q243: the window's last month was fetched on the 10th, the month before is whole; a
    cut-off before the last month is refused before the store is read."""
    _, _, figures = _geo(tmp_path, monkeypatch, cutoff=dt.date(2021, 2, 10))
    titles = {}
    for month in ("2021-01", "2021-02"):
        i7 = json.loads((figures / "geo" / "h3" / month / "af" / "figure.i7.json").read_text())
        titles[month] = i7["title"]
        assert i7["provenance"]["cutoff"] == "2021-02-10"
    assert titles == {"2021-01": "h3 January 2021", "2021-02": "h3 February 2021 (to 10 February)"}
    with pytest.raises(ValueError, match="cut-off 2021-01-20 is before the window's last month"):
        geo_build.build_geo(
            Store.open(tmp_path / "store"), tmp_path, tmp_path, tmp_path, tmp_path,
            geo_build.Month(2021, 1), geo_build.Month(2021, 2), tmp_path / "unwritten",
            tmp_path / "work", {}, {}, _style(tmp_path), created=T0, cutoff=dt.date(2021, 1, 20),
        )  # fmt: skip
    assert not (tmp_path / "unwritten").exists()


def test_a_slot_allowed_empty_is_drawn_blank_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A monthly report's month can have no viruses of a subtype: a slot the round allows empty
    is still drawn (blank maps) and its record says so; an unknown slot name is refused."""
    empty = _doc()
    empty["periods"][0]["locations"] = []
    seen: dict[str, Any] = {}
    _, _, figures = _geo(tmp_path, monkeypatch, allow_empty=("h3",), doc=empty, seen=seen)
    assert seen["empty_subtypes"] == ["A(H3N2)"]  # the outputs make its document anyway
    record = json.loads((figures / "geo" / "af-build.json").read_text())
    assert record["slots"]["h3"]["empty"] is True
    assert record["slots"]["h3"]["drawn"] == {"2021-01": 0, "2021-02": 0}
    i7 = json.loads((figures / "geo" / "h3" / "2021-01" / "af" / "figure.i7.json").read_text())
    assert i7["provenance"]["allowed_empty"] is True
    with pytest.raises(ValueError, match=r"allow_empty names slots that are not drawn: \['bvic'\]"):
        geo_build.build_geo(
            Store.open(tmp_path / "store"), tmp_path, tmp_path, tmp_path, tmp_path,
            geo_build.Month(2021, 1), geo_build.Month(2021, 1), tmp_path / "unwritten",
            tmp_path / "work", {}, {"A(H3N2)": "h3"}, _style(tmp_path), created=T0,
            allow_empty=("bvic",),
        )  # fmt: skip


def test_geo_figures_are_drawn_in_the_rounds_fonts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Style step 3 for geo: the month title is set in the round's [fonts] sans, not
    matplotlib's DejaVu fallback, and a figure-style file without [fonts] is refused before
    the store is read."""
    _, _, figures = _geo(tmp_path, monkeypatch)
    pdf = (figures / "geo" / "h3" / "2021-01" / "af" / "figure.pdf").read_bytes()
    assert b"STIXGeneral" in pdf and b"DejaVu" not in pdf
    store = Store.open(tmp_path / "store")
    with pytest.raises(ConfigError, match="fonts: required table missing"):
        geo_build.build_geo(
            store, tmp_path, tmp_path, tmp_path, tmp_path,
            geo_build.Month(2021, 1), geo_build.Month(2021, 1), tmp_path / "unwritten",
            tmp_path / "work", {}, {}, _style(tmp_path, "", fonts=""), created=T0,
        )  # fmt: skip
    assert not (tmp_path / "unwritten").exists()


def test_the_report_checks_a_geo_slot_against_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report resolves each geo month and runs its provenance checks on the refs; a store
    version moving on after the figures were drawn makes them stale, as for a map."""
    store, refs, figures = _geo(tmp_path, monkeypatch)
    cfg_path = tmp_path / "r.toml"
    cfg_path.write_text(CONFIG)
    cfg = load(cfg_path)
    manifest = tmp_path / "manifest.json"
    figs = report_build.resolve_all(cfg, figures)
    assert sorted(figs) == ["geo/h3/2021-01", "geo/h3/2021-02"]
    use = report_build.check_provenance(cfg, figs, store.root, manifest, deep=False)
    assert sorted(use.refs, key=lambda r: r.kind) == sorted(refs, key=lambda r: r.kind)
    assert use.used_by["sequences", "h3"] == ["geo/h3/2021-01", "geo/h3/2021-02"]
    _publish(store, "serology", "all", text="v2")  # serology moves on: the figures are stale
    with pytest.raises(ProvenanceError, match="stale: geo/h3/2021-01, geo/h3/2021-02 rest on "
                       "serology/all"):  # fmt: skip
        report_build.check_provenance(cfg, figs, store.root, manifest, deep=False)


@pytest.mark.tool(report_build.LATEX)
def test_a_report_builds_from_geo_figures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, refs, figures = _geo(tmp_path, monkeypatch)
    cfg_path = tmp_path / "r.toml"
    cfg_path.write_text(CONFIG)
    manifest = tmp_path / "manifest.json"
    pdf = report_build.build(
        cfg_path, figures, tmp_path / "out", store_root=store.root, manifest_path=manifest
    )
    assert pdf.is_file()
    assert sorted(read_manifest(manifest), key=lambda r: r.kind) == sorted(
        refs, key=lambda r: r.kind
    )
