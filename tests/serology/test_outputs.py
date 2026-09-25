"""Geo and stat from the stores, end to end, with every place and virus invented."""

import datetime
import json
import lzma
from pathlib import Path
from typing import Any

import duckdb
import shapefile  # pyshp, a cartopy dependency

from af.geo.records import Month
from af.serology.outputs import make_geo_and_stat
from af.serology.update import update
from af.store import Provenance, Store, Work
from af.tables.identity import Manifest
from af.tables.store import publish
from tests.seq.test_locations import LOCATIONDB

NOW = datetime.datetime(2026, 9, 25, tzinfo=datetime.UTC)


def _provenance(step: str) -> Provenance:
    return Provenance(step=step, inputs=(), parameters={}, started=NOW, finished=NOW)


def _sequences(store: Store, tmp_path: Path) -> None:
    """Each sequences dataset gets the same few isolates: EXAMPLETOWN is in EXAMPLELAND."""
    source = tmp_path / "isolates.parquet"
    duckdb.execute(
        f"""COPY (SELECT * FROM (VALUES
            ('A(H3N2)/EXAMPLETOWN/1/2021', 'EXAMPLELAND', 'Example Region', []::VARCHAR[]),
            ('A(H3N2)/EXAMPLETOWN/2/2021', 'EXAMPLELAND', 'Example Region', []::VARCHAR[])
        ) AS v(name, country, region, problems)) TO '{source.as_posix()}' (FORMAT parquet)"""
    )
    for dataset in ("h1", "h3", "bvic", "byam"):
        with store.build("sequences", dataset) as builder:
            builder.link(source, "isolates/pull=test/part-0.parquet")
            builder.publish(_provenance("sequences-test"))


def _coastline(tmp_path: Path) -> Path:
    path = tmp_path / "coast"
    with shapefile.Writer(str(path), shapeType=shapefile.POLYGON) as writer:
        writer.field("name", "C")
        writer.poly([[[0.0, 0.0], [0.0, 30.0], [30.0, 30.0], [30.0, 0.0], [0.0, 0.0]]])
        writer.record("square")
    return path.with_suffix(".shp")


def test_geo_and_stat_from_the_stores(tmp_path: Path, syn: Any) -> None:
    store, work = Store.create(tmp_path / "store"), Work.create(tmp_path / "work")
    serum = {"name": syn.virus("Elsewhere", 9), "serum_id": "S-1"}
    here = {"name": syn.virus("EXAMPLETOWN", 1), "passage": "MDCK1", "date": "2021-01-05"}
    lost = {"name": syn.virus("NEVERTOWN", 2), "passage": "MDCK1", "date": "2021-01-06"}
    tables = [syn.table("h3-hi-labx-20210304", [here, lost], [serum], [[["80"]], [["40"]]])]
    publish(store, tables, Manifest.from_tables(tables, inputs=[]), _provenance("tables-test"))
    update(store, work, syn.rules)
    _sequences(store, tmp_path)
    locationdb = tmp_path / "locationdb.json.xz"
    with lzma.open(locationdb, "wt") as handle:
        json.dump(LOCATIONDB, handle)

    out = tmp_path / "out"
    report = make_geo_and_stat(
        store, locationdb, _coastline(tmp_path), Month(2021, 1), Month(2021, 1), out
    )
    names = sorted(p.relative_to(out).as_posix() for p in report.files)
    assert names == [
        "geo/h3-2021-01.pdf",
        "geo/h3-records.json",
        "stat/index.html",
        "stat/stat.json",
    ]
    # two dots in January: EXAMPLETOWN placed, NEVERTOWN reported, never dropped silently
    assert report.geo_drawn == {"A(H3N2)": {"2021-01": 1}}
    assert report.geo_unplaced == {"NEVERTOWN": 1}
    # stat groups by GISAID's region; the unknown location is counted, not guessed
    cells = json.loads((out / "stat" / "stat.json").read_text())["cells"]
    regions = {c["continent"]: c["count"] for c in cells
               if c["measure"] == "antigens" and c["subtype"] == "all" and c["lab"] == "all"
               and c["period"] == "all"}  # fmt: skip
    assert regions == {"all": 2, "Example Region": 1, "UNKNOWN": 1}
    assert report.stat_unknown_region == {"NEVERTOWN": 1}
    assert report.serology == store.current("serology", "all")
