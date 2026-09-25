"""Geo and stat from the stores, end to end, with every place and virus invented."""

import datetime
import json
import lzma
from pathlib import Path
from typing import Any

import duckdb
import shapefile  # pyshp, in the geo extra

from af.clades.colours import ColourEntry, ColourScheme
from af.geo.records import Month
from af.serology.outputs import SubtypeColouring, make_geo_and_stat
from af.serology.update import update
from af.store import Provenance, Store, Work
from af.tables.identity import Manifest
from af.tables.store import publish
from tests.clades.synthetic import build_clone, load_synthetic
from tests.seq.test_locations import LOCATIONDB

NOW = datetime.datetime(2026, 9, 25, tzinfo=datetime.UTC)
H3 = "A(H3N2)"


def _provenance(step: str) -> Provenance:
    return Provenance(step=step, inputs=(), parameters={}, started=NOW, finished=NOW)


def _name(prefix: str, number: int) -> str:
    return "/".join([prefix, "EXAMPLETOWN", str(number), "2021"])


def _sequences(store: Store, tmp_path: Path) -> None:
    """Each sequences dataset gets its own two isolates in EXAMPLETOWN (EXAMPLELAND); only
    h3's are the ones the test's antigens match (an isolate is in one dataset only)."""
    for offset, dataset in enumerate(("h3", "h1", "bvic", "byam")):
        numbers = (1 + 10 * offset, 2 + 10 * offset)
        isolates = tmp_path / f"isolates-{dataset}.parquet"
        sequences = tmp_path / f"sequences-{dataset}.parquet"
        rows = ", ".join(
            f"('EPI_ISL_{n}', 'ACC{n}', '{_name('A', n)}', 'SIAT1', 'EXAMPLELAND', "
            f"'Example Region', 'EXAMPLETOWN', '2021-01-01', []::VARCHAR[])"
            for n in numbers
        )
        duckdb.execute(
            f"COPY (SELECT * FROM (VALUES {rows}) AS v(epi_isl, accession, name, passage, "
            f"country, region, place, collection_date, problems)) "
            f"TO '{isolates.as_posix()}' (FORMAT parquet)"
        )
        seqs = ", ".join(f"('EPI_ISL_{n}', 'ACC{n}', 'hash{n}', '{'K' * 40}')" for n in numbers)
        duckdb.execute(
            f"COPY (SELECT * FROM (VALUES {seqs}) AS v(epi_isl, accession, seq_hash, aa_aligned)) "
            f"TO '{sequences.as_posix()}' (FORMAT parquet)"
        )
        with store.build("sequences", dataset) as builder:
            builder.copy(isolates, "isolates/pull=test/part-0.parquet")
            builder.copy(sequences, "sequences/pull=test/part-0.parquet")
            builder.publish(_provenance("sequences-test"))


def _clades(store: Store, tmp_path: Path) -> None:
    source = tmp_path / "assignments.parquet"
    duckdb.execute(
        f"""COPY (SELECT * FROM (VALUES ('EPI_ISL_1', 'ACC1', 'P.1', 'fallback'))
            AS v(epi_isl, accession, clade, method)) TO '{source.as_posix()}' (FORMAT parquet)"""
    )
    with store.build("clades", "h3") as builder:
        builder.copy(source, "assignments.parquet")
        builder.publish(_provenance("clades-test"))


def _coastline(tmp_path: Path) -> Path:
    path = tmp_path / "coast"
    with shapefile.Writer(str(path), shapeType=shapefile.POLYGON) as writer:
        writer.field("name", "C")
        writer.poly([[[0.0, 0.0], [0.0, 30.0], [30.0, 30.0], [30.0, 0.0], [0.0, 0.0]]])
        writer.record("square")
    return path.with_suffix(".shp")


def _roots(tmp_path: Path, syn: Any) -> tuple[Store, Path, Path]:
    """A store with one table: EXAMPLETOWN/1 (the lab gave its EPI_ISL), NEVERTOWN/2."""
    store, work = Store.create(tmp_path / "store"), Work.create(tmp_path / "work")
    serum = {"name": syn.virus("Elsewhere", 9), "serum_id": "S-1"}
    here = {"name": syn.virus("EXAMPLETOWN", 1), "passage": "MDCK1", "date": "2021-01-05",
            "epi_isl": "EPI_ISL_1", "sequence_pairing": "exact"}  # fmt: skip
    lost = {"name": syn.virus("NEVERTOWN", 2), "passage": "MDCK1", "date": "2021-01-06"}
    tables = [syn.table("h3-hi-labx-20210304", [here, lost], [serum], [[["80"]], [["40"]]])]
    publish(store, tables, Manifest.from_tables(tables, inputs=[]), _provenance("tables-test"))
    update(store, work, syn.rules)
    _sequences(store, tmp_path)
    locationdb = tmp_path / "locationdb.json.xz"
    with lzma.open(locationdb, "wt") as handle:
        json.dump(LOCATIONDB, handle)
    return store, locationdb, _coastline(tmp_path)


def test_geo_and_stat_from_the_stores(tmp_path: Path, syn: Any) -> None:
    store, locationdb, coastline = _roots(tmp_path, syn)
    out = tmp_path / "out"
    report = make_geo_and_stat(store, locationdb, coastline, Month(2021, 1), Month(2021, 1), out)
    names = sorted(p.relative_to(out).as_posix() for p in report.files)
    assert names == ["geo/h3-2021-01.pdf", "geo/h3-records.json", "stat/index.html",
                     "stat/stat.json"]  # fmt: skip
    # two dots in January: EXAMPLETOWN placed, NEVERTOWN reported, never dropped silently
    assert report.geo_drawn == {H3: {"2021-01": 1}}
    assert report.geo_unplaced == {"NEVERTOWN": 1}
    # stat groups by GISAID's region; the unknown location is counted, not guessed
    cells = json.loads((out / "stat" / "stat.json").read_text())["cells"]
    regions = {c["continent"]: c["count"] for c in cells
               if c["measure"] == "antigens" and c["subtype"] == "all" and c["lab"] == "all"
               and c["period"] == "all"}  # fmt: skip
    assert regions == {"all": 2, "Example Region": 1, "UNKNOWN": 1}
    assert report.stat_unknown_region == {"NEVERTOWN": 1}
    assert report.serology == store.current("serology", "all")
    assert report.links is None and report.colours == {}  # no colouring asked for


def test_geo_colours_from_clade_store_and_scheme(tmp_path: Path, syn: Any) -> None:
    store, locationdb, coastline = _roots(tmp_path, syn)
    _clades(store, tmp_path)
    rules = tmp_path / "passage_classes.tsv"
    rules.write_text("pattern\tclass\treason\nSIAT\tcell\ttest\nMDCK\tcell\ttest\n")
    scheme = ColourScheme(
        subtype=H3,
        name="test",
        entries=(
            ColourEntry(order=1, key="P", legend="Clade P", colour="#aa0000", is_group=False),
            ColourEntry(order=2, key="P.1", legend="Clade P.1", colour="#0000aa", is_group=False),
        ),
    )
    clade_set = load_synthetic(build_clone(tmp_path / "clone").parent)
    out = tmp_path / "out"
    report = make_geo_and_stat(
        store, locationdb, coastline, Month(2021, 1), Month(2021, 1), out,
        colouring={H3: SubtypeColouring(scheme, clade_set)}, passage_rules=rules,
    )  # fmt: skip
    assert report.links is not None and report.links.by_status["matched"] == 1
    assert report.colours[H3].coloured == {"Clade P.1": 1}
    assert report.colours[H3].uncoloured == {"no sequence": 1}
    doc = json.loads((out / "geo" / "h3-records.json").read_text())
    points = doc["periods"][0]["locations"][0]["points"]
    assert points == [{"color": "#0000aa", "count": 1, "clade": "Clade P.1"}]
