"""Geo and stat from the stores, end to end, with every place and virus invented."""

import datetime
import json
from pathlib import Path
from typing import Any

import duckdb
import shapefile  # pyshp, in the geo extra

from af.clades.colours import ColourEntry, ColourScheme
from af.geo.records import Month
from af.seq.matching_rules import matching_rules
from af.serology.outputs import SubtypeColouring, make_geo_and_stat
from af.serology.update import update
from af.store import Provenance, Store
from af.tables.identity import Manifest
from af.tables.store import publish
from tests.clades.synthetic import build_clone, load_synthetic
from tests.seq.test_locations import COUNTRIES, PLACE_COLUMNS, PLACES, REGIONS
from tests.seq.test_matching_rules import passage_matcher, write_af_data

NOW = datetime.datetime(2026, 9, 25, tzinfo=datetime.UTC)
H3 = "A(H3N2)"


def _provenance(step: str) -> Provenance:
    return Provenance(step=step, inputs=(), parameters={}, started=NOW, finished=NOW)


def _name(prefix: str, number: int) -> str:
    return "/".join([prefix, "EXAMPLETOWN", str(number), "2021"])


def _sequences(store: Store, tmp_path: Path, extra: tuple[tuple[int, int], ...] = ()) -> None:
    """Each sequences dataset gets its own two isolates in EXAMPLETOWN (EXAMPLELAND); only
    h3's are the ones the test's antigens match (an isolate is in one dataset only).
    ``extra`` adds h3 isolates as (EPI number, name number): two under one name are a tie."""
    for offset, dataset in enumerate(("h3", "h1", "bvic", "byam")):
        numbers = [(1 + 10 * offset,) * 2, (2 + 10 * offset,) * 2]
        if dataset == "h3":
            numbers += list(extra)
        isolates = tmp_path / f"isolates-{dataset}.parquet"
        sequences = tmp_path / f"sequences-{dataset}.parquet"
        rows = ", ".join(
            f"('EPI_ISL_{n}', 'ACC{n}', '{_name('A', m)}', 'SIAT1', 'Exampleland', "
            f"'Example Continent', 'EXAMPLETOWN', '2021-01-01', []::VARCHAR[])"
            for n, m in numbers
        )
        duckdb.execute(
            f"COPY (SELECT * FROM (VALUES {rows}) AS v(epi_isl, accession, name, passage, "
            f"country, region, place, collection_date, problems)) "
            f"TO '{isolates.as_posix()}' (FORMAT parquet)"
        )
        seqs = ", ".join(f"('EPI_ISL_{n}', 'ACC{n}', 'hash{n}', '{'K' * 40}')" for n, _ in numbers)
        duckdb.execute(
            f"COPY (SELECT * FROM (VALUES {seqs}) AS v(epi_isl, accession, seq_hash, aa_aligned)) "
            f"TO '{sequences.as_posix()}' (FORMAT parquet)"
        )
        with store.build("sequences", dataset) as builder:
            builder.copy(isolates, "isolates/pull=test/part-0.parquet")
            builder.copy(sequences, "sequences/pull=test/part-0.parquet")
            builder.publish(_provenance("sequences-test"))


def _clades(store: Store, tmp_path: Path, extra: tuple[tuple[int, str], ...] = ()) -> None:
    source = tmp_path / "assignments.parquet"
    rows = ", ".join(
        f"('EPI_ISL_{n}', 'ACC{n}', '{c}', 'fallback')" for n, c in ((1, "P.1"), *extra)
    )
    duckdb.execute(
        f"""COPY (SELECT * FROM (VALUES {rows})
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
    store = Store.create(tmp_path / "store")
    serum = {"name": syn.virus("Elsewhere", 9), "serum_id": "S-1"}
    here = {"name": syn.virus("EXAMPLETOWN", 1), "passage": "MDCK1", "date": "2021-01-05",
            "epi_isl": "EPI_ISL_1", "sequence_pairing": "exact"}  # fmt: skip
    lost = {"name": syn.virus("NEVERTOWN", 2), "passage": "MDCK1", "date": "2021-01-06"}
    tables = [syn.table("h3-hi-labx-20210304", [here, lost], [serum], [[["80"]], [["40"]]])]
    publish(store, tables, Manifest.from_tables(tables, inputs=[]), _provenance("tables-test"))
    update(store, syn.rules)
    _sequences(store, tmp_path)
    location_dir = tmp_path / "locations"
    location_dir.mkdir()
    (location_dir / "countries.tsv").write_text(COUNTRIES)
    (location_dir / "regions.tsv").write_text(REGIONS)
    rows = [f"{loc}\t{c}\t\t{lat}\t{lon}\tcity\thand\tinvented\t\t\tt\t2026-01-01"
            for loc, c, lat, lon in PLACES]  # fmt: skip
    (location_dir / "places.tsv").write_text("\n".join([PLACE_COLUMNS, *rows]) + "\n")
    return store, location_dir, _coastline(tmp_path)


def test_geo_and_stat_from_the_stores(tmp_path: Path, syn: Any) -> None:
    store, tables, coastline = _roots(tmp_path, syn)
    out = tmp_path / "out"
    report = make_geo_and_stat(
        store, tables, coastline, Month(2021, 1), Month(2021, 1), out, identity_rules=syn.rules
    )
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
    assert regions == {"all": 2, "Example Continent": 1, "UNKNOWN": 1}
    # the location tables a map was drawn from are recorded by content hash
    assert sorted(report.location_tables) == ["countries.tsv", "places.tsv", "regions.tsv"]
    assert all(len(h) == 64 for h in report.location_tables.values())
    assert report.stat_unknown_region == {"NEVERTOWN": 1}
    assert report.serology == store.current("serology", "all")
    assert report.links is None and report.colours == {}  # no colouring asked for


def test_geo_colours_from_clade_store_and_scheme(tmp_path: Path, syn: Any) -> None:
    store, tables, coastline = _roots(tmp_path, syn)
    _clades(store, tmp_path)
    rules = matching_rules(
        write_af_data(tmp_path / "af-data", submitters="", number="", equivalents="")
    )
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
        store, tables, coastline, Month(2021, 1), Month(2021, 1), out,
        colouring={H3: SubtypeColouring(scheme, clade_set)}, matching=rules,
        identity_rules=syn.rules,
    )  # fmt: skip
    assert report.links is not None and report.links.by_status["matched"] == 1
    # the matcher's tables are in the report by content hash, as the colour tables are
    assert report.matching_inputs == rules.provenance() and len(report.matching_inputs) == 6
    assert report.matching_rules == rules.counts()
    assert report.colours[H3].coloured == {"Clade P.1": 1}
    assert report.colours[H3].uncoloured == {"no sequence": 1}
    doc = json.loads((out / "geo" / "h3-records.json").read_text())
    points = doc["periods"][0]["locations"][0]["points"]
    assert points == [{"color": "#0000aa", "count": 1, "clade": "Clade P.1"}]


def test_clade_tables_behind_the_sequence_store_are_reported(tmp_path: Path, syn: Any) -> None:
    from af.seq.matching import read_passage_rules
    from af.seq.matching_rules import MatchingRules
    from af.serology import query
    from af.serology.joins import link_from_store

    store, _, _ = _roots(tmp_path, syn)
    labelled = store.current("sequences", "h3")
    source = tmp_path / "assignments.parquet"
    duckdb.execute(
        f"""COPY (SELECT * FROM (VALUES ('EPI_ISL_1', 'ACC1', 'P.1', 'fallback'))
            AS v(epi_isl, accession, clade, method)) TO '{source.as_posix()}' (FORMAT parquet)"""
    )
    with store.build("clades", "h3") as builder:
        builder.copy(source, "assignments.parquet")
        builder.publish(
            Provenance(step="clades-test", inputs=(labelled,), parameters={}, started=NOW,
                       finished=NOW)
        )  # fmt: skip
    passages = tmp_path / "passage_classes.tsv"
    passages.write_text("pattern\tclass\treason\nSIAT\tcell\ttest\n")
    rules = MatchingRules(
        tuple(read_passage_rules(passages)), frozenset({"LABX"}), {}, {}, (), (),
        passage_matcher(tmp_path),
    )  # fmt: skip
    con = query.connect(store.resolve(store.current("serology", "all")))
    counts = link_from_store(con, store, rules, with_clades=True)
    assert counts.clades_behind == {}  # labels the current sequences version

    newer = tmp_path / "newer.parquet"
    duckdb.execute(f"COPY (SELECT 1 AS x) TO '{newer.as_posix()}' (FORMAT parquet)")
    with store.build("sequences", "h3") as builder:
        builder.link(
            store.resolve(labelled) / "isolates/pull=test/part-0.parquet",
            "isolates/pull=test/part-0.parquet",
        )
        builder.link(
            store.resolve(labelled) / "sequences/pull=test/part-0.parquet",
            "sequences/pull=test/part-0.parquet",
        )
        builder.copy(newer, "isolates/pull=more/extra.txt")
        builder.publish(_provenance("sequences-test"))  # fmt: skip
    con = query.connect(store.resolve(store.current("serology", "all")))
    counts = link_from_store(con, store, rules, with_clades=True)
    current = store.current("sequences", "h3").version
    assert current != labelled.version
    assert counts.clades_behind == {"h3": (labelled.version, current)}


def test_rule_tables_reach_the_matcher(tmp_path: Path, syn: Any) -> None:
    """make_geo_and_stat must pass its rule tables on, not accept and drop them: a location
    equivalent whose GISAID spelling no stored sequence has is refused by the join, so it can
    only be refused from here if the rules got there."""
    import pytest

    store, tables, coastline = _roots(tmp_path, syn)
    _clades(store, tmp_path)
    rules = matching_rules(
        write_af_data(
            tmp_path / "af-data", submitters="", number="",
            equivalents="LABX\tNEVERTOWN\tGHOSTTOWN\t\thand\ttest\ttest\t2026-01-01\t\n",
        )
    )  # fmt: skip
    scheme = ColourScheme(subtype=H3, name="test", entries=())
    clade_set = load_synthetic(build_clone(tmp_path / "clone").parent)
    with pytest.raises(ValueError, match="no stored sequence has"):
        make_geo_and_stat(
            store, tables, coastline, Month(2021, 1), Month(2021, 1), tmp_path / "out",
            colouring={H3: SubtypeColouring(scheme, clade_set)}, matching=rules,
            identity_rules=syn.rules,
        )  # fmt: skip


def test_colouring_without_matching_rules_is_refused(tmp_path: Path, syn: Any) -> None:
    import pytest

    store, tables, coastline = _roots(tmp_path, syn)
    scheme = ColourScheme(subtype=H3, name="test", entries=())
    clade_set = load_synthetic(build_clone(tmp_path / "clone").parent)
    with pytest.raises(ValueError, match="colouring needs matching rules"):
        make_geo_and_stat(
            store, tables, coastline, Month(2021, 1), Month(2021, 1), tmp_path / "out",
            colouring={H3: SubtypeColouring(scheme, clade_set)}, identity_rules=syn.rules,
        )  # fmt: skip


def test_geo_and_stat_refuse_a_serology_store_behind_the_tables(tmp_path: Path, syn: Any) -> None:
    """A lab's tables published after serology/all was built would silently be missing from
    every figure; the build refuses and names them instead."""
    import pytest

    from af.store import StoreError

    store, tables, coastline = _roots(tmp_path, syn)
    serum = {"name": syn.virus("Elsewhere", 9), "serum_id": "S-2"}
    late = {"name": syn.virus("EXAMPLETOWN", 3), "passage": "MDCK1", "date": "2021-01-07"}
    later = [syn.table("h3-hi-laby-20210305", [late], [serum], [[["80"]]], lab="LABY",
                       group="h3-hi-laby")]  # fmt: skip
    publish(store, later, Manifest.from_tables(later, inputs=[]), _provenance("tables-test"))
    with pytest.raises(StoreError, match=r"run af.serology.update.*laby/h3-hi-laby: new since"):
        make_geo_and_stat(
            store, tables, coastline, Month(2021, 1), Month(2021, 1), tmp_path / "out",
            identity_rules=syn.rules,
        )  # fmt: skip


def test_a_refused_tie_is_coloured_from_its_candidates_own_sequences(
    tmp_path: Path, syn: Any
) -> None:
    """End to end (Q81): EXAMPLETOWN/7 has two different stored sequences, both P.1. The match
    stays refused, but the dot is coloured, from the candidates' aligned sequences."""
    store = Store.create(tmp_path / "store")
    serum = {"name": syn.virus("Elsewhere", 9), "serum_id": "S-1"}
    tied = {"name": syn.virus("EXAMPLETOWN", 7), "passage": "MDCK1", "date": "2021-01-05"}
    tables = [syn.table("h3-hi-labx-20210304", [tied], [serum], [[["80"]]])]
    publish(store, tables, Manifest.from_tables(tables, inputs=[]), _provenance("tables-test"))
    update(store, syn.rules)
    _sequences(store, tmp_path, extra=((7, 7), (8, 7)))
    _clades(store, tmp_path, extra=((7, "P.1"), (8, "P.1")))
    _, locations, coastline = _roots(tmp_path / "unused", syn)
    scheme = ColourScheme(
        subtype=H3, name="test",
        entries=(ColourEntry(order=1, key="P.1", legend="Clade P.1", colour="#0000aa",
                             is_group=False),),
    )  # fmt: skip
    clade_set = load_synthetic(build_clone(tmp_path / "clone").parent)
    rules = matching_rules(
        write_af_data(tmp_path / "af-data", submitters="", number="", equivalents="")
    )
    report = make_geo_and_stat(
        store, locations, coastline, Month(2021, 1), Month(2021, 1), tmp_path / "out",
        colouring={H3: SubtypeColouring(scheme, clade_set)}, matching=rules,
        identity_rules=syn.rules,
    )  # fmt: skip
    assert report.links is not None and report.links.by_flag["match.ambiguous"] == 1
    assert report.colours[H3].coloured == {"Clade P.1": 1}
    assert report.colours[H3].ties == {"match.tie-agrees": 1}


def test_a_doubtful_match_ae_uses_is_coloured_and_counted(tmp_path: Path, syn: Any) -> None:
    """End to end (Q81 D): an egg antigen whose only stored sequence is a cell isolate is
    coloured from that sequence's own alignment, and counted under its doubt."""
    store = Store.create(tmp_path / "store")
    serum = {"name": syn.virus("Elsewhere", 9), "serum_id": "S-1"}
    egg = {"name": syn.virus("EXAMPLETOWN", 2), "passage": "E3", "passage_class": "egg",
           "date": "2021-01-05"}  # fmt: skip
    tables = [syn.table("h3-hi-labx-20210304", [egg], [serum], [[["80"]]])]
    publish(store, tables, Manifest.from_tables(tables, inputs=[]), _provenance("tables-test"))
    update(store, syn.rules)
    _sequences(store, tmp_path)  # EXAMPLETOWN/2 is stored as a SIAT1 (cell) isolate only
    _clades(store, tmp_path, extra=((2, "P.1"),))
    _, locations, coastline = _roots(tmp_path / "unused", syn)
    scheme = ColourScheme(
        subtype=H3, name="test",
        entries=(ColourEntry(order=1, key="P.1", legend="Clade P.1", colour="#0000aa",
                             is_group=False),),
    )  # fmt: skip
    clade_set = load_synthetic(build_clone(tmp_path / "clone").parent)
    rules = matching_rules(
        write_af_data(tmp_path / "af-data", submitters="", number="", equivalents="")
    )
    report = make_geo_and_stat(
        store, locations, coastline, Month(2021, 1), Month(2021, 1), tmp_path / "out",
        colouring={H3: SubtypeColouring(scheme, clade_set)}, matching=rules,
        identity_rules=syn.rules,
    )  # fmt: skip
    assert report.links is not None and report.links.by_status["doubtful"] == 1
    assert report.colours[H3].coloured == {"Clade P.1": 1}
    assert report.colours[H3].doubtful == {"match.egg-antigen-non-egg-sequence": 1}
