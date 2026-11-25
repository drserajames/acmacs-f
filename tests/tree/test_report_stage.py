"""af.tree.report_stage: the report tree derived from a published version. Synthetic only."""

from __future__ import annotations

import datetime
import json
from collections import Counter
from pathlib import Path

import pytest

from af.seq.locations import LocationTables
from af.store import Store, StoreRef
from af.tree import report_stage as R
from af.tree.populate import LeafRecord

STARTED = datetime.date(2026, 9, 21)


def record(country: str | None) -> LeafRecord:
    return LeafRecord("EPI_ISL_1", "EPI1", "A/EXAMPLETOWN/1/2024", "ACGT", country=country)


def locations(tmp_path: Path) -> LocationTables:
    head = "code\tspelling\tsource\tevidence\tadded_by\tadded_on\toptional\n"
    rows = [("XAA", "Exampleland"), ("XBB", "Otherland")]
    (tmp_path / "countries.tsv").write_text(
        head + "".join(f"{c}\t{s}\tgisaid\tinvented\ttest\t2026-10-06\t\n" for c, s in rows)
    )
    regions = "scheme\tcountry\tgroup\tevidence\n"
    regions += "continent\tXAA\tEUROPE\tinvented\ncontinent\tXBB\tnot assigned\tinvented\n"
    (tmp_path / "regions.tsv").write_text(regions)
    columns = "location country admin latitude longitude precision source evidence same_as flags"
    (tmp_path / "places.tsv").write_text("\t".join(columns.split()) + "\n")
    return LocationTables.read(tmp_path)


def test_continent_comes_from_the_tables_and_misses_are_counted(tmp_path: Path) -> None:
    continent_of, missing = R.continent_lookup(locations(tmp_path))
    assert continent_of(record("Exampleland")) == "EUROPE"
    assert continent_of(record("Otherland")) is None
    assert continent_of(record("Nowhereland")) is None
    assert continent_of(record(None)) is None
    assert missing == Counter(no_continent_for_country=1, unknown_country_spelling=1, no_country=1)


def test_the_source_must_be_a_named_version(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    with pytest.raises(R.ReportStageError, match="never CURRENT"):
        R.pinned(store, "trees", "h1/weekly")
    with pytest.raises(R.ReportStageError, match="no such version"):
        R.pinned(store, "trees", "h1/weekly@0123456789abcdef")


@pytest.mark.parametrize("purpose", ["weekly", "report-x"])
def test_a_report_tree_never_replaces_its_source(tmp_path: Path, purpose: str) -> None:
    source = StoreRef("trees", "h1/report-x", "0123456789abcdef", "0123456789abcdef" + "0" * 48)
    with pytest.raises(R.ReportStageError, match="never replaces its source"):
        R.publish(Store.create(tmp_path), None, purpose, source, {}, [], {}, STARTED)  # type: ignore[arg-type]


def write_step_record(version: Path, step: str, clock: dict[str, object] | None) -> Path:
    """A minimal populate-step record of the shape af.tree.stages writes beside each file."""
    version.mkdir(parents=True, exist_ok=True)
    record = version / "tree.json.provenance.json"
    parameters: dict[str, object] = {"purpose": "weekly"}
    if clock is not None:
        parameters["clock"] = clock
    record.write_text(json.dumps({"step": step, "parameters": parameters}))
    return record


def test_the_source_s_recorded_clock_thresholds_are_what_the_report_tree_flags_on(
    tmp_path: Path,
) -> None:
    write_step_record(
        tmp_path,
        "h1.populate",
        {"z_threshold": 3.5, "branch_threshold": 0.02, "earliest": "1980-01-01", "min_leaves": 7},
    )
    settings = R.source_clock_settings(tmp_path)
    assert settings.z_threshold == 3.5
    assert settings.branch_threshold == 0.02
    assert settings.earliest == datetime.date(1980, 1, 1)
    assert settings.min_leaves == 7


def test_a_recorded_branch_threshold_of_none_is_kept_as_none(tmp_path: Path) -> None:
    write_step_record(
        tmp_path,
        "h3.populate",
        {"z_threshold": 4.0, "branch_threshold": None, "earliest": "1977-01-01", "min_leaves": 20},
    )
    assert R.source_clock_settings(tmp_path).branch_threshold is None


def test_a_source_that_records_no_clock_settings_is_refused(tmp_path: Path) -> None:
    write_step_record(tmp_path, "h1.populate", None)
    with pytest.raises(R.ReportStageError, match="no populate step record carries clock settings"):
        R.source_clock_settings(tmp_path)


def test_a_record_of_another_step_is_not_mistaken_for_the_populate_one(tmp_path: Path) -> None:
    write_step_record(tmp_path, "h1.publish", {"z_threshold": 4.0})
    with pytest.raises(R.ReportStageError, match="no populate step record carries clock settings"):
        R.source_clock_settings(tmp_path)


def test_clock_settings_that_cannot_be_read_are_refused_rather_than_defaulted(
    tmp_path: Path,
) -> None:
    write_step_record(tmp_path, "h1.populate", {"z_threshold": "four"})
    with pytest.raises(R.ReportStageError, match="cannot read"):
        R.source_clock_settings(tmp_path)


def test_with_a_trim_placement_is_measured_before_it_and_the_clock_after(tmp_path: Path) -> None:
    from af.tree.clock import ClockSettings
    from af.tree.trim import TrimRule, find_trim_node

    from .test_report_filter import full

    populated, ids = full()
    populated.aa_subs[ids["x"]] = ["P3S"]
    trim = find_trim_node(populated, TrimRule("h3", 3, "S", 4, "invented"))
    limits = tmp_path / "placement.tsv"
    limits.write_text("subtype\tmin_correlation\treason\ttrees\nh3\t0.5\tinvented\t1\n")
    clock = ClockSettings(
        z_threshold=4.0, branch_threshold=None, earliest=datetime.date(1977, 1, 1), min_leaves=20
    )
    args = (populated, lambda record: None, Counter[str](), limits, clock)
    plain = R.derive(*args)
    trimmed = R.derive(*args, trim)
    assert trimmed.counts["placement"]["measured_on"].startswith("the untrimmed report tree")
    assert {k: v for k, v in trimmed.counts["placement"].items() if k != "measured_on"} == (
        plain.counts["placement"]
    )
    assert "measured_on" not in plain.counts["placement"]
    assert (
        "outgroup" not in trimmed.counts
        and plain.counts["outgroup"] == trimmed.counts["trim"]["outgroup_removed"]
    )
    assert "clock_direction" in trimmed.counts


def test_with_a_trim_the_clock_flags_come_from_the_untrimmed_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One fit, one meaning of off-clock: the outliers are scored on the tree with the outgroup."""
    from af.tree import clock as C
    from af.tree.trim import TrimRule, find_trim_node

    from .test_report_filter import full

    populated, ids = full()
    populated.aa_subs[ids["x"]] = ["P3S"]
    trim = find_trim_node(populated, TrimRule("h3", 3, "S", 4, "invented"))
    outgroup = populated.counts["outgroup"]
    scored: list[set[str | None]] = []

    def fake(tree, settings):  # every leaf of the scored tree flagged, so carrying is visible
        scored.append({leaf.name for leaf in tree.tree.leaves()})
        result = C.ClockResult(slope=None, intercept=None, residual_scale=None)
        for key in tree.leaves:
            result.flags[key] = [C.CLOCK_OUTLIER]
        return result

    monkeypatch.setattr(R, "find_outliers", fake)
    limits = tmp_path / "placement.tsv"
    limits.write_text("subtype\tmin_correlation\treason\ttrees\nh3\t0.5\tinvented\t1\n")
    settings = C.ClockSettings(
        z_threshold=4.0, branch_threshold=None, earliest=datetime.date(1977, 1, 1), min_leaves=20
    )
    args = (populated, lambda record: None, Counter[str](), limits, settings)
    trimmed = R.derive(*args, trim)
    assert len(scored) == 1 and outgroup in scored[0]  # scored once, on the untrimmed tree
    kept = {leaf.name for leaf in trimmed.tree.leaves()}
    assert outgroup not in kept
    flagged = {leaf.name for leaf in trimmed.tree.leaves() if trimmed.flags.get(leaf.node_id)}
    assert flagged == kept
    assert trimmed.counts["clock_flags_measured_on"] == "the untrimmed report tree"
    assert trimmed.counts["clock_flags_on_kept_leaves"] == {C.CLOCK_OUTLIER: len(kept)}
    plain = R.derive(*args)
    assert "clock_flags_measured_on" not in plain.counts


def test_a_tables_publish_does_not_change_the_report_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sarah, 25 Nov 2026: "the tables shoud not affect the trees". End to end through main():
    the same source before and after a tables publish (one that titrates a leaf) gives the same
    version, and the published version names no tables dataset as an input."""
    from af.store import Provenance
    from af.tables.identity import Manifest as TablesManifest
    from af.tables.model import Antigen, Table
    from af.tables.store import publish as publish_tables
    from af.tree.io import i6

    from .test_report_filter import full
    from .tree_fixtures import KEYS, LEAF_SEQ

    now = datetime.datetime(2026, 11, 25, tzinfo=datetime.UTC)

    def provenance(step: str) -> Provenance:
        return Provenance(step=step, inputs=(), parameters={}, started=now, finished=now)

    store = Store.create(tmp_path / "store")
    populated, _ = full()
    with store.build("trees", "h3/weekly") as build:
        i6.write(populated, build.path, "weekly", {"sequences": {}, "leaves": 5})
        clock = {"z_threshold": 4.0, "earliest": "1977-01-01", "min_leaves": 20}
        write_step_record(build.path, "h3.populate", clock)
        source = build.publish(provenance("test weekly"))
    alignment = {KEYS[n]: LEAF_SEQ[n] for n in "oabcd"}
    fasta = tmp_path / "alignment.fasta"
    fasta.write_text("")
    monkeypatch.setattr(R, "checked_alignment", lambda meta, export: (alignment, fasta))
    (tmp_path / "locations").mkdir()
    locations(tmp_path / "locations")
    limits = tmp_path / "placement.tsv"
    limits.write_text("subtype\tmin_correlation\treason\ttrees\nh3\t0.5\tinvented\t1\n")
    argv = [
        str(store.root),
        f"h3/weekly@{source.version}",
        str(tmp_path / "export"),
        "report-x",
        "--locations",
        str(tmp_path / "locations"),
        "--placement-limits",
        str(limits),
    ]

    def titrating(leaf: str, day: str) -> Table:
        return Table(
            table_id=f"h3-hint-examplelab-{day}",
            group="h3-hint-examplelab",
            lab="EXAMPLELAB",
            subtype="A(H3N2)",
            lineage="",
            assay="HINT",
            rbc="",
            date=f"{day[:4]}-{day[4:6]}-{day[6:]}",
            date_suffix=1,
            source_key="example 1",
            antigens=[Antigen(name=populated.leaves[KEYS[leaf]].name, raw_name="x")],
            sera=[],
            titres=[],
        )

    def tables_state(tables: list[Table]) -> None:
        manifest = TablesManifest.from_tables(tables, inputs=[])
        publish_tables(store, tables, manifest, provenance("test tables"))

    # Two tables states, both present when a report tree is derived; the second titrates another
    # (old, year-only) leaf and adds a table dated after the first.
    tables_state([titrating("a", "20261101")])
    assert R.main(argv) == 0
    before = store.current("trees", "h3/report-x")
    tables_state([titrating("a", "20261101"), titrating("b", "20261124")])
    assert R.main(argv) == 0
    after = store.current("trees", "h3/report-x")

    assert after == before
    published = json.loads((store.resolve(before) / "PROVENANCE.json").read_text())
    kinds = {i["store"]["kind"] for i in published["inputs"] if "store" in i}
    assert kinds == {"trees"}
    assert published["parameters"]["date_rule"].startswith("none")
    for gone in ("cutoff", "tables_through", "table_subtype", "table_lineage"):
        assert gone not in published["parameters"]


@pytest.mark.parametrize("option", ["--cutoff", "--tables-through", "--table-subtype"])
def test_the_date_rule_and_tables_options_are_gone(option: str) -> None:
    with pytest.raises(SystemExit):
        R.main(["store", "h3/weekly@0123456789abcdef", "export", "report-x", option, "x"])
