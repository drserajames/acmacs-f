"""Publishing clade assignments as ``clades/<subtype>``, and what it refuses."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from af.clades.assign import Assignment
from af.clades.nomenclature import CladeSet
from af.clades.store import (
    ASSIGNMENTS_FILE,
    COLUMNS,
    REPORT_FILE,
    CladeRow,
    CladeStoreError,
    build_report,
    dataset_for,
    publish,
    read_report,
    rows_from_assignments,
)
from af.store import ExternalInput, Store, StoreRef

from .synthetic import build_clone, load_synthetic

SUBTYPE = "A(H3N2)"
STARTED = datetime.datetime(2026, 9, 25, 9, 0, tzinfo=datetime.UTC)


def clade_set(tmp_path: Path) -> CladeSet:
    return load_synthetic(build_clone(tmp_path / "clone").parent)


def store(tmp_path: Path) -> Store:
    return Store.create(tmp_path / "store")


def sequences_ref() -> StoreRef:
    """A plausible reference to the sequence version these clades label.

    A version id is the first 16 hex digits of its manifest hash, so the two cannot be
    made up independently.
    """
    manifest_sha256 = "b" * 64
    return StoreRef("sequences", "h3", manifest_sha256[:16], manifest_sha256)


def nomenclature_input(tmp_path: Path) -> ExternalInput:
    path = tmp_path / "clone" / "synthetic_HA" / "subclades"
    return ExternalInput.of(path, version="0123456789abcdef")


def rows(count: int = 2, clade: str | None = "P.1") -> list[CladeRow]:
    return [
        CladeRow(f"EPI_ISL_{index}", f"EPI{index}", SUBTYPE, clade, "tree", 2, 0, f"L{index}")
        for index in range(count)
    ]


def test_dataset_keys_are_short_names() -> None:
    assert dataset_for(SUBTYPE) == "h3"
    assert dataset_for("B/Vic") == "bvic"


def test_byam_has_no_dataset() -> None:
    """Sarah, 25 Sep 2026: B/Yam trees and maps carry no clade labels for now. Asking for
    them must fail, not quietly return nothing."""
    with pytest.raises(CladeStoreError, match="no clade dataset for subtype"):
        dataset_for("B/Yam")


def test_publishes_a_version(tmp_path: Path) -> None:
    opened = store(tmp_path)
    ref = publish(
        opened,
        SUBTYPE,
        rows(),
        clade_set(tmp_path),
        sequences=sequences_ref(),
        nomenclature=[nomenclature_input(tmp_path)],
        started=STARTED,
    )
    assert ref.kind == "clades"
    assert ref.dataset == "h3"
    version = opened.resolve(ref)
    assert (version / ASSIGNMENTS_FILE).is_file()
    assert (version / REPORT_FILE).is_file()
    assert opened.current("clades", "h3") == ref


def test_the_provenance_names_what_the_clades_were_built_from(tmp_path: Path) -> None:
    """The whole point of the version: in the old system a clade was baked in at populate
    time and an edit reached maps only if someone remembered to re-run it."""
    import json

    opened = store(tmp_path)
    ref = publish(
        opened,
        SUBTYPE,
        rows(),
        clade_set(tmp_path),
        sequences=sequences_ref(),
        nomenclature=[nomenclature_input(tmp_path)],
        started=STARTED,
    )
    with (opened.resolve(ref) / "PROVENANCE.json").open() as stream:
        provenance = json.load(stream)
    kinds = [set(entry) for entry in provenance["inputs"]]
    assert {"store"} in kinds and {"external"} in kinds
    assert provenance["parameters"]["clade_set_version"].startswith("synthetic_HA@")
    assert provenance["parameters"]["engine"] == "tree"


def test_the_table_is_readable_with_its_declared_types(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    opened = store(tmp_path)
    ref = publish(
        opened,
        SUBTYPE,
        rows(),
        clade_set(tmp_path),
        sequences=sequences_ref(),
        nomenclature=[nomenclature_input(tmp_path)],
        started=STARTED,
    )
    path = (opened.resolve(ref) / ASSIGNMENTS_FILE).as_posix()
    connection = duckdb.connect()
    columns = connection.execute(f"SELECT * FROM read_parquet('{path}') LIMIT 0").description
    assert [column[0] for column in columns] == list(COLUMNS)
    count = connection.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()
    assert count is not None and count[0] == 2


def test_an_unnamed_clade_stays_null_rather_than_empty(tmp_path: Path) -> None:
    """ "The nomenclature does not name this virus" is an answer, and must not arrive at a
    consumer as the empty string."""
    duckdb = pytest.importorskip("duckdb")
    opened = store(tmp_path)
    ref = publish(
        opened,
        SUBTYPE,
        rows(clade=None),
        clade_set(tmp_path),
        sequences=sequences_ref(),
        nomenclature=[nomenclature_input(tmp_path)],
        started=STARTED,
    )
    path = (opened.resolve(ref) / ASSIGNMENTS_FILE).as_posix()
    nulls = (
        duckdb.connect()
        .execute(f"SELECT count(*) FROM read_parquet('{path}') WHERE clade IS NULL")
        .fetchone()
    )
    assert nulls is not None and nulls[0] == 2


def test_refuses_an_empty_table(tmp_path: Path) -> None:
    """A run that assigned nothing is a failure, not an empty result (design rule 3)."""
    with pytest.raises(CladeStoreError, match="refusing to publish an empty"):
        publish(
            store(tmp_path),
            SUBTYPE,
            [],
            clade_set(tmp_path),
            sequences=sequences_ref(),
            nomenclature=[nomenclature_input(tmp_path)],
            started=STARTED,
        )


def test_refuses_a_repeated_sequence(tmp_path: Path) -> None:
    duplicated = rows() + rows()
    with pytest.raises(CladeStoreError, match="appear more than once"):
        publish(
            store(tmp_path),
            SUBTYPE,
            duplicated,
            clade_set(tmp_path),
            sequences=sequences_ref(),
            nomenclature=[nomenclature_input(tmp_path)],
            started=STARTED,
        )


def test_refuses_rows_of_another_subtype(tmp_path: Path) -> None:
    mixed = [*rows(1), CladeRow("EPI_ISL_9", "EPI9", "B/Vic", "C", "tree")]
    with pytest.raises(CladeStoreError, match="rows carry other subtypes"):
        publish(
            store(tmp_path),
            SUBTYPE,
            mixed,
            clade_set(tmp_path),
            sequences=sequences_ref(),
            nomenclature=[nomenclature_input(tmp_path)],
            started=STARTED,
        )


def test_rows_from_assignments_skips_internal_nodes(tmp_path: Path) -> None:
    assignments = {
        "L1": Assignment("L1", "P.1", 2, 0),
        "N1": Assignment("N1", "P", 1, 0),
    }
    built = rows_from_assignments(assignments, {"L1": ("EPI_ISL_1", "EPI1")}, SUBTYPE)
    assert [row.epi_isl for row in built] == ["EPI_ISL_1"]
    assert built[0].tree_node == "L1"


def test_a_sequence_without_an_assignment_is_fatal(tmp_path: Path) -> None:
    """Otherwise it shows up much later as a virus missing from a map."""
    with pytest.raises(CladeStoreError, match="do not agree"):
        rows_from_assignments({}, {"L1": ("EPI_ISL_1", "EPI1")}, SUBTYPE)


def test_an_unknown_method_is_fatal(tmp_path: Path) -> None:
    with pytest.raises(CladeStoreError, match="unknown method"):
        rows_from_assignments({}, {}, SUBTYPE, method="guessed")


def test_report_counts_clades_and_the_unnamed(tmp_path: Path) -> None:
    clades = clade_set(tmp_path)
    report = build_report([*rows(2, "P.1"), *rows(1, None)], clades)
    assert report["sequences"] == 3
    assert report["unnamed"] == 1
    assert report["counts"] == {"P.1": 2}
    assert "P.2" in report["clades_without_sequences"]


def test_report_is_readable_from_the_store(tmp_path: Path) -> None:
    opened = store(tmp_path)
    ref = publish(
        opened,
        SUBTYPE,
        rows(),
        clade_set(tmp_path),
        sequences=sequences_ref(),
        nomenclature=[nomenclature_input(tmp_path)],
        started=STARTED,
    )
    assert read_report(opened, ref)["sequences"] == 2
