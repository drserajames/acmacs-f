"""The sequence store build: raw pulls, placement, Parquet partitions, the whole step.

Every name, id, date and sequence here is invented.
"""

from __future__ import annotations

import datetime
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import brotli
import pytest

from af.pipeline.config import RunnerSettings
from af.run import Runner
from af.seq import build, processed, pulls
from af.seq.dates import parse as parse_date
from af.seq.gisaid import SequenceRecord, Workbook
from af.seq.nextclade import Aligned
from af.store import PathsConfig, Store, Work, busy
from af.store.ref import StoreError, StoreRef
from af.store.store import Provenance

from .test_nextclade import fake_binary, make_dataset

SUBTYPE = "A / H3N2"


def record(n: int, **over: object) -> SequenceRecord:
    fields: dict[str, object] = dict(
        epi_isl=f"EPI_ISL_{n}", accession=f"EPI{900000 + n}", gisaid_name=f"A/EXAMPLETOWN/{n}/2021",
        name=f"A(H3N2)/EXAMPLETOWN/{n}/2021", nucleotides="ACGTACGT",
        collection_date=parse_date("2021-03"), subtype=SUBTYPE, lineage="", host="Human",
        passage="MDCK1", location="Europe / Exampleland / Exampleshire / Exampleton",
        originating_lab="Example Lab", submitting_lab="", submission_date="2021-04-01",
        embargoed_until="",
    )  # fmt: skip
    fields.update(over)
    return SequenceRecord(**fields)  # type: ignore[arg-type]


def aligned(record: SequenceRecord, **over: object) -> Aligned:
    fields: dict[str, object] = dict(
        seq_id=processed.seq_id(record), error=None, alignment_start=1, alignment_end=30,
        covers_mature=True, nucleotides="ACGT", amino_acids="TV", failed_cds=(), frameshifts=0,
        deleted_aa=0, inserted_aa=0, unknown_aa=0, premature_stop=False, qc_status="good",
    )  # fmt: skip
    fields.update(over)
    return Aligned(**fields)  # type: ignore[arg-type]


RULES = [
    processed.PlacementRule(SUBTYPE, "", "h3", "GISAID states no lineage for A(H3N2)"),
    processed.PlacementRule("B", "Victoria", "bvic", "GISAID's lineage"),
]


class TestPlacement:
    def test_records_go_to_the_dataset_their_rule_names(self) -> None:
        placed = processed.place([record(1), record(2, subtype="B", lineage="Victoria")], RULES)
        assert {k: [r.epi_isl for r in v] for k, v in placed.items()} == {
            "h3": ["EPI_ISL_1"],
            "bvic": ["EPI_ISL_2"],
        }

    def test_an_unplaced_record_is_fatal_with_counts(self) -> None:
        records = [record(1, subtype="B", lineage=""), record(2, subtype="B", lineage="")]
        with pytest.raises(processed.StoreBuildError, match="'B'/'': 2"):
            processed.place(records, RULES)

    def test_two_rules_for_one_pair_are_refused(self) -> None:
        with pytest.raises(processed.StoreBuildError, match="two placement rules"):
            processed.place([], [*RULES, processed.PlacementRule(SUBTYPE, "", "h1", "typo")])


class TestRows:
    def test_isolate_row_keeps_date_precision_and_interval(self) -> None:
        row = processed.isolates_table([record(1)], "h3", "p1").to_pylist()[0]
        assert (row["collection_date"], row["date_precision"]) == ("2021-03", "month")
        assert row["collection_date_first"] == datetime.date(2021, 3, 1)
        assert row["collection_date_last"] == datetime.date(2021, 3, 31)

    def test_location_is_split_and_kept_whole(self) -> None:
        row = processed.isolates_table([record(1)], "h3", "p1").to_pylist()[0]
        assert (row["region"], row["country"], row["place"]) == (
            "Europe", "Exampleland", "Exampleshire / Exampleton",
        )  # fmt: skip
        assert row["location"].startswith("Europe / ")

    def test_a_short_location_leaves_later_parts_empty(self) -> None:
        row = processed.isolates_table([record(1, location="Europe")], "h3", "p").to_pylist()[0]
        assert (row["region"], row["country"], row["place"]) == ("Europe", None, "")

    def test_embargo_is_a_flag_and_a_date(self) -> None:
        rows = processed.isolates_table(
            [record(1), record(2, embargoed_until="2021-09-01")], "h3", "p"
        ).to_pylist()
        assert [(r["embargoed"], r["embargoed_until"]) for r in rows] == [
            (False, None),
            (True, datetime.date(2021, 9, 1)),
        ]

    def test_a_missing_submission_date_is_fatal(self) -> None:
        with pytest.raises(processed.StoreBuildError, match="no Submission_Date"):
            processed.isolates_table([record(1, submission_date="")], "h3", "p")

    def test_rows_are_in_key_order_whatever_the_input_order(self) -> None:
        table = processed.isolates_table([record(3), record(1), record(2)], "h3", "p")
        assert table["epi_isl"].to_pylist() == ["EPI_ISL_1", "EPI_ISL_2", "EPI_ISL_3"]

    def test_unaligned_records_are_kept_with_their_error(self) -> None:
        good, bad = record(1), record(2)
        results = {
            processed.seq_id(good): aligned(good),
            processed.seq_id(bad): aligned(bad, error="seed failed", nucleotides=None,
                                           amino_acids=None, covers_mature=False),
        }  # fmt: skip
        rows = processed.sequences_table([good, bad], results, "p").to_pylist()
        assert [(r["nuc_aligned"], r["align_error"]) for r in rows] == [
            ("ACGT", None),
            (None, "seed failed"),
        ]

    def test_substitutions_are_stored(self) -> None:
        r = record(1)
        rows = processed.sequences_table(
            [r], {processed.seq_id(r): aligned(r, substitutions=12)}, "p"
        )
        assert rows["substitutions"].to_pylist() == [12]

    def test_identical_sequences_share_a_hash_and_stay_two_rows(self) -> None:
        one, two = record(1), record(2)
        results = {processed.seq_id(r): aligned(r) for r in (one, two)}
        rows = processed.sequences_table([one, two], results, "p").to_pylist()
        assert len(rows) == 2 and rows[0]["seq_hash"] == rows[1]["seq_hash"]

    def test_a_record_without_an_alignment_result_is_fatal(self) -> None:
        with pytest.raises(processed.StoreBuildError, match="no alignment result"):
            processed.sequences_table([record(1)], {}, "p")


def publish(store: Store, pull_id: str, records: list[SequenceRecord]) -> StoreRef:
    results = {processed.seq_id(r): aligned(r) for r in records}
    return processed.publish_pull(
        store, "h3", pull_id,
        processed.isolates_table(records, "h3", pull_id),
        processed.sequences_table(records, results, pull_id),
        inputs=[], parameters={"pull": pull_id}, started=datetime.datetime.now(datetime.UTC),
    )  # fmt: skip


class TestPublish:
    def test_a_second_pull_links_the_first_unchanged(self, tmp_path: Path) -> None:
        store = Store.create(tmp_path / "store")
        first = publish(store, "p1", [record(1), record(2)])
        second = publish(store, "p2", [record(3)])
        old = store.version_dir(first) / processed.partition("isolates", "p1")
        new = store.version_dir(second) / processed.partition("isolates", "p1")
        assert old.stat().st_ino == new.stat().st_ino
        table = processed.read_table(store, second, "isolates", ["epi_isl", "source_pull"])
        assert sorted(zip(*table.to_pydict().values(), strict=True)) == [
            ("EPI_ISL_1", "p1"), ("EPI_ISL_2", "p1"), ("EPI_ISL_3", "p2"),
        ]  # fmt: skip

    def test_republishing_a_pull_replaces_only_its_partition(self, tmp_path: Path) -> None:
        store = Store.create(tmp_path / "store")
        publish(store, "p1", [record(1)])
        publish(store, "p2", [record(2)])
        latest = publish(store, "p1", [record(1), record(4)])
        ids = processed.read_table(store, latest, "isolates", ["epi_isl"])["epi_isl"].to_pylist()
        assert sorted(ids) == ["EPI_ISL_1", "EPI_ISL_2", "EPI_ISL_4"]

    def test_the_same_rows_publish_the_same_version(self, tmp_path: Path) -> None:
        store = Store.create(tmp_path / "store")
        assert publish(store, "p1", [record(1)]) == publish(store, "p1", [record(1)])

    def test_a_key_in_two_pulls_is_refused(self, tmp_path: Path) -> None:
        store = Store.create(tmp_path / "store")
        publish(store, "p1", [record(1)])
        with pytest.raises(processed.StoreBuildError, match="1 key.*in two pulls"):
            publish(store, "p2", [record(1)])


# ---- raw pulls and the whole step ------------------------------------------------------

DEFLINE = "{name}_|_a={epi}_|_e=2021-03-01_|_j=Example Lab_|_k=_|_o={acc}_|_"


def extractor_set(root: Path, rng: random.Random, n: int = 3) -> Path:
    """An invented extractor set: one H3 pull, kept .fas.br equal to the delivered FASTA."""
    (root / "sequences").mkdir(parents=True)
    (root / "raw").mkdir()
    deflines = [
        DEFLINE.format(name=f"A/EXAMPLETOWN/{i}/2021", epi=f"EPI_ISL_{i}", acc=f"EPI{900000 + i}")
        for i in range(1, n + 1)
    ]
    text = "".join(
        f">{d}\n" + "".join(rng.choice("ACGT") for _ in range(40)) + "\n" for d in deflines
    )
    (root / "raw" / "epiflu-h3n2-20201213-20210312.fasta").write_text(text)
    (root / "raw" / "epiflu-h3n2-20201213-20210312-metadata.xls").write_bytes(b"workbook")
    (root / "sequences" / "epiflu-2021-0312-h3n2.fas.br").write_bytes(
        brotli.compress(text.encode())
    )
    return root


class TestRawPulls:
    def test_pulls_are_found_from_the_kept_fastas(self, tmp_path: Path) -> None:
        found = pulls.find_pulls(extractor_set(tmp_path / "set", random.Random(1)), "definitive")
        assert [(p.pull_id, p.workbook.name, p.rejoined) for p in found] == [
            ("definitive-2021-0312-h3n2", "epiflu-h3n2-20201213-20210312-metadata.xls", None)
        ]

    def test_a_targeted_pull_is_found_by_its_label(self, tmp_path: Path) -> None:
        root = extractor_set(tmp_path / "set", random.Random(1))
        text = ">" + DEFLINE.format(name="A/EXAMPLETOWN/99/2010", epi="EPI_ISL_99", acc="EPI99")
        text += "\nACGT\n"
        (root / "raw" / "epiflu-targeted-example-roots-20260925-h3n2.fasta").write_text(text)
        (root / "raw" / "epiflu-targeted-example-roots-20260925-h3n2-metadata.xls").write_bytes(
            b"workbook"
        )
        (root / "sequences" / "epiflu-2026-0925-targeted-example-roots-h3n2.fas.br").write_bytes(
            brotli.compress(text.encode())
        )
        found = {p.pull_id: p for p in pulls.find_pulls(root, "targeted")}
        assert sorted(found) == ["targeted-2021-0312-h3n2", "targeted-2026-0925-example-roots-h3n2"]
        pull = found["targeted-2026-0925-example-roots-h3n2"]
        assert pull.subtype == "h3n2" and pull.pull_id.rsplit("-", 1)[-1] == "h3n2"
        assert pull.workbook.name == "epiflu-targeted-example-roots-20260925-h3n2-metadata.xls"

    def test_a_targeted_pull_without_its_workbook_is_fatal(self, tmp_path: Path) -> None:
        root = extractor_set(tmp_path / "set", random.Random(1))
        (root / "raw" / "epiflu-targeted-roots-20260925-b.fasta").write_text(">x\nACGT\n")
        (root / "sequences" / "epiflu-2026-0925-targeted-roots-b.fas.br").write_bytes(b"")
        with pytest.raises(pulls.PullFilesError, match="no metadata workbook"):
            pulls.find_pulls(root, "targeted")

    def test_a_pull_without_its_delivered_fasta_is_fatal(self, tmp_path: Path) -> None:
        root = extractor_set(tmp_path / "set", random.Random(1))
        (root / "raw" / "epiflu-h3n2-20201213-20210312.fasta").unlink()
        with pytest.raises(pulls.PullFilesError, match="found 0"):
            pulls.find_pulls(root, "definitive")

    def test_import_copies_and_a_second_import_publishes_nothing(self, tmp_path: Path) -> None:
        store = Store.create(tmp_path / "store")
        (pull,) = pulls.find_pulls(extractor_set(tmp_path / "set", random.Random(1)), "d")
        first = pulls.import_pull(store, pull)
        assert pulls.import_pull(store, pull) == first
        kept = store.version_dir(first) / pull.sequences.name
        assert kept.stat().st_ino != pull.sequences.stat().st_ino  # copied, not linked
        assert pull.sequences.stat().st_mode & 0o200  # the extractor's file stays writable

    def test_a_kept_fasta_that_is_not_the_delivered_one_is_refused(self, tmp_path: Path) -> None:
        root = extractor_set(tmp_path / "set", random.Random(1))
        (pull,) = pulls.find_pulls(root, "d")
        pull.sequences.write_bytes(brotli.compress(b">other\nACGT\n"))
        with pytest.raises(pulls.PullFilesError, match="no rejoin log"):
            pulls.import_pull(Store.create(tmp_path / "store"), pull)


def _run_pull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, locations: build.LocationsConfig | None = None
) -> tuple[build.SequencesConfig, Store, dict[str, StoreRef]]:
    """Import an invented set and store its one H3 pull with a stand-in Nextclade."""
    rng = random.Random(7)
    root = extractor_set(tmp_path / "set", rng)
    store = Store.create(tmp_path / "store")
    Work.create(tmp_path / "work")
    make_dataset(tmp_path / "ds", rng)
    monkeypatch.setattr(
        build.nc, "fetch_dataset", lambda pin, store: _dataset_ref(store, tmp_path / "ds")
    )
    rows = [
        {"Isolate_Id": f"EPI_ISL_{i}", "Collection_Date": "2021", "Subtype": SUBTYPE,
         "Lineage": "", "Host": "Human", "Passage_History": "MDCK1",
         "Location": "Europe / Exampleland", "Submission_Date": "2021-04-01"}
        for i in (1, 2, 3)
    ]  # fmt: skip
    monkeypatch.setattr(build, "read_workbook", lambda path: Workbook(rows=rows))
    config = build.SequencesConfig(
        paths=PathsConfig(store=tmp_path / "store", work=tmp_path / "work"),
        runner=RunnerSettings(kind="local"),
        nextclade=build.NextcladeConfig(
            binary=fake_binary(tmp_path),
            version="9.9.9",
            max_unknown_aa=10,
            max_deleted_aa=6,
            datasets={"h3": build.DatasetConfig("example/h3", "t1", "0" * 64, 597)},
        ),  # fmt: skip
        placement=[build.PlacementConfig(SUBTYPE, "", "h3", "no lineage for A(H3N2)")],
        sources={"definitive": root},
        source_subtypes={"h3n2": "A(H3N2)"},
        locations=locations,
    )
    build.import_source(config, "definitive")
    refs = build.store_pull(config, PULL, build.make_runner(config.runner))
    return config, store, refs


PULL = "definitive-2021-0312-h3n2"


def _provenance(store: Store, ref: StoreRef) -> dict[str, object]:
    text = (store.version_dir(ref) / "PROVENANCE.json").read_text()
    parameters: dict[str, object] = json.loads(text)["parameters"]
    return parameters


def test_store_pull_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Import, align with a stand-in Nextclade, publish; a re-run changes nothing."""
    config, store, refs = _run_pull(tmp_path, monkeypatch)
    table = processed.read_table(store, refs["h3"], "isolates", ["epi_isl", "date_precision"])
    assert table.to_pylist() == [
        {"epi_isl": f"EPI_ISL_{i}", "date_precision": "year"} for i in (1, 2, 3)
    ]
    sequences = processed.read_table(store, refs["h3"], "sequences", ["covers_mature"])
    assert sequences["covers_mature"].to_pylist() == [False] * 3  # the stand-in aligns 30 nt
    assert build.store_pull(config, PULL, build.make_runner(config.runner)) == refs
    # no [locations]: the provenance says the new-locations check did not run
    assert _provenance(store, refs["h3"])["new_locations"] == {
        "checked": False, "reason": "no [locations] in the sequences config",
    }  # fmt: skip


def _locations(tmp_path: Path) -> build.LocationsConfig:
    """Invented location tables without the pull's town, and a GeoNames snapshot that has it."""
    tables, geo = tmp_path / "locations", tmp_path / "geonames"
    tables.mkdir()
    geo.mkdir()
    meta = "\tadded_by\tadded_on"
    (tables / "countries.tsv").write_text(
        f"code\tspelling\tsource\tevidence{meta}\n"
        "XAA\tExampleland\tgisaid\tsame name\tt\t2026-01-01\n"
    )
    (tables / "regions.tsv").write_text(
        f"scheme\tcountry\tgroup\tevidence{meta}\ngisaid\tXAA\tEurope\tthe store\tt\t2026-01-01\n"
    )
    columns = (
        "location\tcountry\tadmin\tlatitude\tlongitude\tprecision\tsource\tevidence\tsame_as\tflags"
    )
    (tables / "places.tsv").write_text(
        f"{columns}{meta}\nEXAMPLECITY\tXAA\t\t1.0\t2.0\tcity\thand\tinvented\t\t\tt\t2026-01-01\n"
    )
    (geo / "countryInfo.txt").write_text("XA\tXAA\t1\tXA\tExampleland\n")
    (geo / "admin1CodesASCII.txt").write_text("XA.01\tExampleshire\tExampleshire\t901\n")
    town = ["1", "Exampletown", "Exampletown", "", "10.5", "20.25", "P", "PPL", "XA", "", "01"]
    (geo / "cities500.txt").write_text("\t".join([*town, "", "", "", "600"]) + "\n")
    return build.LocationsConfig(tables=tables, geonames=geo)


def test_store_pull_checks_new_locations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A name-location places.tsv lacks is proposed from GeoNames; counts in the provenance."""
    _, store, refs = _run_pull(tmp_path, monkeypatch, _locations(tmp_path))
    report = _provenance(store, refs["h3"])["new_locations"]
    assert isinstance(report, dict)
    assert (report["checked"], report["new_locations"], report["outcomes"]) == (
        True, 1, {"proposed": 1},
    )  # fmt: skip
    area = tmp_path / "work" / "sequences" / "gisaid" / PULL / build.NEW_LOCATIONS / "h3"
    assert report["review"] == str(area / "review.tsv")
    assert (area / "review.tsv").read_text().count("\n") == 1  # header only: nothing to review
    proposed = (area / "proposed-places.tsv").read_text().split("\t")
    assert proposed[:5] == ["EXAMPLETOWN", "XAA", "", "10.5000", "20.2500"]


def _dataset_ref(store: Store, source: Path, key: str = "nextclade/example/h3/t1") -> StoreRef:
    """Put an invented Nextclade dataset in raw/, as fetch_dataset would, without a download."""
    try:
        return store.current("raw", key)
    except StoreError:
        pass
    with store.build("raw", key) as builder:
        builder.copy(source, "dataset")
        now = datetime.datetime.now(datetime.UTC)
        return builder.publish(Provenance("test", (), {}, now, now))


# ---- B lineage by alignment -------------------------------------------------------------

CHECK = processed.LineageCheck("B", {"Victoria": "bvic", "Yamagata": "byam"}, 30, "nearest ref")
B_RULES = [
    processed.PlacementRule("B", "Victoria", "bvic", "stated"),
    processed.PlacementRule("B", "Yamagata", "byam", "stated"),
    processed.PlacementRule("B", "", "bvic", "no evidence either way: kept with B/Vic, flagged"),
]


def distances(vic: int | None, yam: int | None) -> dict[str, Aligned]:
    """One record's alignments against both references; None = did not align."""
    r = record(1, subtype="B")
    return {
        dataset: aligned(r, substitutions=subs) if subs is not None
        else aligned(r, error="seed failed", nucleotides=None, amino_acids=None)
        for dataset, subs in (("bvic", vic), ("byam", yam))
    }  # fmt: skip


class TestChooseLineage:
    @pytest.mark.parametrize(
        ("stated", "vic", "yam", "expected"),
        [
            ("", 20, 160, ("bvic", "lineage.from-alignment")),
            ("", 150, 12, ("byam", "lineage.from-alignment")),
            ("Victoria", 20, 160, ("bvic", None)),
            ("Yamagata", 20, 160, ("bvic", "lineage.disagrees")),
            ("Yamagata-ish", 20, 160, ("bvic", "lineage.unknown-label")),
            ("", 131, 122, (None, "lineage.ambiguous")),  # pre-split: far from both
            ("Yamagata", 128, 130, (None, "lineage.unconfirmed")),
            ("", 40, None, ("bvic", "lineage.from-alignment")),  # only one aligns
            ("", None, None, (None, "lineage.unaligned")),
        ],
    )  # fmt: skip
    def test_cases(self, stated: str, vic: int | None, yam: int | None,
                   expected: tuple[str | None, str | None]) -> None:  # fmt: skip
        assert processed.choose_lineage(stated, distances(vic, yam), CHECK) == expected

    def test_the_margin_is_inclusive(self) -> None:
        assert processed.choose_lineage("", distances(10, 40), CHECK) == (
            "bvic", "lineage.from-alignment",
        )  # fmt: skip
        assert processed.choose_lineage("", distances(10, 39), CHECK)[0] is None


class TestPlaceByAlignment:
    def results(
        self, *pairs: tuple[SequenceRecord, int | None, int | None]
    ) -> dict[str, dict[str, Aligned]]:
        out: dict[str, dict[str, Aligned]] = {"bvic": {}, "byam": {}}
        for r, vic, yam in pairs:
            for dataset, found in distances(vic, yam).items():
                out[dataset][processed.seq_id(r)] = found
        return out

    def test_no_evidence_falls_back_to_the_row_and_keeps_the_flag(self) -> None:
        clear = record(1, subtype="B", lineage="")
        old = record(2, subtype="B", lineage="")
        placed, flags = processed.place_by_alignment(
            [clear, old], B_RULES, CHECK, self.results((clear, 150, 10), (old, 131, 122))
        )
        assert {k: [(r.epi_isl, r.problems[-1]) for r in v] for k, v in placed.items()} == {
            "byam": [("EPI_ISL_1", "lineage.from-alignment")],
            "bvic": [("EPI_ISL_2", "lineage.ambiguous")],
        }
        assert flags == {"lineage.from-alignment": 1, "lineage.ambiguous": 1}

    def test_no_evidence_and_no_row_is_fatal(self) -> None:
        r = record(1, subtype="B", lineage="")
        with pytest.raises(processed.StoreBuildError, match="neither alignment nor"):
            processed.place_by_alignment([r], B_RULES[:2], CHECK, self.results((r, None, None)))


B_ROWS = [
    {"Isolate_Id": f"EPI_ISL_{i}", "Collection_Date": "2021-03-01", "Subtype": "B",
     "Lineage": lineage, "Host": "Human", "Passage_History": "MDCK1",
     "Location": "Europe / Exampleland", "Submission_Date": "2021-04-01"}
    for i, lineage in ((1, "Victoria"), (2, ""), (3, "Yamagata"))
]  # fmt: skip


def b_config(tmp_path: Path, root: Path, **over: object) -> build.SequencesConfig:
    fields: dict[str, object] = dict(
        paths=PathsConfig(store=tmp_path / "store", work=tmp_path / "work"),
        runner=RunnerSettings(kind="local"),
        nextclade=build.NextcladeConfig(
            binary=fake_binary(tmp_path), version="9.9.9", max_unknown_aa=10, max_deleted_aa=6,
            datasets={"bvic": build.DatasetConfig("example/bvic", "t1", "0" * 64, 597),
                      "byam": build.DatasetConfig("example/byam", "t1", "0" * 64, 597)},
        ),
        placement=[build.PlacementConfig(r.gisaid_subtype, r.lineage, r.dataset, r.reason)
                   for r in B_RULES],
        sources={"definitive": root},
        source_subtypes={"b": "B"},
        lineage_check=[build.LineageCheckConfig("B", {"Victoria": "bvic", "Yamagata": "byam"},
                                                30, "nearest reference")],
    )  # fmt: skip
    fields.update(over)
    return build.SequencesConfig(**fields)  # type: ignore[arg-type]


def b_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    rng = random.Random(9)
    root = tmp_path / "set"
    (root / "sequences").mkdir(parents=True)
    (root / "raw").mkdir()
    deflines = [DEFLINE.format(name=f"B/EXAMPLETOWN/{i}/2021", epi=f"EPI_ISL_{i}",
                               acc=f"EPI{900000 + i}") for i in (1, 2, 3)]  # fmt: skip
    text = "".join(
        f">{d}\n" + "".join(rng.choice("ACGT") for _ in range(40)) + "\n" for d in deflines
    )
    (root / "raw" / "epiflu-b-20201213-20210312.fasta").write_text(text)
    (root / "raw" / "epiflu-b-20201213-20210312-metadata.xls").write_bytes(b"workbook")
    (root / "sequences" / "epiflu-2021-0312-b.fas.br").write_bytes(brotli.compress(text.encode()))
    Store.create(tmp_path / "store")
    Work.create(tmp_path / "work")
    for name in ("bvic", "byam"):
        make_dataset(tmp_path / f"ds-{name}", rng)
    monkeypatch.setattr(
        build.nc, "fetch_dataset",
        lambda pin, store: _dataset_ref(store, tmp_path / f"ds-{pin.path.split('/')[-1]}",
                                        f"nextclade/{pin.path}/{pin.tag}"),
    )  # fmt: skip
    monkeypatch.setattr(build, "read_workbook", lambda path: Workbook(rows=B_ROWS))
    return root


def test_b_pull_with_no_evidence_is_placed_by_rows_and_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stand-in Nextclade reports no substitutions, so nothing is placed by evidence."""
    config = b_config(tmp_path, b_set(tmp_path, monkeypatch))
    build.import_source(config, "definitive")
    refs = build.store_pull(config, "definitive-2021-0312-b", build.make_runner(config.runner))
    store = Store.open(tmp_path / "store")
    placed = {
        dataset: processed.read_table(store, ref, "isolates", ["epi_isl", "problems"]).to_pylist()
        for dataset, ref in refs.items()
    }
    assert placed == {
        "bvic": [{"epi_isl": "EPI_ISL_1", "problems": ["lineage.unaligned"]},
                 {"epi_isl": "EPI_ISL_2", "problems": ["lineage.unaligned"]}],
        "byam": [{"epi_isl": "EPI_ISL_3", "problems": ["lineage.unaligned"]}],
    }  # fmt: skip
    provenance = (store.resolve(refs["byam"]) / "PROVENANCE.json").read_text()
    assert "nextclade/example/bvic/t1" in provenance  # both references it was placed against


def test_a_placement_row_outside_the_candidates_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [*B_RULES[:2], processed.PlacementRule("B", "", "h3", "wrong")]
    config = b_config(
        tmp_path, b_set(tmp_path, monkeypatch),
        placement=[build.PlacementConfig(r.gisaid_subtype, r.lineage, r.dataset, r.reason)
                   for r in rows],
    )  # fmt: skip
    build.import_source(config, "definitive")
    with pytest.raises(build.ConfigError, match="not a lineage_check candidate"):
        build.store_pull(config, "definitive-2021-0312-b", build.make_runner(config.runner))


# ---- store runs hold one batch marker (af.store.busy) ------------------------------------

B_PULL = "definitive-2021-0312-b"


def test_a_b_pull_names_both_lineage_datasets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = b_config(tmp_path, b_set(tmp_path, monkeypatch))
    assert build.pull_datasets(config, B_PULL) == ["bvic", "byam"]
    with pytest.raises(build.ConfigError, match="no entry for pull label 'h9n2'"):
        build.pull_datasets(config, "definitive-2021-0312-h9n2")


def test_an_a_pull_names_its_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # source_subtypes says "A(H3N2)" and placement says "A / H3N2": the same subtype
    config = b_config(
        tmp_path, b_set(tmp_path, monkeypatch),
        placement=[build.PlacementConfig(SUBTYPE, "", "h3", "no lineage for A(H3N2)"),
                   *b_config(tmp_path, tmp_path).placement],
        source_subtypes={"h3n2": "A(H3N2)", "b": "B"},
    )  # fmt: skip
    assert build.pull_datasets(config, PULL) == ["h3"]
    assert build.pull_datasets(config, B_PULL) == ["bvic", "byam"]


@pytest.mark.parametrize(
    ("gisaid", "terms"),
    [("A / H3N2", "A(H3N2)"), ("A / H1", "A(H1)"), ("B", "B"), (" A /  H1N1 ", "A(H1N1)")],
)
def test_name_terms(gisaid: str, terms: str) -> None:
    assert build.name_terms(gisaid) == terms


def test_the_marker_is_held_for_the_whole_run_and_refuses_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One marker across every pull, so a reader between two publishes is refused."""
    config = b_config(tmp_path, b_set(tmp_path, monkeypatch))
    root = tmp_path / "store"
    seen: list[tuple[str, list[busy.Marker]]] = []

    def fake_store_pull(config: build.SequencesConfig, pull_id: str, runner: object) -> dict:
        seen.append((pull_id, busy.live_markers(root)))
        with pytest.raises(busy.StoreBusy), Store.open(root).reading("example-map"):
            pass
        return {}

    monkeypatch.setattr(build, "store_pull", fake_store_pull)
    pull_ids = [B_PULL, "definitive-2021-0612-b"]
    assert build.store_pulls(
        config, pull_ids, build.make_runner(config.runner), batch="seq-sweep"
    ) == {p: {} for p in pull_ids}
    assert [pull for pull, _ in seen] == pull_ids
    (first,), (second,) = (markers for _, markers in seen)
    assert first == second  # the same marker, not one taken per pull
    assert first.name == "seq-sweep"
    assert first.datasets == ("sequences/bvic", "sequences/byam")
    assert busy.read_markers(root) == []  # released at the end


def test_a_failed_pull_stops_the_run_and_releases_the_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = b_config(tmp_path, b_set(tmp_path, monkeypatch))
    called: list[str] = []

    def failing(config: build.SequencesConfig, pull_id: str, runner: object) -> dict:
        called.append(pull_id)
        raise StoreError("example failure")

    monkeypatch.setattr(build, "store_pull", failing)
    with pytest.raises(StoreError, match="example failure"):
        build.store_pulls(
            config, [B_PULL, "definitive-2021-0612-b"], build.make_runner(config.runner)
        )
    assert called == [B_PULL]
    assert busy.read_markers(tmp_path / "store") == []


def test_bad_pull_lists_are_refused_before_the_marker_is_taken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = b_config(tmp_path, b_set(tmp_path, monkeypatch))
    runner = build.make_runner(config.runner)
    for pull_ids, message in (
        ([], "no pull named"),
        ([B_PULL, B_PULL], "pull named twice"),
        ([B_PULL, "definitive-2021-0312-h9n2"], "no entry for pull label"),
    ):
        with pytest.raises(build.ConfigError, match=message):
            build.store_pulls(config, pull_ids, runner)
        assert busy.read_markers(tmp_path / "store") == []


def test_a_real_pull_publishes_while_its_own_marker_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must not block the writer that holds it."""
    config = b_config(tmp_path, b_set(tmp_path, monkeypatch))
    build.import_source(config, "definitive")
    real = build.store_pull
    held: list[int] = []

    def spy(config: build.SequencesConfig, pull_id: str, runner: Runner) -> dict[str, StoreRef]:
        held.append(len(busy.live_markers(tmp_path / "store")))
        return real(config, pull_id, runner)

    monkeypatch.setattr(build, "store_pull", spy)
    refs = build.store_pulls(config, [B_PULL], build.make_runner(config.runner))
    assert held == [1]
    assert sorted(refs[B_PULL]) == ["bvic", "byam"]
    assert busy.read_markers(tmp_path / "store") == []


def test_the_command_line_joins_repeated_pulls_and_a_pulls_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listing = tmp_path / "pulls.txt"
    listing.write_text("# example sweep\nexample-2021-0612-b\n\nexample-2021-0912-b  # last\n")
    got: dict[str, object] = {}

    def fake(config: object, pull_ids: list[str], runner: object, *, batch: str) -> dict:
        got.update(pulls=pull_ids, batch=batch)
        return {}

    monkeypatch.setattr(build, "load_config", lambda path, kind: SimpleNamespace(runner=None))
    monkeypatch.setattr(build, "make_runner", lambda settings, path: None)
    monkeypatch.setattr(build, "store_pulls", fake)
    argv = ["cfg.toml", "store", "--pull", "example-2021-0312-b", "--pulls-file", str(listing)]
    assert build.main([*argv, "--batch", "seq-sweep"]) == 0
    assert got == {
        "pulls": ["example-2021-0312-b", "example-2021-0612-b", "example-2021-0912-b"],
        "batch": "seq-sweep",
    }
    build.main(argv[:4])
    assert got["batch"] == build.DEFAULT_BATCH


# ---- reference_check: keeping a dataset to its reference's lineage ----------------------


def _ref_check(**over: object) -> processed.ReferenceCheck:
    fields: dict[str, object] = dict(
        dataset="h1", max_substitutions=200, expect_lineages=("pdm09",),
        reason="example: this dataset's reference is a pdm09 virus",
    )  # fmt: skip
    fields.update(over)
    return processed.ReferenceCheck(**fields)  # type: ignore[arg-type]


def _at(rec: SequenceRecord, substitutions: int | None, **over: object) -> Aligned:
    return aligned(rec, substitutions=substitutions, **over)


def test_a_record_far_from_the_reference_is_held_not_placed() -> None:
    near, far = record(1, lineage="pdm09"), record(2, lineage="pdm09")
    kept, held, flags = processed.hold_far_from_reference(
        [near, far],
        {processed.seq_id(near): _at(near, 30), processed.seq_id(far): _at(far, 300)},
        _ref_check(),
    )
    assert [r.epi_isl for r in kept] == [near.epi_isl]
    assert [r.epi_isl for r in held] == [far.epi_isl]
    assert flags == Counter({processed.FAR_FROM_REFERENCE: 1})


def test_a_record_exactly_at_the_threshold_is_kept() -> None:
    rec = record(1, lineage="pdm09")
    kept, held, _ = processed.hold_far_from_reference(
        [rec], {processed.seq_id(rec): _at(rec, 200)}, _ref_check(max_substitutions=200)
    )
    assert (len(kept), len(held)) == (1, 0)


def test_a_record_that_did_not_align_is_kept_because_distance_says_nothing() -> None:
    # failing to align is a length problem, not lineage evidence: holding it would evict
    # a record on no evidence at all
    short, failed = record(1, lineage="pdm09"), record(2, lineage="pdm09")
    kept, held, flags = processed.hold_far_from_reference(
        [short, failed],
        {processed.seq_id(short): _at(short, None),
         processed.seq_id(failed): _at(failed, None, error="too short to align")},
        _ref_check(),
    )  # fmt: skip
    assert len(held) == 0
    assert flags == Counter({processed.NO_DISTANCE: 2})
    assert all(processed.NO_DISTANCE in r.problems for r in kept)


def test_a_record_with_no_alignment_result_at_all_is_kept_and_flagged() -> None:
    rec = record(1, lineage="pdm09")
    kept, held, flags = processed.hold_far_from_reference([rec], {}, _ref_check())
    assert (len(kept), len(held)) == (1, 0)
    assert flags == Counter({processed.NO_DISTANCE: 1})


def test_a_near_record_whose_label_is_unexpected_is_kept_and_flagged() -> None:
    # the label never decides: it is blank or disagrees too often to be trusted
    rec = record(1, lineage="seasonal")
    kept, held, flags = processed.hold_far_from_reference(
        [rec], {processed.seq_id(rec): _at(rec, 25)}, _ref_check()
    )
    assert (len(held), flags) == (0, Counter({processed.LABEL_UNEXPECTED: 1}))
    assert processed.LABEL_UNEXPECTED in kept[0].problems


def test_a_far_record_is_held_whatever_its_label_claims() -> None:
    rec = record(1, lineage="pdm09")
    _, held, flags = processed.hold_far_from_reference(
        [rec], {processed.seq_id(rec): _at(rec, 400)}, _ref_check()
    )
    assert [r.epi_isl for r in held] == [rec.epi_isl]
    assert flags == Counter({processed.FAR_FROM_REFERENCE: 1})


def test_a_held_record_keeps_its_problems_for_the_raw_pull() -> None:
    rec = record(1, lineage="pdm09", problems=("name.fields",))
    _, held, _ = processed.hold_far_from_reference(
        [rec], {processed.seq_id(rec): _at(rec, 400)}, _ref_check()
    )
    assert held[0].problems == ("name.fields",)


def test_two_reference_check_rows_for_one_dataset_are_refused(tmp_path: Path) -> None:
    rows = [build.ReferenceCheckConfig("h3", 200, ["pdm09"], "why") for _ in range(2)]
    config = b_config(tmp_path, tmp_path, reference_check=rows)
    with pytest.raises(build.ConfigError, match="named twice"):
        build._hold_far_records(config, {"h3": []}, {})


def test_holding_every_record_of_a_dataset_stops_the_run(tmp_path: Path) -> None:
    # a pull entirely of another lineage would otherwise look stored while storing nothing
    rec = record(1, lineage="pdm09")
    config = b_config(
        tmp_path, tmp_path,
        reference_check=[build.ReferenceCheckConfig("h3", 200, ["pdm09"], "why")],
    )  # fmt: skip
    with pytest.raises(processed.StoreBuildError, match="held every record"):
        build._hold_far_records(
            config, {"h3": [rec]}, {"h3": {processed.seq_id(rec): _at(rec, 400)}}
        )
