"""The sequence store build: raw pulls, placement, Parquet partitions, the whole step.

Every name, id, date and sequence here is invented.
"""

from __future__ import annotations

import datetime
import random
from pathlib import Path

import brotli
import pytest

from af.pipeline.config import RunnerSettings
from af.seq import build, processed, pulls
from af.seq.dates import parse as parse_date
from af.seq.gisaid import SequenceRecord, Workbook
from af.seq.nextclade import Aligned
from af.store import PathsConfig, Store, Work
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


def test_store_pull_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Import, align with a stand-in Nextclade, publish; a re-run changes nothing."""
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
    )
    build.import_source(config, "definitive")
    runner = build.make_runner(config.runner)
    refs = build.store_pull(config, "definitive-2021-0312-h3n2", runner)
    table = processed.read_table(store, refs["h3"], "isolates", ["epi_isl", "date_precision"])
    assert table.to_pylist() == [
        {"epi_isl": f"EPI_ISL_{i}", "date_precision": "year"} for i in (1, 2, 3)
    ]
    sequences = processed.read_table(store, refs["h3"], "sequences", ["covers_mature"])
    assert sequences["covers_mature"].to_pylist() == [False] * 3  # the stand-in aligns 30 nt
    assert build.store_pull(config, "definitive-2021-0312-h3n2", runner) == refs


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
