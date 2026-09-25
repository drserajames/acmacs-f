"""All sequences here are invented: random bases, not any real virus.

Most tests write Nextclade-shaped output by hand, so they need no binary. The
end-to-end test runs the real binary against an invented reference and skips when
``nextclade`` is not on PATH.
"""

from __future__ import annotations

import hashlib
import random
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

from af.pipeline.driver import Pipeline
from af.run import LocalRunner
from af.seq import nextclade as nc
from af.store.store import Store

TSV_COLUMNS = [
    "index", "seqName", "qc.overallStatus", "alignmentStart", "alignmentEnd",
    "frameShifts", "aaDeletions", "aaInsertions", "failedCdses", "errors",
]  # fmt: skip

# An invented reference: SigPep 1-6, HA1 7-18, HA2 19-30 (the last codon a stop).
REFERENCE = nc.Reference(
    cds={"SigPep": (1, 6), "HA1": (7, 18), "HA2": (19, 30)},
    mature_start=7,
    mature_end=27,
    trimmed_stop=True,
)


def tsv_row(seq_name: str, **over: str) -> dict[str, str]:
    row = dict.fromkeys(TSV_COLUMNS, "")
    row.update(
        seqName=seq_name, alignmentStart="1", alignmentEnd="30", **{"qc.overallStatus": "good"}
    )
    row.update(over)
    return row


def write_output(
    out: Path,
    rows: list[dict[str, str]],
    aligned: dict[str, str],
    translations: dict[str, dict[str, str]] | None = None,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(TSV_COLUMNS)]
    lines += ["\t".join(row[column] for column in TSV_COLUMNS) for row in rows]
    (out / nc.TSV).write_text("\n".join(lines) + "\n")
    (out / nc.ALIGNED).write_text("".join(f">{k}\n{v}\n" for k, v in aligned.items()))
    for cds, records in (translations or {}).items():
        path = out / f"nextclade.cds_translation.{cds}.fasta"
        path.write_text("".join(f">{k}\n{v}\n" for k, v in records.items()))


def one(out: Path, **over: str) -> nc.Aligned:
    """Reduce a single well-formed record, with TSV fields overridden."""
    write_output(
        out,
        [tsv_row("s1", **over)],
        {"s1": "ATGATG" + "ACGTACGTACGT" + "ACGTACGTATAA"},
        {"HA1": {"s1": "TYVR"}, "HA2": {"s1": "TYV*"}},
    )
    return next(nc.read_alignment(out, REFERENCE))


LIMITS = nc.QcThresholds(max_unknown_aa=1, max_deleted_aa=2)


class TestReduce:
    def test_mature_slice_and_terminal_stop_dropped(self, tmp_path: Path) -> None:
        record = one(tmp_path)
        assert record.nucleotides == "ACGTACGTACGTACGTACGTA"  # 7..27, SigPep and stop gone
        assert record.amino_acids == "TYVRTYV"
        assert (record.covers_mature, record.premature_stop) == (True, False)
        assert nc.qc_failures(record, LIMITS) == []

    def test_sigpep_frameshift_and_insertion_are_ignored(self, tmp_path: Path) -> None:
        """SigPep is outside mature HA; counting it failed ~30 H1 records ae keeps."""
        record = one(tmp_path, frameShifts="SigPep:1-2", aaInsertions="SigPep:1:KL")
        assert (record.frameshifts, record.inserted_aa) == (0, 0)

    def test_mature_frameshift_insertion_and_deletions_count(self, tmp_path: Path) -> None:
        record = one(
            tmp_path,
            frameShifts="HA1:2-3",
            aaInsertions="HA2:1:KL,SigPep:1:A",
            aaDeletions="HA1:T1-,HA1:Y2-,HA2:T1-",
        )
        assert (record.frameshifts, record.inserted_aa, record.deleted_aa) == (1, 2, 3)
        assert nc.qc_failures(record, LIMITS) == ["frameshift", "deleted-aa", "aa-insertion"]

    def test_partial_coverage_is_incomplete(self, tmp_path: Path) -> None:
        record = one(tmp_path, alignmentEnd="20")
        assert nc.qc_failures(record, LIMITS) == ["incomplete"]

    def test_unknown_aa_and_premature_stop(self, tmp_path: Path) -> None:
        write_output(
            tmp_path,
            [tsv_row("s1")],
            {"s1": "A" * 30},
            {"HA1": {"s1": "XX*R"}, "HA2": {"s1": "TYV*"}},
        )
        record = next(nc.read_alignment(tmp_path, REFERENCE))
        assert (record.unknown_aa, record.premature_stop) == (2, True)
        assert nc.qc_failures(record, LIMITS) == ["unknown-aa", "premature-stop"]

    def test_failed_cds_has_no_protein(self, tmp_path: Path) -> None:
        write_output(tmp_path, [tsv_row("s1", failedCdses="HA2")], {"s1": "A" * 30},
                     {"HA1": {"s1": "TYVR"}})  # fmt: skip
        record = next(nc.read_alignment(tmp_path, REFERENCE))
        assert record.amino_acids is None
        assert "cds-failed" in nc.qc_failures(record, LIMITS)

    def test_alignment_error_is_kept_and_flagged(self, tmp_path: Path) -> None:
        write_output(tmp_path, [tsv_row("s1", errors="Unable to align: seed", alignmentStart="",
                                        alignmentEnd="")], {})  # fmt: skip
        record = next(nc.read_alignment(tmp_path, REFERENCE))
        assert (record.nucleotides, record.error) == (None, "Unable to align: seed")
        assert nc.qc_failures(record, LIMITS) == ["not-aligned"]


class TestCheckComplete:
    def test_counts_aligned_and_not_aligned(self, tmp_path: Path) -> None:
        write_output(
            tmp_path,
            [tsv_row("s1"), tsv_row("s2", errors="Unable to align: seed")],
            {"s1": "A" * 30},
        )
        counts = nc.check_complete(["s1", "s2"], tmp_path)
        assert (counts.sequences, counts.aligned, counts.not_aligned) == (2, 1, 1)
        assert counts.errors == {"Unable to align: seed": 1}

    def test_a_sequence_absent_from_the_output_is_fatal(self, tmp_path: Path) -> None:
        """Nextclade drops a sequence containing U from every file and exits 0."""
        write_output(tmp_path, [tsv_row("s1")], {"s1": "A" * 30})
        with pytest.raises(nc.AlignmentError, match="1 input sequence.*absent.*s2"):
            nc.check_complete(["s1", "s2"], tmp_path)

    def test_an_unexpected_id_is_fatal(self, tmp_path: Path) -> None:
        write_output(tmp_path, [tsv_row("s1"), tsv_row("s9")], {"s1": "A", "s9": "A"})
        with pytest.raises(nc.AlignmentError, match="unexpected.*s9"):
            nc.check_complete(["s1"], tmp_path)

    def test_aligned_but_missing_from_the_alignment_is_fatal(self, tmp_path: Path) -> None:
        write_output(tmp_path, [tsv_row("s1"), tsv_row("s2")], {"s1": "A" * 30})
        with pytest.raises(nc.AlignmentError, match="missing s2"):
            nc.check_complete(["s1", "s2"], tmp_path)


class TestWriteInput:
    def test_writes_and_returns_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "in.fasta"
        assert nc.write_input([("EPI_ISL_1/EPI9", "ACGT")], path) == ["EPI_ISL_1/EPI9"]
        assert path.read_text() == ">EPI_ISL_1/EPI9\nACGT\n"

    @pytest.mark.parametrize(
        ("records", "message"),
        [
            ([("a", "ACGU")], "contains U"),
            ([("a", "ACGT"), ("a", "ACGT")], "more than once"),
            ([("a b", "ACGT")], "whitespace"),
            ([], "no sequences"),
        ],
    )
    def test_refuses(self, tmp_path: Path, records: list[tuple[str, str]], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            nc.write_input(records, tmp_path / "in.fasta")


def make_dataset(directory: Path, rng: random.Random) -> str:
    """An invented dataset: random reference, HA2 ending in a stop codon. Returns the reference."""
    codons = [c for c in (a + b + d for a in "ACGT" for b in "ACGT" for d in "ACGT")
              if c not in nc.STOP_CODONS]  # fmt: skip
    body = "".join(rng.choice(codons) for _ in range(199))
    reference = "ATG" * 5 + body + "TAA" + "ACGTACGTAC"  # SigPep 15 nt, mature 597, stop
    directory.mkdir(parents=True)
    (directory / "reference.fasta").write_text(f">EXAMPLE_REF\n{reference}\n")
    end = 15 + 600
    (directory / "genome_annotation.gff3").write_text(
        "##gff-version 3\n"
        f"##sequence-region EXAMPLE_REF 1 {len(reference)}\n"
        'EXAMPLE_REF\tfeature\tgene\t1\t15\t.\t+\t.\tgene_name="SigPep"\n'
        'EXAMPLE_REF\tfeature\tgene\t16\t315\t.\t+\t.\tgene_name="HA1"\n'
        f'EXAMPLE_REF\tfeature\tgene\t316\t{end}\t.\t+\t.\tgene_name="HA2"\n'
    )
    (directory / "pathogen.json").write_text(
        '{"schemaVersion": "3.0.0", "files": {"reference": "reference.fasta",'
        ' "genomeAnnotation": "genome_annotation.gff3", "pathogenJson": "pathogen.json"}}\n'
    )
    return reference


class TestReference:
    def test_terminal_stop_is_not_mature(self, tmp_path: Path) -> None:
        make_dataset(tmp_path / "ds", random.Random(1))
        reference = nc.read_reference(tmp_path / "ds", 597)
        assert (reference.mature_start, reference.mature_end) == (16, 612)
        assert (reference.trimmed_stop, reference.mature_aa) == (True, 199)

    def test_length_disagreeing_with_config_is_refused(self, tmp_path: Path) -> None:
        make_dataset(tmp_path / "ds", random.Random(1))
        with pytest.raises(nc.DatasetError, match="597 nt after dropping the stop.*expects 600"):
            nc.read_reference(tmp_path / "ds", 600)


def zipped(directory: Path, target: Path) -> str:
    with zipfile.ZipFile(target, "w") as bundle:
        for path in sorted(directory.iterdir()):
            bundle.write(path, path.name)
    return hashlib.sha256(target.read_bytes()).hexdigest()


class TestFetch:
    def test_fetch_once_then_reuse(self, tmp_path: Path) -> None:
        make_dataset(tmp_path / "src", random.Random(2))
        release = tmp_path / "server" / "example" / "ha" / "2026-01-01"
        release.mkdir(parents=True)
        sha = zipped(tmp_path / "src", release / nc.DATASET_ZIP)
        pin = nc.DatasetPin("example/ha", "2026-01-01", sha)
        store = Store.create(tmp_path / "store")
        server = (tmp_path / "server").as_uri()

        ref = nc.fetch_dataset(pin, store, server=server)
        assert ref.dataset == "nextclade/example/ha/2026-01-01"
        assert (store.resolve(ref) / nc.DATASET_DIR / "reference.fasta").is_file()

        shutil.rmtree(tmp_path / "server")  # a second fetch must not need the server
        assert nc.fetch_dataset(pin, store, server=server) == ref

    def test_wrong_hash_is_refused_and_nothing_published(self, tmp_path: Path) -> None:
        make_dataset(tmp_path / "src", random.Random(2))
        release = tmp_path / "server" / "example" / "ha" / "t"
        release.mkdir(parents=True)
        zipped(tmp_path / "src", release / nc.DATASET_ZIP)
        store = Store.create(tmp_path / "store")
        with pytest.raises(nc.DatasetError, match="does not match the pinned"):
            nc.fetch_dataset(nc.DatasetPin("example/ha", "t", "0" * 64), store,
                             server=(tmp_path / "server").as_uri())  # fmt: skip
        assert store.list_datasets("raw") == []


FAKE_NEXTCLADE = """#!{python}
# Stands in for nextclade: reports a version, and aligns by copying, dropping `{drop}`.
import sys
from pathlib import Path
if sys.argv[1] == "--version":
    print("nextclade {version}")
    sys.exit(0)
fasta, out = Path(sys.argv[2]), Path(sys.argv[sys.argv.index("-O") + 1])
names = [line[1:].strip() for line in fasta.read_text().splitlines() if line.startswith(">")]
names = [name for name in names if name != "{drop}"]
columns = {columns!r}
rows = ["\\t".join(columns)]
for i, name in enumerate(names):
    rows.append("\\t".join([str(i), name, "good", "1", "30", "", "", "", "", ""]))
(out / "nextclade.tsv").write_text("\\n".join(rows) + "\\n")
(out / "nextclade.aligned.fasta").write_text("".join(">%s\\n%s\\n" % (n, "A" * 30) for n in names))
"""


def fake_binary(tmp_path: Path, *, version: str = "9.9.9", drop: str = "") -> Path:
    path = tmp_path / "nextclade"
    path.write_text(FAKE_NEXTCLADE.format(python=sys.executable, version=version, drop=drop,
                                          columns=TSV_COLUMNS))  # fmt: skip
    path.chmod(0o755)
    return path


def step_parameters(tmp_path: Path, binary: Path, **over: object) -> dict[str, object]:
    make_dataset(tmp_path / "ds", random.Random(3))
    fasta = tmp_path / "in.fasta"
    nc.write_input([("s1", "ACGT"), ("s2", "ACGT")], fasta)
    parameters: dict[str, object] = dict(
        binary=binary, nextclade_version="9.9.9", fasta=fasta, dataset_dir=tmp_path / "ds",
        out_dir=tmp_path / "out", expected_mature_nt=597, max_unknown_aa=10, max_deleted_aa=6,
    )  # fmt: skip
    parameters.update(over)
    return parameters


class TestStep:
    def test_version_mismatch_fails_before_running(self, tmp_path: Path) -> None:
        binary = fake_binary(tmp_path, version="9.9.8")
        with pytest.raises(RuntimeError, match="nextclade 9.9.8, config says 9.9.9"):
            nc.align_step("align", step_parameters(tmp_path, binary))

    def test_missing_parameter_is_fatal(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_deleted_aa"):
            nc.align_step("align", {"binary": "x", "nextclade_version": "1", "fasta": "x",
                                    "dataset_dir": "x", "out_dir": "x",
                                    "expected_mature_nt": 3, "max_unknown_aa": 1})  # fmt: skip

    def test_a_silently_dropped_sequence_fails_the_step(self, tmp_path: Path) -> None:
        step = nc.align_step("align", step_parameters(tmp_path, fake_binary(tmp_path, drop="s2")))
        pipeline = Pipeline([step], state_dir=tmp_path / "state", runner=LocalRunner())
        with pytest.raises(nc.AlignmentError, match="absent.*s2"):
            pipeline.run()

    def test_stale_output_is_removed_before_a_run(self, tmp_path: Path) -> None:
        parameters = step_parameters(tmp_path, fake_binary(tmp_path))
        stale = tmp_path / "out" / "nextclade.cds_translation.HA1.fasta"
        stale.parent.mkdir()
        stale.write_text(">old\nX\n")
        pipeline = Pipeline([nc.align_step("align", parameters)], state_dir=tmp_path / "state",
                            runner=LocalRunner())  # fmt: skip
        assert [outcome.status for outcome in pipeline.run()] == ["ran"]
        assert not stale.exists()


@pytest.mark.skipif(shutil.which("nextclade") is None, reason="nextclade not on PATH")
def test_real_nextclade_end_to_end(tmp_path: Path) -> None:
    """The real binary on an invented reference: a copy, a codon deletion, and junk."""
    rng = random.Random(4)
    reference = make_dataset(tmp_path / "ds", rng)
    binary = Path(str(shutil.which("nextclade")))
    version = nc.nextclade_version(binary)
    deleted = reference[:300] + reference[303:]  # one codon out of HA1
    junk = "".join(rng.choice("ACGT") for _ in range(len(reference)))
    fasta = tmp_path / "in.fasta"
    nc.write_input([("same", reference), ("deleted", deleted), ("junk", junk)], fasta)
    parameters = step_parameters(tmp_path / "unused", binary, nextclade_version=version,
                                 fasta=fasta, dataset_dir=tmp_path / "ds")  # fmt: skip
    Pipeline([nc.align_step("align", parameters)], state_dir=tmp_path / "state",
             runner=LocalRunner()).run()  # fmt: skip

    records = {
        r.seq_id: r
        for r in nc.read_alignment(
            parameters["out_dir"],  # type: ignore[arg-type]
            nc.read_reference(tmp_path / "ds", 597),
        )
    }
    assert records["same"].nucleotides == reference[15:612]
    assert nc.qc_failures(records["same"], LIMITS) == []
    assert records["deleted"].deleted_aa == 1
    assert records["junk"].error is not None


def test_errors_are_grouped_by_kind_not_by_sequence(tmp_path: Path) -> None:
    """Nextclade prefixes every error with the sequence's id; counting must not."""
    write_output(
        tmp_path,
        [
            tsv_row("s1", errors="When processing sequence #1 's1': Unknown nucleotide: L"),
            tsv_row("s2", errors="When processing sequence #2 's2': Unknown nucleotide: L"),
            tsv_row("s3", errors="When processing sequence #3 's3': Unable to align: seed. More"),
        ],
        {},
    )
    counts = nc.check_complete(["s1", "s2", "s3"], tmp_path)
    assert counts.errors == {"Unknown nucleotide: L": 2, "Unable to align: seed": 1}
