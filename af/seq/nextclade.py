"""Align HA sequences to a pinned Nextclade dataset, in mature-HA numbering.

Nextclade does the alignment; this module makes it safe to rely on. Three things it
does not do on its own, each of which has cost time:

- **It drops sequences and still exits 0.** A sequence containing ``U`` is absent from
  every output; one that fails seed alignment is in the TSV with an ``errors`` value
  and nothing else. :func:`check_complete` therefore compares the ids that went in with
  the ids that came out, and a missing id is fatal (design rule 3). An alignment error
  is not fatal: the record is kept, counted and flagged as not aligned (rule 1b).
- **Its dataset downloader fails behind a TLS-intercepting proxy**, so datasets are
  fetched with a plain HTTP client by path and tag, checked against a pinned SHA-256,
  and kept in the store under ``raw/nextclade/<path>/<tag>`` (:func:`fetch_dataset`).
- **A binary upgrade would change results silently.** The step checks
  ``nextclade --version`` against the configured version and records it in its
  parameters, so a new binary fails loudly until the config names it, and then re-runs.

Mature HA is read from the dataset's own annotation: from the start of ``HA1`` to the
end of ``HA2``, less a terminal stop codon when the reference has one there (the
B lineages do; the A subtypes' annotations stop short of it). No per-subtype offsets
live in code; the expected length is a config value that is checked, not used.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.pipeline.driver import Step, StepContext
from af.run.job import Job, Resources
from af.seq.gisaid import read_fasta
from af.store.ref import StoreError, StoreRef
from af.store.store import Provenance, Store
from af.util.artefacts import Artefact

DATASET_SERVER = "https://data.clades.nextstrain.org/v3"
DATASET_ZIP = "dataset.zip"
DATASET_DIR = "dataset"
#: The CDSs that make up mature HA. SigPep is outside it and is ignored everywhere.
MATURE_CDS = ("HA1", "HA2")
STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})

TSV = "nextclade.tsv"
ALIGNED = "nextclade.aligned.fasta"
SUMMARY = "summary.json"

_GENE_NAME = re.compile(r'gene_name="([^"]+)"')
_VERSION = re.compile(r"nextclade\s+(\S+)")
_ERROR_PREFIX = re.compile(r"^When processing sequence #\d+ '.*?': ")


class AlignmentError(RuntimeError):
    """Nextclade's output does not account for every sequence it was given."""


class DatasetError(RuntimeError):
    """A Nextclade dataset is not the pinned one, or cannot give mature-HA coordinates."""


# ---- datasets ---------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetPin:
    """One Nextclade dataset release: its path on the server, its tag and its hash.

    The hash is of the release's ``dataset.zip``. Releases are immutable, so a
    mismatch means the wrong file, never a newer one.
    """

    path: str
    tag: str
    sha256: str

    def url(self, server: str = DATASET_SERVER) -> str:
        return f"{server}/{self.path}/{self.tag}/{DATASET_ZIP}"

    @property
    def store_key(self) -> str:
        return f"nextclade/{self.path}/{self.tag}"

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> DatasetPin:
        missing = sorted({"path", "tag", "sha256"} - set(data))
        if missing:
            raise DatasetError(f"dataset pin is missing {missing}: {dict(data)}")
        return cls(str(data["path"]), str(data["tag"]), str(data["sha256"]))


def fetch_dataset(pin: DatasetPin, store: Store, *, server: str = DATASET_SERVER) -> StoreRef:
    """Put the pinned dataset in the store (once) and return its ref.

    A dataset already in the store is verified, not fetched again. The download is
    checked against ``pin.sha256`` before anything is published.
    """
    try:
        ref = store.current("raw", pin.store_key)
    except StoreError:
        pass
    else:
        store.verify(ref)
        return ref
    started = datetime.datetime.now(datetime.UTC)
    with store.build("raw", pin.store_key) as builder:
        archive = builder.path / DATASET_ZIP
        download(pin.url(server), pin.sha256, archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(builder.path / DATASET_DIR)
        provenance = Provenance(
            step="fetch-nextclade-dataset",
            inputs=(),
            parameters={
                "url": pin.url(server),
                "path": pin.path,
                "tag": pin.tag,
                "sha256": pin.sha256,
            },
            started=started,
            finished=datetime.datetime.now(datetime.UTC),
        )
        return builder.publish(provenance, summary={"tag": pin.tag})


def download(url: str, sha256: str, target: Path, *, curl: str = "curl") -> None:
    """Fetch ``url`` into ``target`` with curl, and keep it only if it hashes to ``sha256``.

    curl, not Python's HTTP client: behind the TLS-intercepting proxy on the development
    machines, urllib's reads came back short on 8 of 10 fetches of one dataset and a
    ranged resume was refused outright (25 Sep 2026), while curl fetched it intact every
    time. Nextclade's own downloader fails there too. The hash check is what makes any
    fetcher safe to use: a short or altered body cannot pass it.
    """
    subprocess.run(
        [
            curl,
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--retry",
            "3",
            "--output",
            str(target),
            url,
        ],
        check=True,
    )
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != sha256:
        target.unlink()
        raise DatasetError(f"{url}: sha256 {actual} does not match the pinned {sha256}")


@dataclass(frozen=True)
class Reference:
    """Mature-HA coordinates of one dataset's reference, 1-based and inclusive."""

    cds: Mapping[str, tuple[int, int]]
    mature_start: int
    mature_end: int
    #: True when the annotated HA2 ends with the stop codon, which is then not mature HA.
    trimmed_stop: bool

    @property
    def mature_nt(self) -> int:
        return self.mature_end - self.mature_start + 1

    @property
    def mature_aa(self) -> int:
        return self.mature_nt // 3


def read_reference(dataset_dir: Path, expected_mature_nt: int) -> Reference:
    """Mature-HA coordinates from the dataset's annotation and reference sequence.

    ``expected_mature_nt`` (1650 H3, 1647 H1, 1710 B/Vic: seqdb's numbering) is a
    check. A dataset whose annotation disagrees is refused rather than trusted.
    """
    cds = _read_annotation(dataset_dir / "genome_annotation.gff3")
    missing = [name for name in MATURE_CDS if name not in cds]
    if missing:
        raise DatasetError(f"{dataset_dir}: annotation has no {missing}; found {sorted(cds)}")
    reference = "".join(sequence for _, sequence in read_fasta(dataset_dir / "reference.fasta"))
    start, end = cds[MATURE_CDS[0]][0], cds[MATURE_CDS[-1]][1]
    trimmed_stop = reference[end - 3 : end].upper() in STOP_CODONS
    if trimmed_stop:
        end -= 3
    found = Reference(cds=cds, mature_start=start, mature_end=end, trimmed_stop=trimmed_stop)
    if found.mature_nt % 3 or found.mature_nt != expected_mature_nt:
        raise DatasetError(
            f"{dataset_dir}: mature HA is {start}-{end} = {found.mature_nt} nt"
            f"{' after dropping the stop codon' if trimmed_stop else ''};"
            f" config expects {expected_mature_nt}"
        )
    return found


def _read_annotation(path: Path) -> dict[str, tuple[int, int]]:
    cds: dict[str, tuple[int, int]] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        name = _GENE_NAME.search(fields[8]) if len(fields) > 8 else None
        if name:
            cds[name.group(1)] = (int(fields[3]), int(fields[4]))
    return cds


# ---- running ----------------------------------------------------------------------


def nextclade_version(binary: Path) -> str:
    """The version the binary reports, e.g. ``3.23.0``."""
    result = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=True)
    found = _VERSION.search(result.stdout)
    if not found:
        raise RuntimeError(f"{binary} --version said {result.stdout!r}; expected 'nextclade X.Y.Z'")
    return found.group(1)


def check_version(binary: Path, expected: str) -> str:
    actual = nextclade_version(binary)
    if actual != expected:
        raise RuntimeError(
            f"{binary} is nextclade {actual}, config says {expected}. Alignments from a"
            " different version are not comparable: update the config to re-run deliberately."
        )
    return actual


def write_input(records: Iterable[tuple[str, str]], path: Path) -> list[str]:
    """Write ``(id, nucleotides)`` pairs as Nextclade's input and return the ids in order.

    Ids become Nextclade's ``seqName``, the only thing joining its output back to ours,
    so they must be unique and contain no whitespace. A ``U`` is refused here: Nextclade
    would drop the record silently (convert first, as :mod:`af.seq.gisaid` does).
    """
    ids: list[str] = []
    seen: set[str] = set()
    with path.open("w") as out:
        for seq_id, nucleotides in records:
            if not seq_id or any(char.isspace() for char in seq_id):
                raise ValueError(f"sequence id {seq_id!r} is empty or contains whitespace")
            if seq_id in seen:
                raise ValueError(f"sequence id {seq_id!r} appears more than once")
            if "U" in nucleotides.upper():
                raise ValueError(f"{seq_id}: contains U; Nextclade would drop it silently")
            seen.add(seq_id)
            ids.append(seq_id)
            out.write(f">{seq_id}\n{nucleotides}\n")
    if not ids:
        raise ValueError(f"no sequences to write to {path}")
    return ids


def align_job(
    binary: Path, fasta: Path, dataset_dir: Path, out_dir: Path, *, threads: int = 4
) -> Job:
    """The Nextclade command. The input comes first: ``-s``-style options that take
    several values would otherwise swallow it."""
    return Job(
        name=f"nextclade-{fasta.stem}",
        command=[binary, "run", fasta, "-D", dataset_dir, "-O", out_dir, "-j", str(threads)],
        cwd=out_dir,
        log=out_dir / "nextclade.log",
        outputs=[Artefact(out_dir / TSV), Artefact(out_dir / ALIGNED, allow_empty=True)],
        resources=Resources(threads=threads),
    )


# ---- reading the output -----------------------------------------------------------


@dataclass
class AlignmentCounts:
    sequences: int = 0
    aligned: int = 0
    not_aligned: int = 0
    errors: Counter[str] = field(default_factory=Counter)

    def to_json(self) -> dict[str, Any]:
        return {
            "sequences": self.sequences,
            "aligned": self.aligned,
            "not_aligned": self.not_aligned,
            "errors": dict(self.errors),
        }


def check_complete(ids: Sequence[str], out_dir: Path) -> AlignmentCounts:
    """Every id that went in is in the TSV, once; aligned ones are in the alignment.

    Raises :class:`AlignmentError` naming the ids that vanished or appeared.
    """
    rows = _read_tsv(out_dir / TSV)
    names = Counter(row["seqName"] for row in rows)
    aligned_ids = [name for name, _ in read_fasta(out_dir / ALIGNED)]
    problems = []
    absent = [seq_id for seq_id in ids if seq_id not in names]
    if absent:
        problems.append(f"{len(absent)} input sequence(s) absent from {TSV}: {_first(absent)}")
    unknown = sorted(set(names) - set(ids))
    if unknown:
        problems.append(f"{len(unknown)} unexpected id(s) in {TSV}: {_first(unknown)}")
    repeated = sorted(name for name, n in names.items() if n > 1)
    if repeated:
        problems.append(f"{len(repeated)} id(s) repeated in {TSV}: {_first(repeated)}")
    failed = {row["seqName"] for row in rows if row.get("errors")}
    expected_aligned = set(names) - failed
    if set(aligned_ids) != expected_aligned or len(aligned_ids) != len(expected_aligned):
        lost = sorted(expected_aligned - set(aligned_ids))
        problems.append(
            f"{ALIGNED} has {len(aligned_ids)} sequences, {TSV} says {len(expected_aligned)}"
            f" aligned{f'; missing {_first(lost)}' if lost else ''}"
        )
    if problems:
        raise AlignmentError(f"{out_dir}: " + "; ".join(problems))
    counts = AlignmentCounts(
        sequences=len(ids), aligned=len(expected_aligned), not_aligned=len(failed)
    )
    for row in rows:
        if row.get("errors"):
            counts.errors[_error_kind(row["errors"])] += 1
    return counts


@dataclass(frozen=True)
class Aligned:
    """One sequence's alignment, reduced to mature HA, with the facts R3 needs.

    ``nucleotides`` and ``amino_acids`` are None when Nextclade could not align the
    sequence (``error`` says why) or could not translate a mature CDS (``failed_cds``).
    """

    seq_id: str
    error: str | None
    alignment_start: int | None
    alignment_end: int | None
    covers_mature: bool
    nucleotides: str | None
    amino_acids: str | None
    failed_cds: tuple[str, ...]
    frameshifts: int
    deleted_aa: int
    inserted_aa: int
    unknown_aa: int
    premature_stop: bool
    qc_status: str


def read_alignment(out_dir: Path, reference: Reference) -> Iterator[Aligned]:
    """Yield every sequence in the TSV, in TSV order, reduced to mature HA."""
    aligned = dict(read_fasta(out_dir / ALIGNED))
    translations = {
        cds: dict(read_fasta(out_dir / f"nextclade.cds_translation.{cds}.fasta"))
        for cds in MATURE_CDS
        if (out_dir / f"nextclade.cds_translation.{cds}.fasta").exists()
    }
    for row in _read_tsv(out_dir / TSV):
        yield _reduce(row, aligned, translations, reference)


def _reduce(
    row: Mapping[str, str],
    aligned: Mapping[str, str],
    translations: Mapping[str, Mapping[str, str]],
    reference: Reference,
) -> Aligned:
    seq_id = row["seqName"]
    error = row.get("errors") or None
    start = _int(row.get("alignmentStart"))
    end = _int(row.get("alignmentEnd"))
    covers = (
        error is None
        and start is not None
        and end is not None
        and start <= reference.mature_start
        and end >= reference.mature_end
    )
    nucleotides = amino_acids = None
    failed = tuple(
        sorted(cds for cds in _cds_list(row.get("failedCdses", "")) if cds in MATURE_CDS)
    )
    if error is None:
        full = aligned[seq_id]
        nucleotides = full[reference.mature_start - 1 : reference.mature_end]
        pieces = [translations.get(cds, {}).get(seq_id) for cds in MATURE_CDS]
        if all(piece is not None for piece in pieces) and not failed:
            amino_acids = "".join(piece for piece in pieces if piece)[: reference.mature_aa]
    return Aligned(
        seq_id=seq_id,
        error=error,
        alignment_start=start,
        alignment_end=end,
        covers_mature=covers,
        nucleotides=nucleotides,
        amino_acids=amino_acids,
        failed_cds=failed,
        frameshifts=_count_in_mature(row.get("frameShifts", "")),
        deleted_aa=_count_in_mature(row.get("aaDeletions", "")),
        inserted_aa=_inserted_aa_in_mature(row.get("aaInsertions", "")),
        unknown_aa=amino_acids.count("X") if amino_acids else 0,
        premature_stop="*" in amino_acids if amino_acids else False,
        qc_status=row.get("qc.overallStatus", ""),
    )


# ---- R3: the sequence-QC rule -----------------------------------------------------


@dataclass(frozen=True)
class QcThresholds:
    """R3's limits (DECISIONS 24 Sep, option C: today's ae thresholds, 10 and 6, applied
    to Nextclade's alignment). Always from config; there are no defaults."""

    max_unknown_aa: int
    max_deleted_aa: int


def qc_failures(record: Aligned, limits: QcThresholds) -> list[str]:
    """R3's reasons for failing a record, empty if it passes.

    Frameshifts and insertions count only inside HA1/HA2 (SigPep is not mature HA);
    Nextclade's own QC status is reported alongside, never used, because it rates an
    849-nt internal deletion ``good``.
    """
    if record.error is not None:
        return ["not-aligned"]
    reasons = []
    if not record.covers_mature:
        reasons.append("incomplete")
    if record.failed_cds or record.amino_acids is None:
        reasons.append("cds-failed")  # no mature protein, whatever the cause
    if record.frameshifts:
        reasons.append("frameshift")
    if record.unknown_aa > limits.max_unknown_aa:
        reasons.append("unknown-aa")
    if record.deleted_aa > limits.max_deleted_aa:
        reasons.append("deleted-aa")
    if record.inserted_aa:
        reasons.append("aa-insertion")
    if record.premature_stop:
        reasons.append("premature-stop")
    return reasons


# ---- the pipeline step ------------------------------------------------------------


def align_step(name: str, parameters: Mapping[str, Any]) -> Step:
    """A pipeline step that aligns one FASTA against one pinned dataset.

    Parameters (all required): ``binary``, ``nextclade_version``, ``fasta``,
    ``dataset_dir``, ``out_dir``, ``expected_mature_nt``, ``max_unknown_aa``,
    ``max_deleted_aa``; optional ``threads``.
    The version check runs now, so a wrong binary fails before anything runs.
    """
    required = [
        "binary",
        "nextclade_version",
        "fasta",
        "dataset_dir",
        "out_dir",
        "expected_mature_nt",
        "max_unknown_aa",
        "max_deleted_aa",
    ]
    missing = [key for key in required if key not in parameters]
    if missing:
        raise ValueError(f"step {name!r}: missing parameter(s) {missing}")
    binary = Path(parameters["binary"])
    version = check_version(binary, str(parameters["nextclade_version"]))
    fasta = Path(parameters["fasta"])
    dataset_dir = Path(parameters["dataset_dir"])
    out_dir = Path(parameters["out_dir"])
    expected = int(parameters["expected_mature_nt"])
    threads = int(parameters.get("threads", 4))
    limits = QcThresholds(int(parameters["max_unknown_aa"]), int(parameters["max_deleted_aa"]))

    def action(context: StepContext) -> None:
        reference = read_reference(dataset_dir, expected)
        ids = [seq_id for seq_id, _ in read_fasta(fasta)]
        if out_dir.exists():
            shutil.rmtree(out_dir)  # never let a previous run's files pass for this one's
        out_dir.mkdir(parents=True)
        context.runner.run(align_job(binary, fasta, dataset_dir, out_dir, threads=threads))
        counts = check_complete(ids, out_dir)
        qc = Counter(
            reason
            for record in read_alignment(out_dir, reference)
            for reason in qc_failures(record, limits) or ["pass"]
        )
        summary = {
            "nextclade_version": version,
            "reference": _reference_json(reference),
            "counts": counts.to_json(),
            "r3": dict(sorted(qc.items())),
        }
        (out_dir / SUMMARY).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    return Step(
        name=name,
        action=action,
        inputs={"fasta": fasta, "dataset": dataset_dir},
        outputs=[Artefact(out_dir, parse=lambda path: json.loads((path / SUMMARY).read_text()))],
        parameters={
            "nextclade_version": version,
            "expected_mature_nt": expected,
            "max_unknown_aa": limits.max_unknown_aa,
            "max_deleted_aa": limits.max_deleted_aa,
        },
    )


def _reference_json(reference: Reference) -> dict[str, Any]:
    return {
        "mature_start": reference.mature_start,
        "mature_end": reference.mature_end,
        "trimmed_stop": reference.trimmed_stop,
        "cds": {name: list(span) for name, span in reference.cds.items()},
    }


# ---- small parsers ----------------------------------------------------------------


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _cds_list(text: str) -> list[str]:
    """``HA1:G200-,HA2:I2-`` -> ``["HA1", "HA2"]``: the CDS of each comma-separated entry.

    Also takes bare names (``failedCdses`` is ``HA1,HA2``)."""
    return [entry.split(":", 1)[0].strip() for entry in text.split(",") if entry.strip()]


def _count_in_mature(text: str) -> int:
    return sum(1 for cds in _cds_list(text) if cds in MATURE_CDS)


def _inserted_aa_in_mature(text: str) -> int:
    """``HA1:123:KL`` is two inserted residues after HA1 position 123."""
    total = 0
    for entry in text.split(","):
        parts = entry.split(":")
        if len(parts) == 3 and parts[0] in MATURE_CDS:
            total += len(parts[2])
    return total


def _error_kind(error: str) -> str:
    """Group Nextclade's error messages for counting: drop the per-sequence prefix
    (``When processing sequence #N '<id>': ``) and keep the first sentence."""
    message = _ERROR_PREFIX.sub("", error)
    return message.split(".", 1)[0].strip()[:80]


def _int(text: str | None) -> int | None:
    return int(text) if text else None


def _first(items: Sequence[str], n: int = 5) -> str:
    return ", ".join(items[:n]) + (" …" if len(items) > n else "")
