"""GISAID pulls: found in gisaid-extractor's output, kept in the store's ``raw/`` kind.

A pull is not reproducible: GISAID edits records, so fetching the same window again
returns different bytes, and each fetch costs hours of serial downloads under a manual
login. So every pull af reads is first copied into ``raw/gisaid/<pull-id>``, which is
never deleted, and everything downstream is built from that copy, never from the
extractor's directory, which its owner may reorganise.

What is kept, per pull, and why:

- the extractor's ``.fas.br``: the delivered FASTA, brotli-compressed (about 65 times
  smaller). For a pull whose headers needed no repair it decompresses to exactly the
  delivered ``.fasta``, and :func:`import_pull` checks that. Where the extractor
  rejoined header fields broken across lines, it is the repaired FASTA, and the
  extractor's rejoin log is kept beside it;
- the metadata workbook (``-metadata.xls``), as delivered: the only source of exact
  collection dates and their precision;
- the rejoin log, when there is one.

The delivered ``.fasta`` itself is recorded in provenance by path and SHA-256, not copied.
"""

from __future__ import annotations

import datetime
import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from af.store.ref import ExternalInput, StoreRef
from af.store.store import Provenance, Store
from af.util.artefacts import sha256_path

RAW_KIND = "raw"
SOURCE = "gisaid"

#: The extractor's names: ``sequences/epiflu-<YYYY-MMDD>-<subtype>.fas.br`` for the kept
#: FASTA, ``raw/epiflu-<subtype>-<YYYYMMDD>-<YYYYMMDD>.fasta`` and ``…-metadata.xls``
#: for what GISAID delivered, and ``metadata/epiflu-<YYYY-MMDD>-<subtype>.rejoined.tsv``.
#:
#: A **targeted** fetch (named isolates, e.g. the tree outgroups) has no submission window, so
#: its names carry a label instead: ``sequences/epiflu-<YYYY-MMDD>-targeted-<label>-<subtype>
#: .fas.br``, ``raw/epiflu-targeted-<label>-<YYYYMMDD>-<subtype>.fasta`` and ``…-metadata.xls``,
#: ``metadata/epiflu-<YYYY-MMDD>-targeted-<label>-<subtype>.rejoined.tsv``. The date is the
#: run date, not a window, and nothing here treats it as one.
_KEPT = re.compile(r"epiflu-(\d{4}-\d{4})-([a-z0-9]+)\.fas\.br")
_KEPT_TARGETED = re.compile(
    r"epiflu-(\d{4}-\d{4})-targeted-([a-z0-9]+(?:-[a-z0-9]+)*)-([a-z0-9]+)\.fas\.br"
)
_DELIVERED = "epiflu-{subtype}-*-{last}.fasta"
_DELIVERED_TARGETED = "epiflu-targeted-{label}-{last}-{subtype}.fasta"
_WORKBOOK_SUFFIX = "-metadata.xls"


class PullFilesError(RuntimeError):
    """A pull whose files cannot be found, paired or trusted."""


@dataclass(frozen=True)
class PullFiles:
    """The files of one pull, as the extractor left them."""

    pull_id: str
    subtype: str  # the extractor's label: h1n1, h3n2, b
    sequences: Path  # the kept .fas.br
    delivered_fasta: Path
    workbook: Path
    rejoined: Path | None

    @property
    def dataset(self) -> str:
        return f"{SOURCE}/{self.pull_id}"


def find_pulls(set_dir: Path, set_name: str) -> list[PullFiles]:
    """Every pull in one extractor set (``definitive``, ``backfill`` …), in name order.

    Pulls are found from the kept FASTAs, not from the extractor's manifests: there is
    one manifest per end date, and where two subtypes share an end date only the last
    one written survives (20 of the 36 definitive pulls are listed, 25 Sep 2026).
    """
    found = []
    for path in sorted((set_dir / "sequences").glob("*.fas.br")):
        if match := _KEPT.fullmatch(path.name):
            stem, subtype = match.groups()
            found.append(_pull_files(set_dir, set_name, stem, subtype, path))
        elif match := _KEPT_TARGETED.fullmatch(path.name):
            stem, label, subtype = match.groups()
            found.append(_pull_files(set_dir, set_name, stem, subtype, path, label))
        else:
            raise PullFilesError(f"{path}: not an extractor sequence file name")
    if not found:
        raise PullFilesError(f"{set_dir}/sequences: no .fas.br pulls")
    return found


def _pull_files(
    set_dir: Path, set_name: str, stem: str, subtype: str, sequences: Path, label: str = ""
) -> PullFiles:
    last = stem.replace("-", "")
    pattern = (
        _DELIVERED_TARGETED.format(label=label, last=last, subtype=subtype)
        if label
        else _DELIVERED.format(subtype=subtype, last=last)
    )
    delivered = sorted((set_dir / "raw").glob(pattern))
    if len(delivered) != 1:
        raise PullFilesError(
            f"{sequences.name}: expected one delivered FASTA ending {last} in {set_dir / 'raw'},"
            f" found {len(delivered)}"
        )
    workbook = delivered[0].with_name(delivered[0].stem + _WORKBOOK_SUFFIX)
    if not workbook.is_file():
        raise PullFilesError(f"{sequences.name}: no metadata workbook {workbook}")
    middle = f"targeted-{label}-" if label else ""
    rejoined = set_dir / "metadata" / f"epiflu-{stem}-{middle}{subtype}.rejoined.tsv"
    return PullFiles(
        # the subtype stays the last token: af.seq.build reads it from there
        pull_id=f"{set_name}-{stem}-{label + '-' if label else ''}{subtype}",
        subtype=subtype,
        sequences=sequences,
        delivered_fasta=delivered[0],
        workbook=workbook,
        rejoined=rejoined if rejoined.is_file() else None,
    )


def import_pull(store: Store, pull: PullFiles) -> StoreRef:
    """Copy one pull into ``raw/gisaid/<pull-id>`` and return its ref.

    Copied, not hard-linked: a published version is made read-only, and a hard link
    would change the permissions of the extractor's own file. Importing the same files
    again publishes nothing new (the version id is the content's).
    """
    started = datetime.datetime.now(datetime.UTC)
    _check_kept_fasta(pull)
    kept: list[Path] = [pull.sequences, pull.workbook]
    if pull.rejoined is not None:
        kept.append(pull.rejoined)
    with store.build(RAW_KIND, pull.dataset) as builder:
        for path in kept:
            shutil.copy2(path, builder.path / path.name)
        provenance = Provenance(
            step="seq.import-pull",
            inputs=tuple(ExternalInput.of(path) for path in [*kept, pull.delivered_fasta]),
            parameters={"pull_id": pull.pull_id, "subtype": pull.subtype},
            started=started,
            finished=datetime.datetime.now(datetime.UTC),
        )
        return builder.publish(provenance, summary={"files": [path.name for path in kept]})


@dataclass(frozen=True)
class RawPull:
    """A pull as the store holds it: the files every later step reads."""

    ref: StoreRef
    pull_id: str
    sequences: Path
    workbook: Path
    rejoined: Path | None


def open_pull(store: Store, pull_id: str) -> RawPull:
    """The CURRENT version of a raw pull, with its files located and checked present."""
    ref = store.current(RAW_KIND, f"{SOURCE}/{pull_id}")
    directory = store.resolve(ref)
    sequences = _one(directory, "*.fas.br")
    workbook = _one(directory, f"*{_WORKBOOK_SUFFIX}")
    rejoined = sorted(directory.glob("*.rejoined.tsv"))
    return RawPull(ref, pull_id, sequences, workbook, rejoined[0] if rejoined else None)


def _one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise PullFilesError(f"{directory}: expected one {pattern}, found {len(matches)}")
    return matches[0]


def _check_kept_fasta(pull: PullFiles) -> None:
    """The kept FASTA must be the delivered one, unless the extractor says it repaired it.

    Without this, a kept file from a different fetch of the same window, or a truncated
    one, would pass for the pull.
    """
    import brotli

    kept = hashlib.sha256(brotli.decompress(pull.sequences.read_bytes())).hexdigest()
    delivered = sha256_path(pull.delivered_fasta)
    if kept != delivered and pull.rejoined is None:
        raise PullFilesError(
            f"{pull.sequences.name} does not decompress to {pull.delivered_fasta.name}, and"
            " there is no rejoin log to say why"
        )
