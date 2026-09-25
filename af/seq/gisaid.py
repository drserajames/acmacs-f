"""Read one GISAID pull: the metadata workbook and the sequences, joined.

A pull is a download of one subtype over one submission-date window, as
gisaid-extractor leaves it: a FASTA of sequences and a metadata workbook. Those two
files say different things, and af needs both.

**The workbook is authoritative for metadata, the FASTA for sequence.** This is not a
preference. GISAID expands a partial collection date to 1 January *in the defline it
serves*, so a virus whose submitter stated only a year arrives in the FASTA as
1 January and is then indistinguishable from one genuinely collected that day. The
workbook keeps what was stated. Anything built from deflines — including the old
pipeline — inherits the expansion, which is where the false 1-January cluster on a
tree's time axis comes from. The workbook also carries the host, the structured
location and the embargo column, none of which are in the defline at all.

The workbook has one row per **isolate**; the FASTA has one record per **sequence**.
An isolate can carry more than one HA sequence, so the join is one-to-many and the
key is (EPI_ISL, segment accession) — EPI_ISL alone is not unique.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from af.seq import names
from af.seq.dates import CollectionDate, DateProblem
from af.seq.dates import parse as parse_date

#: Defline key for the segment accession, which is the per-sequence half of the key.
ACCESSION_KEY = "o"
#: Defline key for the isolate id.
ISOLATE_KEY = "a"
#: Defline key for the collection date. Read only to *check* the workbook, never to use.
DEFLINE_DATE_KEY = "e"

_FIELD = re.compile(r"_\|_|\|")
_UNKNOWN = re.compile(r"N")
_AMBIGUOUS = re.compile(r"[^ACGTN]")  # R, Y, W … : a stated ambiguity, unlike N


class PullError(RuntimeError):
    """A pull that cannot be read as a consistent pair of files."""


@dataclass(frozen=True)
class SequenceRecord:
    """One sequence and the isolate metadata that belongs with it."""

    epi_isl: str
    accession: str
    gisaid_name: str
    name: str
    nucleotides: str
    collection_date: CollectionDate | None
    subtype: str
    lineage: str
    host: str
    passage: str
    location: str
    originating_lab: str
    submitting_lab: str
    submission_date: str
    embargoed_until: str
    problems: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return (self.epi_isl, self.accession)


@dataclass
class PullCounts:
    """What a read of one pull did, so a step can report it (design rule 1 and 3)."""

    sequences: int = 0
    isolates: int = 0
    uracil_converted: int = 0
    date_precision: Counter[str] = field(default_factory=Counter)
    defline_date_differs: int = 0
    unreadable_date: int = 0
    name_problems: Counter[str] = field(default_factory=Counter)
    #: N, i.e. "not known". Counted apart from R/Y/W…, which state a real ambiguity;
    #: Nextclade's QC treats the two differently and so must any threshold of ours.
    unknown_bases: int = 0
    ambiguous_bases: int = 0

    def to_json(self) -> dict[str, object]:
        return {
            "sequences": self.sequences,
            "isolates": self.isolates,
            "uracil_converted": self.uracil_converted,
            "date_precision": dict(self.date_precision),
            "defline_date_differs": self.defline_date_differs,
            "unreadable_date": self.unreadable_date,
            "name_problems": dict(self.name_problems),
            "unknown_bases": self.unknown_bases,
            "ambiguous_bases": self.ambiguous_bases,
        }


def parse_defline(defline: str) -> tuple[str, dict[str, str]]:
    """Split a defline into its name and its ``key=value`` fields.

    GISAID is asked for ``_|_`` and sometimes returns ``|``, so both are accepted.
    """
    parts = _FIELD.split(defline.lstrip(">"))
    fields: dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, _, value = part.partition("=")
            fields[key.strip()] = value.strip()
    return parts[0].strip(), fields


def read_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(defline, sequence)`` for a plain FASTA."""
    defline: str | None = None
    chunks: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line.startswith(">"):
                if defline is not None:
                    yield defline, "".join(chunks)
                defline, chunks = line[1:], []
            elif defline is not None:
                chunks.append(line.strip())
    if defline is not None:
        yield defline, "".join(chunks)


def join(
    records: Iterable[tuple[str, str]],
    workbook_rows: Sequence[dict[str, str]],
    *,
    subtype: str | None = None,
) -> tuple[list[SequenceRecord], PullCounts]:
    """Join FASTA records to workbook rows on the isolate id.

    Every sequence must have a workbook row: without one af has no host, no structured
    location and no stated date precision, and silently keeping such a record would
    reintroduce exactly the defline-only metadata this module exists to avoid. Missing
    rows raise rather than being dropped (design rule 3).
    """
    by_isolate = {row["Isolate_Id"].strip(): row for row in workbook_rows if row.get("Isolate_Id")}
    counts = PullCounts(isolates=len(by_isolate))
    out: list[SequenceRecord] = []
    missing: list[str] = []
    no_accession: list[str] = []

    for defline, sequence in records:
        gisaid_name, fields = parse_defline(defline)
        epi_isl = fields.get(ISOLATE_KEY, "")
        if not fields.get(ACCESSION_KEY, "").strip():
            # The accession is half the key. A header split across lines (older records)
            # loses its later fields, and an empty accession would still make a valid-looking
            # key, so this is fatal rather than defaulted.
            no_accession.append(epi_isl or gisaid_name)
            continue
        row = by_isolate.get(epi_isl)
        if row is None:
            missing.append(epi_isl or gisaid_name)
            continue

        counts.sequences += 1
        nucleotides, converted = _normalise_nucleotides(sequence)
        counts.uracil_converted += converted
        counts.unknown_bases += len(_UNKNOWN.findall(nucleotides))
        counts.ambiguous_bases += len(_AMBIGUOUS.findall(nucleotides))

        problems: list[str] = []
        collection_date = None
        stated = row.get("Collection_Date", "").strip()
        try:
            collection_date = parse_date(stated)
        except DateProblem:
            counts.unreadable_date += 1
            problems.append("date.unreadable")
        else:
            counts.date_precision[collection_date.precision.value] += 1
            if fields.get(DEFLINE_DATE_KEY, "").strip() != str(collection_date):
                # Expected for every partial date: the defline says 1 January where the
                # workbook says a month or a year. Counted, not a problem.
                counts.defline_date_differs += 1

        name = names.normalise(gisaid_name, subtype)
        for problem in name.problems:
            counts.name_problems[problem] += 1
        problems.extend(name.problems)

        out.append(
            SequenceRecord(
                epi_isl=epi_isl,
                accession=fields.get(ACCESSION_KEY, "").strip(),
                gisaid_name=gisaid_name,
                name=name.name,
                nucleotides=nucleotides,
                collection_date=collection_date,
                subtype=row.get("Subtype", "").strip(),
                lineage=row.get("Lineage", "").strip(),
                host=row.get("Host", "").strip(),
                passage=row.get("Passage_History", "").strip(),
                location=row.get("Location", "").strip(),
                originating_lab=row.get("Originating_Lab", "").strip(),
                submitting_lab=row.get("Submitting_Lab", "").strip(),
                submission_date=row.get("Submission_Date", "").strip(),
                embargoed_until=row.get("Publishing_Embargo_Until", "").strip(),
                problems=tuple(problems),
            )
        )

    if no_accession:
        shown = ", ".join(sorted(no_accession)[:5])
        raise PullError(
            f"{len(no_accession)} defline(s) have no segment accession ({ACCESSION_KEY}=), so"
            f" they have no store key; a header broken across lines? {shown}"
            f"{' …' if len(no_accession) > 5 else ''}"
        )
    if missing:
        shown = ", ".join(sorted(missing)[:5])
        raise PullError(
            f"{len(missing)} sequence(s) have no metadata row, so their host, location and"
            f" date precision are unknowable: {shown}"
            f"{' …' if len(missing) > 5 else ''}"
        )
    _check_keys_unique(out)
    return out, counts


def _check_keys_unique(records: Sequence[SequenceRecord]) -> None:
    """(EPI_ISL, accession) must identify one sequence: it is the store's key."""
    seen = Counter(record.key for record in records)
    repeated = [key for key, n in seen.items() if n > 1]
    if repeated:
        shown = ", ".join(f"{epi}/{acc}" for epi, acc in sorted(repeated)[:5])
        raise PullError(
            f"{len(repeated)} (isolate, accession) key(s) appear more than once: {shown}"
        )


def _normalise_nucleotides(sequence: str) -> tuple[str, int]:
    """Upper-case, and write U as T.

    Nextclade refuses a sequence containing U and — worse — leaves it out of every
    output file while exiting 0, so an unconverted record would vanish silently.
    """
    upper = sequence.upper().replace(" ", "")
    if "U" not in upper:
        return upper, 0
    return upper.replace("U", "T"), 1
