"""Which sera on a chart are not ferret sera, how we know, and what we cannot tell.

Sarah, 1 Oct 2026: "The antigenic maps should be ferret sera only." The policy is enforced on
the chain path (af.chain applies it to every table before the merge, in chains and merge_all
alike). A map built any other way, e.g. af.map.build on a stand-in chart or a consumer using
af.chart/af.map directly, is NOT checked unless it calls :func:`non_ferret` itself.

A serum is non-ferret when (a) its species is recorded and is not ferret, or (b) its species is
empty (the lab's default, ferret) but a MARKER row matches its name or serum id: CDC's pooled
vaccinee sera ("2017-18 QUADRIVALENT VACCINE", id "18HumanS0001") have no species at all. The
markers are data with their reasons (acmacs-f-data rules/sera/non_ferret_markers.tsv), not code.
A serum with an empty species that no marker catches is ferret BY DEFAULT: we cannot tell, so
those are counted and reported, never hidden.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from af.chart.model import Chart, Serum

FERRET_ONLY = "The antigenic maps should be ferret sera only. (Sarah, 1 Oct 2026)"
FIELDS = ("name", "serum_id")
COLUMNS = ("field", "pattern", "kind", "reason", "evidence")


class SeraPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class Marker:
    """One row: a regular expression (case-insensitive) on a serum's name or serum id."""

    field: str
    pattern: str
    kind: str  # what it marks, e.g. "human pooled vaccinee"
    reason: str
    evidence: str

    def matches(self, serum: Serum) -> bool:
        value = serum.name if self.field == "name" else serum.serum_id
        return re.search(self.pattern, value, re.IGNORECASE) is not None

    def label(self) -> str:
        return f"marker {self.field}~/{self.pattern}/ ({self.kind})"


def read_markers(path: Path) -> list[Marker]:
    """The markers file. It is required: a missing file, column or reason is an error."""
    if not path.is_file():
        raise SeraPolicyError(f"non-ferret markers file missing: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(
            csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t")
        )
    out = []
    for n, row in enumerate(rows, start=2):
        missing = [c for c in COLUMNS if not (row.get(c) or "").strip()]
        if missing:
            raise SeraPolicyError(f"{path}:{n}: empty {', '.join(missing)}")
        if row["field"] not in FIELDS:
            raise SeraPolicyError(f"{path}:{n}: field must be one of {FIELDS}")
        re.compile(row["pattern"])  # a bad pattern fails here, not on some later serum
        out.append(Marker(*(row[c].strip() for c in COLUMNS)))
    return out


def classify(serum: Serum, markers: list[Marker]) -> str | None:
    """Why `serum` is not a ferret serum ("species SHEEP", "marker …"), or None if ferret."""
    species = serum.species.strip().upper()
    if species and species != "FERRET":
        return f"species {species}"
    if not species:
        for m in markers:
            if m.matches(serum):
                return m.label()
    return None


@dataclass
class SeraReport:
    """What :func:`non_ferret` found on one chart."""

    non_ferret: list[tuple[int, str, str]] = field(
        default_factory=list
    )  # (serum, designation, why)
    ferret_recorded: int = 0  # species says FERRET
    ferret_by_default: int = 0  # species empty and no marker: ferret only because nothing says not

    def to_json(self) -> dict:
        return {
            "non_ferret": [{"serum": d, "caught_by": why} for _, d, why in self.non_ferret],
            "ferret_recorded": self.ferret_recorded,
            "ferret_by_default": self.ferret_by_default,
        }


def non_ferret(chart: Chart, markers: list[Marker]) -> SeraReport:
    report = SeraReport()
    for j, s in enumerate(chart.sera):
        why = classify(s, markers)
        if why is not None:
            report.non_ferret.append((j, s.designation(), why))
        elif s.species.strip():
            report.ferret_recorded += 1
        else:
            report.ferret_by_default += 1
    return report
