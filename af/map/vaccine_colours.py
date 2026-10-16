"""Vaccine antigens take the colour of their named strain's original record (Sarah, Q184, 16 Oct
2026: "vaccines should be colored according to the original clade of the named virus before
passage mutation").

Why: a vaccine preparation is usually egg-passaged, and an egg passage can change the very loci a
clade or a colour row is defined by, so its own record can paint the vaccine as something the
virus never was. The strain's original record (clinical, unpassaged) shows it before passage;
failing that, a cell record, whose passage changes such loci far less often. Never an egg record,
nor one whose passage is mixed or unknown.

Only the antigens a map marks as vaccines (:func:`af.map.vaccines.select_vaccines`) are
recoloured; other preparations of a vaccine strain keep the shared rule. Several records of the
chosen class are resolved as ties are: an unobserved one abstains beside an observed one (Q135),
then a contradicted one beside a supported one; the rest must agree. When they do not, when they
give no colour, or when the strain has no original or cell record, the vaccine keeps its own
preparation's colour and a warning says why (design rule 1: never silent).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from af.geo.colours import DotStyle
from af.seq.matching import CELL, ORIGINAL

#: Which records stand for the virus before passage, best first.
ORIGINAL_FIRST = (ORIGINAL, CELL)


@dataclass(frozen=True)
class StrainRecord:
    """One sequence record of a vaccine's named strain, as the colouring reads it."""

    epi_isl: str
    accession: str
    passage_class: str  # af.seq.matching's class of the record's passage
    clade: str | None
    clade_evidence: str | None = None
    supported_clade: str | None = None


@dataclass(frozen=True)
class VaccineColour:
    """One vaccine antigen's colour, and where it came from."""

    antigen: int
    name: str
    passage_class: str  # the vaccine's own (af.map.vaccines): egg, cell or reassortant
    style: DotStyle
    labels: frozenset[str]
    source: str  # ORIGINAL or CELL when its strain's records coloured it; "" when kept
    records: int  # records of that class considered
    changed: bool
    warning: str = ""  # vaccine_warning()'s sentence when it kept its own colour

    def record(self) -> dict[str, Any]:
        """The figure's provenance entry (decisions.vaccine_colours)."""
        out: dict[str, Any] = {
            "antigen": f"ag{self.antigen}",
            "name": self.name,
            "passage_class": self.passage_class,
            "colour": self.style.label or None,
            "source": self.source or None,
            "records": self.records,
            "changed": self.changed,
        }
        if self.warning:
            out["warning"] = self.warning
        return out


def original_records(records: Sequence[StrainRecord]) -> tuple[str, list[StrainRecord]]:
    """The records that stand for the strain before passage: its original ones, else its cell
    ones; ("", []) when it has neither."""
    for kind in ORIGINAL_FIRST:
        found = [r for r in records if r.passage_class == kind]
        if found:
            return kind, found
    return "", []


def resolve(styled: Sequence[tuple[StrainRecord, DotStyle]]) -> tuple[DotStyle | None, str]:
    """One style from the strain's records, or None with the reason. Unobserved records abstain
    beside observed ones (Q135), then contradicted ones beside supported ones; the rest agree
    on what is drawn (legend and colour), or nothing is chosen."""
    kept = [x for x in styled if x[0].clade_evidence != "unobserved"] or list(styled)
    kept = [x for x in kept if x[0].clade_evidence != "contradicted"] or kept
    looks: dict[tuple[str, str | None], DotStyle] = {}
    counts: dict[str, int] = {}
    for _, style in kept:
        looks.setdefault((style.label, style.colour), style)
        what = style.label or style.state or "uncoloured"
        counts[what] = counts.get(what, 0) + 1
    if len(looks) != 1:
        found = ", ".join(f"{k} {n}" for k, n in sorted(counts.items()))
        return None, f"records disagree ({found})"
    (style,) = looks.values()
    if style.colour is None:
        return None, f"records give it no colour ({style.state or 'uncoloured'})"
    return style, ""


def vaccine_warning(name: str, passage_class: str, reason: str) -> str:
    """The one sentence for a vaccine that keeps its own preparation's colour (Q184), copied
    verbatim by the map's decisions, the build log and the report's how-made notes."""
    return (
        f"WARNING: vaccine {name} ({passage_class}) keeps its own preparation's colour, not its "
        f"strain's original clade: {reason}."
    )
