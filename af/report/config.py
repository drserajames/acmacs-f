"""Report configuration (interface I8): what a report contains and where its figures come from.

The config names *slots* (``map/<chart>/<window>``, ``tree/<subtype>/<variant>``), never file
paths inside the store. The builder resolves each slot to the latest figure (or a pinned
version) and records the choice in the manifest. So the same config rebuilt next month picks up
new trees and maps, and the manifest of each PDF says exactly which ones it used.

Parsed with :func:`af.util.config.load_config` (TOML, unknown and missing keys are errors).
This module adds the checks a schema cannot express: months are ``YYYY-MM`` strings, the
period is ordered, map sections have windows, and every pin names a slot the report uses.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from af.util.config import ConfigError, load_config

REPORT_KINDS = ("monthly", "vcm")
SECTION_KINDS = ("trees", "maps")


@dataclass(frozen=True)
class Period:
    first: str  # "YYYY-MM"; a string, so TOML cannot turn it into anything else
    last: str


@dataclass(frozen=True)
class Report:
    id: str
    kind: str
    title: str
    period: Period
    data_cutoff: dt.date
    centre: str = ""


@dataclass(frozen=True)
class Window:
    """A map time window. ``since`` is explicit data, never derived from today's date.

    No ``since`` means the whole map (TOML has no null, so absence is the explicit form).
    """

    name: str
    title: str
    since: dt.date | None = None


@dataclass(frozen=True)
class Section:
    kind: str
    title: str
    slots: list[str]
    windows: list[Window] = field(default_factory=list)
    grid: list[int] = field(default_factory=lambda: [2, 2])  # columns, rows per map page


@dataclass(frozen=True)
class Figures:
    pins: dict[str, str] = field(default_factory=dict)  # slot -> version
    allow_placeholders: bool = False  # bring-up only; the cover then says DRAFT


@dataclass(frozen=True)
class ReportConfig:
    report: Report
    sections: list[Section]
    figures: Figures = field(default_factory=Figures)

    def all_slots(self) -> list[str]:
        """Every figure slot the report embeds, in page order."""
        out: list[str] = []
        for section in self.sections:
            if section.kind == "maps":
                out += [f"{s}/{w.name}" for w in section.windows for s in section.slots]
            else:
                out += section.slots
        return out


def parse_month(value: str) -> dt.date:
    """``"YYYY-MM"`` -> the first day of that month. Anything else is an error."""
    parts = value.split("-")
    if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 2:
        raise ValueError(f"{value!r} is not a YYYY-MM month")
    return dt.date(int(parts[0]), int(parts[1]), 1)


def load(path: Path) -> ReportConfig:
    """Load and check a report config; every problem is reported in one ConfigError."""
    cfg = load_config(path, ReportConfig)
    problems: list[str] = []
    r = cfg.report
    if r.kind not in REPORT_KINDS:
        problems.append(f"report.kind: {r.kind!r} not one of {REPORT_KINDS}")
    months = {}
    for key in ("first", "last"):
        try:
            months[key] = parse_month(getattr(r.period, key))
        except ValueError as err:
            problems.append(f"report.period.{key}: {err}")
    if len(months) == 2 and months["last"] < months["first"]:
        problems.append("report.period: last is before first")
    if not cfg.sections:
        problems.append("sections: empty")
    for i, section in enumerate(cfg.sections):
        where = f"sections[{i}]"
        if section.kind not in SECTION_KINDS:
            problems.append(f"{where}.kind: {section.kind!r} not one of {SECTION_KINDS}")
        if not section.slots:
            problems.append(f"{where}.slots: empty")
        if section.kind == "maps" and not section.windows:
            problems.append(f"{where}.windows: required for maps")
        if section.kind == "trees" and section.windows:
            problems.append(f"{where}.windows: only for maps")
        if len(section.grid) != 2 or min(section.grid) < 1:
            problems.append(f"{where}.grid: expected [columns, rows], both >= 1")
    slots = cfg.all_slots()
    duplicates = sorted({s for s in slots if slots.count(s) > 1})
    if duplicates:
        problems.append(f"sections: slots used twice {duplicates}")
    unknown_pins = sorted(set(cfg.figures.pins) - set(slots))
    if unknown_pins:
        problems.append(f"figures.pins: slots not in any section {unknown_pins}")
    if problems:
        raise ConfigError(path, problems)
    return cfg


def period_first(cfg: ReportConfig) -> dt.date:
    return parse_month(cfg.report.period.first)


def period_last(cfg: ReportConfig) -> dt.date:
    return parse_month(cfg.report.period.last)
