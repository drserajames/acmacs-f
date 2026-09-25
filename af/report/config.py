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
SECTION_KINDS = ("trees", "maps", "geo")
BLANK = "-"  # a grid cell left empty on purpose (e.g. a lab with no map of this assay)


@dataclass(frozen=True)
class Period:
    first: str  # "YYYY-MM"; a string, so TOML cannot turn it into anything else
    last: str


@dataclass(frozen=True)
class Meeting:
    """The meeting a VCM report is for, as dates: printed on the cover, never derived."""

    start: dt.date
    end: dt.date


@dataclass(frozen=True)
class Report:
    id: str
    kind: str
    title: str
    period: Period
    data_cutoff: dt.date
    centre: str = ""
    subtitle: str = ""  # e.g. "Southern Hemisphere 2027": config, not derived from dates
    meeting: Meeting | None = None


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
    grid: list[int] = field(default_factory=lambda: [2, 2])  # columns, rows per page
    landscape: bool = False
    group: str = ""  # e.g. a subtype: consecutive sections of one group sit under its heading

    def pages(self, months: list[str]) -> list[tuple[str, list[str | None]]]:
        """(heading, cells) per page for a grid section; a blank cell is None.

        maps: one run of pages per window, cells = slots in order (``"-"`` = blank);
        geo: one slot per month of the report period, ``<slot>/<YYYY-MM>``.
        """
        cols, rows = self.grid
        per_page = cols * rows
        runs: list[tuple[str, list[str | None]]] = []
        if self.kind == "maps":
            for w in self.windows:
                cells = [None if s == BLANK else f"{s}/{w.name}" for s in self.slots]
                runs.append((w.title, cells))
        elif self.kind == "geo":
            for slot in self.slots:
                runs.append(("", [f"{slot}/{m}" for m in months]))
        out = []
        for heading, cells in runs:
            for start in range(0, len(cells), per_page):
                out.append((heading if start == 0 else "", cells[start : start + per_page]))
        return out


@dataclass(frozen=True)
class Figures:
    pins: dict[str, str] = field(default_factory=dict)  # slot -> version
    allow_placeholders: bool = False  # bring-up only; the cover then says DRAFT


@dataclass(frozen=True)
class ReportConfig:
    report: Report
    sections: list[Section]
    figures: Figures = field(default_factory=Figures)

    def months(self) -> list[str]:
        """Every month of the report period, "YYYY-MM", first to last inclusive."""
        first, last = parse_month(self.report.period.first), parse_month(self.report.period.last)
        out, y, m = [], first.year, first.month
        while (y, m) <= (last.year, last.month):
            out.append(f"{y:04d}-{m:02d}")
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return out

    def all_slots(self) -> list[str]:
        """Every figure slot the report embeds, in page order (blank cells left out)."""
        out: list[str] = []
        for section in self.sections:
            if section.kind == "trees":
                out += section.slots
            else:
                for _, cells in section.pages(self.months()):
                    out += [c for c in cells if c is not None]
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
    if r.meeting is not None and r.meeting.end < r.meeting.start:
        problems.append("report.meeting: end is before start")
    if len(months) != 2:  # the rest needs a valid period
        raise ConfigError(path, problems)
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
        if section.kind != "maps" and section.windows:
            problems.append(f"{where}.windows: only for maps")
        if section.kind != "maps" and BLANK in section.slots:
            problems.append(f"{where}.slots: blank cells ('{BLANK}') only in map grids")
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
