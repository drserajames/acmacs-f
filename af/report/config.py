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
import zoneinfo
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
class About:
    """A text page after the contents saying what the report is: a monthly for one centre, say,
    whose data end where, and what is new since the last meeting. The words are the report's;
    its facts are placeholders filled from the report config (af.report.text), never typed in."""

    title: str
    text: str


@dataclass(frozen=True)
class Report:
    id: str
    kind: str
    # A string, or the cover's lines as a list (the delivered report breaks its title in fixed
    # places and puts the subtitle on the title's last line).
    title: str | list[str]
    period: Period
    data_cutoff: dt.date
    centre: str | list[str] = ""  # a list: one cover line each, as the delivered report has
    subtitle: str = ""  # e.g. "Southern Hemisphere 2027": config, not derived from dates
    meeting: Meeting | None = None
    about: About | None = None
    end_page: bool = False  # a closing "Report generated <time>" page, as the delivered report has
    # IANA zone the closing page prints its time in (the delivered report: "Europe/London");
    # empty prints UTC. The build record keeps UTC either way.
    timezone: str = ""


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
    # An intro page before the figures (geo and trees): the report's words; facts are
    # placeholders filled from recorded artefacts (af.report.text). Paragraphs: blank lines.
    intro: str = ""
    # geo: the subtype in the geo build's clade key (<figures>/geo/clade-key.json) whose colours
    # the intro page lists, as the delivered report does; the key comes with the figures.
    clade_key: str = ""
    # The key's rows for dots no clade colours, after the clades: the report's words for each
    # uncoloured class the geo build keys (af.geo.records.UNCOLOURED_CLASSES), e.g.
    # {not_sequenced = "not sequenced", clade_not_defined = "clade not defined"} (Sarah, Q215:
    # "mark unsequenced & clade not defined on different lines"). Printed in the key's order.
    clade_key_classes: dict[str, str] = field(default_factory=dict)
    # Why a clade can be not defined (the key's "states" ids) that the intro's words explain
    # (Q215: "explain why clade not defined"); a dot so for any other reason refuses the build.
    clade_not_defined_states: list[str] = field(default_factory=list)

    def pages(self, months: list[str]) -> list[tuple[str, list[str | None]]]:
        """(heading, cells) per page for a grid section; a blank cell is None.

        maps: one run of pages per window, cells = slots in order (``"-"`` = blank);
        geo: one slot per month of the report period, ``<slot>/<YYYY-MM>``.
        """
        return [(heading, cells) for heading, cells, _ in self._pages(months)]

    def run_starts(self, months: list[str]) -> list[bool]:
        """Per page of :meth:`pages`, whether it starts a run (a map window, a geo slot): a
        window titled "" has no heading to tell it by."""
        return [first for _, _, first in self._pages(months)]

    def _pages(self, months: list[str]) -> list[tuple[str, list[str | None], bool]]:
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
                page = cells[start : start + per_page]
                out.append((heading if start == 0 else "", page, start == 0))
        return out


@dataclass(frozen=True)
class Figures:
    pins: dict[str, str] = field(default_factory=dict)  # slot -> version
    allow_placeholders: bool = False  # bring-up only; the cover then says DRAFT


@dataclass(frozen=True)
class TreeMethod:
    """How an imported tree (its provenance says "not an af build") was aligned and built, which
    its ``tree.json`` does not record: declared, with the records it rests on (00, 19 Oct 2026).
    The fields are the ones an af build records (``aligned.region``, ``build.version`` as
    "<tool>/<version>", ``build.parameters.model``, ``build.zero_length_collapsed``)."""

    region: str
    version: str
    model: str
    zero_length_collapsed: bool
    source: str  # the records each fact comes from


@dataclass(frozen=True)
class ReportConfig:
    report: Report
    sections: list[Section]
    figures: Figures = field(default_factory=Figures)
    # Tree dataset -> its declared method, for imported trees only; each must be drawn.
    imported_tree_methods: dict[str, TreeMethod] = field(default_factory=dict)

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


def _key_class_problems(where: str, section: Section) -> list[str]:
    from af.geo.records import CLADE_NOT_DEFINED, UNCOLOURED_CLASSES, UNDEFINED_STATE_IDS

    problems = []
    classes = section.clade_key_classes
    if classes and not section.clade_key:
        problems.append(f"{where}.clade_key_classes: needs clade_key")
    unknown = sorted(set(classes) - set(UNCOLOURED_CLASSES))
    if unknown:
        problems.append(f"{where}.clade_key_classes: unknown {unknown} (of {UNCOLOURED_CLASSES})")
    if any(not label.strip() for label in classes.values()):
        problems.append(f"{where}.clade_key_classes: an empty label")
    if section.clade_not_defined_states and CLADE_NOT_DEFINED not in classes:
        problems.append(f"{where}.clade_not_defined_states: needs a {CLADE_NOT_DEFINED} key row")
    unknown = sorted(set(section.clade_not_defined_states) - set(UNDEFINED_STATE_IDS))
    if unknown:
        problems.append(
            f"{where}.clade_not_defined_states: unknown {unknown} (known: {UNDEFINED_STATE_IDS})"
        )
    if CLADE_NOT_DEFINED in _placeholders(section) and CLADE_NOT_DEFINED not in classes:
        problems.append(f"{where}.intro: {{{CLADE_NOT_DEFINED}}} needs a {CLADE_NOT_DEFINED} row")
    return problems


def _placeholders(section: Section) -> list[str]:
    from af.report.text import placeholders

    try:
        return placeholders(section.intro)
    except ValueError:
        return []  # reported by _intro_problems


def _intro_problems(where: str, section: Section) -> list[str]:
    if section.kind not in ("geo", "trees"):
        return [f"{where}.intro: only for geo and trees sections"]
    return _placeholder_problems(f"{where}.intro", section.intro, section.kind, "section")


def _about_problems(about: About) -> list[str]:
    problems = [f"report.about.{key}: empty" for key in ("title", "text")
                if not getattr(about, key).strip()]  # fmt: skip
    return problems + _placeholder_problems("report.about.text", about.text, "about", "page")


def _placeholder_problems(where: str, text: str, kind: str, noun: str) -> list[str]:
    """Placeholders in ``text`` that are malformed or not filled for a ``kind`` page."""
    from af.report.text import PLACEHOLDERS, placeholders

    try:
        names = placeholders(text)
    except ValueError as err:
        return [f"{where}: {err}"]
    return [
        f"{where}: {{{name}}} is not a placeholder for a{'n' if kind[0] in 'aeiou' else ''} "
        f"{kind} {noun} (known: {sorted(k for k, kinds in PLACEHOLDERS.items() if kind in kinds)})"
        for name in names
        if kind not in PLACEHOLDERS.get(name, ())
    ]


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
    if r.about is not None:
        problems += _about_problems(r.about)
    for key in ("title", "centre"):
        value = getattr(r, key)
        if isinstance(value, list) and not all(line.strip() for line in value):
            problems.append(f"report.{key}: an empty line in the list")
    if r.timezone:
        try:
            zoneinfo.ZoneInfo(r.timezone)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError):
            problems.append(f"report.timezone: {r.timezone!r} is not an IANA time zone")
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
        if section.intro:
            problems += _intro_problems(where, section)
        if section.clade_key and (section.kind != "geo" or not section.intro):
            problems.append(f"{where}.clade_key: only for a geo section with an intro page")
        problems += _key_class_problems(where, section)
    slots = cfg.all_slots()
    duplicates = sorted({s for s in slots if slots.count(s) > 1})
    if duplicates:
        problems.append(f"sections: slots used twice {duplicates}")
    unknown_pins = sorted(set(cfg.figures.pins) - set(slots))
    if unknown_pins:
        problems.append(f"figures.pins: slots not in any section {unknown_pins}")
    for dataset, method in cfg.imported_tree_methods.items():
        where = f"imported_tree_methods.{dataset!r}"
        if any(not v.strip() for v in (method.region, method.model, method.source)):
            problems.append(f"{where}: region, model and source must not be empty")
        tool, _, version = method.version.partition("/")
        if not tool or not version:
            problems.append(f"{where}.version: {method.version!r} is not '<tool>/<version>'")
    if problems:
        raise ConfigError(path, problems)
    return cfg


def lines(value: str | list[str]) -> list[str]:
    """A config text that may be one string or a list of lines, as lines ("" is no line)."""
    return list(value) if isinstance(value, list) else ([value] if value else [])


def period_first(cfg: ReportConfig) -> dt.date:
    return parse_month(cfg.report.period.first)


def period_last(cfg: ReportConfig) -> dt.date:
    return parse_month(cfg.report.period.last)
