"""Mark vaccine antigens on a map: from the curated vaccine list, by name and passage class.

Why: today each lab folder binds its vaccine label rows to an exact designation string (name +
passage + date), and a row that matches nothing is dropped without a message: on the Sep 2026 round
8 vaccine rows were dead, and a superseded vaccine meant to be shrunk was drawn full size because
its row named the wrong passage. Here the curated list names a strain and optionally a passage
class; the antigen is found by name and class, and every hand rule (disable a vaccine, choose a
different preparation) must match something or it is an error (design rules 1 and 9).

Measured on the 18 Sep 2026 map folders: this matching finds all 172 vaccine-tagged antigens, and
the preparation rule below (most source tables, then latest table, then latest passage date) picks
the same antigen as ae except where a person chose otherwise.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Literal

PassageClass = Literal["egg", "cell", "reassortant"]
PASSAGE_CLASSES: tuple[PassageClass, ...] = ("cell", "egg", "reassortant")

# Passage vocabulary. A passage is egg-adapted if it went through eggs at any step, so every token
# counts, not only the last (ae reads only the last element, and an unparsed last token such as
# an SPF-egg or monkey-kidney step left vaccines unmarked).
EGG_TOKENS = frozenset({"E", "SPE", "SPF", "SPFCE", "D", "AM", "AL", "CE", "EGG"})
CELL_PREFIXES = ("MDCK", "SIAT", "HCK", "QMC", "SPFCK", "MK", "CACO")
_DATE = re.compile(r"\((\d{4}-\d{2}-\d{2})\)")
_SUBTYPE_PREFIX = re.compile(r"^(A\(H\d+N\d+\)|A\(H\d+\)|B)/")


class VaccineRuleError(ValueError):
    """A hand vaccine rule matched nothing, or matched more than it may."""


def passage_class(passage: str, reassortant: str) -> PassageClass | None:
    """``reassortant`` if the antigen has a reassortant name; else ``egg`` if any passage step was
    in eggs; else ``cell`` if any step names a cell line; else None (specimen ids, blanks: not a
    passage; such antigens are never vaccine candidates and are counted, not guessed)."""
    if reassortant.strip():
        return "reassortant"
    tokens = re.findall(r"[A-Z]+(?:-[A-Z]+)?", _DATE.sub("", passage.upper()))
    if any(t in EGG_TOKENS or t.startswith("EGG") for t in tokens):
        return "egg"
    if any(t.startswith(CELL_PREFIXES) for t in tokens):
        return "cell"
    return None


def passage_date(passage: str) -> dt.date | None:
    """The date a lab appends to a passage, e.g. ``MDCK1 (2020-01-31)``, parsed (design rule 7)."""
    m = _DATE.search(passage)
    return dt.date.fromisoformat(m.group(1)) if m else None


def strain_name(name: str) -> str:
    """Name without the subtype prefix, upper case: the form the curated list uses."""
    return _SUBTYPE_PREFIX.sub("", name.strip().upper())


@dataclass(frozen=True)
class VaccineRow:
    """One row of the curated vaccine list. ``passage`` None means every class."""

    name: str
    passage: PassageClass | None
    year: str
    surrogate: bool = False


@dataclass(frozen=True)
class MapAntigen:
    """What vaccine selection needs to know about one antigen of a chart."""

    index: int
    name: str
    passage: str
    reassortant: str
    n_tables: int
    has_coordinates: bool
    last_table: dt.date | None = None  # date of the latest source table with a titre for it


@dataclass(frozen=True)
class VaccineDisable:
    """Do not mark this vaccine (e.g. superseded ones a lab no longer wants drawn)."""

    name: str
    passage: PassageClass | Literal["any"]
    reason: str


@dataclass(frozen=True)
class VaccineChoice:
    """Use the preparation whose passage (without its date) is exactly ``passage``, instead of the
    automatic choice. Must match exactly one candidate."""

    name: str
    passage_class: PassageClass
    passage: str
    reason: str


@dataclass(frozen=True)
class VaccineMark:
    row: VaccineRow
    passage_class: PassageClass
    antigen: int
    candidates: int
    chosen_by: str  # "rule" or the name of the VaccineChoice


@dataclass
class VaccineReport:
    marks: list[VaccineMark] = field(default_factory=list)
    rows_without_antigen: list[tuple[str, str]] = field(default_factory=list)
    disabled: list[tuple[str, str, str]] = field(default_factory=list)
    unclassified_passages: int = 0


def select_vaccines(
    antigens: Sequence[MapAntigen],
    rows: Sequence[VaccineRow],
    *,
    disable: Sequence[VaccineDisable] = (),
    choose: Sequence[VaccineChoice] = (),
) -> VaccineReport:
    """One antigen per (curated strain x passage class) present in the chart.

    Among several preparations the automatic choice is the one tested in most source tables, then
    the one in the latest table, then the latest passage date. On the Sep 2026 round this
    reproduces ae's picks except where a person chose otherwise. Rows the
    chart has no antigen for are counted, not errors: a lab need not have tested every vaccine.
    Disable and choose rules that match nothing are errors.
    """
    by_name: dict[str, list[MapAntigen]] = {}
    report = VaccineReport()
    for ag in antigens:
        by_name.setdefault(strain_name(ag.name), []).append(ag)
        if passage_class(ag.passage, ag.reassortant) is None:
            report.unclassified_passages += 1
    used_disable: set[VaccineDisable] = set()
    used_choose: set[VaccineChoice] = set()
    for row in rows:
        for cls in (row.passage,) if row.passage else PASSAGE_CLASSES:
            cands = [
                ag
                for ag in by_name.get(strain_name(row.name), [])
                if ag.has_coordinates and passage_class(ag.passage, ag.reassortant) == cls
            ]
            if not cands:
                report.rows_without_antigen.append((row.name, cls))
                continue
            rule = _disabling(disable, row.name, cls)
            if rule is not None:
                used_disable.add(rule)
                report.disabled.append((row.name, cls, rule.reason))
                continue
            pick, chosen_by = _choose(cands, choose, row.name, cls, used_choose)
            report.marks.append(VaccineMark(row, cls, pick.index, len(cands), chosen_by))
    _require_used("disable", disable, used_disable)
    _require_used("choose", choose, used_choose)
    return report


def _disabling(rules: Sequence[VaccineDisable], name: str, cls: str) -> VaccineDisable | None:
    for rule in rules:
        if strain_name(rule.name) == strain_name(name) and rule.passage in ("any", cls):
            return rule
    return None


def _choose(
    cands: list[MapAntigen],
    rules: Sequence[VaccineChoice],
    name: str,
    cls: str,
    used: set[VaccineChoice],
) -> tuple[MapAntigen, str]:
    for rule in rules:
        if strain_name(rule.name) == strain_name(name) and rule.passage_class == cls:
            hits = [ag for ag in cands if _DATE.sub("", ag.passage).strip() == rule.passage.strip()]
            if len(hits) != 1:
                raise VaccineRuleError(
                    f"vaccine choice {rule.reason!r}: passage {rule.passage!r} matches "
                    f"{len(hits)} of {len(cands)} preparations, need exactly 1"
                )
            used.add(rule)
            return hits[0], rule.reason
    best = max(
        cands,
        key=lambda ag: (
            ag.n_tables,
            ag.last_table or dt.date.min,
            passage_date(ag.passage) or dt.date.min,
        ),
    )
    return best, "rule"


def _require_used(kind: str, rules: Sequence[object], used: Collection[object]) -> None:
    dead = [rule for rule in rules if rule not in used]
    if dead:
        raise VaccineRuleError(f"vaccine {kind} rule(s) matching nothing: {dead}")


# ------------------------------------------------------------------ importer

_TABLE = re.compile(r'"([^"]+)":\s*org_table_to_dict\("""(.*?)"""\)', re.S)


def read_org_vaccine_tables(text: str) -> dict[str, list[VaccineRow]]:
    """Importer for the transition period: read the curated list from acmacs-data's
    ``semantic_vaccines.py`` as text (org-mode tables keyed by subtype), without importing it.
    Every row must have a name; an unknown passage word is an error, not a silent 'all'."""
    tables: dict[str, list[VaccineRow]] = {}
    for key, body in _TABLE.findall(text):
        lines = [ln for ln in body.splitlines() if ln.startswith("|") and not ln.startswith("|-")]
        if not lines:
            raise ValueError(f"vaccine table {key!r} has no header")
        header = [h.strip() for h in lines[0].strip().strip("|").split("|")]
        rows = []
        for ln in lines[1:]:
            cells = dict(
                zip(header, (c.strip() for c in ln.strip().strip("|").split("|")), strict=False)
            )
            if not cells.get("name"):
                continue
            passage = _row_passage(key, cells.get("passage", ""))
            rows.append(
                VaccineRow(
                    name=cells["name"],
                    passage=passage,
                    year=cells.get("year", ""),
                    surrogate=cells.get("surrogate", "").lower() == "true",
                )
            )
        tables[key] = rows
    if not tables:
        raise ValueError("no vaccine tables found")
    return tables


def _row_passage(table: str, word: str) -> PassageClass | None:
    if not word:
        return None
    for cls in PASSAGE_CLASSES:
        if word == cls:
            return cls
    raise ValueError(f"vaccine table {table!r}: unknown passage {word!r}")
