"""Stat counts as a data file and as a web page.

The data file (``stat.json``) is the long form: one record per non-zero cell, with the
window and the rules it was counted under, and what could not be counted (undated
antigens, locations with no continent). The page is rendered from the same counts, so the
two never disagree.

The page has one section per subtype (and per lineage where split): a table of months and
the year/window totals against labs, for antigens, new sera and sera used, and a table of
continents for the window. Labs and continents are whatever the data has, sorted, with
"all" first: no lab list is written into the code (design rule 10).

**Comparison with the previous round.** Given the previous round's ``stat.json``, each
month the previous window also covered shows its change, ``n (+d)``: antigens and sera
reported late, after the previous report was made. Only month rows get a change. A year or
window total is not comparable across rounds whose windows differ, and "sera used" totals
are distinct counts that cannot be summed. A *decrease* means data was removed or
corrected; today's tool clamped those to zero with a warning on stderr. Here they are
shown, highlighted and counted, and listed in ``stat.json``.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from af.geo.records import Month, months
from af.stat.counts import ALL, Key, StatCounts

FORMAT = "af-stat-1"

CellKey = tuple[str, str, str, str, str]  # (measure, subtype, lab, period, continent)


class StatOutputError(ValueError):
    """A previous stat file that cannot be compared with."""


@dataclass(frozen=True)
class Previous:
    """The previous round's counts, and which months its window covered."""

    source: str
    first: Month
    last: Month
    cells: dict[CellKey, int]

    @property
    def months(self) -> set[str]:
        return {str(m) for m in months(self.first, self.last)}


def read_previous(path: Path) -> Previous:
    """Read a previous round's ``stat.json``; a missing or foreign file is fatal."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise StatOutputError(f"cannot read previous stat {path}: {error}") from None
    if doc.get("format") != FORMAT:
        raise StatOutputError(f"{path}: format {doc.get('format')!r}, expected {FORMAT}")
    first, last = (_month(doc["window"][end]) for end in ("first", "last"))
    cells = {
        (c["measure"], c["subtype"], c["lab"], c["period"], c["continent"]): int(c["count"])
        for c in doc["cells"]
    }
    return Previous(source=str(path), first=first, last=last, cells=cells)


def changes(counts: StatCounts, previous: Previous) -> list[dict[str, Any]]:
    """Every month cell (in both windows) whose count differs from the previous round."""
    shared = previous.months
    keys = {
        (measure, *key)
        for measure in MEASURES
        for key in getattr(counts, measure)
        if key[2] in shared
    } | {key for key in previous.cells if key[3] in shared and key[0] in MEASURES}
    out = []
    for key in sorted(keys):
        now, before = _count(counts, key), previous.cells.get(key, 0)
        if now != before:
            measure, subtype, lab, period, continent = key
            out.append(
                {"measure": measure, "subtype": subtype, "lab": lab, "period": period,
                 "continent": continent, "now": now, "before": before}
            )  # fmt: skip
    return out


def _count(counts: StatCounts, key: CellKey) -> int:
    measure, *rest = key
    cells: dict[Key, int] = getattr(counts, measure)
    return cells.get((rest[0], rest[1], rest[2], rest[3]), 0)


def _month(text: str) -> Month:
    year, month = text.split("-")
    return Month(int(year), int(month))


MEASURES = {
    "antigens": "Antigens (by earliest collection date; lab of the earliest table)",
    "sera_new": "New sera (by first titration)",
    "sera_used": "Sera used in titrations (distinct sera per cell)",
}


def stat_document(
    counts: StatCounts, first: Month, last: Month, previous: Previous | None = None
) -> dict[str, Any]:
    """The JSON-ready data: every non-zero cell of every measure, plus what was left out,
    and the changes since the previous round when one is given."""
    cells = [
        {"measure": measure, "subtype": s, "lab": lab, "period": p, "continent": c, "count": n}
        for measure in MEASURES
        for (s, lab, p, c), n in sorted(getattr(counts, measure).items())
        if n
    ]
    doc: dict[str, Any] = {
        "format": FORMAT,
        "window": {"first": str(first), "last": str(last)},
        "measures": MEASURES,
        "cells": cells,
        "undated": dict(counts.undated),
        "unknown_continent": dict(counts.unknown_continent),
    }
    if previous is not None:
        doc["previous"] = {
            "source": previous.source,
            "window": {"first": str(previous.first), "last": str(previous.last)},
            "changes": changes(counts, previous),
        }
    return doc


def write_stat(
    counts: StatCounts,
    first: Month,
    last: Month,
    directory: Path,
    previous: Previous | None = None,
) -> list[Path]:
    """Write ``stat.json`` and ``index.html`` into ``directory``; return both paths."""
    directory.mkdir(parents=True, exist_ok=True)
    doc = stat_document(counts, first, last, previous)
    data = directory / "stat.json"
    data.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    page = directory / "index.html"
    page.write_text(render_html(counts, first, last, previous), encoding="utf-8")
    return [data, page]


def render_html(
    counts: StatCounts, first: Month, last: Month, previous: Previous | None = None
) -> str:
    """A self-contained page (no scripts, no external files)."""
    window = months(first, last)
    subtypes = _subtypes(counts)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Stat {first}..{last}</title><style>{_CSS}</style></head><body>",
        f"<h1>Antigens and sera tested, {first} to {last}</h1>",
        _notes(counts, previous),
    ]
    for subtype in subtypes:
        parts.append(f"<h2>{_e(subtype)}</h2>")
        for measure, title in MEASURES.items():
            parts.append(f"<h3>{_e(title)}</h3>")
            parts.append(_period_table(counts, measure, subtype, window, previous))
        parts.append("<h3>By continent, whole window</h3>")
        parts.append(_continent_table(counts, subtype))
    parts.append("</body></html>")
    return "\n".join(parts)


def _subtypes(counts: StatCounts) -> list[str]:
    seen = {key[0] for measure in MEASURES for key in getattr(counts, measure)}
    return [ALL] + sorted(seen - {ALL})


def _labs(cells: dict[Key, int], subtype: str) -> list[str]:
    labs = {key[1] for key in cells if key[0] == subtype}
    return [ALL] + sorted(labs - {ALL})


def _period_table(
    counts: StatCounts,
    measure: str,
    subtype: str,
    window: list[Month],
    previous: Previous | None,
) -> str:
    cells: dict[Key, int] = getattr(counts, measure)
    labs = _labs(cells, subtype)
    compared = previous.months if previous is not None else set()
    years = sorted({f"{m.year:04d}" for m in window})
    periods = [str(m) for m in window] + years + [ALL]
    head = "".join(f"<th>{_e(lab)}</th>" for lab in labs)
    rows = []
    for period in periods:
        total = period in years or period == ALL
        tds = "".join(
            _cell(cells.get((subtype, lab, period, ALL), 0), (measure, subtype, lab, period, ALL),
                  previous if period in compared else None)
            for lab in labs
        )  # fmt: skip
        css = " class='total'" if total else ""
        rows.append(f"<tr{css}><th>{_e(period)}</th>{tds}</tr>")
    return f"<table><tr><th></th>{head}</tr>{''.join(rows)}</table>"


def _cell(now: int, key: CellKey, previous: Previous | None) -> str:
    """``n``, or ``n (+d)`` against the previous round; a decrease is highlighted."""
    if previous is None:
        return f"<td>{now}</td>"
    change = now - previous.cells.get(key, 0)
    if change == 0:
        return f"<td>{now}</td>"
    css = " class='down'" if change < 0 else ""
    return f"<td{css}>{now} <small>({change:+d})</small></td>"


def _continent_table(counts: StatCounts, subtype: str) -> str:
    cells: dict[str, dict[str, int]] = {}
    for measure in MEASURES:
        for (s, lab, period, continent), n in getattr(counts, measure).items():
            if s == subtype and lab == ALL and period == ALL:
                cells.setdefault(continent, {})[measure] = n
    continents = [ALL] + sorted(set(cells) - {ALL})
    head = "".join(f"<th>{_e(m)}</th>" for m in MEASURES)
    rows = "".join(
        f"<tr><th>{_e(c)}</th>"
        + "".join(f"<td>{cells.get(c, {}).get(m, 0)}</td>" for m in MEASURES)
        + "</tr>"
        for c in continents
    )
    return f"<table><tr><th></th>{head}</tr>{rows}</table>"


def _notes(counts: StatCounts, previous: Previous | None) -> str:
    items = [
        f"<li>{_e(kind.capitalize())} with no collection date anywhere in the store, so in no "
        f"month (whole store, not only this window): {n}</li>"
        for kind, n in sorted(counts.undated.items())
    ]
    if previous is not None:
        found = changes(counts, previous)
        down = sum(1 for c in found if c["now"] < c["before"])
        items.append(
            f"<li>Changes since {_e(previous.source)} (window {previous.first} to "
            f"{previous.last}), for months both windows cover: {len(found) - down} cells "
            f"up, <span class='down'>{down} down</span> (data removed or corrected)</li>"
        )
    unknown = sum(counts.unknown_continent.values())
    if unknown:
        items.append(f"<li>Counted under UNKNOWN continent: {unknown}</li>")
    return f"<ul class='notes'>{''.join(items)}</ul>" if items else ""


def _e(text: str) -> str:
    return html.escape(text, quote=True)


_CSS = """
body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:1.5em;color:#222}
table{border-collapse:collapse;margin:.3em 0 1.2em}
th,td{border:1px solid #ccc;padding:.2em .6em;text-align:right}
tr.total{background:#f3f3f3;font-weight:600}
h2{margin-top:1.6em;border-bottom:1px solid #999}
.notes{color:#555}
.down{background:#fde2e1;color:#8a1c14}
small{color:#666}
"""
