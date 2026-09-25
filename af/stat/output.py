"""Stat counts as a data file and as a web page.

The data file (``stat.json``) is the long form: one record per non-zero cell, with the
window and the rules it was counted under, and what could not be counted (undated
antigens, locations with no continent). The page is rendered from the same counts, so the
two never disagree.

The page has one section per subtype (and per lineage where split): a table of months and
the year/window totals against labs, for antigens, new sera and sera used, and a table of
continents for the window. Labs and continents are whatever the data has, sorted, with
"all" first: no lab list is written into the code (design rule 10).
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from af.geo.records import Month, months
from af.stat.counts import ALL, Key, StatCounts

FORMAT = "af-stat-1"
MEASURES = {
    "antigens": "Antigens (by earliest collection date; lab of the earliest table)",
    "sera_new": "New sera (by first titration)",
    "sera_used": "Sera used in titrations (distinct sera per cell)",
}


def stat_document(counts: StatCounts, first: Month, last: Month) -> dict[str, Any]:
    """The JSON-ready data: every non-zero cell of every measure, plus what was left out."""
    cells = [
        {"measure": measure, "subtype": s, "lab": lab, "period": p, "continent": c, "count": n}
        for measure in MEASURES
        for (s, lab, p, c), n in sorted(getattr(counts, measure).items())
        if n
    ]
    return {
        "format": FORMAT,
        "window": {"first": str(first), "last": str(last)},
        "measures": MEASURES,
        "cells": cells,
        "undated": dict(counts.undated),
        "unknown_continent": dict(counts.unknown_continent),
    }


def write_stat(counts: StatCounts, first: Month, last: Month, directory: Path) -> list[Path]:
    """Write ``stat.json`` and ``index.html`` into ``directory``; return both paths."""
    directory.mkdir(parents=True, exist_ok=True)
    doc = stat_document(counts, first, last)
    data = directory / "stat.json"
    data.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    page = directory / "index.html"
    page.write_text(render_html(counts, first, last), encoding="utf-8")
    return [data, page]


def render_html(counts: StatCounts, first: Month, last: Month) -> str:
    """A self-contained page (no scripts, no external files)."""
    window = months(first, last)
    subtypes = _subtypes(counts)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Stat {first}..{last}</title><style>{_CSS}</style></head><body>",
        f"<h1>Antigens and sera tested, {first} to {last}</h1>",
        _notes(counts),
    ]
    for subtype in subtypes:
        parts.append(f"<h2>{_e(subtype)}</h2>")
        for measure, title in MEASURES.items():
            parts.append(f"<h3>{_e(title)}</h3>")
            parts.append(_period_table(getattr(counts, measure), subtype, window))
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


def _period_table(cells: dict[Key, int], subtype: str, window: list[Month]) -> str:
    labs = _labs(cells, subtype)
    years = sorted({f"{m.year:04d}" for m in window})
    periods = [str(m) for m in window] + years + [ALL]
    head = "".join(f"<th>{_e(lab)}</th>" for lab in labs)
    rows = []
    for period in periods:
        total = period in years or period == ALL
        tds = "".join(f"<td>{cells.get((subtype, lab, period, ALL), 0)}</td>" for lab in labs)
        css = " class='total'" if total else ""
        rows.append(f"<tr{css}><th>{_e(period)}</th>{tds}</tr>")
    return f"<table><tr><th></th>{head}</tr>{''.join(rows)}</table>"


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


def _notes(counts: StatCounts) -> str:
    items = [
        f"<li>{_e(kind.capitalize())} with no collection date anywhere in the store, so in no "
        f"month (whole store, not only this window): {n}</li>"
        for kind, n in sorted(counts.undated.items())
    ]
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
"""
