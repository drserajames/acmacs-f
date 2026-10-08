"""Geo figures for a report: one ``figure.pdf`` + ``figure.i7.json`` per subtype and month.

The report reads figures from ``<figures>/geo/<slot>/<YYYY-MM>/<variant>/`` (the slot is the
report's name for the subtype: "h1", "h3", "bvic"), and the same-science comparison reads only
the I7 beside each PDF (:mod:`af.report.i7`). So each month is written as its own figure, from
the same I7 ``geo`` document the renderer draws (:func:`af.geo.records.to_i7`): what is drawn
and what is compared cannot drift apart.

The clade key is data, not a picture: per slot, every legend entry of the scheme with its
colour and its dots per month and over the window, plus the uncoloured dots. The report's geo
intro page lays it out; the PDFs carry no legend, as the shipped geo pages carry none.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.geo.render import Coordinates, Look, render_period
from af.map.style import ColourScheme as MapColourScheme
from af.report.i7 import I7_NAME, I7_VERSION, validate
from af.util.artefacts import sha256_path

UNCOLOURED_KEY = "uncoloured"


@dataclass
class FiguresReport:
    files: list[Path] = field(default_factory=list)
    drawn: dict[str, int] = field(default_factory=dict)  # month -> dots drawn
    no_coordinates: Counter[str] = field(default_factory=Counter)  # location -> dots


def write_geo_figures(
    doc: Mapping[str, Any],
    slot: str,
    coordinates: Coordinates,
    coastline: Path,
    figures: Path,
    provenance: Mapping[str, Any],
    *,
    variant: str = "af",
    look: Look | None = None,
) -> FiguresReport:
    """Write every period of one subtype's I7 ``geo`` document as a report figure.

    ``provenance`` goes into each I7 as given (the store versions and tables the dots and
    colours came from); the month's dot count and any locations it could not place are added,
    so a thin map says why on its own page.
    """
    report = FiguresReport()
    for period in doc["periods"]:
        month = period["period"]
        folder = figures / "geo" / slot / month / variant
        folder.mkdir(parents=True, exist_ok=True)
        pdf = folder / "figure.pdf"
        unplaced: Counter[str] = Counter()
        report.drawn[month] = render_period(period, coordinates, coastline, pdf, look, unplaced)
        report.no_coordinates.update(unplaced)
        i7 = {
            "i7_version": I7_VERSION,
            "kind": "geo",
            "title": f"{slot} {period.get('title', month)}",
            "placeholder": False,
            "figure": {"pdf": str(pdf), "sha256": sha256_path(pdf), "pages": 1},
            "provenance": {
                **provenance,
                "producer": "af.geo.figures",
                "drawn": report.drawn[month],
                "no_coordinates": dict(sorted(unplaced.items())),
            },
            "geo": {"subtype": slot, "month": month, "locations": period["locations"]},
        }
        validate(i7, str(folder / I7_NAME))
        (folder / I7_NAME).write_text(json.dumps(i7, indent=1, ensure_ascii=False) + "\n")
        report.files += [pdf, folder / I7_NAME]
    return report


def clade_key(doc: Mapping[str, Any], schemes: Sequence[MapColourScheme]) -> dict[str, Any]:
    """One slot's key: each legend entry in scheme order with its dots per month and in total.

    Every entry of the scheme is listed, including ones with no dots in the window, so the
    intro page can choose what to show; ``window`` says which have any. A label on a dot that
    the schemes do not have is an error: the key would miss a colour the maps draw. A slot
    drawing several subtype rows (B: both lineages on one map) lists each row's scheme in turn.
    """
    per_label: dict[str, Counter[str]] = {}
    for period in doc["periods"]:
        for location in period["locations"]:
            for point in location["points"]:
                label = point.get("clade") or UNCOLOURED_KEY
                per_label.setdefault(label, Counter())[period["period"]] += point["count"]
    entries = []
    known = {UNCOLOURED_KEY}
    for row in (r for scheme in schemes for r in scheme.rows):
        if row.legend in known:
            continue  # two rows with one legend draw as one key entry
        known.add(row.legend)
        months = per_label.get(row.legend, Counter())
        entries.append(
            {
                "legend": row.legend,
                "keys": sorted(row.labels),
                "colour": row.colour,
                "per_month": dict(sorted(months.items())),
                "window": sum(months.values()),
            }
        )
    unknown = sorted(set(per_label) - known)
    if unknown:
        raise ValueError(f"{doc.get('subtype')}: dots labelled {unknown} are not in the scheme")
    uncoloured = per_label.get(UNCOLOURED_KEY, Counter())
    return {
        "schemes": [scheme.name for scheme in schemes],
        "months": [p["period"] for p in doc["periods"]],
        "entries": entries,
        "uncoloured": {
            "per_month": dict(sorted(uncoloured.items())),
            "window": sum(uncoloured.values()),
        },
    }
