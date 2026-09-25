"""Geo maps and stat tables for one window, from the stores: the entry point reports call.

Everything is read from published store versions (serology, sequences) and explicit
files (the coastline, locationdb through :mod:`af.seq.locations`), so a report built from
the same refs gets the same figures. Where a dot is drawn, and which region an antigen
counts under, both come from the sequence workstream's location lookup, keyed by the
location part of the strain name: one vocabulary (GISAID's regions) for geo and stat.

Colours need clade assignments; until the clade store exists, ``colour=None`` draws every
dot uncoloured, deliberately.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.geo.colours import DotStyle
from af.geo.records import Month, geo_counts, to_i7
from af.geo.render import render_geo
from af.seq import locations
from af.serology import query
from af.serology.query import Preparation
from af.stat.counts import stat_counts
from af.stat.output import Previous, write_stat
from af.store import Store, StoreRef

#: Store datasets whose isolates the location lookup learns from.
SEQUENCE_DATASETS = ("h1", "h3", "bvic", "byam")
#: File-name prefix per subtype, as today's geo/<st>-YYYY-MM.pdf.
GEO_PREFIX = {"A(H1N1)": "h1", "A(H3N2)": "h3", "B": "b"}


@dataclass
class OutputsReport:
    serology: StoreRef
    files: list[Path] = field(default_factory=list)
    geo_drawn: dict[str, dict[str, int]] = field(default_factory=dict)  # subtype -> month -> dots
    geo_unplaced: dict[str, int] = field(default_factory=dict)  # location -> dots not drawn
    geo_not_counted: dict[str, Any] = field(default_factory=dict)  # undated / no location
    stat_unknown_region: dict[str, int] = field(default_factory=dict)
    lookup: dict[str, object] = field(default_factory=dict)


def make_geo_and_stat(
    store: Store,
    locationdb: Path,
    coastline: Path,
    first: Month,
    last: Month,
    out_dir: Path,
    *,
    previous_stat: Previous | None = None,
    colour: Callable[[Preparation], DotStyle] | None = None,
    split_by_lineage: tuple[str, ...] = ("B",),
) -> OutputsReport:
    """Write ``geo/<st>-records.json``, ``geo/<st>-YYYY-MM.pdf`` and ``stat/`` for a window."""
    serology = store.current("serology", "all")
    con = query.connect(store.resolve(serology))
    lookup = locations.from_store(store, SEQUENCE_DATASETS, locations.LocationDb.read(locationdb))
    preps, uses = query.preparations(con), query.serum_uses(con)
    report = OutputsReport(serology=serology, lookup=lookup.counts.to_json())

    geo = geo_counts(preps, first, last, locations.name_location, style_of=colour)
    geo_dir = out_dir / "geo"
    geo_dir.mkdir(parents=True, exist_ok=True)
    for subtype in sorted({s for s, _, _, _ in geo.dots}):
        prefix = GEO_PREFIX.get(subtype, subtype)
        doc = to_i7(geo, subtype)
        records = geo_dir / f"{prefix}-records.json"
        records.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        drawn = render_geo(doc, lookup.coordinates, coastline, geo_dir, prefix)
        report.files += [records, *drawn.files]
        report.geo_drawn[subtype] = drawn.drawn
        for name, n in drawn.no_coordinates.items():
            report.geo_unplaced[name] = report.geo_unplaced.get(name, 0) + n
    report.geo_not_counted = {
        "undated": dict(geo.undated),
        "no_location": len(geo.no_location),
    }

    def region(location: str) -> str | None:
        found = lookup.lookup(location)
        return found.region if found is not None else None

    counts = stat_counts(
        preps, uses, first, last, locations.name_location, region,
        split_by_lineage=split_by_lineage,
    )  # fmt: skip
    report.files += write_stat(counts, first, last, out_dir / "stat", previous_stat)
    report.stat_unknown_region = dict(counts.unknown_continent)
    return report
