"""Per-slot geo looks from the round's figure-style file (``config/figures.toml``).

The shipped report drew each subtype's dots differently (ae's geographic settings, per subtype):
H1 and H3 coloured dots with no outline, B/Vic's with a grey one, and the uncoloured dots' rings
in different widths. Those are round style, so they are config, one ``[geo.<slot>]`` table per
report slot, beside the fonts (08-maps' ``[fonts]``, read and checked by its own reader):

    [geo.h3]
    packing = "ae-rings"          # or "honeycomb" (Q208)
    coloured_edge = "none"        # or a colour, e.g. "#808080"
    coloured_edge_width = 0.0     # figure points
    uncoloured_edge_width = 0.5   # figure points

Every key is required and an unknown one is an error (design rule 4: no silent defaults), as is
a slot drawn with no table or a table for a slot not drawn (rule 1). The look each slot was drawn
with goes into its figures' provenance (:func:`look_record`), with the file's hash.
"""

from __future__ import annotations

import dataclasses
import re
import tomllib
from collections.abc import Collection
from pathlib import Path
from typing import Any

from af.geo.render import PACKINGS, Look

KEYS = ("packing", "coloured_edge", "coloured_edge_width", "uncoloured_edge_width")
_COLOUR = re.compile(r"#[0-9a-fA-F]{6}")


class GeoStyleError(ValueError):
    """The figure-style file's geo tables are missing, incomplete or wrong."""


def read_geo_looks(path: Path, slots: Collection[str]) -> dict[str, Look]:
    """The look for each of ``slots`` from ``path``'s ``[geo.<slot>]`` tables."""
    if not path.is_file():
        raise GeoStyleError(f"figure-style file not found: {path}")
    with path.open("rb") as f:
        doc = tomllib.load(f)
    tables = doc.get("geo")
    if not isinstance(tables, dict):
        raise GeoStyleError(f"{path}: no [geo.<slot>] tables")
    if missing := sorted(set(slots) - set(tables)):
        raise GeoStyleError(f"{path}: no [geo.<slot>] table for drawn slots {missing}")
    if unused := sorted(set(tables) - set(slots)):
        raise GeoStyleError(f"{path}: [geo.<slot>] tables for slots not drawn: {unused}")
    return {slot: _look(path, slot, tables[slot]) for slot in sorted(slots)}


def _look(path: Path, slot: str, table: Any) -> Look:
    where = f"{path} [geo.{slot}]"
    if not isinstance(table, dict):
        raise GeoStyleError(f"{where}: not a table")
    if unknown := sorted(set(table) - set(KEYS)):
        raise GeoStyleError(f"{where}: unknown keys {unknown} (known: {', '.join(KEYS)})")
    if missing := [k for k in KEYS if k not in table]:
        raise GeoStyleError(f"{where}: missing keys {missing}")
    packing, edge = table["packing"], table["coloured_edge"]
    if packing not in PACKINGS:
        raise GeoStyleError(f"{where}: packing {packing!r}: one of {sorted(PACKINGS)}")
    if not isinstance(edge, str) or not (edge == "none" or _COLOUR.fullmatch(edge)):
        raise GeoStyleError(f"{where}: coloured_edge {edge!r}: '#rrggbb' or 'none'")
    widths = {k: table[k] for k in ("coloured_edge_width", "uncoloured_edge_width")}
    for key, value in widths.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            raise GeoStyleError(f"{where}: {key} {value!r}: a width in points, 0 or more")
    if widths["uncoloured_edge_width"] == 0:
        # an uncoloured dot is only its ring: at 0 it would vanish from the map
        raise GeoStyleError(f"{where}: uncoloured_edge_width 0 would hide uncoloured dots")
    width = widths["coloured_edge_width"]
    if (edge == "none") != (width == 0):
        raise GeoStyleError(f"{where}: coloured_edge {edge!r} with width {width}")
    return dataclasses.replace(
        Look(), packing=packing, coloured_edge=edge,
        coloured_edge_width=float(widths["coloured_edge_width"]),
        uncoloured_edge_width=float(widths["uncoloured_edge_width"]),
    )  # fmt: skip


def look_record(look: Look) -> dict[str, Any]:
    """Every drawing parameter a figure was drawn with, for its provenance (design rule 5)."""
    return dataclasses.asdict(look)
