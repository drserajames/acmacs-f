"""Per-slot geo looks from the round's figure-style file (``config/figures.toml``).

The shipped report drew each subtype's dots differently (ae's geographic settings, per subtype):
H1 and H3 coloured dots with no outline, B/Vic's with a grey one, and the uncoloured dots' rings
in different widths. Those are round style, so they are config, one ``[geo.<slot>]`` table per
report slot, beside the fonts (08-maps' ``[fonts]``, read and checked by its own reader):

    [geo.h3]
    packing = "ae-rings"          # or "honeycomb" (Q208)
    coloured_edge = "none"        # or a colour, e.g. "#808080"
    coloured_edge_width = 0.0     # figure points
    uncoloured_edge_width = 0.5   # figure points: a dot not sequenced, a ring only
    clade_not_defined_fill = "#ffffff"   # a sequenced dot whose clade is not defined (Q215),
    clade_not_defined_edge = "#606060"   # told apart from a not-sequenced one
    clade_not_defined_edge_width = 0.6

Every key is required and an unknown one is an error (design rule 4: no silent defaults), as is
a slot drawn with no table or a table for a slot not drawn (rule 1), and a look whose two kinds
of uncoloured dot would draw alike (Sarah, Q215: they are keyed on different lines). The look each
slot was drawn with goes into its figures' provenance (:func:`look_record`), with the file's hash.
"""

from __future__ import annotations

import dataclasses
import re
import tomllib
from collections.abc import Collection
from pathlib import Path
from typing import Any

from af.geo.records import CLADE_NOT_DEFINED, NOT_SEQUENCED
from af.geo.render import PACKINGS, Look, uncoloured_style

KEYS = (
    "packing", "coloured_edge", "coloured_edge_width", "uncoloured_edge_width",
    "clade_not_defined_fill", "clade_not_defined_edge", "clade_not_defined_edge_width",
)  # fmt: skip
UNLIKE = (NOT_SEQUENCED, CLADE_NOT_DEFINED)  # keyed on different lines, so drawn apart
WIDTHS = ("coloured_edge_width", "uncoloured_edge_width", "clade_not_defined_edge_width")
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
    widths = {k: table[k] for k in WIDTHS}
    for key, value in widths.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            raise GeoStyleError(f"{where}: {key} {value!r}: a width in points, 0 or more")
    if widths["uncoloured_edge_width"] == 0:
        # an uncoloured dot is only its ring: at 0 it would vanish from the map
        raise GeoStyleError(f"{where}: uncoloured_edge_width 0 would hide uncoloured dots")
    width = widths["coloured_edge_width"]
    if (edge == "none") != (width == 0):
        raise GeoStyleError(f"{where}: coloured_edge {edge!r} with width {width}")
    fill, ring = table["clade_not_defined_fill"], table["clade_not_defined_edge"]
    if not isinstance(fill, str) or not (fill == "none" or _COLOUR.fullmatch(fill)):
        raise GeoStyleError(f"{where}: clade_not_defined_fill {fill!r}: '#rrggbb' or 'none'")
    if not isinstance(ring, str) or not _COLOUR.fullmatch(ring):
        raise GeoStyleError(f"{where}: clade_not_defined_edge {ring!r}: '#rrggbb'")
    if fill == "none" and widths["clade_not_defined_edge_width"] == 0:
        raise GeoStyleError(f"{where}: clade_not_defined with no fill and no ring would vanish")
    look = dataclasses.replace(
        Look(), packing=packing, coloured_edge=edge,
        coloured_edge_width=float(widths["coloured_edge_width"]),
        uncoloured_edge_width=float(widths["uncoloured_edge_width"]),
        clade_not_defined_fill=fill, clade_not_defined_edge=ring,
        clade_not_defined_edge_width=float(widths["clade_not_defined_edge_width"]),
    )  # fmt: skip
    alike = [tuple(str(v).lower() for v in uncoloured_style(look, c)) for c in UNLIKE]
    if alike[0] == alike[1]:
        raise GeoStyleError(
            f"{where}: a dot whose clade is not defined would draw as a not-sequenced one"
        )
    return look


def look_record(look: Look) -> dict[str, Any]:
    """Every drawing parameter a figure was drawn with, for its provenance (design rule 5)."""
    return dataclasses.asdict(look)
