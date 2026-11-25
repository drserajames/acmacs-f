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

Where the dots go is a named rule (Q280, :mod:`af.geo.placement`), and the one optional part of
the table, so a round that does not name one draws exactly as it did before Q280:

    placement = "lattice"             # or "piles": each location's own pile (the default)
    lattice_pitch = 6.4               # figure points between cells: the ring step
    lattice_order = "nearest-offer-first"
    lattice_reserve_point = true      # each location takes one cell on its point first
    lattice_max_move = 20.0           # optional: no dot moved further (figure points); unset
                                      # is uncapped, the decided rule

Reverting is one line: ``placement = "piles"`` (recorded as such in every figure), or no
``placement`` key at all (drawn and recorded exactly as before). The ``lattice_*`` keys are
required with ``"lattice"`` and an error without it (design rule 1).

Every other key is required and an unknown one is an error (design rule 4: no silent
defaults), as is a slot drawn with no table or a table for a slot not drawn (rule 1), and a look
whose two kinds of uncoloured dot would draw alike (Sarah, Q215: they are keyed on different
lines). The look each slot was drawn with goes into its figures' provenance
(:func:`look_record`), with the file's hash.
"""

from __future__ import annotations

import dataclasses
import re
import tomllib
from collections.abc import Collection
from pathlib import Path
from typing import Any

from af.geo.placement import LATTICE, ORDERS, PILES, RULES, Placement
from af.geo.records import CLADE_NOT_DEFINED, NOT_SEQUENCED
from af.geo.render import PACKINGS, Look, uncoloured_style

KEYS = (
    "packing", "coloured_edge", "coloured_edge_width", "uncoloured_edge_width",
    "clade_not_defined_fill", "clade_not_defined_edge", "clade_not_defined_edge_width",
)  # fmt: skip
PLACEMENT_KEYS = ("placement", "lattice_pitch", "lattice_order", "lattice_reserve_point",
                  "lattice_max_move")  # fmt: skip
LATTICE_REQUIRED = ("lattice_pitch", "lattice_order", "lattice_reserve_point")
UNLIKE = (NOT_SEQUENCED, CLADE_NOT_DEFINED)  # keyed on different lines, so drawn apart
WIDTHS = ("coloured_edge_width", "uncoloured_edge_width", "clade_not_defined_edge_width")
_COLOUR = re.compile(r"#[0-9a-fA-F]{6}")


class GeoStyleError(ValueError):
    """The figure-style file's geo tables are missing, incomplete or wrong."""


def read_geo_looks(path: Path, slots: Collection[str]) -> dict[str, Look]:
    """The look for each of ``slots`` from ``path``'s ``[geo.<slot>]`` tables."""
    tables = _tables(path, slots)
    return {slot: _look(path, slot, tables[slot]) for slot in sorted(slots)}


def read_geo_placements(path: Path, slots: Collection[str]) -> dict[str, Placement | None]:
    """The placement rule for each of ``slots``; None where the table names none, which draws
    and records exactly as before Q280."""
    tables = _tables(path, slots)
    return {slot: _placement(path, slot, tables[slot]) for slot in sorted(slots)}


def _tables(path: Path, slots: Collection[str]) -> dict[str, Any]:
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
    return tables


def _number(where: str, key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise GeoStyleError(f"{where}: {key} {value!r}: a distance in points, more than 0")
    return float(value)


def _placement(path: Path, slot: str, table: Any) -> Placement | None:
    where = f"{path} [geo.{slot}]"
    if not isinstance(table, dict):
        raise GeoStyleError(f"{where}: not a table")
    lattice_keys = sorted(k for k in table if k.startswith("lattice_"))
    if "placement" not in table:
        if lattice_keys:
            raise GeoStyleError(f'{where}: {lattice_keys} given without placement = "lattice"')
        return None
    rule = table["placement"]
    if rule not in RULES:
        raise GeoStyleError(f"{where}: placement {rule!r}: one of {list(RULES)}")
    if rule == PILES:
        if lattice_keys:
            raise GeoStyleError(f'{where}: {lattice_keys} given with placement = "piles"')
        return Placement(PILES)
    assert rule == LATTICE
    if missing := [k for k in LATTICE_REQUIRED if k not in table]:
        raise GeoStyleError(f'{where}: placement = "lattice" needs {missing}')
    order, reserve = table["lattice_order"], table["lattice_reserve_point"]
    if order not in ORDERS:
        raise GeoStyleError(f"{where}: lattice_order {order!r}: one of {list(ORDERS)}")
    if not isinstance(reserve, bool):
        raise GeoStyleError(f"{where}: lattice_reserve_point {reserve!r}: true or false")
    cap = table.get("lattice_max_move")
    return Placement(
        LATTICE,
        pitch=_number(where, "lattice_pitch", table["lattice_pitch"]),
        order=order,
        reserve_point=reserve,
        max_move=None if cap is None else _number(where, "lattice_max_move", cap),
    )


def _look(path: Path, slot: str, table: Any) -> Look:
    where = f"{path} [geo.{slot}]"
    if not isinstance(table, dict):
        raise GeoStyleError(f"{where}: not a table")
    if unknown := sorted(set(table) - set(KEYS) - set(PLACEMENT_KEYS)):
        known = ", ".join(KEYS + PLACEMENT_KEYS)
        raise GeoStyleError(f"{where}: unknown keys {unknown} (known: {known})")
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
