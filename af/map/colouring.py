"""Map colours from the shared clade colouring: the same path geo uses (Sarah, Q46).

Which entry of a user colour scheme colours a virus is decided once, in
:meth:`af.clades.colours.ColourScheme.entry_for` (reached through
:func:`af.geo.colours.dot_styles`), from the virus's store clade and its aligned sequence. The
map does not decide it again. This module only turns that decision into the map's own terms:

- the scheme becomes the map's legend rows, in the scheme's order, one row per entry, each keyed
  by the entry's key alone;
- each antigen gets the single key of the entry chosen for it, so the map's "last matching row
  wins" painting has exactly one row to find and cannot overrule the shared choice.

Why not paint by the chart's clade labels as the stand-in does: those labels are the old system's,
and painting them in row order is a second precedence rule. Two rules would let a map and a geo
dot draw the same virus in different colours.
"""

from __future__ import annotations

from collections.abc import Mapping

from af.clades.colours import ColourScheme as CladeColourScheme
from af.map.style import ColourRow, ColourScheme


class MapColouringError(ValueError):
    """A scheme cannot be drawn on a map, or a colour names no legend row."""


def map_scheme(scheme: CladeColourScheme) -> ColourScheme:
    """The map's legend rows for a user scheme, in the scheme's order.

    A legend text shared by two entries is an error: the map resolves a chosen colour back to its
    row by legend text, and a shared one would count two entries' antigens under one row.
    """
    entries = sorted(scheme.entries, key=lambda entry: entry.order)
    seen: dict[str, str] = {}
    for entry in entries:
        if entry.legend in seen:
            raise MapColouringError(
                f"colour scheme {scheme.name!r} ({scheme.subtype}): legend {entry.legend!r} "
                f"is used by both {seen[entry.legend]!r} and {entry.key!r}"
            )
        seen[entry.legend] = entry.key
    rows = tuple(ColourRow(e.legend, e.colour, frozenset({e.key})) for e in entries)
    return ColourScheme(f"{scheme.subtype} {scheme.name}", rows)


def key_for_legend(scheme: CladeColourScheme) -> Mapping[str, str]:
    """Legend text -> entry key, to turn a chosen dot style back into the row that drew it."""
    return {entry.legend: entry.key for entry in scheme.entries}


def labels_for(legend: str, keys: Mapping[str, str]) -> frozenset[str]:
    """The one label an antigen carries on the map: the key of the entry chosen for it.

    An empty legend is the shared path's "uncoloured" (no sequence, no clade, not in the
    scheme); the antigen then carries no label and the map leaves it unpainted.
    """
    if not legend:
        return frozenset()
    if legend not in keys:
        raise MapColouringError(f"chosen colour {legend!r} is not a row of the colour scheme")
    return frozenset({keys[legend]})
