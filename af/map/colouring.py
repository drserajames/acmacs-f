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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from af.chart.model import Chart
from af.clades.colours import ColourScheme as CladeColourScheme
from af.clades.colours import shadowed_entries
from af.map.config import ColouringConfig
from af.map.style import ColourRow, ColourScheme
from af.seq.matching_rules import MatchingRules
from af.store.store import Store

if TYPE_CHECKING:
    from af.serology.outputs import SubtypeColouring
    from af.serology.query import Preparation  # noqa: F401  (named in a cast)


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


@dataclass(frozen=True)
class ChartColours:
    """One chart's colours from the shared path, in the map's terms, with what they came from."""

    scheme: ColourScheme
    labels: tuple[frozenset[str], ...]  # per antigen: the chosen entry's key, or nothing
    sequenced: tuple[bool, ...]  # per antigen: matched to a sequence (so it could be painted)
    provenance: dict[str, Any]


class StoreColours:
    """Colours for every map of one run, read from the stores and the user's tables ONCE.

    The antigen -> sequence join and the user's colour tables are the same ones geo uses, so a
    virus is the same colour on a map and on a geo figure (Sarah, Q46). Built once per run: every
    figure of the run is coloured from the same read of the user's tables and the same join.
    """

    def __init__(self, store: Store, cfg: ColouringConfig, rules: MatchingRules) -> None:
        """``rules`` are the matcher's tables, loaded once by the caller
        (:func:`af.seq.matching_rules.matching_rules`), the same loader geo uses."""
        from af.serology import query
        from af.serology.joins import link_from_store, preparation_sequences
        from af.serology.outputs import CLADE_SUBTYPE, aligned_sequences, read_clade_tables

        assert cfg.nomenclature and cfg.acmacs_data  # checked by the config
        self._store = store
        self._clones = cfg.nomenclature
        self.rules = rules
        self.serology = store.current("serology", "all")
        con = query.connect(store.resolve(self.serology))
        self.links = link_from_store(con, store, rules, with_clades=True)
        self._sequences = preparation_sequences(con, rules.passages)
        self._aligned = aligned_sequences(store, con)
        self._user = read_clade_tables(
            store, cfg.nomenclature, cfg.acmacs_data, list(CLADE_SUBTYPE)
        )
        self._colourings: dict[tuple[str, str], SubtypeColouring] = {}

    def _colouring(self, subtype: str, scheme_name: str) -> SubtypeColouring:
        from af.serology.outputs import subtype_colouring

        key = (subtype, scheme_name)
        if key not in self._colourings:
            self._colourings[key] = subtype_colouring(
                self._store, self._clones, self._user, subtype, scheme_name
            )
        return self._colourings[key]

    def for_chart(self, chart: Chart, scheme_name: str) -> ChartColours:
        """Colour each antigen of ``chart`` with the user scheme ``scheme_name``.

        The subtype is the chart's own ("V"), never the folder name's (design rule 10).
        Antigens are looked up by preparation (name, reassortant, annotations, passage), the key
        the serology store groups them by.
        """
        from af.geo.colours import dot_styles

        subtype = str(chart.info.get("V", ""))
        colouring = self._colouring(subtype, scheme_name)
        style, counts = dot_styles(
            self._sequences,
            self._aligned.get_pair,
            colouring.scheme,
            colouring.clade_set,
            colouring.group_set,
        )
        keys = key_for_legend(colouring.scheme)
        labels, sequenced = [], []
        for a in chart.antigens:
            prep = _PreparationKey(subtype, a.name, a.reassortant, tuple(a.annotations), a.passage)
            labels.append(labels_for(style(cast("Preparation", prep)).label, keys))
            sequenced.append(prep.key() in self._sequences)
        scheme = map_scheme(colouring.scheme)
        # Rows a later row always overrides can never colour anything (trap T9). Not an error:
        # row order is the user's (Q80). Listed so a dead legend row is visible, not silent.
        shadowed = shadowed_entries(colouring.scheme, colouring.clade_set, colouring.group_set)
        provenance = {
            "source": "store",
            "scheme": scheme.name,
            "user_tables": {str(i.path): i.sha256 for i in colouring.inputs},
            "matching_rules": self.rules.provenance(),
            "coloured": dict(counts.coloured),
            "uncoloured": dict(counts.uncoloured),
            "shadowed_rows": [str(s) for s in shadowed],
        }
        return ChartColours(scheme, tuple(labels), tuple(sequenced), provenance)

    def store_refs(self) -> list[dict[str, str]]:
        """The store versions every store-coloured figure of this run was drawn from."""
        from af.clades.store import dataset_for
        from af.serology.outputs import CLADE_SUBTYPE

        refs = [self.serology.to_json()]
        for clade_subtype in sorted(set(CLADE_SUBTYPE.values())):
            refs.append(self._store.current("clades", dataset_for(clade_subtype)).to_json())
        return refs


@dataclass(frozen=True)
class _PreparationKey:
    """The fields :func:`af.geo.colours.dot_styles` reads from a preparation, for a chart antigen.

    A map antigen is not a serology :class:`~af.serology.query.Preparation` (it has no first lab
    or first table date), but the colouring reads only these five fields.
    """

    subtype: str
    name: str
    reassortant: str
    annotations: tuple[str, ...]
    passage: str

    def key(self) -> tuple[str, str, str, tuple[str, ...], str]:
        return (self.subtype, self.name, self.reassortant, self.annotations, self.passage)
