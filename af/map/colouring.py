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

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from af.chart.model import Chart
from af.clades.colours import ColourScheme as CladeColourScheme
from af.clades.colours import ColourSchemeError, scheme_from_rows, shadowed_entries
from af.geo.colours import BASIS_GROUP_NO_CLADE
from af.map.config import ColouringConfig
from af.map.style import ColourRow, ColourScheme
from af.seq.matching_rules import MatchingRules
from af.serology.joins import preparation_key
from af.store.ref import StoreRef
from af.store.store import Store
from af.util.subtypes import Subtype

if TYPE_CHECKING:
    from af.clades.groups import GroupSet
    from af.clades.nomenclature import CladeSet
    from af.serology.outputs import SubtypeColouring


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


def groups_sha256(groups: GroupSet) -> str:
    """A content hash of a caller's groups (name, anchor, substitutions), as for its scheme."""
    rows = [[g.name, g.anchor, [str(s) for s in g.substitutions]] for g in groups.groups]
    text = json.dumps({"subtype": groups.subtype, "groups": rows})
    return hashlib.sha256(text.encode()).hexdigest()


def scheme_sha256(scheme: CladeColourScheme) -> str:
    """A content hash of a scheme's rows (order, key, legend, colour, group or clade), so a
    figure coloured by a scheme with no file of its own still says exactly what coloured it."""
    rows = [[e.order, e.key, e.legend, e.colour, e.is_group] for e in scheme.entries]
    text = json.dumps({"subtype": scheme.subtype, "name": scheme.name, "rows": rows})
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class ChartColours:
    """One chart's colours from the shared path, in the map's terms, with what they came from."""

    scheme: ColourScheme
    labels: tuple[frozenset[str], ...]  # per antigen: the chosen entry's key, or nothing
    sequenced: tuple[bool, ...]  # per antigen: matched to a sequence (so it could be painted)
    provenance: dict[str, Any]
    # per antigen: how it got its colour (af.geo.colours.BASIS_*), "" when uncoloured
    basis: tuple[str, ...] = ()


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
        from af.serology.outputs import aligned_sequences, labelled_rows, read_clade_tables

        assert cfg.nomenclature and cfg.acmacs_data  # checked by the config
        self._store = store
        self._clones = cfg.nomenclature
        self.rules = rules
        self.serology = store.current("serology", "all")
        con = query.connect(store.resolve(self.serology))
        self.links = link_from_store(con, store, rules, with_clades=True)
        # The clade tables the join read: clade sets are judged by these same versions, never by
        # a CURRENT re-read a moment later (it may have moved).
        self._clade_refs = {
            ref.dataset: ref for ref in (StoreRef.from_json(r) for r in self.links.refs["clades"])
        }
        self._sequences = preparation_sequences(con, rules.passages)
        self._aligned = aligned_sequences(store, con, self.links)
        self._user = read_clade_tables(
            store, cfg.nomenclature, cfg.acmacs_data, labelled_rows(), self._clade_refs
        )
        self._colourings: dict[tuple[str, str], SubtypeColouring] = {}
        self._sets: dict[str, tuple[CladeSet, GroupSet | None]] = {}

    def _colouring(self, row: str, scheme_name: str) -> SubtypeColouring:
        from af.serology.outputs import subtype_colouring

        key = (row, scheme_name)
        if key not in self._colourings:
            self._colourings[key] = subtype_colouring(
                self._store, self._clones, self._user, row, scheme_name, self._clade_refs.get(row)
            )
        return self._colourings[key]

    def _caller_colouring(
        self, row: str, scheme: CladeColourScheme, groups: GroupSet | None
    ) -> SubtypeColouring:
        """A scheme the caller built, checked exactly as a scheme from the user's tables is.

        Its keys are judged against the same clade set a named scheme is (the one the current
        ``clades/<subtype>`` table was labelled with), by af.clades' one validator, so a row
        naming no known clade or group is an error here too, never a row that silently colours
        nothing (design rule 1). Keys are taken as given: no legacy-name resolution.

        ``groups``, when given, are the caller's own and the only groups its rows may name
        (Sarah, 2 Oct 2026: a caller's groups stay in the caller's config); each anchor must be a
        clade of that same set. Without them, the rows may name the user's groups.
        """
        from af.serology.outputs import SubtypeColouring, clade_set_of
        from af.util.subtypes import subtypes

        clade_subtype = subtypes().by_key(row).name  # the scheme names it: "B/Vic"
        if scheme.subtype != clade_subtype:
            raise MapColouringError(
                f"colour scheme {scheme.name!r} is for {scheme.subtype}, the chart is "
                f"{clade_subtype}"
            )
        if row not in self._sets:  # clade_set_of refuses a row without clade labels (B/Yam)
            self._sets[row] = (
                clade_set_of(self._store, self._clones, row, self._clade_refs.get(row)),
                self._user.group_set(clade_subtype),
            )
        clade_set, group_set = self._sets[row]
        if groups is not None:
            unknown = [g.name for g in groups.groups if g.anchor and g.anchor not in clade_set]
            if groups.subtype != clade_subtype or unknown:
                raise MapColouringError(
                    f"groups for colour scheme {scheme.name!r}: "
                    + (
                        f"for {groups.subtype}, the chart is {clade_subtype}"
                        if groups.subtype != clade_subtype
                        else f"anchors of {', '.join(unknown)} are not clades of {clade_subtype} "
                        f"at {clade_set.version}"
                    )
                )
            group_set = groups
        rows = [
            (
                f"entry {n} ({entry.key!r})",
                {"order": str(entry.order), "key": entry.key, "legend": entry.legend,
                 "colour": entry.colour},
            )
            for n, entry in enumerate(scheme.entries, 1)
        ]  # fmt: skip
        try:
            checked = scheme_from_rows(
                rows, scheme.subtype, scheme.name, clade_set, group_set, scheme.source
            )
        except ColourSchemeError as exc:
            raise MapColouringError(str(exc)) from exc
        # The user's tables are an input only if a row names one of THEIR groups.
        uses_groups = groups is None and any(entry.is_group for entry in checked.entries)
        return SubtypeColouring(
            checked, clade_set, group_set, self._user.inputs if uses_groups else ()
        )

    def for_chart(
        self,
        chart: Chart,
        scheme: str | CladeColourScheme,
        *,
        groups: GroupSet | None = None,
    ) -> ChartColours:
        """Colour each antigen of ``chart`` with a scheme: a name from the user's tables, or a
        :class:`af.clades.colours.ColourScheme` the caller built (chain-pages-af's own rows),
        optionally with the caller's own ``groups`` (an :class:`af.clades.groups.GroupSet`).

        The subtype is the chart's own ("V"), never the folder name's (design rule 10).
        Antigens are looked up by preparation (name, reassortant, annotations, passage), the key
        the serology store groups them by. A caller's scheme is checked as a named one is, and
        its provenance says it came from the caller, with a hash of its rows (and of its groups).

        Each antigen's ``basis`` says how it got its colour: by its clade, or by a group needing
        no clade where the nomenclature names none (:mod:`af.geo.colours`).
        """
        from af.geo.colours import dot_styles

        row = chart_subtype(chart)
        minority = Counter(
            str(a.extra.get("L", "")) or "none"
            for a in chart.antigens
            if str(a.extra.get("L", "")) != row.ace_lineage
        )
        row_key = row.key
        if isinstance(scheme, str):
            if groups is not None:
                raise MapColouringError(
                    f"groups= is for a caller's own scheme; the named scheme {scheme!r} uses the "
                    "user's groups"
                )
            colouring = self._colouring(row_key, scheme)
        else:
            colouring = self._caller_colouring(row_key, scheme, groups)
        style, counts = dot_styles(
            self._sequences,
            self._aligned.get_pair,
            colouring.scheme,
            colouring.clade_set,
            colouring.group_set,
        )
        keys = key_for_legend(colouring.scheme)
        labels, sequenced, basis = [], [], []
        for i in range(chart.n_antigens):
            key = preparation_key(chart, "antigen", i)  # serology's one copy of the key (rule 6)
            dot = style(key)
            labels.append(labels_for(dot.label, keys))
            sequenced.append(key in self._sequences)
            basis.append(dot.basis)
        rows = map_scheme(colouring.scheme)
        # Rows a later row always overrides can never colour anything (trap T9). Not an error:
        # row order is the user's (Q80). Listed so a dead legend row is visible, not silent.
        shadowed = shadowed_entries(colouring.scheme, colouring.clade_set, colouring.group_set)
        provenance: dict[str, Any] = {
            "source": "store",
            "scheme": rows.name,
            "user_tables": {str(i.path): i.sha256 for i in colouring.inputs},
            "matching_rules": self.rules.provenance(),
            "coloured": dict(counts.coloured),
            "uncoloured": dict(counts.uncoloured),
            # How the join reached some of those colours, as geo reports them: refused name ties
            # coloured anyway, doubtful matches used, and preparations whose rows disagreed.
            "ties": dict(counts.ties),
            "doubtful": dict(counts.doubtful),
            "rows": dict(counts.rows),
            "shadowed_rows": [str(s) for s in shadowed],
            # clade tables labelled from another sequences version than the join read: dataset
            # -> [the version they labelled, the version read]. Empty when they agree.
            "clades_behind": {d: list(v) for d, v in sorted(self.links.clades_behind.items())},
            # antigens of another lineage than the map's (coloured by the map's row), by code
            "lineage_minority": dict(sorted(minority.items())),
            # how the coloured antigens got their colour, and which had no clade to go by
            "basis": dict(sorted(counts.basis.items())),
            "coloured_without_clade": [
                f"ag{i}" for i, b in enumerate(basis) if b == BASIS_GROUP_NO_CLADE
            ],
        }
        if not isinstance(scheme, str):
            provenance["scheme_origin"] = "caller-supplied"
            provenance["scheme_file"] = str(scheme.source) if scheme.source else None
            provenance["scheme_sha256"] = scheme_sha256(colouring.scheme)
        if groups is not None:
            provenance["groups_origin"] = "caller-supplied"
            provenance["groups_sha256"] = groups_sha256(groups)
        return ChartColours(rows, tuple(labels), tuple(sequenced), provenance, tuple(basis))

    def store_refs(self) -> list[dict[str, str]]:
        """The store versions every store-coloured figure of this run was drawn from: exactly
        the ones READ (serology, every sequences and clades dataset the join read), never
        CURRENT at the time of asking, which may have moved since."""
        refs = [self.serology.to_json()]
        refs += [self.links.refs["sequences"][d] for d in sorted(self.links.refs["sequences"])]
        refs += sorted(self.links.refs["clades"], key=lambda r: r["dataset"])
        return refs


def chart_subtype(chart: Chart) -> Subtype:
    """The chart's row in af's subtype table, from its own data: the chart's virus type ("V")
    and the lineage code ("L", e.g. "V"; none for A subtypes) most of its antigens carry.

    The one copy of this rule: the map build (titles, vaccines, lineage-minority report) and
    the store colouring both call it. Never from the folder name (design rule 10), and never a
    "B means B/Vic" default: a B chart whose antigens carry no lineage, or two lineages
    equally, is an error. A few antigens of another lineage (a lab testing an old B/Yamagata
    strain against B/Victoria sera) do not change the map's lineage; the build reports them
    (:func:`af.map.build.lineage_minority`), and the store colours them by the map's row.
    """
    from af.util.subtypes import SubtypeError, subtypes

    ranked = Counter(str(a.extra.get("L", "")) for a in chart.antigens).most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        counts = {c or "none": n for c, n in ranked}
        raise MapColouringError(
            f"chart carries antigen lineages equally {counts}: which map is this?"
        )
    try:
        return subtypes().for_chart(str(chart.info.get("V", "")), ranked[0][0] if ranked else "")
    except SubtypeError as exc:
        raise MapColouringError(str(exc)) from exc


def chart_row(chart: Chart) -> str:
    """The subtype row key (``af/subtypes.toml``) a chart is coloured by: :func:`chart_subtype`."""
    return chart_subtype(chart).key
