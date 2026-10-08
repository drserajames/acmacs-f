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
import re
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from af.chart.model import Chart
from af.clades.colours import ColourScheme as CladeColourScheme
from af.clades.colours import ColourSchemeError, scheme_from_rows, shadowed_entries
from af.geo.colours import BASIS_GROUP_NO_CLADE, STATE_UNOBSERVED, DotStyle
from af.map.config import ColouringConfig
from af.map.style import ColourRow, ColourScheme
from af.seq.matching_rules import MatchingRules
from af.serology.joins import PreparationKey, PreparationSequence, preparation_key
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
    # per antigen: the shared path's style, with the clade-evidence state of an unpainted one
    styles: tuple[DotStyle, ...] = ()


@dataclass(frozen=True)
class PreparationColours:
    """Colours for a list of preparations (:data:`af.serology.joins.PreparationKey`), in order.

    The shared rule behind :meth:`StoreColours.for_chart`, for callers whose points are
    preparations rather than chart antigens (geo: one dot per virus, from its preparations).
    """

    scheme: ColourScheme
    labels: tuple[frozenset[str], ...]  # per preparation: the chosen entry's key, or nothing
    sequenced: tuple[bool, ...]  # per preparation: matched to a sequence
    provenance: dict[str, Any]
    basis: tuple[str, ...]  # per preparation: af.geo.colours.BASIS_*, "" when uncoloured
    styles: tuple[DotStyle, ...]  # per preparation: legend text and colour, as geo draws them
    datasets: frozenset[str]  # the sequence datasets these colours can come from


@dataclass(frozen=True)
class Pins:
    """The store versions a caller pinned for :class:`StoreColours`; None: read CURRENT.

    Sequences and clades are pinned all or none: the join reads every ``sequences/*`` and every
    ``clades/*`` dataset, and a partial pin would mix pinned and CURRENT versions in one join.
    """

    serology: StoreRef | None = None
    sequences: Mapping[str, StoreRef] | None = None  # dataset -> ref, every dataset the join reads
    clades: tuple[StoreRef, ...] | None = None  # every clades dataset in the store

    def to_json(self) -> dict[str, str]:
        sequences = self.sequences or {}
        refs = [self.serology] if self.serology else []
        refs += [sequences[d] for d in sorted(sequences)]
        refs += list(self.clades or ())
        return {f"{r.kind}/{r.dataset}": r.version for r in refs}

    def check_clades_labelled(
        self,
        behind: Mapping[str, tuple[str | None, str]],
        datasets: Collection[str] | None = None,
    ) -> None:
        """Refuse a PINNED clades version behind the sequences the join read: its clade calls
        would change if it were relabelled from them. ``behind`` is the join's content-based
        ``clades_behind`` (dataset -> (labelled, read)). A table labelled from another version
        with identical call inputs is not behind, so it is read (and recorded in provenance as
        ``clades_same_content``): refusing it would guard against nothing (ruling, 5 Oct 2026).
        ``datasets``, when given, limits the check to the clades a chart colours with: a B/Vic
        chart is not refused over a behind H1 table it never reads (7 Oct 2026)."""
        wrong = [
            f"clades/{r.dataset}@{r.version} was labelled from "
            + (f"sequences/{r.dataset}@{behind[r.dataset][0]}" if behind[r.dataset][0]
               else "no single sequences version")
            + f", not the sequences/{r.dataset}@{behind[r.dataset][1]} read"
            for r in self.clades or ()
            if r.dataset in behind and (datasets is None or r.dataset in datasets)
        ]  # fmt: skip
        if wrong:
            raise MapColouringError(
                "pinned clades and sequences were not produced together: " + "; ".join(wrong)
            )


PINNABLE = ("serology", "sequences", "clades")


def pinned_refs(store: Store, versions: Mapping[str, str]) -> Pins:
    """Turn ``{"kind/dataset": version}`` into store refs for :class:`StoreColours`.

    Refused (design rule 1): a key StoreColours does not read (another kind, a serology dataset
    other than ``all``, a dataset the store does not hold), a version the store does not hold,
    and a partial pin of sequences or of clades (see :class:`Pins`).
    """
    from af.store.manifest import MANIFEST, Manifest
    from af.store.store import VERSIONS
    from af.util.subtypes import subtypes

    refs: dict[str, dict[str, StoreRef]] = {kind: {} for kind in PINNABLE}
    readable = {
        "serology": ["all"],
        "sequences": sorted(subtypes().keys()),
        "clades": [ref.dataset for ref in store.list_datasets("clades")],
    }
    for key, version in versions.items():
        kind, _, dataset = key.partition("/")
        if kind not in PINNABLE or dataset not in readable[kind]:
            raise MapColouringError(
                f"versions: StoreColours does not read {key!r} (it reads "
                + ", ".join(f"{k}/{d}" for k in PINNABLE for d in readable[k])
                + ")"
            )
        manifest = store.dataset_dir(kind, dataset) / VERSIONS / version / MANIFEST
        if not re.fullmatch(r"[0-9a-f]{16}", version) or not manifest.is_file():
            raise MapColouringError(f"versions: {key}@{version} is not in the store")
        sha = Manifest.from_bytes(manifest.read_bytes()).sha256()
        refs[kind][dataset] = StoreRef(kind, dataset, version, sha)
    for kind in ("sequences", "clades"):
        missing = sorted(set(readable[kind]) - set(refs[kind]))
        if refs[kind] and missing:
            raise MapColouringError(
                f"versions: {kind} are pinned all or none; not pinned: "
                + ", ".join(f"{kind}/{d}" for d in missing)
            )
    return Pins(
        serology=refs["serology"].get("all"),
        sequences=refs["sequences"] or None,
        clades=tuple(refs["clades"][d] for d in sorted(refs["clades"])) or None,
    )


class StoreColours:
    """Colours for every map of one run, read from the stores and the user's tables ONCE.

    The antigen -> sequence join and the user's colour tables are the same ones geo uses, so a
    virus is the same colour on a map and on a geo figure (Sarah, Q46). Built once per run: every
    figure of the run is coloured from the same read of the user's tables and the same join.
    """

    store_read: dict[str, Any] | None = None  # the guarded read, set by the constructor
    pins: Pins = Pins()  # the caller's pinned versions (none: everything read from CURRENT)

    def __init__(
        self,
        store: Store,
        cfg: ColouringConfig,
        rules: MatchingRules,
        *,
        versions: Mapping[str, str] | None = None,
        ignore_busy: bool = False,
    ) -> None:
        """``rules`` are the matcher's tables, loaded once by the caller
        (:func:`af.seq.matching_rules.matching_rules`), the same loader geo uses.

        Every read happens under one guard (:meth:`af.store.Store.reading`): no colouring while
        a batch is writing the store, and none whose inputs moved while it read. ``ignore_busy``
        reads anyway, for diagnosis; ``store_read`` records the guarded read for provenance.

        ``versions`` pins what is read instead of CURRENT, so a caller can reproduce its colours
        after CURRENT moves (design rule 5): ``{"serology/all": v, "sequences/h3": v,
        "clades/h3": v, ...}``. See :func:`pinned_refs` for what is refused. af's own round
        builds never pin: they colour from CURRENT behind the serology staleness guard
        (:func:`af.serology.update.require_current`), which a pinned serology skips."""
        self.pins = pinned_refs(store, versions or {})
        with store.reading("map-colours", override=ignore_busy) as guard:
            self._read(store, cfg, rules)
        self.store_read = guard.to_json()

    def _read(self, store: Store, cfg: ColouringConfig, rules: MatchingRules) -> None:
        from af.serology import query
        from af.serology.joins import link_from_store, preparation_sequences
        from af.serology.outputs import aligned_sequences, labelled_rows, read_clade_tables

        assert cfg.nomenclature and cfg.acmacs_data  # checked by the config
        self._store = store
        self._clones = cfg.nomenclature
        self.rules = rules
        self.serology = self.pins.serology or store.current("serology", "all")
        con = query.connect(store.resolve(self.serology))
        self.links = link_from_store(
            con,
            store,
            rules,
            with_clades=True,
            sequences=self.pins.sequences,
            clades=self.pins.clades,
        )
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
        row = chart_subtype(chart)
        minority = Counter(
            str(a.extra.get("L", "")) or "none"
            for a in chart.antigens
            if str(a.extra.get("L", "")) != row.ace_lineage
        )
        keys = [preparation_key(chart, "antigen", i) for i in range(chart.n_antigens)]
        found = self.for_preparations(
            keys, row.key, scheme, groups=groups, lineage_minority=dict(sorted(minority.items()))
        )
        return ChartColours(
            found.scheme, found.labels, found.sequenced, found.provenance, found.basis,
            found.styles,
        )  # fmt: skip

    def for_preparations(
        self,
        keys: Sequence[PreparationKey],
        row_key: str,
        scheme: str | CladeColourScheme,
        *,
        groups: GroupSet | None = None,
        lineage_minority: Mapping[str, int] | None = None,
    ) -> PreparationColours:
        """Colour preparations of one subtype row (``"h3"``, ``"bvic"``) with a scheme.

        The one copy of the rule: :meth:`for_chart` keys a chart's antigens and calls this.
        Scheme and groups as for :meth:`for_chart`. A pinned clades table behind its sequences
        refuses the call only if these preparations' datasets read it. ``lineage_minority``
        (antigens of another lineage than a chart's, by code) is a chart's, recorded as given;
        ``coloured_without_clade`` names preparations by their index in ``keys`` (``ag<i>``,
        the chart's antigen index when a chart called).
        """
        from af.geo.colours import dot_styles

        datasets = self.preparation_datasets(keys, row_key)
        # A pinned clades table behind its sequences refuses only the calls that colour from it.
        self.pins.check_clades_labelled(self.links.clades_behind, datasets)
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
        legend_keys = key_for_legend(colouring.scheme)
        labels, sequenced, basis, styles = [], [], [], []
        for key in keys:
            dot = style(key)
            labels.append(labels_for(dot.label, legend_keys))
            # An unobserved tree clade is not known: drawn and counted as an unsequenced antigen
            # (Sarah, 8 Oct 2026), not as a sequenced one the scheme failed to paint.
            sequenced.append(key in self._sequences and dot.state != STATE_UNOBSERVED)
            basis.append(dot.basis)
            styles.append(dot)
        rows = map_scheme(colouring.scheme)
        # Rows a later row always overrides can never colour anything (trap T9). Not an error:
        # row order is the user's (Q80). Listed so a dead legend row is visible, not silent.
        shadowed = shadowed_entries(colouring.scheme, colouring.clade_set, colouring.group_set)
        provenance: dict[str, Any] = {
            "source": "store",
            "scheme": rows.name,
            "user_tables": {str(i.path): i.sha256 for i in colouring.inputs},
            "matching_rules": self.rules.provenance(),
            "store_read": self.store_read,
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
            "clades_behind": {
                d: list(v) for d, v in sorted(self.links.clades_behind.items()) if d in datasets
            },
            # clade tables labelled from another sequences version whose calls' inputs are
            # identical: not behind, recorded so the version pair stays visible
            "clades_same_content": {
                d: list(v)
                for d, v in sorted(self.links.clades_same_content.items())
                if d in datasets
            },
            # the sequence datasets this chart's colours come from (its own row, plus any other
            # its antigens matched in): the only ones its figure cites (store_refs(datasets))
            "datasets": sorted(datasets),
            # antigens of another lineage than the map's (coloured by the map's row), by code
            **(
                {"lineage_minority": dict(lineage_minority)} if lineage_minority is not None else {}
            ),
            # how the coloured antigens got their colour, and which had no clade to go by
            "basis": dict(sorted(counts.basis.items())),
            # tree-derived clades: each matched record's clade_evidence, and the supported clades
            # the scheme has no row for (preparations); empty for Nextclade-derived clades
            "clade_evidence": dict(sorted(counts.evidence.items())),
            "unpainted_clades": dict(sorted(counts.unpainted_clades.items())),
            "coloured_without_clade": [
                f"ag{i}" for i, b in enumerate(basis) if b == BASIS_GROUP_NO_CLADE
            ],
            # the caller's pinned versions ("kind/dataset" -> version); empty: all CURRENT
            "pins": self.pins.to_json(),
        }
        if self.pins.serology is not None:
            # The staleness guard compares serology CURRENT with tables CURRENT; a pinned
            # serology is the caller's choice of what to read, so it is not applied.
            provenance["serology_guard"] = "pinned: guard not applied"
        if not isinstance(scheme, str):
            provenance["scheme_origin"] = "caller-supplied"
            provenance["scheme_file"] = str(scheme.source) if scheme.source else None
            provenance["scheme_sha256"] = scheme_sha256(colouring.scheme)
        if groups is not None:
            provenance["groups_origin"] = "caller-supplied"
            provenance["groups_sha256"] = groups_sha256(groups)
        return PreparationColours(
            rows, tuple(labels), tuple(sequenced), provenance, tuple(basis), tuple(styles),
            datasets,
        )  # fmt: skip

    @property
    def sequences(self) -> Mapping[PreparationKey, PreparationSequence]:
        """Each preparation's sequence as the run's one join found it (read-only): geo reads it to
        tell which subtype row a B preparation of unknown lineage is, from the record it matched."""
        return MappingProxyType(self._sequences)

    def store_refs(self, datasets: Collection[str] | None = None) -> list[dict[str, str]]:
        """The store versions a store-coloured figure was drawn from: exactly the ones READ
        (serology, and the sequences and clades the join read), never CURRENT at the time of
        asking, which may have moved since.

        ``datasets`` limits sequences and clades to the ones a chart colours with (its
        provenance ``datasets``, :meth:`chart_datasets`): a figure cites what its colours
        depend on, so a B/Vic figure is not held stale by an H1 clade table (7 Oct 2026).
        Without it, every dataset the run's join read."""
        wanted = (lambda d: True) if datasets is None else (lambda d: d in datasets)
        sequences = self.links.refs["sequences"]
        refs = [self.serology.to_json()]
        refs += [sequences[d] for d in sorted(sequences) if wanted(d)]
        refs += sorted(
            (r for r in self.links.refs["clades"] if wanted(r["dataset"])),
            key=lambda r: r["dataset"],
        )
        return refs

    def chart_datasets(self, chart: Chart, row: str) -> frozenset[str]:
        """The sequence datasets a chart's colours can come from: its own subtype row, plus
        every dataset its antigens' preparations matched a sequence in.

        A preparation is matched only within its own table subtype and lineage
        (:mod:`af.serology.joins`), so this is the chart's row today; a Yamagata-lineage antigen
        on a B/Victoria chart would add byam. The row is always in, so a chart with no
        sequenced antigen still names what it was coloured against."""
        keys = [preparation_key(chart, "antigen", i) for i in range(chart.n_antigens)]
        return self.preparation_datasets(keys, row)

    def preparation_datasets(self, keys: Sequence[PreparationKey], row: str) -> frozenset[str]:
        """:meth:`chart_datasets` for a list of preparations: the row, plus every dataset they
        matched a sequence in."""
        found: set[str] = {row}
        for key in keys:
            prep = self._sequences.get(key)
            if prep is not None:
                found |= prep.datasets
        return frozenset(found)


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
