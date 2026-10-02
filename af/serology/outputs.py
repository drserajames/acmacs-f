"""Geo maps and stat tables for one window, from the stores: the entry point reports call.

Everything is read from published store versions (serology, sequences) and explicit
files (the coastline, af's own location tables through :mod:`af.seq.locations`), so a report
built from the same refs and tables gets the same figures. Where a dot is drawn, and which
region an antigen counts under, both come from the sequence workstream's location lookup,
keyed by the location part of the strain name: one vocabulary (GISAID's regions) for geo and
stat. af no longer reads locationdb here (Sarah: "af should not use locationdb"); the
content hash of each location table is recorded in the report, so a map can be traced to the
table version it was drawn from.

Colours: ``colouring`` gives, per subtype row of the subtype table (``af/subtypes.toml``:
"h1", "h3", "bvic"), a colour scheme with its clade set and groups (:mod:`af.clades.colours`).
A preparation's row is its table subtype and lineage; a B preparation whose tables leave the
lineage unknown takes the lineage of the sequence it matched. A row without clade labels
(B/Yam) is uncoloured with that reason. Each antigen is matched to its sequence
(:mod:`af.serology.joins`, the sequence workstream's matcher, told by ``matching``: the rule
tables :func:`af.seq.matching_rules.matching_rules` reads, the same ones the maps read, with
their hashes in the report), the clade comes from the clade
store, and the aligned sequence from the sequence store, so groups defined by substitutions
are tested on the virus's own sequence. Without ``colouring`` every dot is uncoloured, on
purpose; a subtype missing from it is uncoloured too, and the report says so.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.clades.colours import ColourScheme
from af.clades.groups import GroupSet
from af.clades.importer import UserClades
from af.clades.nomenclature import CladeSet
from af.clades.sequence import AlignedSequence, GapSupport
from af.geo.colours import UNCOLOURED, ColourCounts, DotStyle, dot_styles
from af.geo.records import Month, geo_counts, to_i7
from af.geo.render import render_geo
from af.seq import locations
from af.seq.matching_rules import MatchingRules
from af.serology import query
from af.serology.joins import (
    LinkCounts,
    PreparationSequence,
    link_from_store,
    preparation_sequences,
)
from af.serology.query import Preparation
from af.serology.rows import IdentityRules
from af.serology.update import require_current
from af.stat.counts import stat_counts
from af.stat.output import Previous, write_stat
from af.store import ExternalInput, Store, StoreError, StoreRef
from af.util.artefacts import sha256_path
from af.util.subtypes import Subtype, SubtypeError, subtypes


@dataclass(frozen=True)
class SubtypeColouring:
    """How one subtype's dots are coloured: a scheme and the clade set (and groups) it uses."""

    scheme: ColourScheme
    clade_set: CladeSet
    group_set: GroupSet | None = None
    inputs: tuple[ExternalInput, ...] = ()  # the user's tables the scheme was read from


@dataclass
class OutputsReport:
    serology: StoreRef
    files: list[Path] = field(default_factory=list)
    geo_drawn: dict[str, dict[str, int]] = field(default_factory=dict)  # subtype -> month -> dots
    geo_unplaced: dict[str, int] = field(default_factory=dict)  # location -> dots not drawn
    geo_not_counted: dict[str, Any] = field(default_factory=dict)  # undated / no location
    stat_unknown_region: dict[str, int] = field(default_factory=dict)
    lookup: dict[str, object] = field(default_factory=dict)
    location_tables: dict[str, str] = field(default_factory=dict)  # file -> sha256
    colour_inputs: dict[str, str] = field(default_factory=dict)  # scheme source -> sha256
    matching_inputs: dict[str, str] = field(default_factory=dict)  # matcher's table -> sha256
    matching_rules: dict[str, int] = field(default_factory=dict)  # rows per matcher table
    links: LinkCounts | None = None  # antigen -> sequence matching, when colouring
    colours: dict[str, ColourCounts] = field(default_factory=dict)  # subtype row -> counts
    uncoloured_subtypes: list[str] = field(default_factory=list)  # rows given no colouring
    # table subtype -> preparations left uncoloured because no row could be told: a B
    # preparation of unknown lineage with no matched sequence to say which
    unknown_lineage: dict[str, int] = field(default_factory=dict)


def make_geo_and_stat(
    store: Store,
    location_tables: Path,
    coastline: Path,
    first: Month,
    last: Month,
    out_dir: Path,
    *,
    previous_stat: Previous | None = None,
    colouring: Mapping[str, SubtypeColouring] | None = None,
    matching: MatchingRules | None = None,
    split_by_lineage: tuple[str, ...] | None = None,
    identity_rules: IdentityRules | None = None,
) -> OutputsReport:
    """Write ``geo/<st>-records.json``, ``geo/<st>-YYYY-MM.pdf`` and ``stat/`` for a window.

    Refuses when ``serology/all`` is behind the tables store (:func:`require_current`; the
    identity rules it must have been built with default to the chain rules).
    """
    require_current(store, identity_rules)
    serology = store.current("serology", "all")
    con = query.connect(store.resolve(serology))
    tables = locations.LocationTables.read(location_tables)
    lookup = locations.places_from_store(store, subtypes().keys(), tables)
    preps, uses = query.preparations(con), query.serum_uses(con)
    report = OutputsReport(
        serology=serology,
        lookup=lookup.counts.to_json(),
        location_tables={
            name: sha256_path(location_tables / name)
            for name in ("countries.tsv", "regions.tsv", "places.tsv")
        },
    )

    style_of = None
    if colouring is not None:
        if matching is None:
            raise ValueError("colouring needs matching rules (af.seq.matching_rules)")
        style_of = _styles(store, con, preps, colouring, matching, report)
    geo = geo_counts(preps, first, last, locations.name_location, style_of=style_of)
    geo_dir = out_dir / "geo"
    geo_dir.mkdir(parents=True, exist_ok=True)
    for subtype in sorted({s for s, _, _, _ in geo.dots}):
        prefix = subtypes().table_subtype(subtype).geo
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
        split_by_lineage=(
            subtypes().split_by_lineage() if split_by_lineage is None else split_by_lineage
        ),
    )  # fmt: skip
    report.files += write_stat(counts, first, last, out_dir / "stat", previous_stat)
    report.stat_unknown_region = dict(counts.unknown_continent)
    return report


def _styles(
    store: Store,
    con: Any,
    preps: list[Preparation],
    colouring: Mapping[str, SubtypeColouring],
    matching: MatchingRules,
    report: OutputsReport,
) -> Any:
    """Match antigens to sequences and clades, and give each preparation its dot style."""
    report.links = link_from_store(con, store, matching, with_clades=True)
    report.matching_inputs = matching.provenance()
    report.matching_rules = matching.counts()
    links = preparation_sequences(con, matching.passages)
    aligned = aligned_sequences(store, con, report.links)
    styles = {}
    for row, setting in colouring.items():
        styles[row], report.colours[row] = dot_styles(
            links, aligned.get_pair, setting.scheme, setting.clade_set, setting.group_set
        )
    rows = {p.key(): _row_of(p, links) for p in preps}
    report.uncoloured_subtypes = sorted({r for r in rows.values() if r} - set(colouring))
    report.colour_inputs = {
        str(item.path): item.sha256 for setting in colouring.values() for item in setting.inputs
    }
    seen: set[tuple[str, str, str, tuple[str, ...], str]] = set()

    def uncoloured(row: str, reason: str, key: Any) -> DotStyle:
        if key not in seen:  # counted once per preparation, as dot_styles counts
            seen.add(key)
            report.colours.setdefault(row, ColourCounts()).uncoloured[reason] += 1
        return UNCOLOURED

    def style(prep: Preparation) -> DotStyle:
        key = prep.key()
        row = rows[key]
        if row is None:
            if key not in seen:
                seen.add(key)
                report.unknown_lineage[prep.subtype] = (
                    report.unknown_lineage.get(prep.subtype, 0) + 1
                )
            return UNCOLOURED
        found = styles.get(row)
        if found is not None:
            return found(prep)
        if row not in labelled_rows():
            return uncoloured(row, f"no clade labels for {subtypes().by_key(row).name}", key)
        return uncoloured(row, "no colour scheme given", key)

    return style


def _row_of(prep: Preparation, links: Mapping[Any, PreparationSequence]) -> str | None:
    """The subtype row a preparation is: from its table subtype and lineage, or, for a B
    preparation whose tables leave the lineage unknown, the sequence dataset it matched in.
    None when neither tells (no such subtype, or no match to say which lineage)."""
    try:
        rows = subtypes().for_table(prep.subtype, prep.lineage)
    except SubtypeError:
        return None
    if len(rows) == 1:
        return rows[0].key
    found = links.get(prep.key())
    matched = [r.key for r in rows if found is not None and r.key in found.datasets]
    return matched[0] if len(matched) == 1 else None


class AlignedSequences(dict[tuple[str, str], AlignedSequence]):
    """(epi_isl, accession) -> aligned sequence, with the two-argument lookup dot_styles takes."""

    def get_pair(self, epi_isl: str, accession: str) -> AlignedSequence | None:
        return self.get((epi_isl, accession))


def aligned_sequences(store: Store, con: Any, links: LinkCounts) -> AlignedSequences:
    """Aligned amino acids of every sequence an antigen matched (doubtful matches included:
    colouring uses those ae uses) or tied on, from the sequence store (Q81).

    Nextclade alignments of observed sequences: a gap there is a deletion. Needs the
    ``antigen_sequences`` view (:func:`af.serology.joins.link_sequences`). Public because
    the antigenic maps colour through the same path as geo (Q46). ``links`` is that join's
    result: the alignments come from the sequence versions it read (``links.refs``), pinned
    or CURRENT, never from a version the join did not see.
    """
    if "sequences" not in links.refs:
        raise StoreError("aligned_sequences needs the refs of a link_from_store join")
    paths = [
        path.as_posix()
        for ref in (StoreRef.from_json(r) for r in links.refs["sequences"].values())
        for path in sorted((store.resolve(ref) / "sequences").glob("*/*.parquet"))
    ]
    rows = con.execute(
        "SELECT s.epi_isl, s.accession, s.aa_aligned FROM read_parquet(?) s "
        "JOIN (SELECT epi_isl, accession FROM antigen_sequences "
        "      WHERE status IN ('matched', 'doubtful') "
        "      UNION SELECT unnest(tied, recursive := true) FROM antigen_sequences) m "
        "USING (epi_isl, accession) WHERE s.aa_aligned IS NOT NULL",
        [paths],
    ).fetchall()
    return AlignedSequences(
        {(e, a): AlignedSequence(aa, gaps=GapSupport.OBSERVED) for e, a, aa in rows}
    )


def labelled_rows() -> tuple[str, ...]:
    """The subtype rows (``af/subtypes.toml`` keys) that have clade labels, in table order."""
    from af.clades.subtypes import clade_facts

    return tuple(row.key for row in subtypes() if clade_facts(row.name).labels)


def clade_colouring(
    store: Store,
    clones: Path,
    acmacs_data: Path,
    schemes: Mapping[str, str],
) -> dict[str, SubtypeColouring]:
    """Per subtype row, the clade colouring geo uses (Sarah, Q46: by clade for now).

    ``schemes`` names the scheme per subtype row (``af/subtypes.toml`` key: ``{"h3":
    "clades-v10", "bvic": "clades-v2"}``). A row without clade labels (B/Yam) is an error
    naming the reason, never an empty scheme. The user's schemes and groups are read once, from
    ``acmacs_data``, on this run (:func:`af.clades.importer.read_user_clades`), the same path
    the antigenic maps use, so geo and maps cannot colour a clade differently; see
    :func:`subtype_colouring`.

    This is the seam for other colourings later (the antigenic maps' extra colouring):
    anything that yields a :class:`SubtypeColouring`.
    """
    user = read_clade_tables(store, clones, acmacs_data, list(schemes))
    return {row: subtype_colouring(store, clones, user, row, name) for row, name in schemes.items()}


def read_clade_tables(
    store: Store,
    clones: Path,
    acmacs_data: Path,
    rows: Sequence[str],
    clade_refs: Mapping[str, StoreRef] | None = None,
) -> UserClades:
    """The user's schemes and groups, read once, for the clade sets of these subtype rows.

    Read once per run and handed to :func:`subtype_colouring` for every figure (geo per
    subtype, maps per map), so every figure of a run is coloured from the same read.
    ``clade_refs`` (row -> clades version) are the versions to judge by, e.g. the ones a join
    read; a row without one reads CURRENT.
    """
    from af.clades.importer import read_user_clades

    refs = clade_refs or {}
    return read_user_clades(
        acmacs_data,
        {_labelled(row).name: clade_set_of(store, clones, row, refs.get(row)) for row in rows},
    )


def subtype_colouring(
    store: Store,
    clones: Path,
    user: UserClades,
    row: str,
    scheme_name: str,
    clade_ref: StoreRef | None = None,
) -> SubtypeColouring:
    """One subtype row's colouring with one named scheme from ``user``.

    The clade set is the one the current ``clades/<row>`` table was labelled with
    (:func:`clade_set_of`), so "is this within that clade?" is answered by the nomenclature
    revision that assigned the label. The user's tables' content hashes travel with the
    result, into whatever report the figure goes to.
    """
    name = _labelled(row).name
    return SubtypeColouring(
        user.scheme(name, scheme_name),
        clade_set_of(store, clones, row, clade_ref),
        user.group_set(name),
        inputs=user.inputs,
    )


def clade_set_of(store: Store, clones: Path, row: str, ref: StoreRef | None = None) -> CladeSet:
    """The clade set the ``clades/<row>`` table was labelled with: the one set every colouring
    path (named schemes and a caller's own) judges keys by. ``ref`` is the table version (the
    one a join read, or a pinned one); without it, CURRENT."""
    from af.clades.store import clade_set_for

    dataset = _labelled(row).key  # refused before the store is read
    if ref is not None and ref.dataset != dataset:
        raise ValueError(f"clade_set_of({row!r}): ref is for clades/{ref.dataset}")
    return clade_set_for(store, ref or store.current("clades", dataset), clones)


def _labelled(row: str) -> Subtype:
    """The subtype row ``row``; an unknown row, or one without clade labels, is an error."""
    from af.clades.subtypes import labelled

    found = subtypes().by_key(row)
    labelled(found.name)  # B/Yam: "B/Yam has no clade labels: <the table's reason>"
    return found
