"""Geo maps and stat tables for one window, from the stores: the entry point reports call.

Everything is read from published store versions (serology, sequences) and explicit
files (the coastline, af's own location tables through :mod:`af.seq.locations`), so a report
built from the same refs and tables gets the same figures. Where a dot is drawn, and which
region an antigen counts under, both come from the sequence workstream's location lookup,
keyed by the location part of the strain name: one vocabulary (GISAID's regions) for geo and
stat. af no longer reads locationdb here (Sarah: "af should not use locationdb"); the
content hash of each location table is recorded in the report, so a map can be traced to the
table version it was drawn from.

Colours: ``colouring`` gives, per subtype, a colour scheme with its clade set and groups
(:mod:`af.clades.colours`). Each antigen is matched to its sequence
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
from af.serology.joins import LinkCounts, link_from_store, preparation_sequences
from af.serology.query import Preparation
from af.serology.rows import IdentityRules
from af.serology.update import require_current
from af.stat.counts import stat_counts
from af.stat.output import Previous, write_stat
from af.store import ExternalInput, Store, StoreRef
from af.util.artefacts import sha256_path

#: Store datasets whose isolates the location lookup learns from.
SEQUENCE_DATASETS = ("h1", "h3", "bvic", "byam")
#: File-name prefix per subtype, as today's geo/<st>-YYYY-MM.pdf.
GEO_PREFIX = {"A(H1N1)": "h1", "A(H3N2)": "h3", "B": "b"}


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
    colours: dict[str, ColourCounts] = field(default_factory=dict)  # subtype -> counts
    uncoloured_subtypes: list[str] = field(default_factory=list)


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
    split_by_lineage: tuple[str, ...] = ("B",),
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
    lookup = locations.places_from_store(store, SEQUENCE_DATASETS, tables)
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
    aligned = aligned_sequences(store, con)
    styles = {}
    for subtype, setting in colouring.items():
        styles[subtype], report.colours[subtype] = dot_styles(
            links, aligned.get_pair, setting.scheme, setting.clade_set, setting.group_set
        )
    report.uncoloured_subtypes = sorted({p.subtype for p in preps} - set(colouring))
    report.colour_inputs = {
        str(item.path): item.sha256 for setting in colouring.values() for item in setting.inputs
    }

    def style(prep: Preparation) -> DotStyle:
        found = styles.get(prep.subtype)
        return found(prep) if found is not None else UNCOLOURED

    return style


class AlignedSequences(dict[tuple[str, str], AlignedSequence]):
    """(epi_isl, accession) -> aligned sequence, with the two-argument lookup dot_styles takes."""

    def get_pair(self, epi_isl: str, accession: str) -> AlignedSequence | None:
        return self.get((epi_isl, accession))


def aligned_sequences(store: Store, con: Any) -> AlignedSequences:
    """Aligned amino acids of every sequence an antigen matched (doubtful matches included:
    colouring uses those ae uses) or tied on, from the sequence store (Q81).

    Nextclade alignments of observed sequences: a gap there is a deletion. Needs the
    ``antigen_sequences`` view (:func:`af.serology.joins.link_sequences`). Public because
    the antigenic maps colour through the same path as geo (Q46).
    """
    paths = [
        path.as_posix()
        for ref in store.list_datasets("sequences")
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


#: The clade store's subtype for a table subtype: B tables are coloured by the B/Victoria
#: clade set (there is no B/Yamagata clade table, by design; those dots stay uncoloured).
CLADE_SUBTYPE = {"A(H1N1)": "A(H1N1)", "A(H3N2)": "A(H3N2)", "B": "B/Vic"}


def clade_colouring(
    store: Store,
    clones: Path,
    acmacs_data: Path,
    schemes: Mapping[str, str],
) -> dict[str, SubtypeColouring]:
    """Per table subtype, the clade colouring geo uses (Sarah, Q46: by clade for now).

    ``schemes`` names the scheme per table subtype (e.g. the maps' "clades-v10" for H3). The
    user's schemes and groups are read once, from ``acmacs_data``, on this run
    (:func:`af.clades.importer.read_user_clades`), the same path the antigenic maps use, so
    geo and maps cannot colour a clade differently; see :func:`subtype_colouring`.

    This is the seam for other colourings later (the antigenic maps' extra colouring):
    anything that yields a :class:`SubtypeColouring`.
    """
    user = read_clade_tables(store, clones, acmacs_data, list(schemes))
    return {
        subtype: subtype_colouring(store, clones, user, subtype, name)
        for subtype, name in schemes.items()
    }


def read_clade_tables(
    store: Store, clones: Path, acmacs_data: Path, subtypes: Sequence[str]
) -> UserClades:
    """The user's schemes and groups, read once, for the clade sets of these table subtypes.

    Read once per run and handed to :func:`subtype_colouring` for every figure (geo per
    subtype, maps per map), so every figure of a run is coloured from the same read.
    """
    from af.clades.importer import read_user_clades

    return read_user_clades(
        acmacs_data, {_clade_subtype(s): _clade_set(store, clones, s) for s in subtypes}
    )


def subtype_colouring(
    store: Store, clones: Path, user: UserClades, subtype: str, scheme_name: str
) -> SubtypeColouring:
    """One subtype's colouring with one named scheme from ``user``.

    The clade set is the one the current ``clades/<subtype>`` table was labelled with
    (:func:`af.clades.store.clade_set_for`), so "is this within that clade?" is answered by
    the nomenclature revision that assigned the label. The user's tables' content hashes
    travel with the result, into whatever report the figure goes to.
    """
    clade_subtype = _clade_subtype(subtype)
    return SubtypeColouring(
        user.scheme(clade_subtype, scheme_name),
        _clade_set(store, clones, subtype),
        user.group_set(clade_subtype),
        inputs=user.inputs,
    )


def _clade_subtype(subtype: str) -> str:
    if subtype not in CLADE_SUBTYPE:
        raise ValueError(f"no clade set for table subtype {subtype!r}")
    return CLADE_SUBTYPE[subtype]


def _clade_set(store: Store, clones: Path, subtype: str) -> CladeSet:
    from af.clades.store import clade_set_for, dataset_for

    ref = store.current("clades", dataset_for(_clade_subtype(subtype)))
    return clade_set_for(store, ref, clones)
