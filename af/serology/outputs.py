"""Geo maps and stat tables for one window, from the stores: the entry point reports call.

Everything is read from published store versions (serology, sequences) and explicit
files (the coastline, locationdb through :mod:`af.seq.locations`), so a report built from
the same refs gets the same figures. Where a dot is drawn, and which region an antigen
counts under, both come from the sequence workstream's location lookup, keyed by the
location part of the strain name: one vocabulary (GISAID's regions) for geo and stat.

Colours: ``colouring`` gives, per subtype, a colour scheme with its clade set and groups
(:mod:`af.clades.colours`). Each antigen is matched to its sequence
(:mod:`af.serology.joins`, the sequence workstream's matcher), the clade comes from the clade
store, and the aligned sequence from the sequence store, so groups defined by substitutions
are tested on the virus's own sequence. Without ``colouring`` every dot is uncoloured, on
purpose; a subtype missing from it is uncoloured too, and the report says so.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.clades.colours import ColourScheme
from af.clades.groups import GroupSet
from af.clades.nomenclature import CladeSet
from af.clades.sequence import AlignedSequence, GapSupport
from af.geo.colours import UNCOLOURED, ColourCounts, DotStyle, dot_styles
from af.geo.records import Month, geo_counts, to_i7
from af.geo.render import render_geo
from af.seq import locations
from af.seq.matching import read_passage_rules
from af.serology import query
from af.serology.joins import LinkCounts, link_from_store, preparation_sequences
from af.serology.query import Preparation
from af.stat.counts import stat_counts
from af.stat.output import Previous, write_stat
from af.store import Store, StoreRef

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


@dataclass
class OutputsReport:
    serology: StoreRef
    files: list[Path] = field(default_factory=list)
    geo_drawn: dict[str, dict[str, int]] = field(default_factory=dict)  # subtype -> month -> dots
    geo_unplaced: dict[str, int] = field(default_factory=dict)  # location -> dots not drawn
    geo_not_counted: dict[str, Any] = field(default_factory=dict)  # undated / no location
    stat_unknown_region: dict[str, int] = field(default_factory=dict)
    lookup: dict[str, object] = field(default_factory=dict)
    links: LinkCounts | None = None  # antigen -> sequence matching, when colouring
    colours: dict[str, ColourCounts] = field(default_factory=dict)  # subtype -> counts
    uncoloured_subtypes: list[str] = field(default_factory=list)


def make_geo_and_stat(
    store: Store,
    locationdb: Path,
    coastline: Path,
    first: Month,
    last: Month,
    out_dir: Path,
    *,
    previous_stat: Previous | None = None,
    colouring: Mapping[str, SubtypeColouring] | None = None,
    passage_rules: Path | None = None,
    split_by_lineage: tuple[str, ...] = ("B",),
) -> OutputsReport:
    """Write ``geo/<st>-records.json``, ``geo/<st>-YYYY-MM.pdf`` and ``stat/`` for a window."""
    serology = store.current("serology", "all")
    con = query.connect(store.resolve(serology))
    lookup = locations.from_store(store, SEQUENCE_DATASETS, locations.LocationDb.read(locationdb))
    preps, uses = query.preparations(con), query.serum_uses(con)
    report = OutputsReport(serology=serology, lookup=lookup.counts.to_json())

    style_of = None
    if colouring is not None:
        if passage_rules is None:
            raise ValueError("colouring needs passage_rules (the matcher's passage classes)")
        style_of = _styles(store, con, preps, colouring, passage_rules, report)
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
    passage_rules: Path,
    report: OutputsReport,
) -> Any:
    """Match antigens to sequences and clades, and give each preparation its dot style."""
    report.links = link_from_store(con, store, read_passage_rules(passage_rules), with_clades=True)
    links = preparation_sequences(con)
    aligned = _aligned_sequences(store, con)
    styles = {}
    for subtype, setting in colouring.items():
        styles[subtype], report.colours[subtype] = dot_styles(
            links, aligned.get_pair, setting.scheme, setting.clade_set, setting.group_set
        )
    report.uncoloured_subtypes = sorted({p.subtype for p in preps} - set(colouring))

    def style(prep: Preparation) -> DotStyle:
        found = styles.get(prep.subtype)
        return found(prep) if found is not None else UNCOLOURED

    return style


class _Aligned(dict[tuple[str, str], AlignedSequence]):
    def get_pair(self, epi_isl: str, accession: str) -> AlignedSequence | None:
        return self.get((epi_isl, accession))


def _aligned_sequences(store: Store, con: Any) -> _Aligned:
    """Aligned amino acids of every sequence an antigen matched, from the sequence store.

    Nextclade alignments of observed sequences: a gap there is a deletion.
    """
    paths = [
        path.as_posix()
        for ref in store.list_datasets("sequences")
        for path in sorted((store.resolve(ref) / "sequences").glob("*/*.parquet"))
    ]
    rows = con.execute(
        "SELECT s.epi_isl, s.accession, s.aa_aligned FROM read_parquet(?) s "
        "JOIN (SELECT DISTINCT epi_isl, accession FROM antigen_sequences "
        "      WHERE status = 'matched') m USING (epi_isl, accession) "
        "WHERE s.aa_aligned IS NOT NULL",
        [paths],
    ).fetchall()
    return _Aligned({(e, a): AlignedSequence(aa, gaps=GapSupport.OBSERVED) for e, a, aa in rows})
