"""Where a strain name's location is: country, region and coordinates.

Keyed by the **location part of a normalised strain name** (``A(H3N2)/<location>/…``), so
a table antigen with no sequence finds the same entry as a sequenced virus of the same
place. Two sources, each used for what it is good at:

- **GISAID metadata** (the sequence store) for country and region. It is what the
  submitter stated for that isolate, and the store has it for every sequenced virus.
  Per name-location the majority (country, region) is taken, and the agreement fraction
  is kept, so a location that GISAID places in two countries shows as such.
- **locationdb** (acmacs-data, read through :class:`LocationDb`) for coordinates, and for
  country and region of a name-location GISAID has never seen. Shared facts stay in
  acmacs-data during the transition (DECISIONS, 24 Sep 2026).

The two name countries differently ("RUSSIA" / "Russian Federation"), so a crosswalk
from locationdb's country to GISAID's is built from the data itself: every sequenced
isolate whose name-location resolves in locationdb pairs the two, and the majority wins.
Nobody maintains the table by hand, and its agreement is reported.

Nothing is guessed. A location locationdb cannot resolve has no coordinates; one where
locationdb and GISAID disagree on the country has no coordinates either (locationdb's
point would be in the wrong country); neither falls back to a country centroid. Each
case carries a flag and is counted.

Names are normalised with the location's hyphens made spaces (af.seq.names), but locationdb
keeps them (a Japanese city is ``SAITAMA-C``, apart from the prefecture ``SAITAMA``). So a
location locationdb does not know as written is looked up again with locationdb's own names
normalised the same way, and taken only when that leads to one place; a form that leads to
several stays unresolved and is counted (``LocationDb.hyphen_counts``).
"""

from __future__ import annotations

import json
import lzma
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from af.store import Store

# Flags. Stable identifiers: counted and reported.
NOT_IN_LOCATIONDB = "location.not-in-locationdb"
COUNTRY_DISAGREES = "location.country-disagrees"  # locationdb vs GISAID
GISAID_CONFLICT = "location.gisaid-conflict"  # GISAID states >1 (country, region)
FROM_LOCATIONDB = "location.from-locationdb"  # no sequenced isolate: locationdb only
COUNTRY_UNMAPPED = "location.country-unmapped"  # locationdb country never seen in GISAID


@dataclass(frozen=True)
class LocationDbEntry:
    name: str
    latitude: float
    longitude: float
    country: str
    division: str
    continent: str


class LocationDb:
    """acmacs-data's ``locationdb.json.xz``: replacements -> names -> locations."""

    def __init__(self, data: Mapping[str, Any], source: Path) -> None:
        self.source = source
        self.version = str(data.get("  version", ""))
        self.date = str(data.get(" date", ""))
        self._replacements = _mapping(data, "replacements", source)
        self._names = _mapping(data, "names", source)
        self._locations = _mapping(data, "locations", source)
        self._countries = _mapping(data, "countries", source)
        continents = data.get("continents")
        if not isinstance(continents, list):
            raise ValueError(f"{source}: no continents list")
        self._continents: list[str] = continents
        # locationdb's names and replacements with hyphens made spaces, as names are
        # normalised -> every name they lead to; only a single one is ever used.
        self._spaced: dict[str, set[str]] = defaultdict(set)
        for table in (self._replacements, self._names):
            for key in table:
                target = self._names.get(self._replacements.get(key, key))
                if target is not None and "-" in key:
                    self._spaced[_spaced(key)].add(target)
        # per resolve() call: "hyphen-form" (resolved that way) or "hyphen-ambiguous"
        self.hyphen_counts: Counter[str] = Counter()

    @classmethod
    def read(cls, path: Path) -> LocationDb:
        with lzma.open(path) as handle:
            return cls(json.load(handle), Path(path))

    def resolve(self, location: str) -> LocationDbEntry | None:
        name = self._names.get(self._replacements.get(location, location))
        if name is None and location in self._spaced:
            targets = self._spaced[location]
            if len(targets) == 1:
                (name,) = targets
                self.hyphen_counts["hyphen-form"] += 1
            else:
                self.hyphen_counts["hyphen-ambiguous"] += 1
        entry = self._locations.get(name) if name is not None else None
        if name is None or entry is None:
            return None
        latitude, longitude, country, division = entry
        index = self._countries.get(str(country))
        continent = self._continents[index] if index is not None else ""
        return LocationDbEntry(
            name, float(latitude), float(longitude), str(country), str(division), continent
        )


def _spaced(name: str) -> str:
    """A locationdb name as af.seq.names normalises a location part."""
    return name.replace("-", " ").strip()


def _mapping(data: Mapping[str, Any], key: str, source: Path) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{source}: no {key!r} table")
    return value


@dataclass(frozen=True)
class Location:
    """What af knows about one name-location."""

    name_location: str
    country: str | None  # in GISAID's naming
    region: str | None  # GISAID's continent
    latitude: float | None
    longitude: float | None
    isolates: int  # sequenced isolates behind country/region (0: locationdb only)
    agreement: float | None  # share of those isolates in the majority (country, region)
    flags: tuple[str, ...] = ()


@dataclass
class LookupCounts:
    name_locations: int = 0
    flags: Counter[str] = field(default_factory=Counter)
    #: Isolates whose locationdb country (crosswalked) matched / did not match GISAID's.
    country_agree_isolates: int = 0
    country_disagree_isolates: int = 0

    def to_json(self) -> dict[str, object]:
        return {
            "name_locations": self.name_locations,
            "flags": dict(sorted(self.flags.items())),
            "country_agree_isolates": self.country_agree_isolates,
            "country_disagree_isolates": self.country_disagree_isolates,
        }


class LocationLookup:
    """Name-location -> :class:`Location`, built by :func:`build_lookup`."""

    def __init__(
        self,
        entries: dict[str, Location],
        locationdb: LocationDb,
        crosswalk: Mapping[str, str],
        regions: Mapping[str, str],
        counts: LookupCounts,
    ) -> None:
        self._entries = entries
        self._locationdb = locationdb
        self._crosswalk = crosswalk
        self._regions = regions
        self.counts = counts

    def lookup(self, name_location: str) -> Location | None:
        """The entry for a name-location; for one no isolate had, locationdb's, or None."""
        found = self._entries.get(name_location)
        if found is not None:
            return found
        entry = self._locationdb.resolve(name_location)
        if entry is None:
            return None
        country = self._crosswalk.get(entry.country)
        if country is None:
            return Location(name_location, None, None, entry.latitude, entry.longitude, 0,
                            None, (FROM_LOCATIONDB, COUNTRY_UNMAPPED))  # fmt: skip
        return Location(name_location, country, self._regions.get(country), entry.latitude,
                        entry.longitude, 0, None, (FROM_LOCATIONDB,))  # fmt: skip

    def coordinates(self, name_location: str) -> tuple[float, float] | None:
        """(longitude, latitude), the shape af.geo takes; None when af does not know."""
        found = self.lookup(name_location)
        if found is None or found.latitude is None or found.longitude is None:
            return None
        return found.longitude, found.latitude


def name_location(name: str) -> str | None:
    """The location part of a normalised name (``TYPE/LOCATION/ISOLATE/YEAR``)."""
    parts = name.split("/")
    return parts[1] if len(parts) == 4 and parts[1] else None


def build_lookup(
    isolates: Iterable[tuple[str, str | None, str | None]], locationdb: LocationDb
) -> LocationLookup:
    """Build from ``(normalised name, GISAID country, GISAID region)`` per sequenced isolate.

    Pass only cleanly parsed names: a name of the wrong shape has no reliable location part.
    """
    stated: dict[str, Counter[tuple[str | None, str | None]]] = defaultdict(Counter)
    for name, country, region in isolates:
        location = name_location(name)
        if location is not None:
            stated[location][(country, region)] += 1
    resolved = {location: locationdb.resolve(location) for location in stated}
    crosswalk = _crosswalk(stated, resolved)
    regions = _country_regions(stated)

    counts = LookupCounts(name_locations=len(stated))
    entries = {}
    for location, pairs in stated.items():
        entry = _entry(location, pairs, resolved[location], crosswalk, counts)
        counts.flags.update(entry.flags)
        entries[location] = entry
    return LocationLookup(entries, locationdb, crosswalk, regions, counts)


def _entry(
    location: str,
    pairs: Counter[tuple[str | None, str | None]],
    ldb: LocationDbEntry | None,
    crosswalk: Mapping[str, str],
    counts: LookupCounts,
) -> Location:
    (country, region), top = pairs.most_common(1)[0]
    total = sum(pairs.values())
    flags = [GISAID_CONFLICT] if len(pairs) > 1 else []
    latitude = longitude = None
    if ldb is None:
        flags.append(NOT_IN_LOCATIONDB)
    elif crosswalk.get(ldb.country) == country:
        latitude, longitude = ldb.latitude, ldb.longitude
        counts.country_agree_isolates += total
    else:
        flags.append(COUNTRY_DISAGREES)
        counts.country_disagree_isolates += total
    return Location(
        location, country, region, latitude, longitude, total, top / total, tuple(flags)
    )


def _crosswalk(
    stated: Mapping[str, Counter[tuple[str | None, str | None]]],
    resolved: Mapping[str, LocationDbEntry | None],
) -> dict[str, str]:
    """locationdb country -> GISAID country, by isolate majority."""
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for location, pairs in stated.items():
        entry = resolved[location]
        if entry is None:
            continue
        for (country, _), n in pairs.items():
            if country is not None:
                votes[entry.country][country] += n
    return {ldb_country: c.most_common(1)[0][0] for ldb_country, c in votes.items()}


def _country_regions(
    stated: Mapping[str, Counter[tuple[str | None, str | None]]],
) -> dict[str, str]:
    """GISAID country -> GISAID region, by isolate majority."""
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for pairs in stated.values():
        for (country, region), n in pairs.items():
            if country is not None and region is not None:
                votes[country][region] += n
    return {country: c.most_common(1)[0][0] for country, c in votes.items()}


def from_store(store: Store, datasets: Iterable[str], locationdb: LocationDb) -> LocationLookup:
    """Build from the CURRENT versions of ``sequences/<dataset>``, cleanly parsed names only."""
    paths = [
        str(store.resolve(store.current("sequences", dataset)) / "isolates" / "*" / "*.parquet")
        for dataset in datasets
    ]
    if not paths:
        raise ValueError("no sequence datasets given")
    rows = duckdb.execute(
        "select name, country, region from read_parquet(?)"
        " where len(list_filter(problems, x -> x like 'name.%')) = 0",
        [paths],
    ).fetchall()
    return build_lookup(rows, locationdb)
