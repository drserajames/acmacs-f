"""Where a strain name's location is: country, region and coordinates, from af's own tables.

Keyed by the **location part of a normalised strain name** (``A(H3N2)/<location>/…``), so a table
antigen with no sequence finds the same entry as a sequenced virus of the same place. Two
sources, each for what it is good at:

- **GISAID metadata** (the sequence store) for country and region: what the submitter stated for
  each isolate. Per name-location the majority (country, region) is taken and the agreement is
  kept, so a location GISAID places in two countries shows as such.
- **acmacs-f-data** ``rules/locations/`` (:mod:`af.seq.places`) for coordinates, country identity
  and region schemes. Country is an ISO 3166 code from countries.tsv; a place row is keyed by
  (location, country), so a name several places share is never resolved silently.

af no longer reads locationdb here (LOCATIONS-PROPOSAL §5 step 6). places.tsv was seeded from the
locationdb-based lookup this module used to hold, and matched it dot for dot on the live store
before that lookup was removed, apart from listed rows (notes/sequences/PARITY-PLACES.md).

Nothing is guessed. A location with no row, or a row only in another country, has no
coordinates and is flagged and counted, never placed at a country centroid.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from af.seq.places import AmbiguousPlace, Countries, Place, Places, PlacesError, RegionSchemes
from af.store import Store

# Flags. Stable identifiers: counted and reported.
COUNTRY_DISAGREES = "location.country-disagrees"  # rows exist, none in GISAID's country
GISAID_CONFLICT = "location.gisaid-conflict"  # GISAID states >1 (country, region)
NO_PLACE = "location.no-place"  # places.tsv has no row for the name-location
AMBIGUOUS = "location.ambiguous"  # several rows and no GISAID country to choose
NO_COORDINATES = "location.no-coordinates"  # the row records none
NOT_SEQUENCED = "location.not-sequenced"  # no sequenced isolate: places.tsv only


@dataclass(frozen=True)
class Location:
    """What af knows about one name-location."""

    name_location: str
    country: str | None  # a code from countries.tsv
    region: str | None  # GISAID's continent
    latitude: float | None
    longitude: float | None
    isolates: int  # sequenced isolates behind country/region (0: places.tsv only)
    agreement: float | None  # share of those isolates in the majority (country, region)
    flags: tuple[str, ...] = ()


@dataclass
class LookupCounts:
    name_locations: int = 0
    flags: Counter[str] = field(default_factory=Counter)
    #: Isolates whose location has a places row in GISAID's country / only in another one.
    country_agree_isolates: int = 0
    country_disagree_isolates: int = 0

    def to_json(self) -> dict[str, object]:
        return {
            "name_locations": self.name_locations,
            "flags": dict(sorted(self.flags.items())),
            "country_agree_isolates": self.country_agree_isolates,
            "country_disagree_isolates": self.country_disagree_isolates,
        }


def name_location(name: str) -> str | None:
    """The location part of a normalised name (``TYPE/LOCATION/ISOLATE/YEAR``)."""
    parts = name.split("/")
    return parts[1] if len(parts) == 4 and parts[1] else None


def _store_rows(store: Store, datasets: Iterable[str]) -> list[tuple[str, str | None, str | None]]:
    """(name, GISAID country, GISAID region) per isolate whose name parsed cleanly."""
    paths = [
        str(store.resolve(store.current("sequences", dataset)) / "isolates" / "*" / "*.parquet")
        for dataset in datasets
    ]
    if not paths:
        raise ValueError("no sequence datasets given")
    rows: list[tuple[str, str | None, str | None]] = duckdb.execute(
        "select name, country, region from read_parquet(?)"
        " where len(list_filter(problems, x -> x like 'name.%')) = 0",
        [paths],
    ).fetchall()
    return rows


@dataclass(frozen=True)
class LocationTables:
    """acmacs-f-data ``rules/locations/``: the three tables, read and cross-checked."""

    countries: Countries
    regions: RegionSchemes
    places: Places

    @classmethod
    def read(cls, directory: Path) -> LocationTables:
        countries = Countries.read(directory / "countries.tsv")
        return cls(
            countries,
            RegionSchemes.read(directory / "regions.tsv", countries),
            Places.read(directory / "places.tsv", countries),
        )


class PlacesLookup:
    """Name-location -> :class:`Location` from af's tables; built by :func:`build_places_lookup`.

    ``lookup``, ``coordinates`` (the shape af.geo takes), ``counts`` for reports, and
    :meth:`group_of` for a region scheme (stat and the tree legend inject one).
    """

    def __init__(
        self, entries: dict[str, Location], tables: LocationTables, counts: LookupCounts
    ) -> None:
        self._entries = entries
        self._tables = tables
        self._unsequenced: dict[str, Location | None] = {}
        self.counts = counts

    def lookup(self, name_location: str) -> Location | None:
        """The entry for a name-location; for one no isolate had, its places.tsv row, or None."""
        found = self._entries.get(name_location)
        if found is not None:
            return found
        if name_location not in self._unsequenced:
            self._unsequenced[name_location] = self._unsequenced_entry(name_location)
        return self._unsequenced[name_location]

    def _unsequenced_entry(self, name_location: str) -> Location | None:
        """No GISAID statement: the bare name must be one place, its region GISAID's for it."""
        place, flags = _place(self._tables.places, name_location, None)
        if place is None:
            return (
                None
                if flags == (NO_PLACE,)
                else _location(name_location, None, None, 0, None, (NOT_SEQUENCED, *flags))
            )
        region = self._tables.regions.assigned("gisaid", place.country)
        return _location(name_location, place, region, 0, None, (NOT_SEQUENCED, *flags))

    def coordinates(self, name_location: str) -> tuple[float, float] | None:
        """(longitude, latitude), the shape af.geo takes; None when af does not know."""
        found = self.lookup(name_location)
        if found is None or found.latitude is None or found.longitude is None:
            return None
        return found.longitude, found.latitude

    def group_of(self, scheme: str) -> Callable[[str], str | None]:
        """name-location -> its country's group in ``scheme``; None when there is none.

        The scheme is checked now, so a misspelt one fails at wiring, not on the first dot.
        """
        regions = self._tables.regions
        if scheme not in regions.schemes:
            raise PlacesError(f"no region scheme {scheme!r}; have {sorted(regions.schemes)}")

        def group(name_location: str) -> str | None:
            found = self.lookup(name_location)
            if found is None or found.country is None:
                return None
            return regions.assigned(scheme, found.country)

        return group


def build_places_lookup(
    isolates: Iterable[tuple[str, str | None, str | None]], tables: LocationTables
) -> PlacesLookup:
    """Build from ``(normalised name, GISAID country, GISAID region)`` per sequenced isolate.

    Pass only cleanly parsed names: a name of the wrong shape has no reliable location part.
    GISAID's majority (country, region) per name-location; the coordinates are the places.tsv
    row for that name-location in GISAID's country.
    """
    stated: dict[str, Counter[tuple[str | None, str | None]]] = defaultdict(Counter)
    for name, country, region in isolates:
        location = name_location(name)
        if location is not None:
            stated[location][(country, region)] += 1
    spellings = {c for pairs in stated.values() for c, _ in pairs if c is not None}
    if unknown := sorted(s for s in spellings if ("gisaid", s) not in tables.countries.by_spelling):
        raise PlacesError(f"GISAID countries with no row in countries.tsv: {unknown}")

    counts = LookupCounts(name_locations=len(stated))
    entries = {}
    for location, pairs in stated.items():
        (country, region), top = pairs.most_common(1)[0]
        total = sum(pairs.values())
        code = tables.countries.code(country, "gisaid") if country is not None else None
        place, flags = _place(tables.places, location, code)
        if len(pairs) > 1:
            flags = (GISAID_CONFLICT, *flags)
        if place is not None:
            counts.country_agree_isolates += total
        elif COUNTRY_DISAGREES in flags:
            counts.country_disagree_isolates += total
        entry = _location(location, place, region, total, top / total, flags, code)
        counts.flags.update(entry.flags)
        entries[location] = entry
    return PlacesLookup(entries, tables, counts)


def _place(places: Places, location: str, code: str | None) -> tuple[Place | None, tuple[str, ...]]:
    """The row for a name-location (in a country when GISAID gives one), and why not if none."""
    try:
        place = places.resolve(location, code)
    except AmbiguousPlace:
        return None, (AMBIGUOUS,)
    if place is None:
        elsewhere = code is not None and bool(places.by_location.get(location))
        return None, (COUNTRY_DISAGREES if elsewhere else NO_PLACE,)
    return place, (() if place.coordinates is not None else (NO_COORDINATES,))


def _location(
    location: str,
    place: Place | None,
    region: str | None,
    isolates: int,
    agreement: float | None,
    flags: tuple[str, ...],
    code: str | None = None,
) -> Location:
    """A Location from a places row; GISAID's country when there is no row to give one."""
    country = place.country if place is not None else code
    latitude = place.latitude if place is not None else None
    longitude = place.longitude if place is not None else None
    return Location(location, country, region, latitude, longitude, isolates, agreement, flags)


def places_from_store(
    store: Store, datasets: Iterable[str], tables: LocationTables
) -> PlacesLookup:
    """Build from the CURRENT versions of ``sequences/<dataset>``, cleanly parsed names only."""
    return build_places_lookup(_store_rows(store, datasets), tables)
