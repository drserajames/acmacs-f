"""af's own place tables: country identity and region schemes (replacing locationdb).

Why these exist: locationdb was a database nobody here maintains, and it spelled countries its
own way, so every consumer inherited its spellings and a crosswalk. Here (acmacs-f-data
``rules/locations/``, LOCATIONS-PROPOSAL.md §6):

- **Country identity is a code** (ISO 3166 alpha-3, ``XKX`` for Kosovo as WHO uses, and
  ``non-country:*`` for placeholders). ``countries.tsv`` maps each source's spelling to it,
  one row per (code, spelling, source) with a reason. A spelling with no row is an error that
  names the value: a new one stops the run instead of silently becoming "unknown".
- **Region groupings are schemes** (``regions.tsv``: scheme, country, group). Every country has
  a value in every scheme. "Not assigned" is a recorded value with a reason, never a blank,
  and af never fills a blank itself (Sarah: blank not filled from consensus). A new scheme is
  new rows, not code.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from af.seq.names import normalise

REQUIRED_COUNTRY = ("code", "spelling", "source", "evidence")
REQUIRED_REGION = ("scheme", "country", "group", "evidence")


class PlacesError(ValueError):
    """A place table that is inconsistent, or a value it has no row for."""


def _rows(path: Path, required: tuple[str, ...]) -> list[tuple[int, dict[str, str]]]:
    """(line, row) for each data row; comments and blank lines skipped, cells stripped."""
    with path.open(newline="", encoding="utf-8") as handle:
        numbered = [
            (n, line) for n, line in enumerate(handle, 1) if line.strip() and line[0] != "#"
        ]
    reader = csv.DictReader([line for _, line in numbered], delimiter="\t")
    if missing := [c for c in required if c not in (reader.fieldnames or [])]:
        raise PlacesError(f"{path}: missing columns {missing}")
    out = []
    for (line, _), row in zip(numbered[1:], reader, strict=True):
        if None in row:
            raise PlacesError(f"{path}:{line}: more cells than columns")
        values = {k: (v or "").strip() for k, v in row.items()}
        if not values["evidence"]:
            raise PlacesError(f"{path}:{line}: no evidence")
        out.append((line, values))
    return out


@dataclass(frozen=True)
class Countries:
    """Spelling -> country code, per source (gisaid, locationdb, who-gho, ...)."""

    by_spelling: dict[tuple[str, str], str]  # (source, spelling) -> code
    codes: frozenset[str]

    @classmethod
    def read(cls, path: Path) -> Countries:
        found: dict[tuple[str, str], str] = {}
        for line, row in _rows(path, REQUIRED_COUNTRY):
            key = (row["source"], row["spelling"])
            if key in found and found[key] != row["code"]:
                raise PlacesError(
                    f"{path}:{line}: {row['source']} spelling {row['spelling']!r} is both"
                    f" {found[key]} and {row['code']}"
                )
            found[key] = row["code"]
        return cls(found, frozenset(found.values()))

    def code(self, spelling: str, source: str) -> str:
        """The country a source's spelling means; an unknown spelling is an error, not a guess."""
        try:
            return self.by_spelling[(source, spelling)]
        except KeyError:
            raise PlacesError(
                f"country {spelling!r} from {source} has no row in countries.tsv"
            ) from None


@dataclass(frozen=True)
class RegionSchemes:
    """(scheme, country code) -> group, complete for every scheme."""

    groups: dict[tuple[str, str], str]
    schemes: frozenset[str]

    @classmethod
    def read(cls, path: Path, countries: Countries) -> RegionSchemes:
        groups: dict[tuple[str, str], str] = {}
        covered: dict[str, set[str]] = defaultdict(set)
        for line, row in _rows(path, REQUIRED_REGION):
            scheme, code, group = row["scheme"], row["country"], row["group"]
            if code not in countries.codes:
                raise PlacesError(f"{path}:{line}: country {code!r} is not in countries.tsv")
            if not group:
                raise PlacesError(
                    f"{path}:{line}: blank group; record 'not assigned' with a reason"
                )
            if (scheme, code) in groups:
                raise PlacesError(f"{path}:{line}: {scheme} {code} has two rows")
            groups[(scheme, code)] = group
            covered[scheme].add(code)
        gaps = {s: sorted(countries.codes - c) for s, c in covered.items() if countries.codes - c}
        if gaps:
            raise PlacesError(f"{path}: countries with no group: {gaps}")
        return cls(groups, frozenset(covered))

    def group_of(self, scheme: str, code: str) -> str:
        if scheme not in self.schemes:
            raise PlacesError(f"no region scheme {scheme!r}; have {sorted(self.schemes)}")
        return self.groups[(scheme, code)]

    def members(self, scheme: str) -> dict[str, list[str]]:
        """group -> country codes, for reports."""
        out: dict[str, list[str]] = defaultdict(list)
        for (s, code), group in sorted(self.groups.items()):
            if s == scheme:
                out[group].append(code)
        return dict(out)


def unknown_spellings(countries: Countries, spellings: Iterable[str], source: str) -> list[str]:
    """Spellings a source uses that countries.tsv lacks, for a report before a run stops."""
    return sorted({s for s in spellings if (source, s) not in countries.by_spelling})


PRECISIONS = frozenset({"city", "district", "province", "country", "not recorded"})
FLAGS = frozenset({"inherited-guess", "no-coordinates", "country-unknown"})  # plus "today:*" (seed)
REQUIRED_PLACE = (
    "location", "country", "admin", "latitude", "longitude", "precision", "source", "evidence",
    "same_as", "flags",
)  # fmt: skip


@dataclass(frozen=True)
class Place:
    location: str  # the name-location, as af.seq.names normalises it
    country: str  # a code from countries.tsv
    admin: str  # the admin area where one country has two places of this name, else ""
    latitude: float | None
    longitude: float | None
    precision: str
    source: str
    flags: tuple[str, ...]

    @property
    def coordinates(self) -> tuple[float, float] | None:
        """(longitude, latitude), the shape af.geo takes; None when af does not know."""
        if self.latitude is None or self.longitude is None:
            return None
        return self.longitude, self.latitude


class AmbiguousPlace(PlacesError):
    """A bare name that is several places: never resolved to one silently."""


def is_normalised(location: str) -> bool:
    """True when af.seq.names would leave this location part unchanged.

    Asked of af.seq.names itself, not re-implemented. Runs of spaces are compared collapsed:
    af.seq.names collapses spaces before it turns hyphens into spaces, so "X - Y" becomes
    "X  Y", and that form is what the store holds (one name-location, 26 Sep 2026).
    """
    name = normalise(f"A/{location}/1/2000")
    return (
        name.ok
        and location == location.strip()
        and " ".join(name.name.split("/")[1].split()) == " ".join(location.split())
    )


@dataclass(frozen=True)
class Places:
    """(location, country, admin) -> Place; a bare lookup must name one place or fail."""

    by_key: dict[tuple[str, str, str], Place]
    by_location: dict[str, list[Place]]

    @classmethod
    def read(cls, path: Path, countries: Countries) -> Places:
        by_key: dict[tuple[str, str, str], Place] = {}
        same_as: dict[tuple[str, str, str], tuple[str, int]] = {}
        for line, row in _rows(path, REQUIRED_PLACE):
            key = (row["location"], row["country"], row["admin"])
            where = f"{path}:{line}"
            if not is_normalised(row["location"]):
                raise PlacesError(
                    f"{where}: location {row['location']!r} is not in normalised form"
                )
            if row["country"] not in countries.codes:
                raise PlacesError(f"{where}: country {row['country']!r} is not in countries.tsv")
            if row["precision"] not in PRECISIONS:
                raise PlacesError(f"{where}: precision {row['precision']!r}")
            flags = tuple(f for f in row["flags"].split(";") if f)
            if bad := [f for f in flags if f not in FLAGS and not f.startswith("today:")]:
                raise PlacesError(f"{where}: unknown flags {bad}")
            if key in by_key:
                raise PlacesError(f"{where}: {key} has two rows")
            latitude = _degrees(row["latitude"], 90, where)
            longitude = _degrees(row["longitude"], 180, where)
            by_key[key] = Place(row["location"], row["country"], row["admin"], latitude,
                                longitude, row["precision"], row["source"], flags)  # fmt: skip
            if row["same_as"]:
                same_as[key] = (row["same_as"], line)
        for key, (target, line) in same_as.items():
            _check_same_as(path, line, key, target, by_key, same_as)
        by_location: dict[str, list[Place]] = defaultdict(list)
        for place in by_key.values():
            by_location[place.location].append(place)
        return cls(by_key, dict(by_location))

    def resolve(self, location: str, country: str | None = None) -> Place | None:
        """The place a name-location means, or None when af has no row (unplaced, counted).

        With a country, the row for that country. Without one, the name must be a single
        place: several rows are an :class:`AmbiguousPlace` error, never a silent choice.
        """
        candidates = self.by_location.get(location, [])
        if country is not None:
            candidates = [p for p in candidates if p.country == country]
        if len(candidates) > 1:
            raise AmbiguousPlace(
                f"{location!r} is {len(candidates)} places: "
                + ", ".join(
                    sorted(f"{p.country}/{p.admin}" if p.admin else p.country for p in candidates)
                )
            )
        return candidates[0] if candidates else None


def _degrees(text: str, limit: int, where: str) -> float | None:
    if not text:
        return None  # a blank stays blank: af never fills a coordinate
    value = float(text)
    if not -limit <= value <= limit:
        raise PlacesError(f"{where}: {value} is outside ±{limit}")
    return value


def _check_same_as(
    path: Path,
    line: int,
    key: tuple[str, str, str],
    target: str,
    by_key: dict[tuple[str, str, str], Place],
    same_as: dict[tuple[str, str, str], tuple[str, int]],
) -> None:
    """One hop, to a row of the same country that has coordinates."""
    target_key = (target, key[1], key[2])
    if target_key not in by_key:
        raise PlacesError(f"{path}:{line}: same_as {target!r} has no row in {key[1]}")
    if target_key in same_as:
        raise PlacesError(f"{path}:{line}: same_as {target!r} is itself a same_as: one hop only")
    if by_key[target_key].coordinates is None:
        raise PlacesError(f"{path}:{line}: same_as {target!r} has no coordinates")
