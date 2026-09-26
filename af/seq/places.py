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
