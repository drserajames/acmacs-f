"""New locations: propose places.tsv rows from GeoNames, and list the rest for review.

A name-location that places.tsv has no row for has no coordinates, so its sequences and antigens
are unplaced (counted, never dropped). This step proposes rows for new ones automatically
(LOCATIONS-PROPOSAL §6c; Sarah approved GeoNames as the source, Q56):

- Only **within GISAID's stated country** (countries.tsv -> ISO code -> GeoNames' country). A name
  is never looked up worldwide: a bare name is several places in several countries (§6a).
- Narrowed to a **province** when GISAID's place, or the name's own leading words, name one of that
  country's first-level regions (GeoNames admin1).
- Matched on GeoNames' primary, ASCII and **alternate** names, all normalised the way af.seq.names
  normalises a location. Alternate names are safe here only because the country is fixed.
- A proposal only for a **unique** GeoNames place. Several candidates, a region-level name
  (GeoNames' admin1 file carries no coordinates, so region rows are hand rows) or no match at all
  go on the review list with the reason and the candidates, never a guess.

GeoNames' cities500 holds places of 500 people or more, so a real village can be missing: a miss is
a limit of the source, not an error to report to GISAID.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from af.seq.locations import LocationTables
from af.seq.places import Countries, is_normalised
from af.store import Store

CITIES = "cities500.txt"
ADMIN1 = "admin1CodesASCII.txt"
COUNTRY_INFO = "countryInfo.txt"


def norm(text: str) -> str:
    """A name as af.seq.names leaves a location: ASCII upper case, hyphens as spaces."""
    ascii_ = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().upper()
    return " ".join(
        ascii_.replace("-", " ").replace("_", " ").replace("'", "").replace(".", "").split()
    )


@dataclass(frozen=True)
class GeoPlace:
    geonameid: str
    name: str
    latitude: float
    longitude: float
    feature: str
    country: str  # ISO 3166 alpha-2, as GeoNames keys it
    admin1: str


@dataclass
class GeoNames:
    """The parts of a GeoNames snapshot this step reads, indexed by (country, normalised name)."""

    places: dict[tuple[str, str], list[GeoPlace]] = field(default_factory=lambda: defaultdict(list))
    admin1: dict[tuple[str, str], str] = field(default_factory=dict)  # (cc, norm name) -> code
    admin1_name: dict[tuple[str, str], str] = field(default_factory=dict)  # (cc, code) -> name
    iso2: dict[str, str] = field(default_factory=dict)  # ISO3 -> ISO2
    country_names: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )  # cc -> norm

    @classmethod
    def read(cls, directory: Path) -> GeoNames:
        missing = [n for n in (CITIES, ADMIN1, COUNTRY_INFO) if not (directory / n).is_file()]
        if missing:
            raise FileNotFoundError(f"{directory}: missing {missing}")
        geo = cls()
        for line in (directory / COUNTRY_INFO).read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                cells = line.split("\t")
                geo.iso2[cells[1]] = cells[0]
                geo.country_names[cells[0]].add(norm(cells[4]))
        for line in (directory / ADMIN1).read_text(encoding="utf-8").splitlines():
            key, name, ascii_name, _ = line.split("\t")
            cc, code = key.split(".", 1)
            geo.admin1_name[(cc, code)] = name
            for n in {norm(name), norm(ascii_name)} - {""}:
                geo.admin1[(cc, n)] = code
        with (directory / CITIES).open(encoding="utf-8") as handle:
            for line in handle:
                c = line.rstrip("\n").split("\t")
                place = GeoPlace(c[0], c[1], float(c[4]), float(c[5]), c[7], c[8], c[10])
                for n in {norm(x) for x in (c[1], c[2], *c[3].split(","))} - {""}:
                    geo.places[(c[8], n)].append(place)
        return geo


@dataclass(frozen=True)
class Stated:
    """What GISAID states for a name-location's isolates: the majority country and place."""

    country: str | None  # GISAID's spelling
    place: str  # GISAID's place below the country, e.g. "Guangdong / Liwan"; "" if none
    isolates: int


@dataclass(frozen=True)
class Outcome:
    location: str
    status: str  # "proposed" or a review reason
    country: str = ""  # ISO3
    place: GeoPlace | None = None
    matched: str = ""
    candidates: tuple[str, ...] = ()
    isolates: int = 0

    def row(self) -> list[str]:
        """A places.tsv row (columns of af.seq.places.REQUIRED_PLACE plus added_by, added_on)."""
        assert self.place is not None
        p = self.place
        return [self.location, self.country, "", f"{p.latitude:.4f}", f"{p.longitude:.4f}",
                "city", "geonames",
                f"GeoNames {p.geonameid} {p.name} ({p.feature}),"
                f" the only place named {self.matched!r}"
                f" in {self.country}{', admin1 ' + p.admin1 if p.admin1 else ''}; GISAID states"
                f" {self.country} for {self.isolates} isolate(s)", "", ""]  # fmt: skip


PROPOSED = "proposed"
NO_COUNTRY = "no GISAID country"
NO_GEONAMES_COUNTRY = "country not in GeoNames"
REGION = "region-level name: needs a hand row (GeoNames admin1 has no coordinates)"
COUNTRY_LEVEL = "country-level name: needs a hand row (a country is not a point)"
AMBIGUOUS = "several GeoNames places"
GENERIC = "a common name: several places in the country, and the one in this province is no seat"
#: Administrative seats down to county level. A name common across a country (a district name
#: repeated in many provinces) is taken inside a province only when the match is such a seat:
#: GeoNames may lack the district itself and hold only a same-named township elsewhere in it.
SEATS = frozenset({"PPLC", "PPLA", "PPLA2", "PPLA3"})
NO_MATCH = "no GeoNames match in GISAID's country"
NOT_NORMALISED = "location not in normalised form (places.tsv would refuse it)"


def resolve(location: str, stated: Stated, countries: Countries, geo: GeoNames) -> Outcome:
    """Propose a place for one new name-location, or say why it needs a person."""
    if not is_normalised(location):
        return Outcome(location, NOT_NORMALISED, isolates=stated.isolates)
    if stated.country is None:
        return Outcome(location, NO_COUNTRY, isolates=stated.isolates)
    code = countries.code(stated.country, "gisaid")
    cc = geo.iso2.get(code)
    if cc is None:
        return Outcome(location, NO_GEONAMES_COUNTRY, code, isolates=stated.isolates)
    name = norm(location)
    if name in geo.country_names[cc] | _spellings(countries, code):
        return Outcome(location, COUNTRY_LEVEL, code, isolates=stated.isolates)
    if (cc, name) in geo.admin1:
        return Outcome(location, REGION, code, isolates=stated.isolates)
    parts = [norm(p) for p in stated.place.split("/") if norm(p)]
    stated_admin = next((geo.admin1[(cc, p)] for p in parts if (cc, p) in geo.admin1), None)
    for text, implied in _candidates(name, parts, cc, geo):
        admin = implied or stated_admin
        found = {
            p.geonameid: p for p in geo.places.get((cc, text), []) if not admin or p.admin1 == admin
        }
        if len(found) == 1:
            (place,) = found.values()
            nationwide = {p.geonameid for p in geo.places.get((cc, text), [])}
            if admin and len(nationwide) > 1 and place.feature not in SEATS:
                return Outcome(
                    location, GENERIC, code, None, text, (place.geonameid,), stated.isolates
                )
            return Outcome(location, PROPOSED, code, place, text, isolates=stated.isolates)
        if len(found) > 1:
            names = tuple(sorted(f"{p.geonameid}:{p.name}:{p.admin1}" for p in found.values()))
            return Outcome(location, AMBIGUOUS, code, None, text, names, stated.isolates)
    return Outcome(location, NO_MATCH, code, isolates=stated.isolates)


def _spellings(countries: Countries, code: str) -> set[str]:
    """Every spelling countries.tsv records for a country, normalised."""
    return {norm(s) for (_, s), c in countries.by_spelling.items() if c == code}


def _candidates(
    name: str, parts: list[str], cc: str, geo: GeoNames
) -> list[tuple[str, str | None]]:
    """Strings to look up, most specific first, each with the province it implies (if any).

    The name itself; the name without leading words that name a province ("EXAMPLEPROV
    EXAMPLETOWN" -> "EXAMPLETOWN" in that province). GISAID's place parts only when the name
    ends with them: a finer place GISAID names (a city inside a state the name is) is a different
    place, not a spelling of this one.
    """
    out: list[tuple[str, str | None]] = [(name, None)]
    words = name.split()
    for n in range(1, min(3, len(words) - 1) + 1):
        head = " ".join(words[:n])
        if (cc, head) in geo.admin1:
            out.append((" ".join(words[n:]), geo.admin1[(cc, head)]))
    out += [(p, None) for p in reversed(parts) if name.endswith(" " + p)]
    seen: set[str] = set()
    return [(t, a) for t, a in out if t and not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]


def stated_by_location(
    isolates: Iterable[tuple[str, str | None, str | None]],
) -> dict[str, Stated]:
    """(name-location, GISAID country, GISAID place) per isolate -> the majority statement."""
    votes: dict[str, Counter[tuple[str | None, str]]] = defaultdict(Counter)
    for location, country, place in isolates:
        votes[location][(country, place or "")] += 1
    out = {}
    for location, counter in votes.items():
        (country, place), _ = counter.most_common(1)[0]
        out[location] = Stated(country, place, sum(counter.values()))
    return out


def write(outcomes: Iterable[Outcome], directory: Path, by: str, on: str) -> dict[str, int]:
    """``proposed-places.tsv`` (places.tsv rows) and ``review.tsv``; returns counts by status."""
    directory.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with (
        (directory / "proposed-places.tsv").open("w", newline="") as p,
        (directory / "review.tsv").open("w", newline="") as r,
    ):
        pw = csv.writer(p, delimiter="\t", lineterminator="\n")
        rw = csv.writer(r, delimiter="\t", lineterminator="\n")
        rw.writerow(["location", "country", "isolates", "reason", "matched", "candidates"])
        for o in sorted(outcomes, key=lambda o: (-o.isolates, o.location)):
            counts[o.status] += 1
            if o.status == PROPOSED:
                pw.writerow([*o.row(), by, on])
            else:
                rw.writerow([o.location, o.country, o.isolates, o.status, o.matched,
                             "; ".join(o.candidates)])  # fmt: skip
    return dict(counts)


def new_locations(stated: Mapping[str, Stated], known: Iterable[str]) -> dict[str, Stated]:
    """The name-locations that places.tsv has no row for."""
    have = set(known)
    return {loc: s for loc, s in stated.items() if loc not in have}


def from_store(
    store: Store, datasets: Iterable[str], tables: LocationTables, geo: GeoNames
) -> list[Outcome]:
    """Every name-location in the CURRENT sequence datasets that places.tsv has no row for."""
    paths = [
        str(store.resolve(store.current("sequences", d)) / "isolates" / "*" / "*.parquet")
        for d in datasets
    ]
    rows = duckdb.execute(
        "select split_part(name, '/', 2), country, place from read_parquet(?, union_by_name=true)"
        " where len(list_filter(problems, x -> x like 'name.%')) = 0",
        [paths],
    ).fetchall()
    stated = new_locations(stated_by_location(rows), tables.places.by_location)
    return [resolve(loc, s, tables.countries, geo) for loc, s in sorted(stated.items())]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument(
        "--locations", type=Path, required=True, help="acmacs-f-data rules/locations"
    )
    parser.add_argument(
        "--geonames", type=Path, required=True, help="a GeoNames snapshot directory"
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--by", required=True, help="added_by for proposed rows")
    parser.add_argument(
        "--on", required=True, help="added_on (YYYY-MM-DD); never taken from the clock"
    )
    args = parser.parse_args(argv)
    outcomes = from_store(
        Store.open(args.store), args.datasets, LocationTables.read(args.locations),
        GeoNames.read(args.geonames),
    )  # fmt: skip
    counts = write(outcomes, args.out, args.by, args.on)
    print(json.dumps({"new_locations": len(outcomes), **counts}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
