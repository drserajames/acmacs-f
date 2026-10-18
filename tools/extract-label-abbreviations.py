#!/usr/bin/env python3
"""Write acmacs-f-data's rules/locations/label-abbreviations.tsv: ae's vaccine-label rule for US
places, as data.

    tools/extract-label-abbreviations.py <acmacs-data checkout> <acmacs-f-data checkout> <out.tsv>

ae (ae/cc/locdb/v3/locdb.cc, Db::abbreviation) labels a vaccine from a place in the USA with that
place's CDC abbreviation, found by name in locationdb's ``cdc_abbreviations`` (code -> name); any
other place falls back to initials / two letters. This tool keeps every CDC (code, name) whose
name is a place that af's places.tsv puts in the USA, as a (location, abbreviation) row. Places
outside the USA are left to af's own fallback, as ae left them.

The table is a derived copy: when locationdb's codes or af's places change, run this again and
commit the result in acmacs-f-data. Its header records which commits and which file hash it came
from. Read-only on both checkouts; writes only <out.tsv>.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import subprocess
import sys
from pathlib import Path

from af.map.labels import LabelAbbreviations
from af.seq.locations import LocationTables

LOCATIONDB = "locationdb.json.xz"
LOCATIONS = Path("rules/locations")
COUNTRY = "USA"


def last_commit(checkout: Path, path: Path | str, form: str) -> str:
    """The last commit that touched ``path``, so the header says which version was read."""
    command = ["git", "-C", str(checkout), "log", "-1", f"--format={form}", "--date=short"]
    out = subprocess.run(
        [*command, "--", str(path)], capture_output=True, text=True, check=True
    ).stdout.strip()
    if not out:
        raise SystemExit(f"{checkout}: {path} has no commit; the header could not name its version")
    return out


def rows(codes: dict[str, str], us_places: set[str]) -> list[tuple[str, str]]:
    """(location, abbreviation) for every CDC code whose name is a US place, sorted by location."""
    found = sorted((name, code) for code, name in codes.items() if name in us_places)
    seen: dict[str, str] = {}
    for name, code in found:
        if name in seen:
            raise SystemExit(f"{name!r} has two CDC codes ({seen[name]}, {code})")
        seen[name] = code
    if not found:
        raise SystemExit(f"no CDC code names a place that places.tsv puts in {COUNTRY}")
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("acmacs_data", type=Path, help="acmacs-data checkout (has locationdb.json.xz)")
    ap.add_argument("af_data", type=Path, help="acmacs-f-data checkout (has rules/locations/)")
    ap.add_argument("out", type=Path, help="the TSV to write")
    args = ap.parse_args(argv)

    db_path = args.acmacs_data / LOCATIONDB
    codes: dict[str, str] = json.loads(lzma.decompress(db_path.read_bytes()))["cdc_abbreviations"]
    tables = LocationTables.read(args.af_data / LOCATIONS)
    us_places = {p.location for p in tables.places.by_key.values() if p.country == COUNTRY}
    table = rows(codes, us_places)

    db_commit = last_commit(args.acmacs_data, LOCATIONDB, "%h %cd")
    places_commit = last_commit(args.af_data, LOCATIONS / "places.tsv", "%h")
    sha = hashlib.sha256(db_path.read_bytes()).hexdigest()
    lines = [
        "# Place abbreviations for vaccine labels on af's maps (af.map.labels.LabelAbbreviations;",
        "# named by a round's maps.toml label_abbreviations). A display short form, not a place",
        "# identity: places.tsv (and its same_as) stays the one copy of what a place is.",
        "# Why: the shipped maps labelled a US vaccine by its CDC code (ae's rule, ae/cc/locdb/v3/",
        "# locdb.cc Db::abbreviation); without this table af abbreviated those places by its",
        "# fallback (initials / two letters), giving two states one prefix (style row 08-27).",
        f"# Source: acmacs-data {LOCATIONDB} @ {db_commit}, sha256 {sha}, key cdc_abbreviations",
        f"# ({len(codes)} codes), kept where the name is a place places.tsv @ {places_commit}",
        f"# puts in {COUNTRY}. A derived copy: never edit by hand; regenerate with acmacs-f",
        "# tools/extract-label-abbreviations.py when either source changes.",
        "location\tabbreviation",
        *(f"{name}\t{code}" for name, code in table),
    ]
    args.out.write_text("\n".join(lines) + "\n")
    LabelAbbreviations.read(args.out)  # the file af will read must load (rule 3)
    print(f"{len(table)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
