"""Country identity and region schemes. Every country, code and region here is invented."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.seq import places as P

COUNTRIES = """# invented
code\tspelling\tsource\tevidence\tadded_by\tadded_on
XAA\tExampleland\tgisaid\tsame name\tt\t2026-01-01
XAA\tEXAMPLELAND\tatlas\tsame name\tt\t2026-01-01
XAA\tExampleland, Republic of\tgisaid\tlong form\tt\t2026-01-01
XBB\tOtherland\tgisaid\tsame name\tt\t2026-01-01
non-country:NOWHERE\tNOWHERE\tatlas\ta placeholder, not a country\tt\t2026-01-01
"""

REGIONS = """scheme\tcountry\tgroup\tevidence\tadded_by\tadded_on
blocs\tXAA\tNORTH\tfrom the atlas\tt\t2026-01-01
blocs\tXBB\tSOUTH\tfrom the atlas\tt\t2026-01-01
blocs\tnon-country:NOWHERE\tnot a country\trecorded, not blank\tt\t2026-01-01
offices\tXAA\tOFFICE-1\tpublished list\tt\t2026-01-01
offices\tXBB\tnot assigned\tnot in the published list\tt\t2026-01-01
offices\tnon-country:NOWHERE\tnot a country\trecorded, not blank\tt\t2026-01-01
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.fixture
def countries(tmp_path: Path) -> P.Countries:
    return P.Countries.read(write(tmp_path, "countries.tsv", COUNTRIES))


def test_each_sources_spelling_maps_to_one_code(countries: P.Countries) -> None:
    assert countries.code("Exampleland, Republic of", "gisaid") == "XAA"
    assert countries.code("EXAMPLELAND", "atlas") == "XAA"
    assert countries.codes == {"XAA", "XBB", "non-country:NOWHERE"}


def test_an_unknown_spelling_is_an_error_naming_it(countries: P.Countries) -> None:
    with pytest.raises(P.PlacesError, match="'Exampleland ' from gisaid has no row"):
        countries.code("Exampleland ", "gisaid")
    with pytest.raises(P.PlacesError, match="from atlas"):
        countries.code("Exampleland", "atlas")  # the spelling is per source
    assert P.unknown_spellings(countries, ["Otherland", "Newland"], "gisaid") == ["Newland"]


def test_one_spelling_with_two_codes_is_refused(tmp_path: Path) -> None:
    text = COUNTRIES + "XBB\tExampleland\tgisaid\tclash\tt\t2026-01-01\n"
    with pytest.raises(P.PlacesError, match="both XAA and XBB"):
        P.Countries.read(write(tmp_path, "c.tsv", text))


def test_every_row_needs_evidence(tmp_path: Path) -> None:
    text = COUNTRIES + "XCC\tThirdland\tgisaid\t\tt\t2026-01-01\n"
    with pytest.raises(P.PlacesError, match="no evidence"):
        P.Countries.read(write(tmp_path, "c.tsv", text))


def test_schemes_give_a_group_for_every_country(tmp_path: Path, countries: P.Countries) -> None:
    schemes = P.RegionSchemes.read(write(tmp_path, "r.tsv", REGIONS), countries)
    assert schemes.schemes == {"blocs", "offices"}
    assert schemes.group_of("offices", "XBB") == "not assigned"
    assert schemes.members("blocs") == {
        "NORTH": ["XAA"], "SOUTH": ["XBB"], "not a country": ["non-country:NOWHERE"],
    }  # fmt: skip
    with pytest.raises(P.PlacesError, match="no region scheme 'zones'"):
        schemes.group_of("zones", "XAA")


def test_a_country_missing_from_a_scheme_is_refused(tmp_path: Path, countries: P.Countries) -> None:
    text = "\n".join(line for line in REGIONS.splitlines() if "offices\tXBB" not in line) + "\n"
    with pytest.raises(P.PlacesError, match=r"no group: \{'offices': \['XBB'\]\}"):
        P.RegionSchemes.read(write(tmp_path, "r.tsv", text), countries)


def test_a_blank_group_is_refused_not_filled(tmp_path: Path, countries: P.Countries) -> None:
    text = REGIONS.replace("offices\tXBB\tnot assigned", "offices\tXBB\t")
    with pytest.raises(P.PlacesError, match="blank group"):
        P.RegionSchemes.read(write(tmp_path, "r.tsv", text), countries)


def test_a_country_not_in_the_countries_table_is_refused(
    tmp_path: Path, countries: P.Countries
) -> None:
    text = REGIONS + "blocs\tXZZ\tNORTH\tstray\tt\t2026-01-01\n"
    with pytest.raises(P.PlacesError, match="'XZZ' is not in countries.tsv"):
        P.RegionSchemes.read(write(tmp_path, "r.tsv", text), countries)


def test_a_repeated_scheme_country_is_refused(tmp_path: Path, countries: P.Countries) -> None:
    text = REGIONS + "blocs\tXAA\tSOUTH\tsecond\tt\t2026-01-01\n"
    with pytest.raises(P.PlacesError, match="blocs XAA has two rows"):
        P.RegionSchemes.read(write(tmp_path, "r.tsv", text), countries)


COLUMNS = ["location", "country", "admin", "latitude", "longitude", "precision", "source",
           "evidence", "same_as", "flags", "added_by", "added_on"]  # fmt: skip
HEADER = "\t".join(COLUMNS) + "\n"


def place(location: str, country: str = "XAA", lat: str = "10.5", lon: str = "20.25",
          same_as: str = "", flags: str = "", admin: str = "") -> str:  # fmt: skip
    cells = [location, country, admin, lat, lon, "city", "hand", "invented", same_as, flags, "t",
             "2026-01-01"]  # fmt: skip
    return "\t".join(cells) + "\n"


def places(tmp_path: Path, countries: P.Countries, *rows: str) -> P.Places:
    return P.Places.read(write(tmp_path, "places.tsv", HEADER + "".join(rows)), countries)


def test_a_place_resolves_and_gives_longitude_first(tmp_path: Path, countries: P.Countries) -> None:
    table = places(tmp_path, countries, place("EXAMPLETOWN"), place("QUIETTOWN", lat="", lon=""))
    found = table.resolve("EXAMPLETOWN")
    assert found is not None and found.coordinates == (20.25, 10.5)
    quiet = table.resolve("QUIETTOWN")
    assert quiet is not None and quiet.coordinates is None  # a blank stays blank
    assert table.resolve("NOWHERETOWN") is None  # no row: unplaced, never guessed


def test_a_bare_name_that_is_two_places_is_an_error(tmp_path: Path, countries: P.Countries) -> None:
    table = places(
        tmp_path, countries, place("TWINTOWN", "XAA"), place("TWINTOWN", "XBB", "1", "2")
    )
    with pytest.raises(P.AmbiguousPlace, match="'TWINTOWN' is 2 places: XAA, XBB"):
        table.resolve("TWINTOWN")
    found = table.resolve("TWINTOWN", "XBB")
    assert found is not None and found.coordinates == (2.0, 1.0)


def test_one_country_two_places_needs_the_admin_area(
    tmp_path: Path, countries: P.Countries
) -> None:
    table = places(tmp_path, countries, place("TWINTOWN", admin="NORTH"),
                   place("TWINTOWN", admin="SOUTH", lat="1", lon="2"))  # fmt: skip
    with pytest.raises(P.AmbiguousPlace, match="XAA/NORTH, XAA/SOUTH"):
        table.resolve("TWINTOWN", "XAA")


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (place("Exampletown"), "not in normalised form"),
        (place("EXAMPLE-TOWN"), "not in normalised form"),
        (place("EXAMPLETOWN", "XZZ"), "'XZZ' is not in countries.tsv"),
        (place("EXAMPLETOWN", lat="91"), "outside ±90"),
        (place("EXAMPLETOWN", flags="guessed"), "unknown flags"),
        (place("EXAMPLETOWN") + place("EXAMPLETOWN"), "two rows"),
    ],
)
def test_a_bad_row_is_refused(
    tmp_path: Path, countries: P.Countries, row: str, message: str
) -> None:
    with pytest.raises(P.PlacesError, match=message):
        places(tmp_path, countries, row)


def test_same_as_is_one_hop_to_a_place_with_coordinates(
    tmp_path: Path, countries: P.Countries
) -> None:
    alias = place("EXAMPLE TOWN", same_as="EXAMPLETOWN")
    table = places(tmp_path, countries, place("EXAMPLETOWN"), alias)
    assert table.resolve("EXAMPLE TOWN") is not None
    chain = place("EXAMPLE TOWN X", same_as="EXAMPLE TOWN")
    with pytest.raises(P.PlacesError, match="one hop only"):
        places(tmp_path, countries, place("EXAMPLETOWN"), alias, chain)
    with pytest.raises(P.PlacesError, match="has no coordinates"):
        places(tmp_path, countries, place("EXAMPLETOWN", lat="", lon=""),
               place("EXAMPLE TOWN", same_as="EXAMPLETOWN"))  # fmt: skip
    with pytest.raises(P.PlacesError, match="has no row in XAA"):
        places(tmp_path, countries, place("EXAMPLE TOWN", same_as="ELSEWHERE"))


@pytest.mark.parametrize(
    ("location", "ok"),
    [("EXAMPLETOWN", True), ("EXAMPLE TOWN", True), ("EXAMPLE  TOWN", True),
     ("Exampletown", False), ("EXAMPLE-TOWN", False), (" EXAMPLETOWN", False)],
)  # fmt: skip
def test_keys_are_what_af_seq_names_produces(location: str, ok: bool) -> None:
    # "EXAMPLE - TOWN" normalises to "EXAMPLE  TOWN": spaces collapse before hyphens go
    assert P.is_normalised(location) is ok
