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
