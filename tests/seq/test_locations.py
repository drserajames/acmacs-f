"""Location lookup from af's tables. Every place, country and coordinate here is invented."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.seq import locations as L


def isolates(
    location: str, country: str | None, region: str | None, n: int = 1
) -> list[tuple[str, str | None, str | None]]:
    return [(f"A(H3N2)/{location}/{i}/2021", country, region) for i in range(n)]


def test_name_location_needs_a_four_part_name() -> None:
    assert L.name_location("A(H3N2)/EXAMPLETOWN/1/2021") == "EXAMPLETOWN"
    assert L.name_location("EXAMPLETOWN/1/2021") is None


# --- the lookup from af's own tables -------------------------------------------------------

COUNTRIES = """code\tspelling\tsource\tevidence\tadded_by\tadded_on
XAA\tExampleland\tgisaid\tsame name\tt\t2026-01-01
XBB\tOtherland\tgisaid\tsame name\tt\t2026-01-01
"""
REGIONS = """scheme\tcountry\tgroup\tevidence\tadded_by\tadded_on
gisaid\tXAA\tExample Continent\tthe store\tt\t2026-01-01
gisaid\tXBB\tOther Continent\tthe store\tt\t2026-01-01
blocs\tXAA\tNORTH\tan atlas\tt\t2026-01-01
blocs\tXBB\tnot assigned\tnot in the atlas\tt\t2026-01-01
"""
PLACE_COLUMNS = ("location\tcountry\tadmin\tlatitude\tlongitude\tprecision\tsource\tevidence"
                 "\tsame_as\tflags\tadded_by\tadded_on")  # fmt: skip
PLACES = [
    ("EXAMPLETOWN", "XAA", "10.5", "20.25"),
    ("OTHERTOWN", "XBB", "-5.0", "7.0"),
    ("BORDERTOWN", "XAA", "1.0", "2.0"),  # GISAID says Otherland: no row there
    ("EXAMPLE TOWN", "XAA", "10.5", "20.25"),
    ("QUIETTOWN", "XAA", "6.0", "8.0"),  # never sequenced
    ("TWIN FORD", "XAA", "2.0", "3.0"),  # two places of one name, never sequenced
    ("TWIN FORD", "XBB", "4.0", "5.0"),
    ("BLANKTOWN", "XAA", "", ""),  # a row that records no coordinates
]


@pytest.fixture
def tables(tmp_path: Path) -> L.LocationTables:
    (tmp_path / "countries.tsv").write_text(COUNTRIES)
    (tmp_path / "regions.tsv").write_text(REGIONS)
    rows = [f"{loc}\t{c}\t\t{lat}\t{lon}\tcity\thand\tinvented\t\t\tt\t2026-01-01"
            for loc, c, lat, lon in PLACES]  # fmt: skip
    (tmp_path / "places.tsv").write_text("\n".join([PLACE_COLUMNS, *rows]) + "\n")
    return L.LocationTables.read(tmp_path)


def build_places(tables: L.LocationTables) -> L.PlacesLookup:
    rows = (
        isolates("EXAMPLETOWN", "Exampleland", "Example Continent", 5)
        + isolates("OTHERTOWN", "Otherland", "Other Continent", 3)
        + isolates("BORDERTOWN", "Otherland", "Other Continent", 2)
        + isolates("EXAMPLE TOWN", "Exampleland", "Example Continent", 3)
        + isolates("EXAMPLE TOWN", "Otherland", "Other Continent", 1)
        + isolates("NOT A PLACE", "Exampleland", "Example Continent", 1)
        + isolates("BLANKTOWN", "Exampleland", "Example Continent", 1)
    )
    return L.build_places_lookup(rows, tables)


class TestPlacesLookup:
    def test_the_row_in_gisaids_country_gives_coordinates(self, tables: L.LocationTables) -> None:
        lookup = build_places(tables)
        entry = lookup.lookup("EXAMPLETOWN")
        assert entry is not None
        assert (entry.country, entry.region, entry.isolates, entry.flags) == (
            "XAA", "Example Continent", 5, (),
        )  # fmt: skip
        assert lookup.coordinates("EXAMPLETOWN") == (20.25, 10.5)

    def test_a_row_only_in_another_country_is_not_used(self, tables: L.LocationTables) -> None:
        entry = build_places(tables).lookup("BORDERTOWN")
        assert entry is not None
        assert (entry.country, entry.latitude, entry.flags) == ("XBB", None, (L.COUNTRY_DISAGREES,))

    def test_flags_and_counts(self, tables: L.LocationTables) -> None:
        lookup = build_places(tables)
        conflict = lookup.lookup("EXAMPLE TOWN")
        assert conflict is not None and conflict.flags == (L.GISAID_CONFLICT,)
        assert conflict.agreement == 0.75
        assert lookup.counts.to_json() == {
            "name_locations": 6,
            "flags": {
                L.COUNTRY_DISAGREES: 1, L.GISAID_CONFLICT: 1, L.NO_COORDINATES: 1, L.NO_PLACE: 1,
            },
            "country_agree_isolates": 13,  # EXAMPLETOWN, OTHERTOWN, EXAMPLE TOWN, BLANKTOWN
            "country_disagree_isolates": 2,
        }  # fmt: skip
        assert lookup.coordinates("BLANKTOWN") is None  # blank stays blank

    def test_a_table_only_name_needs_exactly_one_row(self, tables: L.LocationTables) -> None:
        lookup = build_places(tables)
        quiet = lookup.lookup("QUIETTOWN")
        assert quiet is not None
        assert (quiet.country, quiet.region, quiet.flags) == (
            "XAA", "Example Continent", (L.NOT_SEQUENCED,),
        )  # fmt: skip
        twin = lookup.lookup("TWIN FORD")  # no GISAID country to choose: never a silent pick
        assert twin is not None
        assert (twin.country, twin.latitude, twin.flags) == (
            None,
            None,
            (L.NOT_SEQUENCED, L.AMBIGUOUS),
        )
        assert lookup.lookup("NOWHERE AT ALL") is None

    def test_group_of_a_scheme(self, tables: L.LocationTables) -> None:
        lookup = build_places(tables)
        blocs = lookup.group_of("blocs")
        assert [blocs(x) for x in ("EXAMPLETOWN", "OTHERTOWN", "TWIN FORD", "NOT A PLACE")] == [
            "NORTH", None, None, "NORTH",  # not assigned -> None; NOT A PLACE: GISAID's country
        ]  # fmt: skip
        with pytest.raises(L.PlacesError, match="no region scheme 'zones'"):
            lookup.group_of("zones")

    def test_a_gisaid_country_with_no_row_stops_the_build(self, tables: L.LocationTables) -> None:
        rows = isolates("EXAMPLETOWN", "Newland", "Example Continent")
        with pytest.raises(L.PlacesError, match=r"no row in countries.tsv: \['Newland'\]"):
            L.build_places_lookup(rows, tables)


def test_the_committed_tables_read(af_data: Path) -> None:
    """acmacs-f-data's rules/locations/ passes every check the readers make."""
    directory = af_data / "rules" / "locations"
    if not (directory / "places.tsv").exists():
        pytest.skip("rules/locations/places.tsv not committed yet")
    tables = L.LocationTables.read(directory)
    assert {"gisaid", "continent", "who"} <= tables.regions.schemes
    assert tables.places.by_key
