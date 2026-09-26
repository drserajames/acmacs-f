"""Location lookup. Every place, country and coordinate here is invented."""

from __future__ import annotations

import json
import lzma
from pathlib import Path

import pytest

from af.seq import locations as L

LOCATIONDB = {
    "  version": "13",
    " date": "2026-01-01",
    "continents": ["EXAMPLE-CONTINENT", "OTHER-CONTINENT"],
    "countries": {"EXAMPLELAND": 0, "OTHERLAND": 1, "NOWHERELAND": 1},
    "locations": {
        "EXAMPLETOWN": [10.5, 20.25, "EXAMPLELAND", "NORTH"],
        "OTHERTOWN": [-5.0, 7.0, "OTHERLAND", "SOUTH"],
        "BORDERTOWN": [1.0, 2.0, "EXAMPLELAND", "EAST"],
        "LONELYTOWN": [3.0, 4.0, "NOWHERELAND", "WEST"],
        "QUIETTOWN": [6.0, 8.0, "EXAMPLELAND", "WEST"],
        "HILLTOWN-C": [9.0, 9.5, "EXAMPLELAND", "NORTH"],
        "TWIN-FORD": [2.0, 3.0, "EXAMPLELAND", "EAST"],
        "TWIN FORD": [4.0, 5.0, "OTHERLAND", "WEST"],
    },
    "names": {
        "EXAMPLETOWN": "EXAMPLETOWN", "OTHERTOWN": "OTHERTOWN", "BORDERTOWN": "BORDERTOWN",
        "LONELYTOWN": "LONELYTOWN", "QUIETTOWN": "QUIETTOWN",
        "HILLTOWN-C": "HILLTOWN-C", "TWIN-FORD": "TWIN-FORD", "TWIN FORD": "TWIN FORD",
    },
    "replacements": {"EXAMPLE TOWN": "EXAMPLETOWN"},
}  # fmt: skip


@pytest.fixture
def locationdb(tmp_path: Path) -> L.LocationDb:
    path = tmp_path / "locationdb.json.xz"
    with lzma.open(path, "wt") as handle:
        json.dump(LOCATIONDB, handle)
    return L.LocationDb.read(path)


def isolates(
    location: str, country: str | None, region: str | None, n: int = 1
) -> list[tuple[str, str | None, str | None]]:
    return [(f"A(H3N2)/{location}/{i}/2021", country, region) for i in range(n)]


def build(locationdb: L.LocationDb) -> L.LocationLookup:
    rows = (
        isolates("EXAMPLETOWN", "Exampleland", "Example Continent", 5)
        + isolates("OTHERTOWN", "Otherland", "Other Continent", 3)
        # GISAID says the bordertown is in Otherland; locationdb says Exampleland.
        + isolates("BORDERTOWN", "Otherland", "Other Continent", 2)
        # Two GISAID countries for one name-location.
        + isolates("EXAMPLE TOWN", "Exampleland", "Example Continent", 3)
        + isolates("EXAMPLE TOWN", "Otherland", "Other Continent", 1)
        + isolates("NOT A PLACE", "Exampleland", "Example Continent", 1)
    )
    return L.build_lookup(rows, locationdb)


class TestLocationDb:
    def test_replacement_then_name_then_location(self, locationdb: L.LocationDb) -> None:
        entry = locationdb.resolve("EXAMPLE TOWN")
        assert entry is not None
        assert (entry.name, entry.country, entry.continent) == (
            "EXAMPLETOWN", "EXAMPLELAND", "EXAMPLE-CONTINENT",
        )  # fmt: skip
        assert locationdb.resolve("NOT A PLACE") is None

    def test_a_hyphenated_name_is_found_from_its_normalised_form(
        self, locationdb: L.LocationDb
    ) -> None:
        entry = locationdb.resolve("HILLTOWN C")  # af.seq.names made the hyphen a space
        assert entry is not None
        assert (entry.name, entry.latitude) == ("HILLTOWN-C", 9.0)
        assert locationdb.hyphen_counts == {"hyphen-form": 1}

    def test_an_exact_name_wins_over_the_normalised_form(self, locationdb: L.LocationDb) -> None:
        exact = locationdb.resolve("TWIN FORD")
        assert exact is not None and exact.country == "OTHERLAND"  # not TWIN-FORD's
        assert not locationdb.hyphen_counts

    def test_a_form_leading_to_two_places_resolves_to_neither(self) -> None:
        data = json.loads(json.dumps(LOCATIONDB))
        data["locations"]["LAKE-SIDE"] = [1.0, 1.0, "EXAMPLELAND", "NORTH"]
        data["names"]["LAKE-SIDE"] = "LAKE-SIDE"
        data["replacements"]["-LAKE SIDE"] = "QUIETTOWN"  # normalises to LAKE SIDE too
        db = L.LocationDb(data, Path("invented.json.xz"))
        assert db.resolve("LAKE SIDE") is None
        assert db.hyphen_counts == {"hyphen-ambiguous": 1}
        assert db.resolve("LAKE-SIDE") is not None  # as written, still found

    def test_a_file_without_a_table_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json.xz"
        with lzma.open(path, "wt") as handle:
            json.dump({"continents": []}, handle)
        with pytest.raises(ValueError, match="no 'replacements' table"):
            L.LocationDb.read(path)


class TestLookup:
    def test_sequenced_location_takes_gisaid_country_and_locationdb_coordinates(
        self, locationdb: L.LocationDb
    ) -> None:
        found = build(locationdb).lookup("EXAMPLETOWN")
        assert found == L.Location("EXAMPLETOWN", "Exampleland", "Example Continent",
                                   10.5, 20.25, 5, 1.0, ())  # fmt: skip

    def test_a_country_disagreement_gives_no_coordinates(self, locationdb: L.LocationDb) -> None:
        found = build(locationdb).lookup("BORDERTOWN")
        assert found is not None
        assert (found.country, found.latitude, found.flags) == (
            "Otherland", None, (L.COUNTRY_DISAGREES,),
        )  # fmt: skip

    def test_a_gisaid_conflict_keeps_the_majority_and_its_share(
        self, locationdb: L.LocationDb
    ) -> None:
        found = build(locationdb).lookup("EXAMPLE TOWN")
        assert found is not None
        assert (found.country, found.agreement, found.flags) == ("Exampleland", 0.75,
                                                                 (L.GISAID_CONFLICT,))  # fmt: skip

    def test_not_in_locationdb_is_flagged_without_coordinates(
        self, locationdb: L.LocationDb
    ) -> None:
        found = build(locationdb).lookup("NOT A PLACE")
        assert found is not None
        assert (found.country, found.latitude, found.flags) == (
            "Exampleland", None, (L.NOT_IN_LOCATIONDB,),
        )  # fmt: skip

    def test_unsequenced_location_uses_the_crosswalked_country(
        self, locationdb: L.LocationDb
    ) -> None:
        found = build(locationdb).lookup("QUIETTOWN")
        assert found == L.Location("QUIETTOWN", "Exampleland", "Example Continent", 6.0, 8.0,
                                   0, None, (L.FROM_LOCATIONDB,))  # fmt: skip

    def test_a_country_gisaid_never_named_is_flagged_not_guessed(
        self, locationdb: L.LocationDb
    ) -> None:
        found = build(locationdb).lookup("LONELYTOWN")
        assert found is not None
        assert (found.country, found.region, found.flags) == (
            None, None, (L.FROM_LOCATIONDB, L.COUNTRY_UNMAPPED),
        )  # fmt: skip

    def test_unknown_everywhere_is_none(self, locationdb: L.LocationDb) -> None:
        lookup = build(locationdb)
        assert lookup.lookup("NOWHERE AT ALL") is None
        assert lookup.coordinates("NOWHERE AT ALL") is None

    def test_coordinates_are_longitude_first(self, locationdb: L.LocationDb) -> None:
        assert build(locationdb).coordinates("EXAMPLETOWN") == (20.25, 10.5)

    def test_counts(self, locationdb: L.LocationDb) -> None:
        counts = build(locationdb).counts.to_json()
        assert counts == {
            "name_locations": 5,
            "flags": {L.COUNTRY_DISAGREES: 1, L.GISAID_CONFLICT: 1, L.NOT_IN_LOCATIONDB: 1},
            "country_agree_isolates": 12,  # 5 + 3 + the 4 of EXAMPLE TOWN
            "country_disagree_isolates": 2,
        }


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
