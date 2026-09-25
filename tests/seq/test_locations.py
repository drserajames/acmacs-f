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
    },
    "names": {
        "EXAMPLETOWN": "EXAMPLETOWN", "OTHERTOWN": "OTHERTOWN", "BORDERTOWN": "BORDERTOWN",
        "LONELYTOWN": "LONELYTOWN", "QUIETTOWN": "QUIETTOWN",
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
