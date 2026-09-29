"""New locations from GeoNames. Every country, region, place and id here is invented."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.seq import newplaces as N
from af.seq.places import Countries, Places

COUNTRY_INFO = (
    "#ISO\tISO3\tISO-Numeric\tfips\tCountry\n"
    "XA\tXAA\t1\tXA\tExampleland\n"
    "XB\tXBB\t2\tXB\tOtherland\n"
)
ADMIN1 = (
    "XA.01\tExampleprov\tExampleprov\t901\n"
    "XA.02\tOtherprov\tOtherprov\t902\n"
    "XB.01\tFarprov\tFarprov\t903\n"
)


def city(gid: str, name: str, cc: str, admin: str, feature: str = "PPL", alt: str = "") -> str:
    cells = [gid, name, name, alt, "10.5", "20.25", "P", feature, cc, "", admin, "", "", "", "1000"]
    return "\t".join(cells) + "\n"


CITIES = (
    city("1", "Exampletown", "XA", "01", "PPLA2")
    + city("2", "Exampletown", "XB", "01")  # the same name in another country
    + city("3", "Twinford", "XA", "01")  # twice in one province
    + city("4", "Twinford", "XA", "01")
    + city("5", "Commonburg", "XA", "01", "PPLA4")  # common name, no seat, in one province
    + city("6", "Commonburg", "XA", "02", "PPLA4")
    + city("7", "Seatburg", "XA", "01", "PPLA3")  # common name, but a seat here
    + city("8", "Seatburg", "XA", "02", "PPLA4")
    + city("9", "Oldtown", "XA", "02", alt="Exampleolden")  # found by an alternate name
    + city("10", "Finertown", "XA", "01")  # a finer place GISAID may name
)
COUNTRIES = (
    "code\tspelling\tsource\tevidence\tadded_by\tadded_on\n"
    "XAA\tExampleland\tgisaid\tsame name\tt\t2026-01-01\n"
    "XBB\tOtherland\tgisaid\tsame name\tt\t2026-01-01\n"
)


@pytest.fixture
def geo(tmp_path: Path) -> N.GeoNames:
    (tmp_path / N.COUNTRY_INFO).write_text(COUNTRY_INFO)
    (tmp_path / N.ADMIN1).write_text(ADMIN1)
    (tmp_path / N.CITIES).write_text(CITIES)
    return N.GeoNames.read(tmp_path)


@pytest.fixture
def countries(tmp_path: Path) -> Countries:
    path = tmp_path / "countries.tsv"
    path.write_text(COUNTRIES)
    return Countries.read(path)


def run(geo: N.GeoNames, countries: Countries, location: str, place: str = "",
        country: str | None = "Exampleland") -> N.Outcome:  # fmt: skip
    return N.resolve(location, N.Stated(country, place, 3), countries, geo)


def test_a_unique_place_in_gisaids_country_is_proposed(
    geo: N.GeoNames, countries: Countries
) -> None:
    o = run(geo, countries, "EXAMPLETOWN")
    assert (o.status, o.country, o.place and o.place.geonameid) == (N.PROPOSED, "XAA", "1")
    assert run(geo, countries, "EXAMPLETOWN", country="Otherland").place.geonameid == "2"  # type: ignore[union-attr]


def test_a_province_prefix_narrows_the_search(geo: N.GeoNames, countries: Countries) -> None:
    assert run(geo, countries, "OTHERPROV OLDTOWN").status == N.PROPOSED
    assert run(geo, countries, "EXAMPLEPROV OLDTOWN").status == N.NO_MATCH


def test_alternate_names_count_within_the_country(geo: N.GeoNames, countries: Countries) -> None:
    o = run(geo, countries, "EXAMPLEOLDEN")
    assert o.place is not None and o.place.geonameid == "9"


def test_several_places_go_to_review_with_the_candidates(
    geo: N.GeoNames, countries: Countries
) -> None:
    o = run(geo, countries, "TWINFORD")
    assert o.status == N.AMBIGUOUS and len(o.candidates) == 2


def test_a_common_name_is_taken_in_a_province_only_at_a_seat(
    geo: N.GeoNames, countries: Countries
) -> None:
    assert run(geo, countries, "EXAMPLEPROV COMMONBURG").status == N.GENERIC
    assert run(geo, countries, "EXAMPLEPROV SEATBURG").status == N.PROPOSED


def test_regions_and_countries_are_not_points(geo: N.GeoNames, countries: Countries) -> None:
    assert run(geo, countries, "EXAMPLEPROV").status == N.REGION
    assert run(geo, countries, "EXAMPLELAND").status == N.COUNTRY_LEVEL


def test_gisaids_finer_place_is_not_a_spelling_of_the_name(
    geo: N.GeoNames, countries: Countries
) -> None:
    # the name is a place GeoNames lacks; GISAID names a city inside it: not the same place
    assert run(geo, countries, "EXAMPLEMISSING", "Exampleprov / Finertown").status == N.NO_MATCH
    # the name ends with GISAID's part: that part is the place
    o = run(geo, countries, "EXAMPLEWORD FINERTOWN", "Exampleprov / Finertown")
    assert o.place is not None and o.place.geonameid == "10"


def test_no_country_or_an_unnormalised_name_goes_to_review(
    geo: N.GeoNames, countries: Countries
) -> None:
    assert run(geo, countries, "EXAMPLETOWN", country=None).status == N.NO_COUNTRY
    assert run(geo, countries, "Exampletown").status == N.NOT_NORMALISED


def test_proposals_are_rows_the_places_reader_accepts(
    tmp_path: Path, geo: N.GeoNames, countries: Countries
) -> None:
    outcomes = [run(geo, countries, "EXAMPLETOWN"), run(geo, countries, "TWINFORD")]
    counts = N.write(outcomes, tmp_path / "out", "t", "2026-01-01")
    assert counts == {N.PROPOSED: 1, N.AMBIGUOUS: 1}
    columns = ["location", "country", "admin", "latitude", "longitude", "precision", "source",
               "evidence", "same_as", "flags", "added_by", "added_on"]  # fmt: skip
    header = "\t".join(columns)
    table = tmp_path / "places.tsv"
    table.write_text(header + "\n" + (tmp_path / "out" / "proposed-places.tsv").read_text())
    places = Places.read(table, countries)
    assert places.resolve("EXAMPLETOWN", "XAA").coordinates == (20.25, 10.5)  # type: ignore[union-attr]
    assert "TWINFORD" in (tmp_path / "out" / "review.tsv").read_text()


def test_a_missing_geonames_file_is_fatal(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        N.GeoNames.read(tmp_path)
