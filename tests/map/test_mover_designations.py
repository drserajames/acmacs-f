"""Movers name antigens by designation; twin preparations are told apart by their passage date."""

from af.chart.model import Antigen, Chart, Titres
from af.map.build import designation, mover_designations

NAME = "B/EXAMPLETOWN/{}/2019"  # invented


def test_twin_preparations_are_named_with_their_passage_date() -> None:
    antigens = [
        Antigen(NAME.format(1), "MDCK2 (2020-01-12)"),
        Antigen(NAME.format(1), "MDCK2 (2020-11-19)"),  # the same name and passage: twins
        Antigen(NAME.format(2), "E3 (2021-10-08)"),
    ]
    chart = Chart({}, antigens, [], Titres([[] for _ in antigens]))
    assert mover_designations(chart) == [
        NAME.format(1) + " MDCK2 (2020-01-12)",
        NAME.format(1) + " MDCK2 (2020-11-19)",
        NAME.format(2) + " E3",  # unique: the dateless form, as configs write it
    ]
    # a hide by name still means every preparation of it
    assert designation(antigens[0]) == designation(antigens[1]) == NAME.format(1) + " MDCK2"
