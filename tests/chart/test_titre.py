"""Titre parsing and the lispmds merge (inventory D §4, T22/T23)."""

import math

import pytest

from af.chart.titre import (
    MergeOutcome,
    MergeSettings,
    MoreThanOnly,
    Titre,
    TitreError,
    TitreType,
    column_basis,
    from_logged,
    merge_titres,
)


def merged(*texts, **kw):
    t, why = merge_titres([Titre.parse(x) for x in texts], **kw)
    return str(t), why


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (["40", "80"], "57"),  # geometric mean, C lround
        (["<10", "<20"], "<10"),
        (["<20", "40"], "<80"),
        ([">1280", "2560"], ">1280"),
        (["<10", ">1280"], "*"),
        (["10", "1280"], "*"),  # SD > 1
        ([">640", "320"], ">160"),
    ],
)
def test_t22_merge_cases(inputs, expected):
    assert merged(*inputs)[0] == expected


def test_more_than_only_is_dropped_but_kept_for_column_bases():
    assert merged(">5120") == ("*", MergeOutcome.MORE_THAN_ONLY_TO_DONT_CARE)
    t, _ = merge_titres([Titre.parse(">5120")], MoreThanOnly.ADJUST_TO_NEXT)
    assert str(t) == ">5120" and t.logged_for_column_bases() == pytest.approx(10.0)


def test_sd_limit_is_inclusive_and_configurable():
    assert merged("10", "40")[0] == "20"  # SD exactly 1.0 is kept
    assert merged("10", "40", settings=MergeSettings(sd_limit=0.5))[0] == "*"
    assert merged("10", "1280", settings=MergeSettings(sd_limit=math.nan))[0] == "113"


def test_parse_and_logs():
    assert Titre.parse("<10").type == TitreType.LESS_THAN
    assert Titre.parse("*").is_missing
    assert Titre.parse("~80").type == TitreType.DODGY
    assert Titre.parse("80").logged() == pytest.approx(3.0)
    assert Titre.parse("<40").logged_with_thresholded() == pytest.approx(1.0)
    for bad in ("x", "0", "<", "1/2", "-40"):
        with pytest.raises(TitreError):
            Titre.parse(bad)


def test_dodgy_cannot_be_merged():
    with pytest.raises(TitreError):
        merge_titres([Titre.parse("~40"), Titre.parse("40")])


def test_from_logged_rounds_half_away_from_zero():
    assert str(from_logged(math.log2(2.5))) == "25"


def test_column_basis_floor_and_rules():
    def parse(*xs):
        return [Titre.parse(x) for x in xs]

    assert column_basis(parse("<10", "*")) == 0.0
    assert column_basis(parse("40", "<160")) == pytest.approx(4.0)
    assert column_basis(parse("40", ">160")) == pytest.approx(5.0)
    assert column_basis(parse("40", "~640")) == pytest.approx(2.0)
