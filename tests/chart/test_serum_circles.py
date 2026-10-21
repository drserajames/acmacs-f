"""Serum circles and coverage (af.chart.serum_circles) on invented charts.

Each chart is small enough to work the expected radius out by hand; the docstrings show it.
"""

import json
import math

import numpy as np
import pytest

from af.chart.model import Antigen, Chart, Serum, Titres
from af.chart.serum_circles import (
    SerumCircleError,
    coverage_antigen,
    homologous_antigens,
    serum_circles,
)
from af.chart.titre import Titre

NAME = "A(H3N2)/EXAMPLETOWN/1/2020"


def chart(antigens: list[Antigen], sera: list[Serum], columns: list[list[str]]) -> Chart:
    """``columns[s][a]`` is serum s's titre against antigen a."""
    table = [[Titre.parse(columns[s][a]) for s in range(len(sera))] for a in range(len(antigens))]
    return Chart(info={}, antigens=antigens, sera=sera, titres=Titres(table))


def ruler(*distances: float, serum: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Antigens on the x axis at ``distances`` from one serum at ``serum``."""
    return np.array([[serum[0] + d, serum[1]] for d in distances] + [list(serum)])


def others(n: int) -> list[Antigen]:
    return [Antigen(f"A(H3N2)/EXAMPLEVILLE/{i}/2020", "MDCK1") for i in range(n)]


def one_serum(titres: list[str], distances: list[float], fold: float = 2.0, **kw):
    """Antigen 0 is the serum's homologous antigen (same name, cell passage)."""
    antigens = [Antigen(NAME, "MDCK2"), *others(len(titres) - 1)]
    c = chart(antigens, [Serum(NAME, "F1/20", "MDCK1")], [titres])
    return serum_circles(c, ruler(*distances), fold, **kw)[0]


def test_empirical_radius_separates_protected_from_unprotected():
    """Column basis log2(640/10) = 6; boundary 6 - 2 = 4: 640 and 320 are protected, 40 and 20
    not. Radius 2.5 (midway between the 2nd and 3rd antigens) puts nothing on the wrong side.
    Theoretical: 2 + 6 - 6 = 2."""
    rec = one_serum(["640", "320", "40", "20"], [1, 2, 3, 4])
    (h,) = rec.homologous
    assert h.antigen == 0 and h.status == "good"
    assert h.empirical == 2.5 and h.theoretical == 2.0
    assert rec.circle.empirical == 2.5 and rec.circle.theoretical == 2.0
    assert rec.circle.status == "good" and not rec.circle.floored
    assert rec.column_basis == 6.0


def test_tied_radii_are_averaged():
    """Protected: 1st, 2nd, 4th. Radius 2.5 leaves the 4th outside, 4.5 leaves the 3rd inside:
    one antigen on the wrong side each, the fewest, so the radius is their mean 3.5."""
    rec = one_serum(["640", "320", "40", "320", "20"], [1, 2, 3, 4, 5])
    assert rec.circle.empirical == 3.5


def test_both_radii_are_floored_at_two_whatever_the_fold():
    """Fold 1: theoretical 1 + 6 - 6 = 1, floored to 2. Empirical: boundary 6 - 1 = 5 protects
    only 640; radii 1 and 1.5 both leave nothing on the wrong side, so 1.25, floored to 2."""
    rec = one_serum(["640", "160", "40", "20"], [1, 2, 3, 4], fold=1.0)
    assert rec.homologous[0].theoretical == 1.0 and rec.homologous[0].empirical == 1.25
    assert rec.circle.theoretical == 2.0 and rec.circle.empirical == 2.0 and rec.circle.floored


def test_a_low_homologous_titre_has_no_empirical_radius():
    """Boundary min(cb, log2(40/10) = 2) - 2 = 0 < 1: ae's "titre too low"."""
    rec = one_serum(["40", "640", "20", "20"], [1, 2, 3, 4])
    assert rec.homologous[0].empirical is None and rec.homologous[0].status == "titre too low"
    assert rec.circle.empirical is None and rec.circle.theoretical is not None
    assert rec.coverage.status == "too_low"  # threshold 2 - 2 = 0


def test_a_more_than_homologous_titre_has_an_empirical_radius_only():
    rec = one_serum([">640", "320", "40", "20"], [1, 2, 3, 4])
    (h,) = rec.homologous
    assert h.theoretical is None and h.empirical is not None and h.status == "good"
    assert rec.coverage.status == "non_regular"  # ae refuses a non-regular homologous titre


def test_coverage_splits_titrated_antigens_at_the_homologous_titre_less_fold():
    """Threshold log2(640/10) - 2 = 4, i.e. 160. Missing and dodgy titres are in neither set."""
    rec = one_serum(["640", "160", "80", "<10", "*", "~640"], [1, 2, 3, 4, 5, 6])
    assert rec.coverage.status == "good" and rec.coverage.threshold == 4.0
    assert rec.coverage.within == (0, 1) and rec.coverage.outside == (2, 3)
    assert rec.coverage.dont_care == 2


def test_homologous_antigens_match_name_reassortant_and_egg_in_index_order():
    antigens = [
        Antigen(NAME, "E4"),  # egg: not the cell serum's
        Antigen(NAME, "MDCK1"),
        Antigen(NAME, "SIAT2"),  # another cell passage still matches
        Antigen(NAME, "MDCK1", reassortant="NYMC X-1"),  # a reassortant does not
        Antigen("A(H3N2)/EXAMPLETOWN/2/2020", "MDCK1"),
    ]
    cell = Serum(NAME, "F1/20", "MDCK1")
    egg = Serum(NAME, "F2/20", "E3/E1")  # ae reads the last step
    no_passage = Serum(NAME, "EGG F3/20")  # passage type in the serum id
    assert homologous_antigens(antigens, cell) == [1, 2]
    assert homologous_antigens(antigens, egg) == [0]
    assert homologous_antigens(antigens, no_passage) == [0]


def test_annotations_count_unless_no_antigen_matches_with_them():
    antigens = [Antigen(NAME, "MDCK1"), Antigen(NAME, "MDCK1", annotations=("CONC 2:1",))]
    assert homologous_antigens(antigens, Serum(NAME, "F1", "MDCK1", annotations=("CONC 2:1",))) == [
        1
    ]
    assert homologous_antigens(antigens, Serum(NAME, "F1", "MDCK1", annotations=("X",))) == [0, 1]


def test_the_coverage_antigen_is_the_best_ranked_one_with_a_titre():
    antigens = [Antigen(NAME, "MDCK2"), Antigen(NAME, "MDCK1"), Antigen(NAME, "E2")]
    serum = Serum(NAME, "F1", "MDCK1")
    t = [Titre.parse(v) for v in ("320", "160", "640")]
    assert coverage_antigen(antigens, serum, t) == 1  # same passage: rank 0
    t[1] = Titre.parse("*")
    assert coverage_antigen(antigens, serum, t) == 0  # passage differs (4), same type
    t[0] = Titre.parse("*")
    assert coverage_antigen(antigens, serum, t) == 2  # passage and its type differ (6)


def test_the_circle_and_coverage_rules_are_named_and_swappable():
    """Two homologous antigens. Antigen 1 (640, boundary 4) protects the antigens at 1 and 1.5
    only: radius 1.75. Antigen 0 (320, boundary 3) protects 80 at 2 as well: radius 2.5. The
    circle takes the smaller; the coverage antigen is the exact passage, antigen 0."""
    antigens = [Antigen(NAME, "MDCK1"), Antigen(NAME, "SIAT1"), *others(3)]
    c = chart(antigens, [Serum(NAME, "F1", "MDCK1")], [["320", "640", "80", "40", "20"]])
    layout = ruler(1.5, 1, 2, 3, 4)
    (a,) = serum_circles(c, layout)
    assert a.circle.rule == "ae-circle" and a.coverage.rule == "ae-coverage"
    assert [h.empirical for h in a.homologous] == [2.5, 1.75]
    assert a.circle.empirical_antigen == 1 and a.circle.empirical == 2.0  # floored
    assert a.coverage.antigen == 0
    (b,) = serum_circles(c, layout, coverage_rule="circle-antigen")
    assert b.coverage.antigen == 1 and b.coverage.rule == "circle-antigen"
    assert b.circle == a.circle
    (d,) = serum_circles(c, layout, circle_rule="coverage-antigen")
    assert d.circle.empirical_antigen == 0 and d.circle.empirical == 2.5
    assert d.coverage == a.coverage
    with pytest.raises(SerumCircleError, match="circle rule"):
        serum_circles(c, layout, circle_rule="nearest")
    with pytest.raises(SerumCircleError, match="layout has 5 points"):
        serum_circles(c, layout[:-1])


def test_a_serum_without_coordinates_or_homologous_antigen_has_no_circle():
    c = chart(
        [Antigen(NAME, "MDCK1"), *others(2)], [Serum(NAME, "F1", "MDCK1")], [["640", "40", "20"]]
    )
    layout = ruler(1, 2, 3)
    layout[-1] = np.nan
    (rec,) = serum_circles(c, layout)
    assert rec.xy is None and rec.homologous[0].status == "serum disconnected"
    assert rec.circle.empirical is None and rec.circle.theoretical == 2.0
    c.antigens[0] = Antigen("A(H3N2)/EXAMPLEOTHER/1/2020", "MDCK1")
    (rec,) = serum_circles(c, ruler(1, 2, 3))
    assert rec.homologous == () and rec.circle.status == "no_homologous"
    assert rec.coverage.status == "no_homologous_titre"


def test_the_record_is_json():
    rec = one_serum(["640", "320", "40", "20"], [1, 2, 3, 4])
    data = json.loads(json.dumps(rec.to_json()))
    assert data["circle"]["empirical"] == 2.5 and data["passage_class"] == "cell"
    assert data["homologous"][0]["titre"] == "640" and data["fold"] == 2.0
    assert not math.isnan(data["column_basis"])


def test_distances_are_the_same_with_and_without_math_fma(monkeypatch):
    """Python 3.11 and 3.12 have no math.fma; the exact-rational fallback rounds the same."""
    from af.chart import serum_circles as sc

    layout = np.array([[0.1, 0.7], [1e-9, 3.3], [np.nan, np.nan], [0.3, 0.2]])
    with_fma = sc._distances(layout, 3, 3)
    monkeypatch.delattr(math, "fma", raising=False)
    without = sc._distances(layout, 3, 3)
    assert np.array_equal(with_fma, without, equal_nan=True) and np.isnan(without[2])
