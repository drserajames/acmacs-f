"""A titre's log is the same number on every machine (af.chart.exact).

The values pinned here were computed on one machine; this test passing on another (CI runs a
different processor, compiler and C library) is the evidence that they do not depend on it.
"""

import hashlib
import math
from decimal import Decimal, localcontext

import pytest

from af.chart import control, exact
from af.chart.model import minimum_column_basis_value
from af.chart.titre import Titre
from af.serology.titre import parse_reading

# integers v for which one C library's log2(v/10) is the other neighbour of the true value (all
# five below 100,000 on the machine that was measured), with the correctly rounded answer
OTHER_NEIGHBOUR = {
    39785: "0x1.7ea8023f12b07p+3",
    53947: "0x1.8cb6e7269725dp+3",
    71403: "0x1.99a81768dd8c9p+3",
    79570: "0x1.9ea8023f12b07p+3",
    98473: "0x1.a87f140f96b23p+3",
}
EVERY_TITRE_TO_20000 = "764362121c927c4a90dc490bf4f90958b3dfaacc6ccde1a14afbeae847ddf6bf"


def test_a_power_of_two_is_exact():
    assert [exact.log2(2.0**k) for k in range(-1074, 1024, 7)] == [
        float(k) for k in range(-1074, 1024, 7)
    ]
    assert [Titre.parse(str(10 * 2**k)).logged() for k in range(12)] == [
        float(k) for k in range(12)
    ]
    assert minimum_column_basis_value("1280") == 7.0 and minimum_column_basis_value("none") == 0.0


def test_the_values_are_the_same_bits_everywhere():
    h = hashlib.sha256()
    for v in range(1, 20001):
        h.update(exact.log2(v / 10.0).hex().encode())
    assert h.hexdigest() == EVERY_TITRE_TO_20000
    assert {v: exact.log2(v / 10.0).hex() for v in OTHER_NEIGHBOUR} == OTHER_NEIGHBOUR
    assert Titre.parse("57").logged().hex() == "0x1.4167333e661c4p+1"  # 40 and 80 merged
    assert Titre.parse("<113").logged().hex() == "0x1.bfc6af369e297p+1"  # the prefix is ignored


def test_every_titre_log_is_the_defined_one():
    # the serology store's readings and the HI-control port log as the optimiser's input does
    for v, want in OTHER_NEIGHBOUR.items():
        assert parse_reading(str(v)).log.hex() == want
        assert parse_reading(f"<{v}").log.hex() == want  # the face value
        assert control.log_num([str(v)])[0].hex() == want
        assert control.log_num([f"<{2 * v}"])[0].hex() == want  # half the detection limit
    assert control.log_num(["inf", "0"]) == [math.inf, -math.inf]
    assert all(math.isnan(x) for x in control.log_num(["-40", "*", ">1280"]))


def _nearest(x: float) -> float:
    """The double nearest to log2(x), from far more digits than the function starts with."""
    with localcontext() as context:
        context.prec = 200
        return float(Decimal(x).ln() / Decimal(2).ln())


@pytest.mark.parametrize("x", [0.1, 0.3, 1.1, 5.7, 11.3, 3978.5, 1e-300, 1e300, 5e-324])
def test_it_is_the_nearest_double(x):
    assert exact.log2(x) == _nearest(x)


def test_the_c_library_is_never_more_than_one_step_away():
    # so adopting the definition moves no titre's log by more than the last bit
    for v in (*range(1, 3000), *OTHER_NEIGHBOUR):
        ours, theirs = exact.log2(v / 10.0), math.log2(v / 10.0)
        assert abs(ours - theirs) <= math.ulp(ours)


@pytest.mark.parametrize("x", [0.0, -1.0, math.inf, math.nan])
def test_only_a_positive_finite_number_has_a_log(x):
    with pytest.raises(ValueError, match="not a positive finite number"):
        exact.log2(x)
