"""log2 with one answer on every machine.

Why: a titre reaches the optimiser as log2(value/10), and the optimiser turns one bit of difference
in its input into a different local minimum. The system's log2 (`math.log2`) is accurate to about
one unit in the last place, but which neighbour it returns is the C library's choice: one machine
measured here returns the correctly rounded value for every titre value that occurs in a map today
and the other neighbour for 87 of the integers 1..1,000,000 (the lowest is 39,785). A map must not
depend on that, so the value is defined here: the double nearest to the true log2.

Only titres that are not ten times a power of two are affected at all (a power of two is exact
everywhere), and the definition changes nothing where the C library already rounds correctly.
"""

from __future__ import annotations

import functools
import math
from decimal import Decimal, localcontext

_DIGITS = 40  # decimal digits of the first attempt; doubled until the rounding is certain
_SLACK = 8  # digits given up to the three rounded operations: far more than their error


@functools.cache  # a chart has a few thousand distinct titre values and millions of cells
def log2(x: float) -> float:
    """The double nearest to log2(x): correctly rounded, the same bits on every machine."""
    if not 0.0 < x < math.inf:
        raise ValueError(f"log2 of {x!r}: not a positive finite number")
    mantissa, exponent = math.frexp(x)
    if mantissa == 0.5:
        return float(exponent - 1)  # a power of two: exact
    digits = _DIGITS
    while True:
        # log2 of any other double is irrational, so it is never exactly halfway between two
        # doubles: with enough digits both ends of the error interval round to the same one.
        with localcontext() as context:
            context.prec = digits
            value = Decimal(x).ln() / Decimal(2).ln()  # Decimal(x) is exact
            error = abs(value).scaleb(_SLACK - digits)
            low, high = float(value - error), float(value + error)
        if low == high:
            return low
        digits *= 2
