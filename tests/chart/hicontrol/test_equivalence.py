"""af.chart.control is equivalent to hicontrol: every case R produced (make_fixtures.R) must match.

The fixtures are synthetic series, not lab data. R indices are 1-based and af's 0-based.
"""

import json
import math
from pathlib import Path

import pytest

from af.chart import control

FIXTURES = json.loads((Path(__file__).parent / "hicontrol-cases.json").read_text())
CASES = FIXTURES["cases"]
INDEX_RULES = {"rule_xyz", "rule_diff"}
RUN_RULES = {"rule_noC", "rule_onlyC", "rule_noCrelax", "rule_alt", "rule_nodiff"}
HI_ORDER = (
    "rule_xyz",
    "rule_xyz",
    "rule_xyz",
    "rule_noCrelax",
    "rule_nodiff",
    "rule_alt",
    "rule_trend",
    "rule_diff",
)


def series(values):
    return [math.nan if v is None else float(v) for v in values]


def expected(fn, r):
    """R's encoded result in af's form."""
    if r["kind"] == "null":
        return None
    if fn in INDEX_RULES:
        return [int(v) - 1 for v in r["values"]]
    if r["kind"] == "matrix" and r["ncol"] == 0:
        return []
    rows = r["rows"]
    if fn == "rule_trend":
        return [
            (None if s is None else int(s) - 1, int(n), int(d))
            for s, n, d in zip(*rows, strict=True)
        ]
    if fn in RUN_RULES:
        return [(int(s) - 1, int(n)) for s, n in zip(*rows, strict=True)]
    raise AssertionError(f"unexpected result for {fn}: {r}")


def test_fixtures_come_from_the_ported_commit():
    assert FIXTURES["commit"] == control.HICONTROL_COMMIT
    assert len(CASES) > 3000


@pytest.mark.parametrize(
    "case", CASES, ids=[f"{i}-{c['fn']}-n{len(c['dat'])}" for i, c in enumerate(CASES)]
)
def test_case_matches_r(case):
    fn, dat, params, r = case["fn"], series(case["dat"]), case["params"], case["result"]
    if r["kind"] == "error":
        with pytest.raises(ValueError):
            getattr(control, "hi_rules" if fn == "hi_rules2" else fn)(dat, **params)
        return
    if fn in ("hi_rules", "hi_rules2"):
        got = control.hi_rules(dat, **params)
        items, numbers = r["items"]
        assert list(got.results) == [
            expected(f, x) for f, x in zip(HI_ORDER, items["items"], strict=True)
        ]
        if fn == "hi_rules2":
            assert list(got.counts) == [int(v) for v in numbers["values"]]
        else:
            assert list(got.rates(len(dat))) == numbers["values"]
        return
    assert getattr(control, fn)(dat, **params) == expected(fn, r)


@pytest.mark.parametrize("case", FIXTURES["log_num"], ids=lambda c: "|".join(c["titres"]))
def test_log_num_matches_r(case):
    got = control.log_num(case["titres"])
    want = [math.nan if v is None else v for v in case["result"]["values"]]
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert (math.isnan(g) and math.isnan(w)) or g == w
