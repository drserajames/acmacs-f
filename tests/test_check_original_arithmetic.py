"""tools/check-original-arithmetic.py: its three answers, and one real comparison.

A release build acts on its exit status (0 the same bits, 1 different, anything else could not
compare), so each is pinned here. The real comparison is this build's original core against
itself through a second python process: the path a release takes against an earlier release.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "check-original-arithmetic.py"


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_original_arithmetic", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_original_core_equals_itself_through_another_process(tool, capsys):
    assert tool.main(["--reference", sys.executable]) == tool.SAME == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert "invented table: 40 starts" in lines[0] and "the same bits" in lines[0]
    assert "iterations" in lines[0] and "layout sha256" in lines[0]  # stated, not just "equal"
    assert lines[-1].startswith("original arithmetic: the same bits on 1 problem(s)")


def test_a_differing_bit_is_exit_1(tool, monkeypatch, capsys):
    real = tool.reference

    def one_bit_off(args):
        result = real(args)
        problem = result["problems"]["invented table"]
        problem["layout"] = "0" * 64
        return result

    monkeypatch.setattr(tool, "reference", one_bit_off)
    assert tool.main(["--reference", sys.executable]) == tool.DIFFERENT == 1
    assert "original arithmetic: DIFFERENT bits on invented table" in capsys.readouterr().out


def test_a_reference_that_will_not_run_is_exit_2(tool, capsys, tmp_path):
    assert (
        tool.main(["--reference", str(tmp_path / "no-such-python")]) == tool.COULD_NOT_COMPARE == 2
    )
    assert "COULD NOT COMPARE" in capsys.readouterr().out


def test_an_unforeseen_failure_is_exit_2_not_pythons_exit_1(tool, monkeypatch, capsys):
    # python exits 1 on an unhandled exception, and 1 here means "compared, and different"
    def broken(args):
        raise RuntimeError("something nobody planned for")

    monkeypatch.setattr(tool, "emit", broken)
    assert tool.main(["--reference", sys.executable]) == tool.COULD_NOT_COMPARE
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith("original arithmetic: COULD NOT COMPARE: RuntimeError")


def test_a_build_without_the_original_cannot_be_compared(tool):
    class ContractOnly:  # an af whose optimiser has the contract and nothing else
        NUMERICS = "strict-fp-1"

    class Before:  # an af from before the contract: its only arithmetic is the original
        pass

    class Both:
        NUMERICS = "strict-fp-1"
        NUMERICS_OFFERED = ("strict-fp-1", "original")

    with pytest.raises(tool.CannotCompare, match="no original core"):
        tool.original_options(ContractOnly)
    assert tool.original_options(Before) == {}
    assert tool.original_options(Both) == {"numerics": "original"}
