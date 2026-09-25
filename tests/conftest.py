"""Shared pytest fixtures.

Real WHO CC data never lives in this (public) repo. Tests that need it take the
``af_data`` fixture, which finds the private data repo and skips the test cleanly
when it is absent, so a public CI machine never fails for want of private data.
"""

import os
from pathlib import Path

import pytest

from tests import helpers
from tests.helpers import resolve_af_data


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-tools",
        action="store_true",
        help="fail, not skip, a test whose external tool (@pytest.mark.tool) is missing",
    )
    parser.addoption(
        "--skip-tool",
        action="append",
        default=[],
        metavar="NAME",
        help="skip tests needing this tool, even under --require-tools (repeatable)",
    )


def pytest_configure(config: pytest.Config) -> None:
    helpers.REQUIRE_TOOLS = bool(config.getoption("--require-tools"))
    helpers.SKIP_TOOLS = frozenset(config.getoption("--skip-tool"))


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Resolve every ``@pytest.mark.tool(name)`` before the test runs."""
    for mark in item.iter_markers(name="tool"):
        for name in mark.args:
            helpers.require_tool(name)


@pytest.fixture(scope="session")
def af_data() -> Path:
    """Root of the private acmacs-f-data repo; skips the test if it is not there."""
    path = resolve_af_data()
    if not path.is_dir():
        source = "AF_DATA" if os.environ.get("AF_DATA") else "default ../acmacs-f-data"
        pytest.skip(f"private data repo not found at {path} ({source}); real-data test skipped")
    return path
