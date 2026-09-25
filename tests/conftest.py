"""Shared pytest fixtures.

Real WHO CC data never lives in this (public) repo. Tests that need it take the
``af_data`` fixture, which finds the private data repo and skips the test cleanly
when it is absent, so a public CI machine never fails for want of private data.
"""

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_af_data() -> Path:
    """Where the private data repo should be: ``$AF_DATA``, else ``../acmacs-f-data``.

    ``AF_DATA`` is the one environment variable af reads, and only in tests
    (design rule 4). A relative ``AF_DATA`` is taken relative to the repo root,
    not the current directory, so the answer does not depend on where pytest runs.
    """
    configured = os.environ.get("AF_DATA")
    path = Path(configured) if configured else Path("..") / "acmacs-f-data"
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


@pytest.fixture(scope="session")
def af_data() -> Path:
    """Root of the private acmacs-f-data repo; skips the test if it is not there."""
    path = resolve_af_data()
    if not path.is_dir():
        source = "AF_DATA" if os.environ.get("AF_DATA") else "default ../acmacs-f-data"
        pytest.skip(f"private data repo not found at {path} ({source}); real-data test skipped")
    return path
