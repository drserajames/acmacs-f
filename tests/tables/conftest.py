"""Fixtures for the table tests; the synthetic rules themselves are in synthetic_rules.py
(tests import helpers from there, never from a conftest)."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.tables.rules import Rules

from .synthetic_rules import write_rules


@pytest.fixture
def rules_dir(tmp_path: Path) -> Path:
    return write_rules(tmp_path / "rules")


@pytest.fixture
def rules(rules_dir: Path) -> Rules:
    return Rules(rules_dir)
