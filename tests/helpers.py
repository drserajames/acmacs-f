"""Helpers shared by tests. Import them as ``from tests.helpers import ...``.

Kept out of conftest.py: a test must never import conftest by name, because pytest
loads conftest itself and a second import under another name is a second module.
"""

import os
import shutil
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


REQUIRE_TOOLS = False
"""Set by conftest from ``--require-tools``: a missing tool fails instead of skipping."""

SKIP_TOOLS: frozenset[str] = frozenset()
"""Set by conftest from ``--skip-tool NAME``: tools deliberately not tested on this run."""


def require_tool(name: str) -> str:
    """Path of an external tool; skip without it, or fail under ``--require-tools``.

    Tests mark their tools with ``@pytest.mark.tool("cmaple")`` (conftest calls this).
    The CI job built from environment.yml runs ``pytest --require-tools -m tool``, so
    a tool that environment should provide but doesn't is a failure, never a skip.
    """
    if name in SKIP_TOOLS:
        pytest.skip(f"tool skipped by --skip-tool: {name}")
    path = shutil.which(name)
    if path is not None:
        return path
    if REQUIRE_TOOLS:
        pytest.fail(f"required tool not installed: {name} (--require-tools)", pytrace=False)
    pytest.skip(f"tool not installed: {name}")
