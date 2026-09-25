"""Helpers shared by tests. Import them as ``from tests.helpers import ...``.

Kept out of conftest.py: a test must never import conftest by name, because pytest
loads conftest itself and a second import under another name is a second module.
"""

import os
from pathlib import Path

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
