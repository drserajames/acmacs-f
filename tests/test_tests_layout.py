"""Every directory under tests/ is a package, so each test helper has exactly one name.

Without an __init__.py a directory's modules can be imported under a second, top-level
name, which a clean checkout rejects (mypy: "Source file found twice"). CLAUDE.md, "Tests".
"""

from tests.helpers import REPO_ROOT

SKIP = {"__pycache__"}


def test_every_tests_directory_has_init() -> None:
    tests = REPO_ROOT / "tests"
    directories = [tests, *(d for d in tests.rglob("*") if d.is_dir() and not SKIP & set(d.parts))]
    missing = sorted(
        str(d.relative_to(REPO_ROOT))
        for d in directories
        if any(d.glob("*.py")) and not (d / "__init__.py").exists()
    )
    assert missing == [], f"add an __init__.py to: {missing}"
