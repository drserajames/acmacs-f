"""Tool-gated tests skip on a bare machine and fail in CI's tools job (--require-tools)."""

import pytest

from tests import helpers

ABSENT = "af-no-such-tool-xyz"


def test_present_tool_resolves() -> None:
    assert helpers.require_tool("sh").endswith("/sh")


def test_missing_tool_skips_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helpers, "REQUIRE_TOOLS", False)
    with pytest.raises(pytest.skip.Exception, match=f"tool not installed: {ABSENT}"):
        helpers.require_tool(ABSENT)


def test_missing_tool_fails_under_require_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helpers, "REQUIRE_TOOLS", True)
    with pytest.raises(pytest.fail.Exception, match=f"required tool not installed: {ABSENT}"):
        helpers.require_tool(ABSENT)


def test_skip_tool_wins_over_require_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helpers, "REQUIRE_TOOLS", True)
    monkeypatch.setattr(helpers, "SKIP_TOOLS", frozenset({"sh"}))
    with pytest.raises(pytest.skip.Exception, match="tool skipped by --skip-tool: sh"):
        helpers.require_tool("sh")
