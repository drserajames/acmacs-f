"""Named hand overrides for the tree figure, read from a TOML file per report or round.

The automatic rules decide almost everything; what is left is a short list of named choices
(show or hide one clade), each with its reason and who decided it. They live in data (the
private data repo), not in code, and the draw report prints every one that was applied.

File layout (one table per subtype; a subtype with no table has no overrides)::

    [[subtypes.h1.hide_clades]]
    clade = "D.3"
    reason = "repeats D.3.1's bracket: 97.4% of its leaves are the same"
    decided = "Sarah, 25 Sep 2026"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from af.util.config import load_config

from .figure import Overrides


class OverrideFileError(ValueError):
    """The same clade is both shown and hidden, or listed twice."""


@dataclass(frozen=True)
class NamedClade:
    clade: str
    reason: str
    decided: str


@dataclass(frozen=True)
class SubtypeOverrides:
    hide_clades: list[NamedClade] = field(default_factory=list)
    show_clades: list[NamedClade] = field(default_factory=list)


@dataclass(frozen=True)
class OverridesFile:
    subtypes: dict[str, SubtypeOverrides]


def load_overrides(path: Path, subtype: str) -> Overrides:
    """The overrides for ``subtype`` from ``path`` (every row needs a reason and who decided)."""
    doc = load_config(path, OverridesFile)
    rows = doc.subtypes.get(subtype, SubtypeOverrides())
    hide = [r.clade for r in rows.hide_clades]
    show = [r.clade for r in rows.show_clades]
    for kind, names in (("hide_clades", hide), ("show_clades", show)):
        if len(set(names)) != len(names):
            raise OverrideFileError(f"{path}: {subtype} {kind} lists a clade twice")
    both = sorted(set(hide) & set(show))
    if both:
        raise OverrideFileError(f"{path}: {subtype} both shows and hides {both}")
    reasons = {
        f"{kind}:{r.clade}": f"{r.reason} ({r.decided})"
        for kind, group in (("hide", rows.hide_clades), ("show", rows.show_clades))
        for r in group
    }
    return Overrides(show_clades=frozenset(show), hide_clades=frozenset(hide), reasons=reasons)
