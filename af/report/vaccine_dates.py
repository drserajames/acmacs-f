"""When each vaccine rule was decided, derived from recorded sources when config has no date.

Sarah, 8 Oct (Q126): "if dates can not be derived from who recommendations folder or git, then
put 'not known'". So a rule's date comes from, in order:

1. the config entry's own ``decided`` (a hand date always wins; the note says when it overrides
   a derived one);
2. for a rule that stops marking a whole vaccine (``disable``, passage ``any``), WHO's
   published recommendations: the first recommendation, in any hemisphere or platform, after
   the last one that named the vaccine (when it was no longer recommended anywhere);
3. the git history of the config file: the first commit that added the rule's row;
4. otherwise "not known", printed as such (a known unknown, not a missing record).

Each date says which source it came from. WHO's file is read, never edited; names are matched
by af's normalisation of both sides (type prefix dropped, case and leading zeros of numbers
ignored), so ``<type>/<Place>/02/<year>`` and ``<PLACE>/2/<year>`` are one vaccine.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NOT_KNOWN = "not known"


def normalise(name: str) -> str:
    """A vaccine name both sides can compare: no type prefix, upper case, numbers unpadded."""
    parts = name.strip().upper().split("/")
    if parts and parts[0] in ("A", "B"):
        parts = parts[1:]
    return "/".join(p.lstrip("0") or "0" if p.isdigit() else p for p in parts)


@dataclass(frozen=True)
class Recommendation:
    date: dt.date
    season_id: str
    vaccines: frozenset[str]  # normalised names of every component, any platform


def load_who(path: Path) -> list[Recommendation]:
    """WHO's recommendations (who-vaccine-recommendations data/*.json), oldest first."""
    document = json.loads(path.read_text())
    out = []
    for season in document["seasons"]:
        vaccines = frozenset(
            normalise(c["vaccine_virus"]) for c in season["components"] if c.get("vaccine_virus")
        )
        out.append(Recommendation(dt.date.fromisoformat(season["recommendation_date"]),
                                  season["season_id"], vaccines))  # fmt: skip
    return sorted(out, key=lambda r: (r.date, r.season_id))


def superseded(who: list[Recommendation], name: str) -> tuple[str, str] | None:
    """(date, source) of the first recommendation after the vaccine's last, or None."""
    key = normalise(name)
    named = [r for r in who if key in r.vaccines]
    if not named:
        return None
    last = named[-1]
    after = [r for r in who if r.date > last.date]
    if not after:
        return None  # still recommended in the latest recommendation
    first = after[0]
    return first.date.isoformat(), (
        f"WHO: last recommended in {last.season_id} ({last.date}); {first.season_id} "
        f"({first.date}) is the first recommendation after it"
    )


def git_introduced(path: Path, needle: str) -> tuple[str, str] | None:
    """(date, source) of the first commit adding ``needle`` to ``path``, or None."""
    try:
        out = subprocess.run(
            ["git", "-C", str(path.parent), "log", "--reverse", "--date=short",
             "--format=%h %ad", "-S", needle, "--", path.name],
            capture_output=True, text=True, check=True,
        ).stdout.split()  # fmt: skip
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if len(out) < 2:
        return None
    return out[1], f"git: commit {out[0]} added the rule to {path.name}"


@dataclass
class VaccineDates:
    """Resolves each vaccine rule's date; ``files`` maps a rule scope to its config file."""

    who: list[Recommendation] = field(default_factory=list)
    files: dict[str, Path] = field(default_factory=dict)  # "map" / "subtype default" -> file

    def resolve(self, rule: dict[str, Any]) -> tuple[str, str]:
        """(date or "not known", where it came from) for one applied vaccine rule."""
        derived = self._derived(rule)
        if rule.get("decided"):
            note = "config decided"
            if derived and derived[0] != str(rule["decided"]):
                note += f", overriding {derived[0]} ({derived[1]})"
            return str(rule["decided"]), note
        if derived:
            return derived
        return NOT_KNOWN, "neither WHO's recommendations nor git history give a date"

    def _derived(self, rule: dict[str, Any]) -> tuple[str, str] | None:
        name = str(rule.get("name", ""))
        if rule.get("rule") == "disable" and rule.get("passage") in (None, "", "any"):
            found = superseded(self.who, name)
            if found:
                return found
        path = self.files.get(str(rule.get("scope", "")))
        return git_introduced(path, f'name = "{name}"') if path else None
