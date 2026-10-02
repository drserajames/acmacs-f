"""Points a chain leaves out: named rules applied to every table as it enters the merge.

A round's map is often a selection of the lab's tables, not all of them (today's `0do` files
drop, e.g., every egg antigen and serum from an egg-free map). Here that selection is data in
the chain config, one rule per `[[select.remove]]` entry, each with its reason:

    [[select.remove]]
    what = "antigens"             # or "sera"
    passage = "egg"               # ANY passage step in eggs (af.map.vaccines.passage_class)
    reason = "egg-free map (Sarah, 29 Sep 2026)"

A rule has exactly one selector: `passage` (the class of the whole passage history, see below),
`name` (a substring of the virus name) or `designation` (a substring of the designation, which
adds the passage). `passage = "egg"` means ANY step was in eggs, so E3/E2SIAT1 is egg. ae's
`is_egg()` looks at the LAST step only and would call it cell; the two differ on egg-then-cell
passages (Sarah chose any-step, 29 Sep 2026).

Why applied per table, before the merge: a removed point then never reaches the merged titres,
the column bases or the map, which is what the round's per-merge removal does. A rule that
matches nothing in any of the chain's tables is an error (a typo would otherwise remove
nothing, silently); the matches are counted per step.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from af.chart.ace import read_chart
from af.chart.model import Antigen, Chart, Serum, Titres
from af.chart.sera import FERRET_ONLY, Marker, non_ferret
from af.chart.titre import MISSING_TITRE
from af.map.vaccines import passage_class

WHAT = ("antigens", "sera")
PASSAGE_CLASSES = ("egg", "cell")


class SelectError(ValueError):
    pass


@dataclass(frozen=True)
class RemoveRule:
    what: str  # "antigens" or "sera"
    reason: str
    passage: str | None = None  # "egg" (any step in eggs) or "cell"
    name: str | None = None  # substring of the name
    designation: str | None = None  # substring of the designation

    def __post_init__(self) -> None:
        if self.what not in WHAT:
            raise SelectError(f"select.remove: what must be antigens or sera, not {self.what!r}")
        if not self.reason.strip():
            raise SelectError("select.remove: every rule needs a reason")
        chosen = [k for k in ("passage", "name", "designation") if getattr(self, k)]
        if len(chosen) != 1:
            raise SelectError(
                f"select.remove ({self.reason}): exactly one of passage, name, designation"
                f" (got {chosen or 'none'})"
            )
        if self.passage is not None and self.passage not in PASSAGE_CLASSES:
            raise SelectError(f"select.remove: passage must be one of {PASSAGE_CLASSES}")

    def matches(self, point: Antigen | Serum) -> bool:
        if self.passage is not None:
            # the passage history alone: a reassortant grown in eggs is egg here, as in ae
            return passage_class(point.passage, "") == self.passage
        if self.name is not None:
            return self.name in point.name
        assert self.designation is not None
        return self.designation in point.designation()

    def to_json(self) -> dict[str, str]:
        return {k: v for k, v in vars(self).items() if v is not None}


@dataclass(frozen=True)
class DropCell:
    """One reading kept out of the MAP, the table keeping the lab's value: `[[select.drop_cell]]`
    {table, antigen, serum, reason, decided}. Antigen and serum are exact designations; the rule
    must match exactly one present reading in that one table, or the chain refuses to start
    (design rule 1). Sarah, 2 Oct 2026 (Q98): an implausible ">10240" set a serum's column basis
    ~2 units high; the lab is asked (LAB-QUESTIONS L11) and the rule goes when it answers."""

    table: str  # table id, e.g. <group>-20221208
    antigen: str
    serum: str
    reason: str
    decided: str

    def __post_init__(self) -> None:
        empty = [
            k
            for k in ("table", "antigen", "serum", "reason", "decided")
            if not getattr(self, k).strip()
        ]
        if empty:
            raise SelectError(f"select.drop_cell: empty {', '.join(empty)}")

    def to_json(self) -> dict[str, str]:
        return dict(vars(self))


def drop_cells(rules: list[DropCell], table_id: str, chart: Chart) -> tuple[Chart, list[dict]]:
    """`chart` (one table) with each rule for this table's reading made missing, and what each
    rule dropped (the reading as the lab wrote it)."""
    mine = [r for r in rules if r.table == table_id]
    if not mine:
        return chart, []
    table = [list(row) for row in chart.titres.table]
    report = []
    for rule in mine:
        i, j = _cell(rule, chart)
        report.append({**rule.to_json(), "titre": str(table[i][j])})
        table[i][j] = MISSING_TITRE
    layers = [{k: v for k, v in layer.items() if k != (i, j)} for layer in chart.titres.layers]
    return replace(chart, titres=Titres(table, layers)), report


def _cell(rule: DropCell, chart: Chart) -> tuple[int, int]:
    ags = [i for i, a in enumerate(chart.antigens) if a.designation() == rule.antigen]
    srs = [j for j, s in enumerate(chart.sera) if s.designation() == rule.serum]
    if len(ags) != 1 or len(srs) != 1:
        raise SelectError(
            f"select.drop_cell {rule.table}: antigen {rule.antigen!r} matches {len(ags)},"
            f" serum {rule.serum!r} matches {len(srs)}; each must match exactly one"
        )
    if chart.titres.table[ags[0]][srs[0]].is_missing:
        raise SelectError(f"select.drop_cell {rule.table}: that reading is already missing")
    return ags[0], srs[0]


def check_cells_match(rules: list[DropCell], tables: dict[str, Path]) -> None:
    """Every drop_cell rule names one of the chain's tables and one present reading in it,
    checked before any step runs."""
    for rule in rules:
        if rule.table not in tables:
            raise SelectError(
                f"select.drop_cell: table {rule.table} is not one of the chain's tables"
            )
        _cell(rule, read_chart(tables[rule.table]))


@dataclass(frozen=True)
class Selection:
    """The `[select]` table of a chain config."""

    remove: list[RemoveRule] = field(default_factory=list)
    drop_cell: list[DropCell] = field(default_factory=list)
    # Ferret sera only (af.chart.sera.FERRET_ONLY): "exclude" is the default for every map; a
    # map that needs other sera says "keep" and why.
    non_ferret_sera: str = "exclude"
    non_ferret_reason: str = ""
    # The markers file; default: <data repo>/rules/sera/non_ferret_markers.tsv next to chains/
    non_ferret_markers: Path | None = None

    def __post_init__(self) -> None:
        if self.non_ferret_sera not in ("exclude", "keep"):
            raise SelectError("select.non_ferret_sera must be exclude or keep")
        if self.non_ferret_sera == "keep" and not self.non_ferret_reason.strip():
            raise SelectError("select.non_ferret_sera = keep needs a non_ferret_reason")


@dataclass(frozen=True)
class SeraPolicy:
    """The ferret-only policy as one chain applies it."""

    mode: str = "exclude"  # or "keep" (with reason)
    reason: str = ""
    markers: tuple[Marker, ...] = ()  # empty: species only (a chain built in code, e.g. tests)
    markers_path: str | None = None
    markers_sha256: str | None = None

    def to_json(self) -> dict:
        out: dict[str, object] = {"mode": self.mode, "policy": FERRET_ONLY}
        if self.reason:
            out["reason"] = self.reason
        if self.markers_path:
            out["markers"] = {"path": self.markers_path, "sha256": self.markers_sha256}
        return out


def apply_sera_policy(policy: SeraPolicy, chart: Chart) -> tuple[Chart, dict]:
    """`chart` without its non-ferret sera (unless the chain keeps them), and the report: what
    was removed and by which test, and how many sera are ferret only by default."""
    report = non_ferret(chart, list(policy.markers))
    out = {**report.to_json(), "removed": policy.mode == "exclude" and bool(report.non_ferret)}
    if policy.mode == "keep" or not report.non_ferret:
        return chart, out
    gone = {j for j, _, _ in report.non_ferret}
    return chart.select(
        range(chart.n_antigens), [j for j in range(chart.n_sera) if j not in gone]
    ), out


def removed_by(rules: list[RemoveRule], chart: Chart) -> list[dict]:
    """Per rule, the designations it removes from `chart` (empty lists included)."""
    out = []
    for rule in rules:
        points = chart.antigens if rule.what == "antigens" else chart.sera
        out.append(
            {**rule.to_json(), "points": [p.designation() for p in points if rule.matches(p)]}
        )
    return out


def apply(rules: list[RemoveRule], chart: Chart) -> tuple[Chart, list[dict]]:
    """`chart` without the points any rule removes, and what each rule removed.

    The removal itself is `Chart.select`, the one copy of how points are taken out of a chart
    (titres, layers, per-point map settings, plot styles). Chains apply rules to tables, before
    the merge; a chart with maps would come back with each map's stress unset (stale)."""
    report = removed_by(rules, chart)
    if not any(r["points"] for r in report):
        return chart, report
    keep_ag = [i for i, a in enumerate(chart.antigens) if not _removed(rules, "antigens", a)]
    keep_sr = [j for j, s in enumerate(chart.sera) if not _removed(rules, "sera", s)]
    return chart.select(keep_ag, keep_sr), report


def _removed(rules: list[RemoveRule], what: str, point: Antigen | Serum) -> bool:
    return any(r.what == what and r.matches(point) for r in rules)


def check_rules_match(rules: list[RemoveRule], tables: list[Path]) -> None:
    """Every rule must remove something from at least one of the chain's tables (design rule 1).
    Checked before any step runs, so a typo fails in seconds, not after hours of maps."""
    unmatched = list(rules)
    for path in tables:
        if not unmatched:
            return
        chart = read_chart(path)
        unmatched = [rule for rule in unmatched if not removed_by([rule], chart)[0]["points"]]
    if unmatched:
        raise SelectError(
            "select.remove rules match nothing in the chain's tables: "
            + "; ".join(str(r.to_json()) for r in unmatched)
        )
