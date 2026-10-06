"""Merging charts: which antigens/sera are the same, and the layer merge.

This reproduces ae's `chart_v3.merge(match="strict", merge_type=...)` as the chains use
it (`ae/cc/chart/v3/merge.cc`, `common.cc`, `titers.cc`), because the chains must replay
today's science. Each rule names the ae source it copies.

Which points are the same across tables is `identity.py`.
"""

from __future__ import annotations

import copy
import enum
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from af.chart.identity import antigen_identity, serum_identity
from af.chart.model import Antigen, Chart, Layer, Projection, Serum, Titres, empty_table
from af.chart.titre import (
    MISSING_TITRE,
    MergeOutcome,
    MergeSettings,
    MoreThanOnly,
    Titre,
    TitreType,
    column_basis,
    merge_titres,
)


class MergeError(ValueError):
    pass


class MergeType(enum.Enum):
    SIMPLE = "simple"  # no projections in the result
    INCREMENTAL = "incremental"  # copy chart1's best layout; new points NaN


class ColumnBasisConvention(enum.Enum):
    """How a merged chart's column bases treat `>` titres the merge discards.

    ADJUST_TO_NEXT is ae's behaviour today (`titers.cc:544-566`): if any layer has a `>`,
    column bases come from a first pass that keeps `>X` (counted at log2(X/10)+1) and are
    then forced for every serum. TABLE_ONLY computes bases from the merged table only.
    Which is the default is Sarah's decision (T22/T23).
    """

    ADJUST_TO_NEXT = "adjust-to-next"
    TABLE_ONLY = "table-only"


@dataclass(frozen=True)
class MergeOptions:
    merge_type: MergeType = MergeType.INCREMENTAL
    combine_copied_references: bool = False
    titres: MergeSettings = MergeSettings()
    column_bases: ColumnBasisConvention = ColumnBasisConvention.ADJUST_TO_NEXT


@dataclass
class MergeReport:
    """What the merge did, for the step record and the review page."""

    common_antigens: int = 0
    common_sera: int = 0
    new_antigens: list[int] = field(default_factory=list)  # indexes in the merge
    new_sera: list[int] = field(default_factory=list)
    copied_references: bool = False
    skipped_reference_antigens: int = 0
    outcomes: Counter = field(default_factory=Counter)  # MergeOutcome -> cells
    # Cells a merge rule turned into `*` although they had readings: (antigen, serum, outcome).
    # Counted above too; listed so a missing titre can say why (sd_limit, `<` meeting `>`).
    dropped: list[tuple[int, int, str]] = field(default_factory=list)
    column_basis_slack: dict[int, float] = field(
        default_factory=dict
    )  # serum -> forced - table-only
    # merge_many only: each table's own point counts, in order (the fields above describe the
    # last step and the final table, as a fold's last report does)
    steps: list[MergeStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe, with stable keys: ints, floats, strings and lists only."""
        return {
            **MergeStep.of(self).to_dict(),
            "outcomes": {str(k): int(v) for k, v in sorted(self.outcomes.items())},
            "dropped": [[int(i), int(j), str(o)] for i, j, o in self.dropped],
            "column_basis_slack": {
                str(int(j)): float(v) for j, v in sorted(self.column_basis_slack.items())
            },
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class MergeStep:
    """What one table did to the points of a merge_many (its cells are merged at the end)."""

    common_antigens: int
    common_sera: int
    new_antigens: int
    new_sera: int
    copied_references: bool
    skipped_reference_antigens: int

    @classmethod
    def of(cls, report: MergeReport) -> MergeStep:
        return cls(
            report.common_antigens,
            report.common_sera,
            len(report.new_antigens),
            len(report.new_sera),
            report.copied_references,
            report.skipped_reference_antigens,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "common_antigens": int(self.common_antigens),
            "common_sera": int(self.common_sera),
            "new_antigens": int(self.new_antigens),
            "new_sera": int(self.new_sera),
            "copied_references": bool(self.copied_references),
            "skipped_reference_antigens": int(self.skipped_reference_antigens),
        }


# ----------------------------------------------------------------------
# identity


def _antigen_key(a: Antigen):
    return antigen_identity(a.name, a.reassortant, a.annotations, a.passage)


def _serum_key(s: Serum):
    return serum_identity(s.name, s.reassortant, s.annotations, s.serum_id)


def match_points(primary: list, secondary: list, key) -> dict[int, int]:
    """secondary index -> primary index for points with the same identity key.

    When a key occurs more than once, pairs are taken greedily in (primary, secondary)
    index order, each point used once (ae `common.cc` sort + mark_match_use).
    """
    by_key: dict[tuple, list[int]] = {}
    for i, p in enumerate(primary):
        if (k := key(p)) is not None:
            by_key.setdefault(k, []).append(i)
    pairs = sorted(
        (i, j)
        for j, s in enumerate(secondary)
        if (k := key(s)) is not None
        for i in by_key.get(k, ())
    )
    used_p: set[int] = set()
    result: dict[int, int] = {}
    for i, j in pairs:
        if i not in used_p and j not in result:
            used_p.add(i)
            result[j] = i
    return result


def reference_antigens(chart: Chart) -> list[int]:
    """Antigens whose name and annotations equal a serum's (ae `Chart::reference`)."""
    sera = {(s.name, s.annotations) for s in chart.sera}
    return [
        i
        for i, a in enumerate(chart.antigens)
        if not a.distinct and (a.name, a.annotations) in sera
    ]


def antigen_key(a: Antigen) -> tuple:
    """Identity within one chart (ae `designation_key`); DISTINCT antigens are never compared."""
    return (a.name, " ".join(a.annotations), a.reassortant, a.passage)


def serum_key(s: Serum) -> tuple:
    return (s.name, " ".join(s.annotations), s.reassortant, s.serum_id)


def find_duplicates(chart: Chart) -> list[str]:
    ag = Counter(antigen_key(a) for a in chart.antigens if not a.distinct)
    sr = Counter(serum_key(s) for s in chart.sera if not s.distinct)
    return [f"AG {k}" for k, n in ag.items() if n > 1] + [f"SR {k}" for k, n in sr.items() if n > 1]


# ----------------------------------------------------------------------
# layers


def chart_layers(chart: Chart) -> list[Layer]:
    """A chart's layers; a single table is one layer."""
    if len(chart.titres.layers) > 1:
        return chart.titres.layers
    return [
        {
            (i, j): t
            for i, row in enumerate(chart.titres.table)
            for j, t in enumerate(row)
            if not t.is_missing
        }
    ]


DROPPING = (MergeOutcome.SD_TOO_BIG, MergeOutcome.LESS_AND_MORE_THAN)


def merge_layers(
    layers: list[Layer],
    n_ag: int,
    n_sr: int,
    options: MergeOptions,
    report: MergeReport | None = None,
) -> tuple[list[list[Titre]], np.ndarray | None]:
    """Merged table and, when the convention forces them, the column bases.

    Returns (table, forced_column_bases or None).
    """
    cells: dict[tuple[int, int], list[Titre]] = {}
    for layer in layers:
        for key, t in layer.items():
            cells.setdefault(key, []).append(t)

    def build(more_than: MoreThanOnly, count: bool) -> list[list[Titre]]:
        table = empty_table(n_ag, n_sr)
        for (i, j), ts in cells.items():
            merged, outcome = merge_titres(ts, more_than, options.titres)
            table[i][j] = merged
            if count and report is not None:
                report.outcomes[outcome.value] += 1
                if outcome in DROPPING:
                    report.dropped.append((i, j, outcome.value))
        return table

    table = build(MoreThanOnly.TO_DONT_CARE, count=True)
    forced = None
    has_more_than = any(t.type == TitreType.MORE_THAN for ts in cells.values() for t in ts)
    if options.column_bases == ColumnBasisConvention.ADJUST_TO_NEXT and has_more_than:
        adjusted = Titres(build(MoreThanOnly.ADJUST_TO_NEXT, count=False))
        forced = np.array([column_basis(adjusted.serum_titres(j)) for j in range(n_sr)])
        if report is not None:
            plain = np.array([column_basis(Titres(table).serum_titres(j)) for j in range(n_sr)])
            report.column_basis_slack = {
                j: float(forced[j] - plain[j]) for j in range(n_sr) if forced[j] != plain[j]
            }
    return table, forced


# ----------------------------------------------------------------------
# merge


def merge(
    chart1: Chart, chart2: Chart, options: MergeOptions | None = None
) -> tuple[Chart, MergeReport]:
    """Merge chart2 (normally one new table) into chart1 (the running merge).

    The result shares chart1's layer dicts and every point's `extra` dict with the inputs
    (titres are immutable). Copying them at every step of a fold would cost time that grows
    with the square of the number of tables; `merge_many` builds its own and shares nothing.
    """
    options = options or MergeOptions()
    for chart, which in ((chart1, "primary"), (chart2, "secondary")):
        if dups := find_duplicates(chart):
            raise MergeError(f"{which} chart has duplicates: {dups[:5]}")
    report = MergeReport()
    antigens = [Antigen(**vars(a)) for a in chart1.antigens]
    sera = [Serum(**vars(s)) for s in chart1.sera]
    layers = list(chart_layers(chart1))  # chart1's indexes are unchanged in the merge
    layers += _join(chart1, chart2, antigens, sera, options, report, _shallow)

    table, forced = merge_layers(layers, len(antigens), len(sera), options, report)
    merged = Chart(
        info=_merge_info(chart1, chart2),
        antigens=antigens,
        sera=sera,
        titres=Titres(table=table, layers=layers),
        forced_column_bases=forced,
    )
    if dups := find_duplicates(merged):
        raise MergeError(f"merge has duplicates: {dups[:5]}")
    if options.merge_type == MergeType.INCREMENTAL and chart1.projections:
        merged.projections = [_incremental_projection(chart1, merged)]
    return merged, report


def merge_many(
    charts: Sequence[Chart], options: MergeOptions | None = None
) -> tuple[Chart, MergeReport]:
    """Merge charts in order: the same chart as folding `merge` over them, cells merged once.

    A fold re-merges every cell of the running merge at each step, so its cost grows with the
    square of the number of tables. Here each step only matches points and maps the new table's
    layers, exactly as `merge` does (`_join`); the layers are merged once at the end. Points,
    their order, copied references, the table, forced column bases and dropped cells are those
    of the fold (tests/chart/test_merge_many.py). The report is the fold's last report plus
    `steps`, each table's point counts. Nothing mutable is shared with the inputs.

    Simple merges only: an incremental merge carries chart1's layout, which a fold rebuilds
    at every step. One chart is returned as a copy, unmerged, with an empty report.
    """
    options = options or MergeOptions(merge_type=MergeType.SIMPLE)
    if options.merge_type != MergeType.SIMPLE:
        raise MergeError("merge_many makes simple merges only")
    if not charts:
        raise MergeError("merge_many needs at least one chart")
    for k, chart in enumerate(charts):
        if dups := find_duplicates(chart):
            raise MergeError(f"chart {k} has duplicates: {dups[:5]}")
    first = charts[0]
    if len(charts) == 1:
        return copy.deepcopy(first), MergeReport()
    antigens: list[Antigen] = [_own(a) for a in first.antigens]  # type: ignore[misc]
    sera: list[Serum] = [_own(s) for s in first.sera]  # type: ignore[misc]
    layers = [dict(layer) for layer in chart_layers(first)]
    ag_keys = {antigen_key(a): i for i, a in enumerate(antigens) if not a.distinct}
    sr_keys = {serum_key(s): i for i, s in enumerate(sera) if not s.distinct}
    running, info, steps = first, first.info, []
    for chart2 in charts[1:]:
        report = MergeReport()
        layers += _join(running, chart2, antigens, sera, options, report, _own)
        info = _merge_info(running, chart2)
        dups = _new_duplicates(antigens, report.new_antigens, ag_keys, antigen_key, "AG")
        dups += _new_duplicates(sera, report.new_sera, sr_keys, serum_key, "SR")
        if dups:
            raise MergeError(f"merge has duplicates: {dups[:5]}")
        steps.append(MergeStep.of(report))
        # the running merge as the next step reads it: its points and layers, never its table
        running = Chart(info=info, antigens=antigens, sera=sera, titres=Titres([], layers))
    table, forced = merge_layers(layers, len(antigens), len(sera), options, report)
    report.steps = steps
    merged = Chart(
        info=copy.deepcopy(info),
        antigens=antigens,
        sera=sera,
        titres=Titres(table=table, layers=layers),
        forced_column_bases=forced,
    )
    return merged, report


def _new_duplicates(
    points: list, new: list[int], keys: dict[tuple, int], key: Callable, kind: str
) -> list[str]:
    """The fold's duplicate check on the merged chart, for the points this step added: the
    points already merged were checked before, and matching never changes a point's key.
    Listed in the order the fold lists them (first occurrence in the merged chart)."""
    dups: dict[tuple, int] = {}
    for i in new:
        if points[i].distinct:
            continue
        k = key(points[i])
        if k in keys:
            dups.setdefault(k, keys[k])
        else:
            keys[k] = i
    return [f"{kind} {k}" for k in sorted(dups, key=dups.__getitem__)]


def _shallow(point: Antigen | Serum) -> Antigen | Serum:
    return type(point)(**vars(point))


def _own(point: Antigen | Serum) -> Antigen | Serum:
    return type(point)(**{**vars(point), "extra": copy.deepcopy(point.extra)})


def _join(
    chart1: Chart,
    chart2: Chart,
    antigens: list[Antigen],
    sera: list[Serum],
    options: MergeOptions,
    report: MergeReport,
    copy_point: Callable,
) -> list[Layer]:
    """Match chart2's points to chart1's and return chart2's layers in the merge's numbering.

    `antigens` and `sera` hold copies of chart1's points, in chart1's order: matched points are
    updated in place and chart2's new points appended (copied by `copy_point`), so all of
    chart1 keeps its indexes and the new points follow in chart2's order.
    """
    ag_match = match_points(chart1.antigens, chart2.antigens, _antigen_key)
    sr_match = match_points(chart1.sera, chart2.sera, _serum_key)

    secondary_antigens = list(range(chart2.n_antigens))
    if options.combine_copied_references:
        secondary_antigens = _copied_reference_test_antigens(
            chart1, chart2, ag_match, sr_match, report
        )

    ag2_target: dict[int, int] = {}
    for j in secondary_antigens:
        if j in ag_match:
            ag2_target[j] = ag_match[j]
            _update_antigen(antigens[ag_match[j]], chart2.antigens[j])
            report.common_antigens += 1
        else:
            ag2_target[j] = len(antigens)
            report.new_antigens.append(len(antigens))
            antigens.append(copy_point(chart2.antigens[j]))
    sr2_target: dict[int, int] = {}
    for j in range(chart2.n_sera):
        if j in sr_match:
            sr2_target[j] = sr_match[j]
            report.common_sera += 1
        else:
            sr2_target[j] = len(sera)
            report.new_sera.append(len(sera))
            sera.append(copy_point(chart2.sera[j]))

    return [
        {(ag2_target[i], sr2_target[j]): t for (i, j), t in layer2.items() if i in ag2_target}
        for layer2 in chart_layers(chart2)
    ]


def _update_antigen(target: Antigen, src: Antigen) -> None:
    """ae `Antigen::update_with`: fill a missing date, add lab ids not yet present."""
    if not target.date:
        target.date = src.date
    target.lab_ids = target.lab_ids + tuple(i for i in src.lab_ids if i not in target.lab_ids)


def _merge_info(chart1: Chart, chart2: Chart) -> dict:
    sources = list(chart1.info.get("S") or [chart1.info]) + list(
        chart2.info.get("S") or [chart2.info]
    )
    info = {k: v for k, v in chart1.info.items() if k in ("v", "V", "A", "l", "r")}
    info["S"] = [{k: v for k, v in s.items() if k != "S"} for s in sources]
    return info


def _incremental_projection(chart1: Chart, merged: Chart) -> Projection:
    """Copy chart1's best layout into the merge; points new to the merge are NaN.

    chart1's antigens and sera keep their indexes in the merge, so the copy is two slices
    (ae `merge.cc:310-328`).
    """
    best = chart1.best_projection()
    assert best is not None
    dim = best.dimensions
    layout = np.full((merged.n_points, dim), np.nan)
    layout[: chart1.n_antigens] = best.layout[: chart1.n_antigens]
    layout[merged.n_antigens : merged.n_antigens + chart1.n_sera] = best.layout[chart1.n_antigens :]
    disconnected = tuple(
        p if p < chart1.n_antigens else p - chart1.n_antigens + merged.n_antigens
        for p in best.disconnected
    )
    return Projection(
        layout=layout,
        minimum_column_basis=best.minimum_column_basis,
        transformation=None if best.transformation is None else best.transformation.copy(),
        disconnected=disconnected,
    )


def _copied_reference_test_antigens(
    chart1: Chart,
    chart2: Chart,
    ag_match: dict[int, int],
    sr_match: dict[int, int],
    report: MergeReport,
) -> list[int]:
    """If chart2 repeats chart1's reference block exactly, merge only its test antigens.

    A table with "copied references" repeats the reference antigens and sera with the same titres
    alongside new test viruses. Merging those references again would weight them twice,
    so ae keeps only the test antigens (`merge.cc:134-209`).
    """
    refs = reference_antigens(chart2)
    if not all(i in ag_match for i in refs) or not all(j in sr_match for j in range(chart2.n_sera)):
        return list(range(chart2.n_antigens))

    def same(layer_titre) -> bool:
        return all(
            chart2.titres.table[i][j] == layer_titre(ag_match[i], sr_match[j])
            for i in refs
            for j in range(chart2.n_sera)
        )

    if len(chart1.titres.layers) == 0:
        titres_same = same(lambda a, s: chart1.titres.table[a][s])
    else:
        titres_same = any(
            same(lambda a, s, L=layer: L.get((a, s), MISSING_TITRE))
            for layer in chart1.titres.layers
        )
    if not titres_same:
        return list(range(chart2.n_antigens))
    test = [i for i in range(chart2.n_antigens) if i not in set(refs)]
    if not test:
        raise MergeError(
            "copied references and chart has no test antigens: "
            "remove the table or disable combine_copied_references"
        )
    report.copied_references = True
    report.skipped_reference_antigens = len(refs)
    return test
