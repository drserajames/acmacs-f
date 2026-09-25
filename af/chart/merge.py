"""Merging charts: which antigens/sera are the same, and the layer merge.

This reproduces ae's `chart_v3.merge(match="strict", merge_type=...)` as the chains use
it (`ae/cc/chart/v3/merge.cc`, `common.cc`, `titers.cc`), because the chains must replay
today's science. Each rule names the ae source it copies.

Which points are the same across tables is `identity.py`.
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from af.chart.identity import antigen_identity, serum_identity
from af.chart.model import Antigen, Chart, Layer, Projection, Serum, Titres, empty_table
from af.chart.titre import (
    MISSING_TITRE,
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
    combine_cheating_assays: bool = False
    titres: MergeSettings = MergeSettings()
    column_bases: ColumnBasisConvention = ColumnBasisConvention.ADJUST_TO_NEXT


@dataclass
class MergeReport:
    """What the merge did, for the step record and the review page."""

    common_antigens: int = 0
    common_sera: int = 0
    new_antigens: list[int] = field(default_factory=list)  # indexes in the merge
    new_sera: list[int] = field(default_factory=list)
    cheating_assay: bool = False
    skipped_reference_antigens: int = 0
    outcomes: Counter = field(default_factory=Counter)  # MergeOutcome -> cells
    column_basis_slack: dict[int, float] = field(
        default_factory=dict
    )  # serum -> forced - table-only


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
    """Merge chart2 (normally one new table) into chart1 (the running merge)."""
    options = options or MergeOptions()
    for chart, which in ((chart1, "primary"), (chart2, "secondary")):
        if dups := find_duplicates(chart):
            raise MergeError(f"{which} chart has duplicates: {dups[:5]}")
    report = MergeReport()

    ag_match = match_points(chart1.antigens, chart2.antigens, _antigen_key)
    sr_match = match_points(chart1.sera, chart2.sera, _serum_key)

    secondary_antigens = list(range(chart2.n_antigens))
    if options.combine_cheating_assays:
        secondary_antigens = _cheating_assay_test_antigens(
            chart1, chart2, ag_match, sr_match, report
        )

    # target indexes: all of chart1, then chart2's new points in their order
    antigens = [Antigen(**vars(a)) for a in chart1.antigens]
    ag2_target: dict[int, int] = {}
    for j in secondary_antigens:
        if j in ag_match:
            ag2_target[j] = ag_match[j]
            _update_antigen(antigens[ag_match[j]], chart2.antigens[j])
            report.common_antigens += 1
        else:
            ag2_target[j] = len(antigens)
            report.new_antigens.append(len(antigens))
            antigens.append(Antigen(**vars(chart2.antigens[j])))
    sera = [Serum(**vars(s)) for s in chart1.sera]
    sr2_target: dict[int, int] = {}
    for j in range(chart2.n_sera):
        if j in sr_match:
            sr2_target[j] = sr_match[j]
            report.common_sera += 1
        else:
            sr2_target[j] = len(sera)
            report.new_sera.append(len(sera))
            sera.append(Serum(**vars(chart2.sera[j])))

    layers = list(chart_layers(chart1))  # chart1's indexes are unchanged in the merge
    for layer2 in chart_layers(chart2):
        layers.append(
            {(ag2_target[i], sr2_target[j]): t for (i, j), t in layer2.items() if i in ag2_target}
        )

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


def _cheating_assay_test_antigens(
    chart1: Chart,
    chart2: Chart,
    ag_match: dict[int, int],
    sr_match: dict[int, int],
    report: MergeReport,
) -> list[int]:
    """If chart2 repeats chart1's reference block exactly, merge only its test antigens.

    A "cheating assay" re-runs the reference antigens and sera with the same titres
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
            "cheating assay and chart has no test antigens: "
            "remove the table or disable combine_cheating_assays"
        )
    report.cheating_assay = True
    report.skipped_reference_antigens = len(refs)
    return test
