"""The chart model: antigens, sera, titre table and layers, column bases, projections.

A `Chart` is what the chains merge and what the optimiser maps. It is deliberately a
plain data holder: parsing and writing live in `ace.py`, merging in `merge.py`, and the
optimiser only ever sees the arrays from `Chart.optimiser_arrays()` (interface I1).

Every antigen/serum field the `.ace` format carries that af does not interpret is kept
in `extra`, so a chart read and written again loses nothing (kateri and Racmacs read
what we write).
"""

from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from af.chart.titre import MISSING_TITRE, Titre, TitreType, column_basis

# ----------------------------------------------------------------------
# antigens and sera


@dataclass
class Antigen:
    name: str
    passage: str = ""
    reassortant: str = ""
    annotations: tuple[str, ...] = ()
    date: str = ""
    lab_ids: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def distinct(self) -> bool:
        return "DISTINCT" in self.annotations

    def designation(self) -> str:
        """Name, reassortant, annotations and passage: the point as a person reads it."""
        return " ".join(
            p for p in (self.name, self.reassortant, *self.annotations, self.passage) if p
        )


@dataclass
class Serum:
    name: str
    serum_id: str = ""
    passage: str = ""
    reassortant: str = ""
    annotations: tuple[str, ...] = ()
    species: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def distinct(self) -> bool:
        return "DISTINCT" in self.annotations

    def designation(self) -> str:
        return " ".join(
            p for p in (self.name, self.reassortant, *self.annotations, self.serum_id) if p
        )


# ----------------------------------------------------------------------
# titres

Layer = dict[tuple[int, int], Titre]  # sparse: (antigen, serum) -> titre, missing omitted


@dataclass
class Titres:
    """The merged table plus, for a merged chart, the per-source layers it came from.

    `table` is dense (n_ag x n_sr, MISSING where absent). `layers` is empty for a single
    table; for a merge there is one sparse layer per source table.
    """

    table: list[list[Titre]]
    layers: list[Layer] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        n_ag = len(self.table)
        return n_ag, (len(self.table[0]) if n_ag else 0)

    def serum_titres(self, sr: int) -> list[Titre]:
        return [row[sr] for row in self.table]

    def regular_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """Number of REGULAR titres per antigen and per serum (the disconnection rule)."""
        n_ag, n_sr = self.shape
        ag = np.zeros(n_ag, dtype=int)
        sr = np.zeros(n_sr, dtype=int)
        for i, row in enumerate(self.table):
            for j, t in enumerate(row):
                if t.type == TitreType.REGULAR:
                    ag[i] += 1
                    sr[j] += 1
        return ag, sr


# ----------------------------------------------------------------------
# projections


@dataclass
class Projection:
    """One map. `layout` is float [n_points, dim] with NaN rows for disconnected points."""

    layout: np.ndarray
    stress: float | None = None
    minimum_column_basis: str = "none"
    forced_column_bases: np.ndarray | None = None
    transformation: np.ndarray | None = None  # 2x2 (row-major "t" in .ace)
    disconnected: tuple[int, ...] = ()
    unmovable: tuple[int, ...] = ()
    dodgy_is_regular: bool = False
    avidity_adjusts: np.ndarray | None = None
    gradient_multipliers: np.ndarray | None = None
    comment: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def stress_value(self) -> float:
        """The stress, for a projection that must have one (a map the optimiser made)."""
        if self.stress is None:
            raise ValueError("projection has no stress")
        return float(self.stress)

    @property
    def dimensions(self) -> int:
        return int(self.layout.shape[1]) if self.layout.ndim == 2 else 0

    def transformed_layout(self) -> np.ndarray:
        if self.transformation is None:
            return self.layout.copy()
        return self.layout @ self.transformation


# ----------------------------------------------------------------------
# the chart


def minimum_column_basis_value(mcb: str) -> float:
    """ "none" -> 0; "1280" -> log2(1280/10) = 7."""
    if mcb in ("none", ""):
        return 0.0
    return math.log2(float(mcb) / 10.0)


@dataclass
class Chart:
    info: dict[str, Any]
    antigens: list[Antigen]
    sera: list[Serum]
    titres: Titres
    forced_column_bases: np.ndarray | None = None
    projections: list[Projection] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)  # "p", "R", ... kept verbatim

    @property
    def n_antigens(self) -> int:
        return len(self.antigens)

    @property
    def n_sera(self) -> int:
        return len(self.sera)

    @property
    def n_points(self) -> int:
        return self.n_antigens + self.n_sera

    # --- column bases ---

    def raw_column_bases(self) -> np.ndarray:
        return np.array([column_basis(self.titres.serum_titres(j)) for j in range(self.n_sera)])

    def column_bases(
        self, minimum_column_basis: str = "none", projection: Projection | None = None
    ) -> np.ndarray:
        """Column bases a map of this chart uses.

        Order of precedence, as ae: the projection's forced bases, used as they are; then the
        chart's forced bases (a merge with `>` titres sets these) or, without them, the bases
        computed from the table. The minimum column basis applies to both of the latter, each
        basis = max(basis, minimum), as ae's Chart::column_bases does (mcb.apply on the forced
        and the computed basis alike, cc/chart/v3/chart.cc). A projection's own forced bases
        are not raised: ae's stress takes them verbatim (cc/chart/v3/stress.cc).
        """
        if projection is not None and projection.forced_column_bases is not None:
            return projection.forced_column_bases.copy()
        mcb = minimum_column_basis_value(minimum_column_basis)
        if self.forced_column_bases is not None:
            return np.maximum(np.asarray(self.forced_column_bases, dtype=float), mcb)
        return np.maximum(self.raw_column_bases(), mcb)

    # --- disconnection ---

    def disconnected(self, threshold: int = 3) -> np.ndarray:
        """Points with fewer than `threshold` REGULAR titres (ae `titers.cc:448-464`)."""
        ag, sr = self.titres.regular_counts()
        return np.concatenate([ag < threshold, sr < threshold])

    # --- interface I1 ---

    def optimiser_arrays(
        self,
        minimum_column_basis: str = "none",
        projection: Projection | None = None,
        disconnect_threshold: int | None = 3,
    ) -> dict[str, Any]:
        """The arrays interface I1 hands to the optimiser core (COORDINATION.md)."""
        n_ag, n_sr = self.titres.shape
        value = np.full((n_ag, n_sr), np.nan)
        kind = np.zeros((n_ag, n_sr), dtype=np.int8)
        for i, row in enumerate(self.titres.table):
            for j, t in enumerate(row):
                if not t.is_missing:
                    value[i, j] = t.logged()
                    kind[i, j] = int(t.type)
        if disconnect_threshold is None:
            disconnected = np.zeros(self.n_points, dtype=bool)
        else:
            disconnected = self.disconnected(disconnect_threshold)
        if projection is not None and projection.disconnected:
            disconnected[list(projection.disconnected)] = True
        arrays: dict[str, Any] = {
            "titre_value": value,
            "titre_type": kind,
            "column_bases": self.column_bases(minimum_column_basis, projection),
            "disconnected": disconnected,
            "dodgy_is_regular": bool(projection.dodgy_is_regular) if projection else False,
        }
        if projection is not None:
            if projection.avidity_adjusts is not None:
                arrays["avidity_adjust"] = projection.avidity_adjusts
            if projection.unmovable:
                unmovable = np.zeros(self.n_points, dtype=bool)
                unmovable[list(projection.unmovable)] = True
                arrays["unmovable"] = unmovable
        return arrays

    # --- convenience ---

    def best_projection(self) -> Projection | None:
        with_stress = [p for p in self.projections if p.stress is not None]
        if not with_stress:
            return self.projections[0] if self.projections else None
        return min(with_stress, key=lambda p: p.stress)  # type: ignore[arg-type,return-value]

    def best(self) -> Projection:
        """The lowest-stress projection; a chart without one is an error here."""
        best = self.best_projection()
        if best is None:
            raise ValueError("chart has no projections")
        return best

    def source_dates(self) -> list[str]:
        sources = self.info.get("S") or []
        if sources:
            return [s.get("D", "") for s in sources]
        return [self.info.get("D", "")]

    def date_range(self) -> str:
        dates = [d for d in self.source_dates() if d]
        if not dates:
            return ""
        return dates[0] if len(dates) == 1 else f"{min(dates)}-{max(dates)}"

    # --- selecting points ---

    def select(self, keep_ag: Sequence[int], keep_sr: Sequence[int]) -> Chart:
        """A new chart with only the antigens `keep_ag` and sera `keep_sr` (indices into this
        chart, in their original order), its maps kept.

        Everything indexed by point follows its point, so no coordinate, setting or style
        ends up on another point: the titre table and layers; each projection's layout rows,
        forced column bases (per serum), avidity adjusts and gradient multipliers (per point),
        and its disconnected / unmovable / unmovable-in-last-dimension ("u") index lists,
        renumbered with removed points dropped; sera's homologous antigens ("h"); the plot
        spec "p" (per-point style index, drawing order, points shown on all maps), REMAPPED
        rather than dropped so each kept point keeps its style; and the antigen/serum index
        selectors ("!i") of the semantic plot specs "R". Each projection keeps its
        transformation, but its stress is set to None: the stored value was for the old point
        set and is stale until the map is relaxed or the stress recomputed. None means unknown,
        not zero: it is not a perfect fit.

        Indices, not names: callers resolve their selection first (design rule 2 is about what
        a user writes, not this call). Reordering is not supported: indices must be unique,
        in range and are kept in their original order. A selector "!i" that names a removed
        point, or one whose antigen/serum switch "A" is missing, is an error rather than a guess.
        """
        ag = _keep_indices(keep_ag, self.n_antigens, "antigen")
        sr = _keep_indices(keep_sr, self.n_sera, "serum")
        new_ag = {old: new for new, old in enumerate(ag)}
        new_sr = {old: new for new, old in enumerate(sr)}
        points = ag + [self.n_antigens + j for j in sr]
        new_point = {old: new for new, old in enumerate(points)}
        titres = Titres(
            [[self.titres.table[i][j] for j in sr] for i in ag],
            [
                {
                    (new_ag[i], new_sr[j]): t
                    for (i, j), t in layer.items()
                    if i in new_ag and j in new_sr
                }
                for layer in self.titres.layers
            ],
        )
        return Chart(
            copy.deepcopy(self.info),
            [copy.deepcopy(self.antigens[i]) for i in ag],
            [_select_serum(self.sera[j], new_ag) for j in sr],
            titres,
            None
            if self.forced_column_bases is None
            else np.asarray(self.forced_column_bases)[sr].copy(),
            [_select_projection(p, points, sr, new_point) for p in self.projections],
            _select_extra(self.extra, points, new_point, new_ag, new_sr),
        )


def _keep_indices(keep: Sequence[int], n: int, what: str) -> list[int]:
    out = [int(i) for i in keep]
    if len(set(out)) != len(out):
        raise ValueError(f"select: repeated {what} index")
    if any(i < 0 or i >= n for i in out):
        raise ValueError(f"select: {what} index out of range 0..{n - 1}")
    if out != sorted(out):
        raise ValueError(f"select: {what} indices must be in their original order (no reordering)")
    return out


def _renumber(indices: Any, new: dict[int, int]) -> list[int]:
    """Indices into the old numbering -> the kept ones in the new numbering."""
    return [new[int(i)] for i in indices if int(i) in new]


def _select_serum(serum: Serum, new_ag: dict[int, int]) -> Serum:
    s = copy.deepcopy(serum)
    if "h" in s.extra:  # homologous antigens: antigen indices
        s.extra["h"] = _renumber(s.extra["h"], new_ag)
    return s


def _per_point(a: np.ndarray | None, rows: list[int]) -> np.ndarray | None:
    return None if a is None else np.asarray(a)[rows].copy()


def _select_projection(
    p: Projection, points: list[int], sr: list[int], new_point: dict[int, int]
) -> Projection:
    extra = copy.deepcopy(p.extra)
    if "u" in extra:  # unmovable in the last dimension: point indices
        extra["u"] = _renumber(extra["u"], new_point)
    return dataclasses.replace(
        p,
        layout=np.asarray(p.layout)[points].copy(),
        stress=None,  # stale: computed for the old point set
        forced_column_bases=_per_point(p.forced_column_bases, sr),
        transformation=None if p.transformation is None else np.array(p.transformation, copy=True),
        disconnected=tuple(_renumber(p.disconnected, new_point)),
        unmovable=tuple(_renumber(p.unmovable, new_point)),
        avidity_adjusts=_per_point(p.avidity_adjusts, points),
        gradient_multipliers=_per_point(p.gradient_multipliers, points),
        extra=extra,
    )


def _select_extra(
    extra: dict[str, Any],
    points: list[int],
    new_point: dict[int, int],
    new_ag: dict[int, int],
    new_sr: dict[int, int],
) -> dict[str, Any]:
    out = copy.deepcopy(extra)
    plot = out.get("p")
    if isinstance(plot, dict):  # legacy plot spec, point-indexed lists
        if "p" in plot:  # style index for each point, antigens then sera
            plot["p"] = [plot["p"][i] for i in points]
        for key in ("d", "s"):  # drawing order; points shown on all maps
            if key in plot:
                plot[key] = _renumber(plot[key], new_point)
    if "R" in out:
        _renumber_selectors(out["R"], new_ag, new_sr)
    return out


def _renumber_selectors(node: Any, new_ag: dict[int, int], new_sr: dict[int, int]) -> None:
    """Semantic plot specs select single points with {"T": {"!i": index}, "A": 1|0}: an index
    among antigens (A = 1) or sera (A = 0). Renumber in place; refuse what cannot be kept."""
    if isinstance(node, list):
        for x in node:
            _renumber_selectors(x, new_ag, new_sr)
        return
    if not isinstance(node, dict):
        return
    t = node.get("T")
    if isinstance(t, dict) and "!i" in t:
        kind = node.get("A")
        if kind not in (0, 1, True, False):
            raise ValueError(
                "select: a '!i' selector without A = antigens or sera cannot be renumbered"
            )
        new, what = (new_ag, "antigen") if kind in (1, True) else (new_sr, "serum")
        old = int(t["!i"])
        if old not in new:
            raise ValueError(f"select: a semantic plot spec selects removed {what} {old}")
        t["!i"] = new[old]
    for value in node.values():
        _renumber_selectors(value, new_ag, new_sr)


def empty_table(n_ag: int, n_sr: int) -> list[list[Titre]]:
    return [[MISSING_TITRE] * n_sr for _ in range(n_ag)]
