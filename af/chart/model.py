"""The chart model: antigens, sera, titre table and layers, column bases, projections.

A `Chart` is what the chains merge and what the optimiser maps. It is deliberately a
plain data holder: parsing and writing live in `ace.py`, merging in `merge.py`, and the
optimiser only ever sees the arrays from `Chart.optimiser_arrays()` (interface I1).

Every antigen/serum field the `.ace` format carries that af does not interpret is kept
in `extra`, so a chart read and written again loses nothing (kateri and Racmacs read
what we write).
"""

from __future__ import annotations

import math
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

        Order of precedence, as ae: the projection's forced bases, then the chart's forced
        bases (a merge with `>` titres sets these), then computed from the table. The
        minimum column basis applies to computed bases only.
        """
        if projection is not None and projection.forced_column_bases is not None:
            return projection.forced_column_bases.copy()
        if self.forced_column_bases is not None:
            return self.forced_column_bases.copy()
        return np.maximum(self.raw_column_bases(), minimum_column_basis_value(minimum_column_basis))

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


def empty_table(n_ag: int, n_sr: int) -> list[list[Titre]]:
    return [[MISSING_TITRE] * n_sr for _ in range(n_ag)]
