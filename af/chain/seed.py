"""A seed chart's layout as the chain's step 0: orientation, which projection, and the check.

A chain can start from a map made elsewhere (`first_map`), e.g. the map as shipped for a
report. Three things decide whether the chain continues the map that was drawn rather than a
different map that happens to share its titres.

**Orientation.** A chart stores a layout and, separately, a transformation (`t` in .ace) that
is applied when it is drawn: rotation and reflection chosen by hand, so the map reads the way
the people who use it expect. Nothing in a chain's steps carries it (`engine._projections`
builds each step's projections without one), and nothing realigns a stored layout either, so a
seed's transformation would be dropped at step 1 and every later step would sit in the seed's
raw frame while the seed itself was published in its drawn frame. Worse, a step whose chosen
map came from the scratch run is in a frame of its own: a transformation carried along from the
seed would then be applied to coordinates it was never measured for. So the transformation is
*baked* at step 0 — applied to the coordinates once, the stored layout becoming the drawn
layout and the transformation becoming the identity — and recorded in the step, so that the
conversion is declared and can be undone exactly. Baking is only sound where the
transformation preserves distances: a scaling would change every map distance and so the
stress. `bake_transformation` refuses a transformation that is not orthogonal rather than
quietly changing the map (all 18 maps of the Sep 2026 round are orthogonal to 1.4e-15).

**Which projection.** `Chart.best_projection` means "the lowest stress", which is what the rest
of the code wants. A chart made elsewhere keeps 1-17 projections, and the one a chain continues
is P[0], which is not always the lowest. Measured on the Sep 2026 round's 18 maps: in the file
the chains seed from (the chain output, before any hand moves) P[0] is the lowest in all 18, but
in the same maps as drawn it is not in 6 of them, by as much as 200 stress units. So the seed's
projection is named in the config (`[seed_chart] projection`, default 0) and passed explicitly
where the step-0 layout is taken, rather than deduced: which file seeds a chain is a ruling that
can change, and taking the lowest would silently chain from a map nobody has seen.

**The check.** Step 0 records the seed's own stored stress, which comes from the program that
made the seed. `seed_stress_check` recomputes it here, from the seed's titres under its own
stored column bases and its own disconnected points, and records both numbers with their
difference, refusing past a tolerance. That turns "step 0 reproduces the shipped map" from a
claim into a number. A seed may store no stress at all (2 of the 18 maps): then the computed
value is recorded alone, with no comparison, because there is nothing to disagree with.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from af.chart.model import Chart, Projection

# A relative tolerance, because the stresses in use span 470 to 33,780. The two numbers are the
# same sum of the same terms computed by two programs, so they differ only in summation order:
# well under 1e-12 relative in practice. Anything that matters -- a column basis read
# differently, a point paired wrongly, a titre dropped -- moves the stress by far more than
# 1e-6 of it, so this default never trips on arithmetic and never hides a real disagreement.
STRESS_TOLERANCE = 1e-6

# How nearly orthogonal a transformation must be to be baked: |t @ t.T - I| at most this.
# Measured on the Sep 2026 round the worst is 1.4e-15, so this allows ~700x that and still
# refuses any real scaling.
ORTHOGONAL_TOLERANCE = 1e-12


class SeedError(ValueError):
    pass


# ----------------------------------------------------------------------
# which projection


def seed_projection(chart: Chart, index: int) -> Projection:
    """The seed's projection by index: the layout the chain continues, named not deduced."""
    if not chart.projections:
        raise SeedError("seed chart has no projections")
    if not 0 <= index < len(chart.projections):
        raise SeedError(
            f"[seed_chart] projection {index} but the seed has {len(chart.projections)}"
            f" (0..{len(chart.projections) - 1})"
        )
    return chart.projections[index]


def projection_stresses(chart: Chart) -> list[float | None]:
    """Every projection's stored stress, in order: what the step records beside the chosen one."""
    return [None if p.stress is None else float(p.stress) for p in chart.projections]


def lowest_stress_index(chart: Chart) -> int | None:
    """Which projection has the lowest stored stress, or None if none stores one."""
    known = [(s, i) for i, s in enumerate(projection_stresses(chart)) if s is not None]
    return min(known)[1] if known else None


# ----------------------------------------------------------------------
# orientation


def _orthogonality_error(t: np.ndarray) -> float:
    return float(np.abs(t @ t.T - np.eye(t.shape[0])).max())


@dataclass(frozen=True)
class Baked:
    """A chart whose projections' transformations are in their coordinates, and the record."""

    chart: Chart
    transformations: list[dict[str, Any]]  # one per projection that had one

    def to_json(self) -> dict[str, Any]:
        return {"baked": self.transformations}


def bake_transformation(chart: Chart, tolerance: float = ORTHOGONAL_TOLERANCE) -> Baked:
    """Apply each projection's own transformation to its own coordinates; drop the matrix.

    The stored layout becomes the layout as drawn, so every later step, which starts from the
    previous step's stored layout, is in the drawn frame too and is directly comparable with the
    map that was published. Each matrix is kept in the record, so the conversion is declared and
    exactly invertible (the inverse of an orthogonal matrix is its transpose).

    Refuses a transformation that is not orthogonal within `tolerance`: that would be a scaling
    or a shear, which changes map distances and so the stress, and no record would make the
    resulting chart the same map.
    """
    projections = []
    record: list[dict[str, Any]] = []
    for i, p in enumerate(chart.projections):
        if p.transformation is None:
            projections.append(p)
            continue
        t = np.asarray(p.transformation, dtype=float)
        if t.ndim != 2 or t.shape[0] != t.shape[1]:
            raise SeedError(f"projection {i}: transformation is not a square matrix")
        if t.shape[0] != p.dimensions:
            raise SeedError(
                f"projection {i}: transformation is {t.shape[0]}-d, the layout {p.dimensions}-d"
            )
        err = _orthogonality_error(t)
        if err > tolerance:
            raise SeedError(
                f"projection {i}: transformation is not orthogonal (|t t' - I| = {err:.3g} >"
                f" {tolerance:.3g}), so baking it would change the map distances and the stress"
            )
        projections.append(
            dataclasses.replace(p, layout=p.transformed_layout(), transformation=None)
        )
        record.append(
            {
                "projection": i,
                "transformation": [[float(v) for v in row] for row in t],
                "determinant": float(np.linalg.det(t)),
                "orthogonality_error": err,
                # a reflection (negative determinant) is a flip, not a rotation: worth seeing
                "reflection": bool(np.linalg.det(t) < 0),
            }
        )
    return Baked(chart=dataclasses.replace(chart, projections=projections), transformations=record)


# ----------------------------------------------------------------------
# the step-0 stress check


@dataclass(frozen=True)
class SeedStress:
    """The seed's stored stress beside the stress recomputed here."""

    stored: float | None
    computed: float
    difference: float | None  # computed - stored
    relative: float | None  # |difference| / |stored|
    tolerance: float

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": self.stored,
            "computed": self.computed,
            "difference": self.difference,
            "relative": self.relative,
            "tolerance": self.tolerance,
        }


def seed_stress(chart: Chart, projection: Projection) -> float:
    """The stress of a seed's projection, computed here from the seed's own titres and bases.

    The seed's numbers, not the chain's: its projection's own minimum column basis, its forced
    bases where it has them, its own disconnected points (so no point is disconnected here that
    the seed kept), and its avidity adjustments. Nothing is optimised; the layout is the seed's.
    """
    from af.map.optimise import MapProblem, stress

    arrays = chart.optimiser_arrays(
        projection.minimum_column_basis or "none", projection, disconnect_threshold=None
    )
    problem = MapProblem(
        titre_value=arrays["titre_value"],
        titre_type=arrays["titre_type"],
        column_bases=np.asarray(arrays["column_bases"], dtype=float),
        disconnected=np.asarray(arrays["disconnected"], dtype=bool),
        dodgy_is_regular=bool(arrays["dodgy_is_regular"]),
        avidity_adjust=arrays.get("avidity_adjust"),
        unmovable=arrays.get("unmovable"),
    )
    return stress(problem, np.asarray(projection.layout, dtype=float))


def seed_stress_check(
    chart: Chart, projection: Projection, tolerance: float = STRESS_TOLERANCE
) -> SeedStress:
    """Recompute the seed's stress and compare it with the stored one; refuse past `tolerance`.

    A seed with no stored stress is not an error: the computed value is recorded with no
    comparison, because there is nothing to disagree with.
    """
    computed = seed_stress(chart, projection)
    stored = None if projection.stress is None else float(projection.stress)
    if stored is None:
        return SeedStress(None, computed, None, None, tolerance)
    difference = computed - stored
    relative = abs(difference) / abs(stored) if stored else abs(difference)
    if relative > tolerance:
        raise SeedError(
            f"step 0 does not reproduce the seed's stress: stored {stored!r}, computed"
            f" {computed!r} (relative difference {relative:.3g} > {tolerance:.3g}). The seed's"
            " titres, column bases or layout are not being read as the program that made it"
            " wrote them."
        )
    return SeedStress(stored, computed, difference, relative, tolerance)


# ----------------------------------------------------------------------
# layers


def layer_count(chart: Chart) -> int:
    """How many titre layers the chart keeps: one per source table where it has them."""
    return len(chart.titres.layers)


def check_layers(chart: Chart, tables: int) -> dict[str, Any]:
    """The seed's layers, and whether they can serve as the bridge.

    A chart with one layer or none is read as a single table (af.chart.merge), so its tables
    cannot be told apart and no store table can be paired with its own layer. A seed that is to
    be bridged needs one layer per source table.
    """
    layers = layer_count(chart)
    return {
        "layers": layers,
        "source_tables": tables,
        "per_table": layers == tables,
        # <= 1 layer reads as one table: the bridge has nothing to pair against
        "single_table": layers <= 1,
    }
