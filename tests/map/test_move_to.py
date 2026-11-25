"""move_to (Q354): named antigens to given positions, measured relative to the rest of the map.

Every chart here is synthetic, with invented names: antigens and sera on a plane, and titres made
from their distances, so the optimiser has a real problem to relax.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pytest

from af.chart.ace import content_hash, read_chart, write_chart
from af.chart.model import Antigen, Chart, Projection, Serum, Titre, Titres, empty_table
from af.map.build import (
    BuildError,
    apply_moves_to,
    check_moves_to,
    mover_designations,
    orientation_for,
    relaxer_for,
)
from af.map.config import Defaults, MapConfig, MapsConfig, MoveConfig, MoveToConfig
from af.map.curate import CurationError, MoveToOverride, apply_move_to
from af.map.orient import procrustes, rotation

COLUMN_BASE = 8.0  # log2(2560 / 10)
SERA = np.array([[4.0, 0.0], [0.0, 4.0], [-4.0, 0.0], [0.0, -4.0], [3.0, 3.0], [-3.0, -3.0]])
CORE = np.array(
    [[x, y] for x in (-2.0, -1.0, 0.0, 1.0, 2.0) for y in (-1.5, 0.0, 1.5)]
)  # 15 well-measured antigens
STRAY = np.array([[2.5, -2.5], [3.0, -2.0], [2.0, -2.0]])  # three movers, a block in one corner
NAMES = [f"A(H9N9)/EXAMPLETOWN/{i}/2020" for i in range(len(CORE))] + [
    f"A(H9N9)/EXAMPLESTRAY/{i}/2020" for i in range(len(STRAY))
]
MOVERS = tuple(n + " MDCK1" for n in NAMES[len(CORE) :])
ROUND = "2025-0101-example"


def _titre(d: float) -> Titre:
    return Titre.parse(str(max(10, round(10 * 2 ** (COLUMN_BASE - d)))))


def _chart(antigens: np.ndarray, layout: np.ndarray | None = None, **projection) -> Chart:
    """Titres from the distances of ``antigens`` to SERA; the projection holds ``layout``."""
    table = empty_table(len(antigens), len(SERA))
    for i, a in enumerate(antigens):
        for j, s in enumerate(SERA):
            table[i][j] = _titre(float(np.linalg.norm(a - s)))
    chart = Chart(
        info={"D": "20200115"},
        antigens=[Antigen(n, passage="MDCK1") for n in NAMES],
        sera=[
            Serum(f"A(H9N9)/EXAMPLESERUM/{j}/2019", serum_id=f"LAB F{j}/19")
            for j in range(len(SERA))
        ],
        titres=Titres(table, []),
        projections=[
            Projection(np.vstack([antigens, SERA]) if layout is None else layout, **projection)
        ],
    )
    return chart


TRUE = np.vstack([CORE, STRAY])


@pytest.fixture(scope="module")
def relaxed() -> tuple[Chart, np.ndarray]:
    """The synthetic chart at its own relaxed minimum (the start every move is measured from)."""
    chart = _chart(TRUE)
    relax, _ = relaxer_for(chart)
    start = chart.projections[0].layout
    layout, _ = relax(start, np.ones(len(start), dtype=bool))
    return _chart(TRUE, layout), layout


def rule(**changes: object) -> MoveToOverride:
    base: dict[str, object] = dict(
        name="strays to the core",
        reason="test",
        decided="2026-01-01",
        movers=MOVERS,
        max_stress_rise=1e9,
        max_from_target=0.5,
        relax=False,
    )
    base.update(changes)
    return MoveToOverride(**base)  # type: ignore[arg-type]


def _block_targets(layout: np.ndarray, degrees: float, shift: tuple[float, float]) -> np.ndarray:
    """The movers' block turned about its centroid and shifted: a rigid image of itself."""
    block = layout[len(CORE) : len(TRUE)]
    c = block.mean(axis=0)
    return (block - c) @ rotation(degrees) + c + np.asarray(shift)


def _move(chart: Chart, layout: np.ndarray, targets: np.ndarray, relax_fn=None, **changes):
    real_relax, stress = relaxer_for(chart)
    return apply_move_to(
        rule(**changes),
        layout,
        mover_designations(chart),
        targets,
        stress_before=stress(layout),
        stress=stress,
        relax=relax_fn or real_relax,
    )


def test_relax_off_places_the_movers_exactly_and_costs_the_arrangement(relaxed) -> None:
    chart, layout = relaxed
    targets = _block_targets(layout, 0.0, (-3.0, 2.0))
    result = _move(chart, layout, targets)
    _, stress = relaxer_for(chart)
    rows = list(result.rows)
    np.testing.assert_allclose(result.layout[rows], targets)
    start = layout.copy()
    start[rows] = targets
    assert result.stress_after == pytest.approx(stress(start), rel=1e-12)
    assert result.stress_after - result.stress_before > 0
    assert max(result.relative_from_target) == pytest.approx(0.0, abs=1e-9)
    assert result.rest_fit_rmsd == pytest.approx(0.0, abs=1e-9)
    assert result.rest_fit_det == pytest.approx(1.0)
    assert min(result.relative_moved) == pytest.approx(math.hypot(3.0, 2.0))


def test_a_rigid_block_moved_rigidly_relative_to_the_rest_is_a_real_move(relaxed) -> None:
    """The companion of the trap below: the same congruent targets, relax off. It is displaced
    relative to the rest of the map, so it passes, at the cost of the arrangement."""
    chart, layout = relaxed
    targets = _block_targets(layout, 30.0, (-2.5, 1.0))
    result = _move(chart, layout, targets)
    assert max(result.relative_from_target) == pytest.approx(0.0, abs=1e-9)
    assert result.stress_after > result.stress_before


@pytest.mark.parametrize("nudge, guard", [(0.3, "from_target_relative"), (0.0, "moved_relative")])
def test_the_rigid_block_trap_fails_however_cheap(relaxed, nudge: float, guard: str) -> None:
    """A relax that satisfies the targets by carrying the WHOLE map rigidly onto them, as a real
    optimiser did on a real map (+0.0065% stress, nothing changed). With a nudge the movers shift a
    little relative to the rest (the real case: 0.05-0.45 u) and from_target_relative catches it;
    without one, moved_relative does."""
    chart, layout = relaxed
    targets = _block_targets(layout, 30.0, (-2.5, 1.0))
    rows = slice(len(CORE), len(CORE) + len(STRAY))
    fit = procrustes(layout[rows], targets)  # the rigid motion taking the block onto its targets
    _, stress = relaxer_for(chart)

    def carry_the_map(start: np.ndarray, movable: np.ndarray) -> tuple[np.ndarray, float]:
        out = layout @ fit.matrix + fit.translation
        out[rows] += [nudge, 0.0]
        return out, stress(out)

    with pytest.raises(CurationError) as exc:
        _move(chart, layout, targets, relax_fn=carry_the_map, relax=True)
    assert exc.value.guard == guard
    # in absolute coordinates every mover is within the nudge of its target: the trap
    out, _ = carry_the_map(layout, np.ones(len(layout), dtype=bool))
    assert np.linalg.norm(out[rows] - targets, axis=1).max() == pytest.approx(nudge, abs=1e-9)


def test_a_relax_that_walks_the_movers_back_fails(relaxed) -> None:
    chart, layout = relaxed
    targets = _block_targets(layout, 0.0, (-3.0, 2.0))  # well measured: the titres pull them back
    with pytest.raises(CurationError, match="relative to the rest") as exc:
        _move(chart, layout, targets, relax=True)
    assert exc.value.guard in ("from_target_relative", "moved_relative")


def test_every_distance_is_the_same_in_a_rotated_reflected_translated_copy(relaxed) -> None:
    chart, layout = relaxed
    targets = _block_targets(layout, 20.0, (-2.0, 1.5))
    q, c = rotation(130.0, reflect=True), np.array([7.0, -3.0])
    a = _move(chart, layout, targets)
    copy = _chart(TRUE, layout @ q + c)
    b = _move(copy, layout @ q + c, targets @ q + c)
    np.testing.assert_allclose(a.relative_moved, b.relative_moved, atol=1e-9)
    np.testing.assert_allclose(a.relative_from_target, b.relative_from_target, atol=1e-9)
    np.testing.assert_allclose(a.moved, b.moved, atol=1e-9)
    assert a.largest_other_move == pytest.approx(b.largest_other_move, abs=1e-9)
    assert a.stress_after - a.stress_before == pytest.approx(
        b.stress_after - b.stress_before, rel=1e-9
    )


def test_a_mover_matching_none_or_two_antigens_raises(relaxed) -> None:
    chart, layout = relaxed
    targets = np.zeros((1, 2))
    with pytest.raises(CurationError, match="matches 0 antigens"):
        _move(chart, layout, targets, movers=("A(H9N9)/EXAMPLENOWHERE/1/2020 MDCK1",))
    names = mover_designations(chart)
    with pytest.raises(CurationError, match="matches 2 antigens"):
        apply_move_to(
            rule(movers=(names[0],)),
            layout,
            [names[0], *names[:-1]],  # the first designation twice
            targets,
            stress_before=0.0,
            stress=lambda x: 0.0,
            relax=lambda s, m: (s, 0.0),
        )


def test_the_stress_cap_fails_with_the_measured_numbers(relaxed) -> None:
    chart, layout = relaxed
    targets = _block_targets(layout, 0.0, (-3.0, 2.0))
    with pytest.raises(CurationError, match=r"stress rose \d+\.\d{3} \(cap 0\.001, relax off\)"):
        _move(chart, layout, targets, max_stress_rise=0.001)


# --- through the build: targets from a reference round, the final frame ----------------------

PREVIOUS = "2025-0101-previous"  # the round the build orients to
SHIPPED = "2025-0601-shipped"  # the round whose drawn map holds the positions to restore
Q, C = rotation(130.0, reflect=True), np.array([5.0, -2.0])  # how SHIPPED drew this map
Q_PREV, C_PREV = rotation(-40.0), np.array([-3.0, 1.0])  # how PREVIOUS drew it: another turn
DRAWN_STRAY = np.array([[-1.0, 3.5], [-0.5, 4.0], [-1.5, 4.2]])  # where SHIPPED drew the movers


def _round(
    tmp_path: Path, name: str, layout: np.ndarray, q: np.ndarray, c: np.ndarray, strays=None
) -> Path:
    """A round whose map shows the same points turned (q, c) and drawn y DOWN, the movers
    optionally drawn elsewhere. Its own projection carries a further transformation, as a
    styled.ace does."""
    drawn = layout @ q + c
    if strays is not None:
        drawn[len(CORE) : len(TRUE)] = strays
    y_down = drawn * [1.0, -1.0]
    t = rotation(25.0)
    folder = tmp_path / name / "h9-example"
    folder.mkdir(parents=True, exist_ok=True)
    write_chart(_chart(TRUE, y_down @ t.T, transformation=t), folder / "styled.ace")
    return tmp_path / name


def _configs(tmp_path: Path, layout: np.ndarray, **move) -> tuple[MapConfig, MapsConfig]:
    previous = _round(tmp_path, PREVIOUS, layout, Q_PREV, C_PREV)
    shipped = _round(tmp_path, SHIPPED, layout, Q, C, DRAWN_STRAY)
    base: dict[str, object] = dict(
        name="strays as shipped",
        reason="test",
        decided=dt.date(2026, 1, 1),
        movers=MOVERS,
        target="shipped",
        reference=None,
        reference_round=shipped,
        max_stress_rise=1e9,
        max_from_target=0.5,
        relax=False,
    )
    if move.get("target", "shipped") != "shipped":
        base.update(reference=PREVIOUS, reference_round=None)
    moves = move.pop("moves", ())
    base.update(move)
    cfg = MapConfig(
        folder="h9-example",
        clade_scheme="clades",
        layout_stand_in=tmp_path / "unused.ace",
        moves=moves,
        moves_to=(MoveToConfig(**base),),  # type: ignore[arg-type]
    )
    config = MapsConfig(
        defaults=Defaults(
            must_show_since=dt.date(2020, 1, 1),
            windows=(),
            min_common_points=3,
            previous_round=previous,
        ),
        frames=(),
        maps=(),
    )
    return cfg, config


def _build(chart: Chart, layout: np.ndarray, cfg: MapConfig, config: MapsConfig):
    relax, stress = relaxer_for(chart)
    moved, pending = apply_moves_to(chart, layout, cfg, config, relax, stress)
    ori = orientation_for(chart, moved, cfg, config, cfg.clade_scheme)
    reports = check_moves_to(pending, moved, ori, cfg.folder, config.defaults.min_common_points)
    return reports, moved, ori


def test_shipped_lands_on_the_reference_rounds_drawn_positions(tmp_path, relaxed) -> None:
    """The build orients to PREVIOUS; the positions come from SHIPPED, which drew the map turned,
    mirrored and y down. The movers land at SHIPPED's DRAWN positions, not at its raw layout and
    not at PREVIOUS's."""
    chart, layout = relaxed
    cfg, config = _configs(tmp_path, layout)
    (report,), moved, ori = _build(chart, layout, cfg, config)
    achieved = np.array([m["achieved"] for m in report["per_mover"]])
    np.testing.assert_allclose(achieved, DRAWN_STRAY, atol=1e-3)
    np.testing.assert_allclose(
        [m["requested"] for m in report["per_mover"]], DRAWN_STRAY, atol=1e-4
    )
    assert max(m["from_target_final"] for m in report["per_mover"]) < 1e-3
    # independently: the moved block, fitted onto SHIPPED's drawing over the non-movers
    shipped_drawn = layout @ Q + C
    rest = slice(0, len(CORE))
    fit = procrustes(moved[rest], shipped_drawn[rest])
    np.testing.assert_allclose(
        moved[len(CORE) : len(TRUE)] @ fit.matrix + fit.translation, DRAWN_STRAY, atol=1e-3
    )
    # SHIPPED's RAW layout puts the movers somewhere else entirely: not that
    styled = tmp_path / SHIPPED / "h9-example" / "styled.ace"
    raw = read_chart(styled).projections[0].layout[len(CORE) : len(TRUE)]
    assert np.linalg.norm(achieved - raw, axis=1).min() > 1.0
    # and the frame the map is drawn in (PREVIOUS's) is not where they were asked to be
    drawn_now = moved[len(CORE) : len(TRUE)] @ ori.matrix + ori.translation
    assert np.linalg.norm(drawn_now - DRAWN_STRAY, axis=1).min() > 1.0
    assert report["reference"]["round"] == SHIPPED
    assert report["reference"]["sha256"] == content_hash(styled)
    assert report["fit"]["placement_common"] == len(CORE) + len(SERA)  # movers excluded
    assert report["fit"]["final_common"] == len(CORE) + len(SERA)
    assert report["relax"] is False and report["frame"] == f"{SHIPPED} drawn"
    assert report["rest_alignment"]["det"] == pytest.approx(1.0)


def test_an_explicit_target_lands_there_in_the_drawn_frame(tmp_path, relaxed) -> None:
    """The mover is placed exactly relative to the rest. In the final drawn frame it is off by the
    pull it exerts on the map's own orientation fit, which keeps the movers in it (Q354 plan Q3):
    about its displacement over the number of points fitted, measured and capped, never hidden."""
    chart, layout = relaxed
    was = layout[len(CORE)] @ Q_PREV + C_PREV  # where the map is drawn now
    want = (float(was[0] + 0.8), float(was[1] + 0.6))  # one unit away
    cfg, config = _configs(tmp_path, layout, target=want, frame_rotations=(), movers=MOVERS[:1])
    (report,), moved, ori = _build(chart, layout, cfg, config)
    (m,) = report["per_mover"]
    assert m["from_target_relative"] == pytest.approx(0.0, abs=1e-4)
    drawn = moved[len(CORE)] @ ori.matrix + ori.translation
    np.testing.assert_allclose(m["achieved"], drawn, atol=1e-4)
    assert m["from_target_final"] == pytest.approx(math.dist(drawn, want), abs=1e-4)
    # the moved point also turns the fit a little (leverage), so allow a few times d / n
    assert 0 < m["from_target_final"] < 4 * 1.0 / report["fit"]["final_common"]
    assert report["frame"] == "this map drawn" and report["reference"] == PREVIOUS


@pytest.mark.parametrize(
    "change, message",
    [
        (dict(target=(0.0, 0.0), frame_rotations=(), reference="2024-0101-other"),
         "decided against 2024-0101-other"),
        (dict(target=(0.0, 0.0), frame_rotations=("turned",)), "re-read them in the current frame"),
        (dict(movers=("A(H9N9)/EXAMPLESERUM/0/2019",)), "is a serum"),
        (dict(movers=("A(H9N9)/EXAMPLENOWHERE/1/2020 MDCK1",)), "matches 0 antigens"),
        (dict(max_from_target=1e-6, relax=True), "relative to the rest"),
        (dict(reference_round=Path("/nonexistent/round")), "is not a directory"),
    ],
)  # fmt: skip
def test_the_build_refuses(tmp_path, relaxed, change: dict, message: str) -> None:
    chart, layout = relaxed
    cfg, config = _configs(tmp_path, layout, **change)
    with pytest.raises(BuildError, match=message):
        _build(chart, layout, cfg, config)


def test_a_reference_round_without_the_map_is_refused(tmp_path, relaxed) -> None:
    chart, layout = relaxed
    (tmp_path / "2025-0701-empty").mkdir()
    cfg, config = _configs(tmp_path, layout, reference_round=tmp_path / "2025-0701-empty")
    with pytest.raises(BuildError, match="has no h9-example/styled.ace"):
        _build(chart, layout, cfg, config)


def test_a_mover_the_reference_does_not_draw_is_refused(tmp_path, relaxed) -> None:
    chart, layout = relaxed
    fewer = _chart(TRUE[:-1])  # the reference round never had the last mover
    fewer = Chart(
        info=fewer.info,
        antigens=fewer.antigens[:-1],
        sera=fewer.sera,
        titres=Titres([row for row in fewer.titres.table[:-1]], []),
        projections=[Projection(np.vstack([TRUE[:-1], SERA]))],
    )
    (tmp_path / "2025-0801-fewer" / "h9-example").mkdir(parents=True)
    write_chart(fewer, tmp_path / "2025-0801-fewer" / "h9-example" / "styled.ace")
    cfg, config = _configs(tmp_path, layout, reference_round=tmp_path / "2025-0801-fewer")
    with pytest.raises(BuildError, match="is not drawn, once, on 2025-0801-fewer"):
        _build(chart, layout, cfg, config)


def test_a_point_in_both_moves_and_moves_to_is_refused(tmp_path, relaxed) -> None:
    chart, layout = relaxed
    also = MoveConfig("also", "test", dt.date(2026, 1, 1), MOVERS[:1], "K", 1.0, 1.0)
    cfg, config = _configs(tmp_path, layout, moves=(also,))
    with pytest.raises(BuildError, match="also moved by a moves rule"):
        _build(chart, layout, cfg, config)
