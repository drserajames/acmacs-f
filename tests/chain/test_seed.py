"""A seed chart's layout as step 0 (af.chain.seed): orientation, which projection, the check.

Invented names only (acmacs-f is public). A seed made by another program keeps a transformation
it is drawn through, and several projections of which the drawn one is not always the lowest
stress; neither survives a chain's steps on its own.
"""

import dataclasses
import json
import math

import numpy as np
import pytest

from af.chain.backend import StubOptimiser
from af.chain.config import ChainConfig, ChainConfigError, MapOptions, SeedChart
from af.chain.engine import CHOSEN, run_chain
from af.chain.seed import (
    ORTHOGONAL_TOLERANCE,
    STRESS_TOLERANCE,
    SeedError,
    bake_transformation,
    check_layers,
    layer_count,
    lowest_stress_index,
    projection_stresses,
    seed_projection,
    seed_stress,
    seed_stress_check,
)
from af.chain.synthetic import make_tables
from af.chart.ace import read_chart, write_chart
from af.chart.merge import MergeError, MergeOptions, MergeType, merge
from af.chart.model import Chart, Projection, Titres, empty_table
from af.chart.titre import Titre

from .sera_markers import select_block

GROUP = "h9-hi-test-lab"
OPTS = MapOptions(
    scratch_starts=2, incremental_starts=2, projections_to_keep=2, resolve_best=1, dimensions=2
)

ROTATE_90 = np.array([[0.0, 1.0], [-1.0, 0.0]])  # orthogonal: a quarter turn
FLIP_X = np.array([[-1.0, 0.0], [0.0, 1.0]])  # orthogonal, a reflection (determinant -1)
SCALE_2 = np.array([[2.0, 0.0], [0.0, 2.0]])  # NOT orthogonal: changes every distance


def chart_with(projections, n_ag=3, n_sr=2):
    """A small chart whose titres are real enough for a stress, with the given projections."""
    table = empty_table(n_ag, n_sr)
    for i in range(n_ag):
        for j in range(n_sr):
            table[i][j] = Titre.parse(str(40 * (i + 1) * (j + 1)))
    from af.chart.model import Antigen, Serum

    return Chart(
        {"D": "2021-01-15"},
        [Antigen(name=f"TEST-{i + 1}", passage="MDCK1") for i in range(n_ag)],
        [Serum(name=f"TEST-{j + 1}", serum_id=f"F00{j}", passage="E3") for j in range(n_sr)],
        Titres(table),
        projections=list(projections),
    )


def layout(n, seed=4):
    return np.random.default_rng(seed).normal(size=(n, 2))


# ----------------------------------------------------------------------
# which projection is the seed


def test_seed_projection_is_the_named_one_not_the_lowest_stress():
    a, b = layout(5, 1), layout(5, 2)
    chart = chart_with([Projection(a, stress=99.0), Projection(b, stress=1.0)])
    assert seed_projection(chart, 0).layout is a  # P[0], although P[1] has the lower stress
    assert seed_projection(chart, 1).layout is b
    assert lowest_stress_index(chart) == 1
    assert projection_stresses(chart) == [99.0, 1.0]


def test_seed_projection_out_of_range_is_refused():
    chart = chart_with([Projection(layout(5), stress=1.0)])
    with pytest.raises(SeedError, match="projection 3 but the seed has 1"):
        seed_projection(chart, 3)


def test_seed_projection_needs_a_projection():
    with pytest.raises(SeedError, match="no projections"):
        seed_projection(chart_with([]), 0)


def test_lowest_stress_index_with_no_stored_stress():
    chart = chart_with([Projection(layout(5)), Projection(layout(5, 9))])
    assert lowest_stress_index(chart) is None
    assert projection_stresses(chart) == [None, None]


# ----------------------------------------------------------------------
# orientation


def test_bake_puts_the_transformation_in_the_coordinates():
    raw = layout(5)
    chart = chart_with([Projection(raw, stress=3.0, transformation=ROTATE_90)])
    baked = bake_transformation(chart)
    p = baked.chart.projections[0]
    assert p.transformation is None
    assert p.layout.tobytes() == (raw @ ROTATE_90).tobytes()  # exactly as drawn
    assert p.stress == 3.0  # a turn moves no point relative to another
    (rec,) = baked.transformations
    assert rec["projection"] == 0
    assert rec["transformation"] == [[0.0, 1.0], [-1.0, 0.0]]
    assert rec["determinant"] == pytest.approx(1.0)
    assert rec["reflection"] is False
    assert rec["orthogonality_error"] < ORTHOGONAL_TOLERANCE


def test_bake_records_a_reflection_as_one():
    chart = chart_with([Projection(layout(5), stress=3.0, transformation=FLIP_X)])
    (rec,) = bake_transformation(chart).transformations
    assert rec["determinant"] == pytest.approx(-1.0)
    assert rec["reflection"] is True


def test_bake_leaves_a_projection_without_a_transformation_alone():
    raw = layout(5)
    chart = chart_with([Projection(raw, stress=3.0), Projection(raw, transformation=ROTATE_90)])
    baked = bake_transformation(chart)
    assert baked.chart.projections[0].layout is raw  # untouched
    assert [r["projection"] for r in baked.transformations] == [1]


def test_bake_is_exactly_invertible_through_the_record():
    raw = layout(6)
    chart = chart_with([Projection(raw, stress=3.0, transformation=ROTATE_90)])
    baked = bake_transformation(chart)
    t = np.asarray(baked.transformations[0]["transformation"], dtype=float)
    back = baked.chart.projections[0].layout @ t.T  # orthogonal: the inverse is the transpose
    assert np.allclose(back, raw, rtol=0, atol=1e-15)


def test_bake_refuses_a_transformation_that_changes_distances():
    chart = chart_with([Projection(layout(5), stress=3.0, transformation=SCALE_2)])
    with pytest.raises(SeedError, match="not orthogonal"):
        bake_transformation(chart)


def test_bake_refuses_a_transformation_of_the_wrong_shape():
    chart = chart_with([Projection(layout(5), stress=3.0, transformation=np.ones((2, 3)))])
    with pytest.raises(SeedError, match="not a square matrix"):
        bake_transformation(chart)
    chart = chart_with([Projection(layout(5), stress=3.0, transformation=np.eye(3))])
    with pytest.raises(SeedError, match="3-d, the layout 2-d"):
        bake_transformation(chart)


def test_baking_does_not_move_the_stress():
    """A rotation or a flip moves no point relative to another, so the stress is the same."""
    chart = chart_with([Projection(layout(5), transformation=FLIP_X @ ROTATE_90)])
    before = seed_stress(chart, chart.projections[0])
    baked = bake_transformation(chart)
    after = seed_stress(baked.chart, baked.chart.projections[0])
    assert after == pytest.approx(before, rel=1e-12)


# ----------------------------------------------------------------------
# the step-0 stress check


def test_stress_check_passes_and_records_both_numbers():
    chart = chart_with([Projection(layout(5))])
    computed = seed_stress(chart, chart.projections[0])
    chart.projections[0].stress = computed
    checked = seed_stress_check(chart, chart.projections[0])
    assert checked.stored == computed
    assert checked.computed == pytest.approx(computed, rel=1e-15)
    assert checked.relative is not None and checked.relative <= STRESS_TOLERANCE
    assert checked.to_json()["tolerance"] == STRESS_TOLERANCE


def test_stress_check_refuses_a_stress_that_does_not_reproduce():
    chart = chart_with([Projection(layout(5), stress=1.0)])
    with pytest.raises(SeedError, match="does not reproduce the seed's stress"):
        seed_stress_check(chart, chart.projections[0])


def test_stress_check_without_a_stored_stress_records_the_computed_one():
    chart = chart_with([Projection(layout(5))])
    checked = seed_stress_check(chart, chart.projections[0])
    assert checked.stored is None and checked.difference is None and checked.relative is None
    assert checked.computed > 0
    assert math.isfinite(checked.computed)


def test_stress_check_tolerance_is_relative_and_configurable():
    chart = chart_with([Projection(layout(5))])
    computed = seed_stress(chart, chart.projections[0])
    chart.projections[0].stress = computed * (1 + 1e-4)
    with pytest.raises(SeedError):
        seed_stress_check(chart, chart.projections[0], 1e-6)
    checked = seed_stress_check(chart, chart.projections[0], 1e-3)
    assert checked.relative == pytest.approx(1e-4, rel=1e-3)


# ----------------------------------------------------------------------
# layers


def test_layer_count_and_check(tmp_path):
    paths = make_tables(tmp_path / "t", GROUP, n_tables=3)
    chart = read_chart(paths[0])
    assert layer_count(chart) == 0
    assert check_layers(chart, 1) == {
        "layers": 0,
        "source_tables": 1,
        "per_table": False,
        "single_table": True,
    }


def test_check_layers_one_per_table(tmp_path):
    paths = make_tables(tmp_path / "t", GROUP, n_tables=3)
    from af.chain.config import TableRef, tables_from_directory

    refs = tables_from_directory(tmp_path / "t", GROUP)
    assert isinstance(refs[0], TableRef)
    results = run_chain(ChainConfig(GROUP, refs, OPTS, seed=3), tmp_path / "c", StubOptimiser())
    made = read_chart(results[-1].directory / CHOSEN)
    assert layer_count(made) == len(paths)
    assert check_layers(made, len(paths)) == {
        "layers": 3,
        "source_tables": 3,
        "per_table": True,
        "single_table": False,
    }


# ----------------------------------------------------------------------
# config


def test_seed_chart_defaults():
    sc = SeedChart(bridge=True)
    assert sc.projection == 0
    assert sc.bake_transformation is True
    assert sc.stress_tolerance == STRESS_TOLERANCE


def test_seed_chart_refuses_a_negative_projection_or_tolerance():
    with pytest.raises(ChainConfigError, match="projection must be 0 or more"):
        SeedChart(bridge=True, projection=-1)
    with pytest.raises(ChainConfigError, match="stress_tolerance must be above 0"):
        SeedChart(bridge=True, stress_tolerance=0.0)


def test_seed_chart_layout_settings_apply_without_a_bridge():
    """They are about the seed's own layout, so they do not need the identity bridge."""
    sc = SeedChart(bridge=False, projection=2, bake_transformation=False, stress_tolerance=1e-3)
    assert (sc.projection, sc.bake_transformation, sc.stress_tolerance) == (2, False, 1e-3)


def test_seed_chart_settings_are_read_from_the_chain_file(tmp_path):
    from af.chain.config import load_chain_config

    make_tables(tmp_path / "t", GROUP, n_tables=2)
    write_chart(read_chart(sorted((tmp_path / "t").glob("*.ace"))[0]), tmp_path / "seed.ace")
    (tmp_path / "c.toml").write_text(
        f'name = "{GROUP}"\nseed = 3\nfirst_map = "seed.ace"\n'
        f'[tables]\ndirectory = "t"\ngroup = "{GROUP}"\n'
        '[options]\nminimum_column_basis = "none"\n'
        "[seed_chart]\nbridge = false\nprojection = 1\nbake_transformation = false\n"
        "stress_tolerance = 0.001\n" + select_block(tmp_path)
    )
    cfg = load_chain_config(tmp_path / "c.toml")
    assert cfg.seed_chart is not None
    assert cfg.seed_chart.projection == 1
    assert cfg.seed_chart.bake_transformation is False
    assert cfg.seed_chart.stress_tolerance == 0.001
    assert json.loads(json.dumps(cfg.seed_chart.to_json()))["projection"] == 1


# ----------------------------------------------------------------------
# the merge takes the named projection, not the best


def merge_two(tmp_path, projection):
    paths = make_tables(tmp_path / "t", GROUP, n_tables=2)
    chart1, chart2 = read_chart(paths[0]), read_chart(paths[1])
    high, low = layout(chart1.n_points, 1), layout(chart1.n_points, 2)
    chart1.projections = [Projection(high, stress=99.0), Projection(low, stress=1.0)]
    options = MergeOptions(merge_type=MergeType.INCREMENTAL, chart1_projection=projection)
    merged, _ = merge(chart1, chart2, options)
    return merged, high, low, chart1


def test_merge_copies_the_named_projection(tmp_path):
    merged, high, low, c1 = merge_two(tmp_path, 0)
    start = merged.projections[0].layout[: c1.n_antigens]
    assert start.tobytes() == high[: c1.n_antigens].tobytes()


def test_merge_without_a_named_projection_takes_the_best(tmp_path):
    merged, high, low, c1 = merge_two(tmp_path, None)
    start = merged.projections[0].layout[: c1.n_antigens]
    assert start.tobytes() == low[: c1.n_antigens].tobytes()


def test_merge_refuses_a_projection_it_does_not_have(tmp_path):
    with pytest.raises(MergeError, match="chart1_projection 5"):
        merge_two(tmp_path, 5)


# ----------------------------------------------------------------------
# end to end: a seed with a transformation and alternative projections


@pytest.fixture
def seed_with_orientation(tmp_path):
    """A chain's own map turned into a seed that carries a quarter turn and a lower-stress
    alternative projection, as a map made by another program does."""
    from af.chain.config import tables_from_directory

    make_tables(tmp_path / "t", GROUP, n_tables=4)
    refs = tables_from_directory(tmp_path / "t", GROUP)
    run_chain(ChainConfig(GROUP, refs[:2], OPTS, seed=3), tmp_path / "pre", StubOptimiser())
    made = read_chart(tmp_path / "pre" / GROUP / "steps" / "0001" / CHOSEN)
    drawn = dataclasses.replace(
        made.projections[0], transformation=ROTATE_90, stress=made.projections[0].stress
    )
    alternative = Projection(  # a different map that fits better: never the one to continue
        layout(made.n_points, 7), stress=drawn.stress_value * 0.5
    )
    seed = dataclasses.replace(made, projections=[drawn, alternative])
    write_chart(seed, tmp_path / "seed.ace")
    return tmp_path, refs, seed


def seeded_config(tmp_path, refs, **kw):
    return ChainConfig(
        GROUP,
        refs,
        OPTS,
        seed=3,
        first_map=tmp_path / "seed.ace",
        seed_chart=SeedChart(bridge=False, **kw),
    )


def test_step0_bakes_the_orientation_and_records_it(seed_with_orientation):
    tmp_path, refs, seed = seed_with_orientation
    results = run_chain(seeded_config(tmp_path, refs), tmp_path / "s", StubOptimiser())
    step0 = results[0]
    chosen = read_chart(step0.directory / CHOSEN)
    # the stored layout is now the layout as drawn, bit for bit
    assert (
        chosen.projections[0].layout.tobytes() == seed.projections[0].transformed_layout().tobytes()
    )
    assert chosen.projections[0].transformation is None
    rec = step0.record["transformation"]
    assert rec["baked"] is True
    assert rec["projections"][0]["transformation"] == [[0.0, 1.0], [-1.0, 0.0]]
    assert rec["projections"][0]["reflection"] is False


def test_step0_without_baking_keeps_the_raw_layout(seed_with_orientation):
    tmp_path, refs, seed = seed_with_orientation
    results = run_chain(
        seeded_config(tmp_path, refs, bake_transformation=False), tmp_path / "s", StubOptimiser()
    )
    chosen = read_chart(results[0].directory / CHOSEN)
    assert chosen.projections[0].layout.tobytes() == seed.projections[0].layout.tobytes()
    assert chosen.projections[0].transformation is not None
    rec = results[0].record["transformation"]
    assert rec["baked"] is False
    assert rec["projections_with_one"] == [0]


def test_step0_records_which_projection_and_the_stress_check(seed_with_orientation):
    tmp_path, refs, seed = seed_with_orientation
    results = run_chain(seeded_config(tmp_path, refs), tmp_path / "s", StubOptimiser())
    rec = results[0].record
    assert rec["seed_projection"]["index"] == 0
    assert rec["seed_projection"]["of"] == 2
    assert rec["seed_projection"]["lowest_stress_index"] == 1
    assert rec["seed_projection"]["is_lowest_stress"] is False  # the drawn map is not the lowest
    assert rec["stress"]["seed"] == pytest.approx(seed.projections[0].stress_value)
    assert rec["stress"]["computed"] == pytest.approx(seed.projections[0].stress_value, rel=1e-9)
    assert abs(rec["stress"]["relative"]) <= STRESS_TOLERANCE
    assert rec["layers"]["layers"] == 2


def test_step1_starts_from_the_named_projection_not_the_lowest(seed_with_orientation):
    tmp_path, refs, seed = seed_with_orientation
    results = run_chain(seeded_config(tmp_path, refs), tmp_path / "s", StubOptimiser())
    merged = read_chart(results[1].directory / "merge.ace")
    step0 = read_chart(results[0].directory / CHOSEN)
    n_ag = step0.n_antigens
    # the incremental start is the drawn map carried forward, not the better-fitting alternative
    assert (
        merged.projections[0].layout[:n_ag].tobytes()
        == step0.projections[0].layout[:n_ag].tobytes()
    )
    alternative = seed.projections[1].layout[:n_ag]
    assert not np.array_equal(merged.projections[0].layout[:n_ag], alternative)


def test_step0_refuses_a_seed_whose_stress_does_not_reproduce(seed_with_orientation):
    tmp_path, refs, seed = seed_with_orientation
    broken = dataclasses.replace(
        seed,
        projections=[dataclasses.replace(seed.projections[0], stress=1.0), seed.projections[1]],
    )
    write_chart(broken, tmp_path / "seed.ace")
    with pytest.raises(SeedError, match="does not reproduce the seed's stress"):
        run_chain(seeded_config(tmp_path, refs), tmp_path / "s", StubOptimiser())


def test_a_first_map_chain_without_seed_chart_is_unchanged(seed_with_orientation):
    """No [seed_chart] table: step 0 copies the seed as it stands, as it did before -- which is
    what the three settings are for. Without them the orientation is left behind and the stress
    recorded is the lowest-stress projection's, not the drawn one's."""
    tmp_path, refs, seed = seed_with_orientation
    cfg = ChainConfig(GROUP, refs, OPTS, seed=3, first_map=tmp_path / "seed.ace")
    results = run_chain(cfg, tmp_path / "s", StubOptimiser())
    chosen = read_chart(results[0].directory / CHOSEN)
    assert chosen.projections[0].layout.tobytes() == seed.projections[0].layout.tobytes()
    assert chosen.projections[0].transformation is not None
    assert results[0].record["stress"] == {"seed": seed.projections[1].stress_value}
    assert "seed_projection" not in results[0].record
