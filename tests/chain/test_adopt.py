"""af.chain.adopt: a merged map's next version takes another published chain's layout of the same
tables, only once the two are shown to be the same problem, point for point and cell for cell."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from af.chain.__main__ import main as chain_main
from af.chain.adopt import AdoptError, Pairing, check_same_problem, pair_points
from af.chain.adopt import main as adopt_main
from af.chain.synthetic import make_tables
from af.store.store import Store
from af.store.work import Work

from .sera_markers import write_markers

GROUP = "h9-hi-test-lab"
MERGED, MAIN = "testlab/h9/merged", "testlab/h9/main"


# ---- pairing and identity, on small stand-ins ------------------------------------------------


def _ag(name, passage="E1"):
    return SimpleNamespace(name=name, reassortant="", annotations=(), passage=passage)


def _sr(name, serum_id):
    return SimpleNamespace(name=name, reassortant="", annotations=(), serum_id=serum_id)


def _chart(antigens, sera):
    return SimpleNamespace(antigens=antigens, sera=sera, n_antigens=len(antigens), n_sera=len(sera))


def _arrays(values, n_points):
    v = np.array(values, dtype=float)
    return {
        "titre_value": v,
        "titre_type": np.where(np.isnan(v), 0, 1).astype(np.int8),
        "column_bases": np.nanmax(v, axis=0),
        "disconnected": np.zeros(n_points, dtype=bool),
        "dodgy_is_regular": False,
    }


NAN = float("nan")
ANTIGENS = [_ag("TEST-1"), _ag("TEST-2"), _ag("TEST-3")]
SERA = [_sr("TEST-1", "F1"), _sr("TEST-2", "F2")]
VALUES = [[7.0, 5.0], [4.0, 6.0], [3.0, NAN]]


def test_the_same_points_pair_by_key_in_any_order():
    target = _chart(ANTIGENS, SERA)
    order = [2, 0, 1]  # the source holds the antigens in another order
    source = _chart([ANTIGENS[i] for i in order], SERA)
    t, s = _arrays(VALUES, 5), _arrays([VALUES[i] for i in order], 5)
    p = pair_points(target, t, source, s)
    assert p.antigens.tolist() == [1, 2, 0] and p.sera.tolist() == [0, 1] and not p.by_titres
    check_same_problem(t, s, p, source.n_antigens)  # the same problem: no error


def test_a_renamed_antigen_pairs_by_its_titres_and_is_recorded():
    target = _chart(ANTIGENS, SERA)
    source = _chart([ANTIGENS[0], _ag("TEST-2-RENAMED"), ANTIGENS[2]], SERA)
    p = pair_points(target, _arrays(VALUES, 5), source, _arrays(VALUES, 5))
    assert p.antigens.tolist() == [0, 1, 2]
    [renamed] = p.by_titres
    assert renamed["what"] == "antigen" and renamed["target"] == 1 and renamed["source"] == 1
    assert renamed["source_name"] == "TEST-2-RENAMED"


def test_a_point_with_no_unique_match_is_refused():
    target = _chart(ANTIGENS, SERA)
    extra = _chart([*ANTIGENS, _ag("TEST-4")], SERA)
    with pytest.raises(AdoptError, match="0 target and 1 source antigens pair with nothing"):
        pair_points(target, _arrays(VALUES, 5), extra, _arrays([*VALUES, [1.0, 1.0]], 6))
    # two renamed antigens with the same titres cannot say which is which
    twins = [[7.0, 5.0], [4.0, 6.0], [4.0, 6.0]]
    renamed = _chart([ANTIGENS[0], _ag("TEST-X"), _ag("TEST-Y")], SERA)
    with pytest.raises(AdoptError, match="pair with nothing"):
        pair_points(target, _arrays(twins, 5), renamed, _arrays(twins, 5))


def test_a_key_held_twice_is_refused():
    target = _chart([ANTIGENS[0], ANTIGENS[0]], SERA)
    with pytest.raises(AdoptError, match="1 antigen identity keys held twice in the target"):
        pair_points(target, _arrays(VALUES[:2], 4), target, _arrays(VALUES[:2], 4))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda a: a["titre_value"].__setitem__((0, 0), 8.0), r"titre values \(1 cells\)"),
        (lambda a: a["titre_type"].__setitem__((1, 1), 2), r"titre types \(1 cells\)"),
        (lambda a: a["column_bases"].__setitem__(1, 9.0), r"column bases \(1 sera\)"),
        (lambda a: a["disconnected"].__setitem__(4, True), "disconnected"),
        (lambda a: a.__setitem__("dodgy_is_regular", True), "dodgy_is_regular"),
        (lambda a: a.__setitem__("unmovable", np.ones(5, dtype=bool)), "unmovable"),
    ],
)
def test_any_difference_after_pairing_is_refused(change, message):
    t, s = _arrays(VALUES, 5), _arrays(VALUES, 5)
    change(s)
    p = Pairing(np.arange(3), np.arange(2), ())
    with pytest.raises(AdoptError, match=message):
        check_same_problem(t, s, p, 3)


# ---- end to end: a merged map adopts the main chain's layout ----------------------------------


@pytest.fixture
def both(tmp_path):
    """A merged map and the incremental chain of the same tables, both published in one store."""
    pytest.importorskip("af.map.optimise")
    make_tables(tmp_path / "tables", GROUP, n_tables=3)
    write_markers(tmp_path / "rules" / "sera" / "non_ferret_markers.tsv")
    tables = f'[tables]\ndirectory = "{tmp_path / "tables"}"\ngroup = "{GROUP}"\n'
    head = f'name = "{GROUP}"\nseed = 3\n{tables}[options]\nminimum_column_basis = "none"\n'
    chains = tmp_path / "chains" / "testlab" / "h9"
    chains.mkdir(parents=True)
    merged = chains / "merged.toml"
    merged.write_text(head + "merge_all = true\nscratch_starts = 6\n")
    main = chains / "main.toml"
    main.write_text(head + "scratch_starts = 6\nincremental_starts = 6\n")
    Work.create(tmp_path / "work")
    Store.create(tmp_path / "store")
    run = tmp_path / "run.toml"
    run.write_text(
        'optimiser = "core"\nnumerics = "strict-fp-1"\nthreads = 1\npublish = true\n'
        f'[paths]\nstore = "{tmp_path / "store"}"\nwork = "{tmp_path / "work"}"\n'
        '[runner]\nkind = "local"\n'
    )
    assert chain_main([str(merged), str(run)]) == 0
    assert chain_main([str(main), str(run)]) == 0
    store = Store.open(tmp_path / "store")
    return SimpleNamespace(
        root=tmp_path,
        merged=merged,
        run=run,
        version=store.current("chains", MERGED).version,
        main=store.current("chains", MAIN).version,
    )


def _adopt(both, source: str | None = None, *extra: str) -> int:
    spec = source or f"{MAIN}@{both.main}"
    return adopt_main([str(both.merged), str(both.run), "--version", both.version, "--from", spec,
                       *extra])  # fmt: skip


def _vdir(both, dataset: str, version: str) -> Path:
    return both.root / "store" / "chains" / dataset / "versions" / version


def test_the_main_chains_layout_becomes_the_next_merged_version(both):
    from af.chart.ace import read_chart
    from af.chart.procrustes import procrustes

    assert _adopt(both) == 0
    store = Store.open(both.root / "store")
    new = store.current("chains", MERGED).version
    assert new != both.version
    history = store.history("chains", MERGED)
    assert history[-1]["parent"] == both.version
    vdir = _vdir(both, MERGED, new)
    step = json.loads((vdir / "steps/0000/step.json").read_text())
    assert step["chosen"] == "adopted"
    assert step["adopted_from"]["dataset"] == MAIN and step["adopted_from"]["version"] == both.main
    assert step["adopted_onto"]["version"] == both.version
    adopt = step["adopt"]
    # the same problem: the layout, placed, has the main chain's own stress, and only goes lower
    assert adopt["stress_placed"] == pytest.approx(step["stress"]["source"], rel=1e-9)
    assert adopt["stress_after"] <= adopt["stress_placed"] * (1 + 1e-12)
    # its provenance names the main chain's version as an input beside the tables
    provenance = json.loads((vdir / "PROVENANCE.json").read_text())
    stores = [i["store"] for i in provenance["inputs"] if "store" in i]
    assert {"kind": "chains", "dataset": MAIN, "version": both.main} in [
        {k: s[k] for k in ("kind", "dataset", "version")} for s in stores
    ]
    assert provenance["parameters"]["adopted_from"]["version"] == both.main
    # the map drawn is the main chain's, moved by at most the last bits of a fine minimise
    main_dir = _vdir(both, MAIN, both.main)
    last = sorted((main_dir / "steps").iterdir())[-1]
    a = np.asarray(read_chart(vdir / "steps/0000/chosen.ace").best().layout)
    b = np.asarray(read_chart(last / "chosen.ace").best().layout)
    ok = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
    assert procrustes(b[ok], a[ok]).rmsd < 1e-3


def test_a_dataset_does_not_adopt_its_own_layout(both):
    with pytest.raises(AdoptError, match="not its own"):
        _adopt(both, f"{MERGED}@{both.version}")


def test_an_unpublished_source_is_refused(both):
    with pytest.raises(AdoptError, match="is not a published version"):
        _adopt(both, f"{MAIN}@{'0' * 16}")
    with pytest.raises(AdoptError, match="expected <dataset>@<version>"):
        _adopt(both, MAIN)


def test_a_step_that_is_not_there_is_refused(both):
    with pytest.raises(AdoptError, match="has no steps/0099"):
        _adopt(both, None, "--step", "0099")


def test_a_different_problem_is_refused(both):
    """An earlier step of the main chain holds fewer tables: not the merged map's problem."""
    with pytest.raises(AdoptError, match="not the same problem"):
        _adopt(both, None, "--step", "0000")
    assert Store.open(both.root / "store").current("chains", MERGED).version == both.version


# ---- the optimiser's own refusal: the layout placed point for point --------------------------


def _problem_and_map():
    pytest.importorskip("af.map.optimise")
    from af.chain.backend import CoreOptimiser
    from af.chain.synthetic import make_tables as _make
    from af.chart.ace import read_chart

    return CoreOptimiser(threads=1), _make, read_chart


def test_the_optimiser_refuses_a_layout_not_placed_point_for_point(tmp_path):
    """The stress check is what proves the placement: two connected points swapped is refused,
    the layout as it is (to the last bits) is accepted, and the stand-in refuses outright."""
    core, make, read = _problem_and_map()
    import af.map.optimise as optimise
    from af.chain.backend import StubOptimiser

    make(tmp_path / "t", GROUP, n_tables=1)
    chart = read(sorted((tmp_path / "t").glob("*.ace"))[0])
    arrays = chart.optimiser_arrays("none", disconnect_threshold=None)
    problem = core._problem(arrays)
    rng = np.random.default_rng(1)
    layout = optimise.optimise(problem, rng.normal(size=(chart.n_points, 2))).layout
    recorded = optimise.stress(problem, layout)
    made = core.adopt(arrays, layout, recorded * (1 + 1e-15))
    assert made["adopt"]["stress_placed"] == pytest.approx(recorded, rel=1e-15)
    swapped = layout.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    assert optimise.stress(problem, swapped) != recorded
    with pytest.raises(ValueError, match="is not that map's layout"):
        core.adopt(arrays, swapped, recorded)
    with pytest.raises(ValueError, match="cannot adopt"):
        StubOptimiser().adopt(arrays, layout, recorded)
