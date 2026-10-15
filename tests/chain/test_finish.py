"""af.chain.finish: the next version of a merged map made from its own published layout by the
trapped-point loop under a named rule, without new starts, and only when the chain's input is
still the map's own."""

import functools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from af.chain.__main__ import main as chain_main
from af.chain.backend import FinishRule
from af.chain.config import ChainConfigError
from af.chain.finish import FinishError
from af.chain.finish import main as finish_main
from af.chain.synthetic import make_tables
from af.store.store import Store
from af.store.work import Work

from .sera_markers import write_markers

pytest.importorskip("af.map.optimise")

GROUP = "h9-hi-test-lab"
DATASET = "testlab/h9/merged"


@pytest.fixture
def made(tmp_path):
    """A merged map built and published under the original arithmetic."""
    make_tables(tmp_path / "tables", GROUP, n_tables=3)
    chain = tmp_path / "chains" / "testlab" / "h9" / "merged.toml"
    chain.parent.mkdir(parents=True)
    write_markers(tmp_path / "rules" / "sera" / "non_ferret_markers.tsv")
    chain.write_text(
        f'name = "{GROUP}"\nseed = 3\n[tables]\ndirectory = "{tmp_path / "tables"}"\n'
        f'group = "{GROUP}"\n[options]\nminimum_column_basis = "none"\nmerge_all = true\n'
        "scratch_starts = 6\n"
    )
    Work.create(tmp_path / "work")
    Store.create(tmp_path / "store")
    run = _run(tmp_path, "original")
    assert chain_main([str(chain), str(run)]) == 0
    return SimpleNamespace(root=tmp_path, chain=chain, run=run, version=_current(tmp_path))


def _run(root: Path, numerics: str | None, optimiser: str = "core", name: str = "run.toml") -> Path:
    path = root / name
    path.write_text(
        f'optimiser = "{optimiser}"\n'
        + (f'numerics = "{numerics}"\n' if numerics else "")
        + f'threads = 1\npublish = true\n[paths]\nstore = "{root / "store"}"\n'
        f'work = "{root / "work"}"\n[runner]\nkind = "local"\n'
    )
    return path


def _current(root: Path) -> str:
    return Store.open(root / "store").current("chains", DATASET).version


def _version(root: Path, version: str) -> Path:
    return root / "store" / "chains" / DATASET / "versions" / version


def _finish(made, *extra: str, run: Path | None = None) -> int:
    args = [str(made.chain), str(run or made.run), "--version", made.version, *extra]
    return finish_main(args)


def test_a_rule_is_required():
    with pytest.raises(ValueError, match="must turn something on"):
        FinishRule()


def test_a_finish_that_moves_nothing_makes_no_version(made):
    assert _finish(made, "--move-groups") == 0  # the synthetic map has no group to move
    assert _current(made.root) == made.version


def test_a_finished_map_is_the_next_version_and_says_how_it_was_made(made, monkeypatch):
    import af.map.optimise as optimise

    real = optimise.resolve_trapped
    seen = {}

    def lower(problem, layout, **kwargs):  # a rule that found a lower place for one point
        seen.update(kwargs)
        result = real(problem, layout, **kwargs)
        moved = np.asarray(result.projection.layout).copy()
        moved[0] += 0.5
        p = SimpleNamespace(
            layout=moved,
            stress=result.projection.stress - 1.0,
            dimensions=result.projection.dimensions,
            n_iterations=result.projection.n_iterations,
        )
        return SimpleNamespace(projection=p, rounds=2, moved=1, groups=[], threads=1)

    monkeypatch.setattr(optimise, "resolve_trapped", lower)
    assert _finish(made, "--move-groups") == 0
    assert seen["move_groups"] is True
    new = _current(made.root)
    assert new != made.version
    history = (made.root / "store/chains" / DATASET / "HISTORY.jsonl").read_text().splitlines()
    assert json.loads(history[-1])["parent"] == made.version
    old_dir, new_dir = _version(made.root, made.version), _version(made.root, new)
    step = json.loads((new_dir / "steps/0000/step.json").read_text())
    source = json.loads((old_dir / "steps/0000/step.json").read_text())
    assert step["chosen"] == "finished"
    assert step["finished_from"]["version"] == made.version
    assert step["finish"]["rule"] == {
        "loop": "af.map.optimise.resolve_trapped",
        "move_groups": True,
        "move_lower": False,
    }
    assert step["finish"]["stress_after"] == pytest.approx(step["finish"]["stress_before"] - 1.0)
    assert step["stress"]["source"] == source["stress"]["scratch"]
    assert "numerics" not in step["platform"]  # finished under the arithmetic it was made with
    # the merged table is the published one, byte for byte; only the map moved
    assert (new_dir / "steps/0000/merge.ace").read_bytes() == (
        old_dir / "steps/0000/merge.ace"
    ).read_bytes()
    params = json.loads((new_dir / "PROVENANCE.json").read_text())["parameters"]
    assert params["finished_from"]["version"] == made.version and params["finish"]["move_groups"]
    assert "finished from" in (new_dir / "review/index.html").read_text()

    from af.chain.continuity import record, sentences

    thresholds = made.root / "t.toml"
    thresholds.write_text(
        '[movers]\ndistance = 1.0\nreason = "invented for the test"\n'
        '[[apart]]\ndistance = 0.5\nreason = "invented for the test"\n'
        '[[apart]]\ndistance = 1.0\nreason = "invented for the test"\n'
    )
    rec = record(made.root / "store/chains", DATASET, new, thresholds, made.root / "records")
    assert rec["same_problem"] is True and rec["parameters_differ"] == []
    assert rec["search_differ"]["finish"][0] is None
    assert "finish none -> move_groups" in sentences(rec)[1]


def test_only_the_current_version_is_finished(made):
    made.version = "0" * 16
    with pytest.raises(FinishError, match="is not CURRENT"):
        _finish(made, "--move-groups")


def test_a_table_changed_since_the_map_was_made_is_refused(made):
    path = sorted((made.root / "tables").glob("*.ace"))[1]
    from af.chart.ace import read_chart, write_chart
    from af.chart.titre import Titre

    chart = read_chart(path)
    before = path.read_bytes()
    chart.titres.table[0][0] = Titre.parse(
        "<10" if str(chart.titres.table[0][0]) != "<10" else "40"
    )
    write_chart(chart, path)
    assert path.read_bytes() != before
    with pytest.raises(FinishError, match="not the ones it was built from"):
        _finish(made, "--move-groups")
    assert _current(made.root) == made.version


def test_a_finish_never_mixes_arithmetics(made):
    import af.map.optimise as optimise

    strict = _run(made.root, optimise.NUMERICS, name="strict.toml")
    with pytest.raises(FinishError, match="must not mix arithmetics"):
        _finish(made, "--move-groups", run=strict)


def test_the_stand_in_cannot_finish(made):
    with pytest.raises(ChainConfigError, match="needs the core optimiser"):
        _finish(made, "--move-groups", run=_run(made.root, None, "stub", "stub.toml"))


def _finish_root(made) -> Path:
    return made.root / "work/chains" / DATASET / "finish" / made.version


def test_move_lower_runs_the_rule_and_records_it(made, monkeypatch):
    import af.map.optimise as optimise

    real = optimise.resolve_trapped
    seen = {}

    @functools.wraps(real)  # the rule's round cap is read from resolve_trapped's signature
    def spy(problem, layout, **kwargs):
        seen.update(kwargs)
        return real(problem, layout, **kwargs)

    monkeypatch.setattr(optimise, "resolve_trapped", spy)
    assert _finish(made, "--move-lower") == 0
    assert seen["move_lower"] is True and seen["move_groups"] is False
    step = json.loads((_finish_root(made) / "steps/0000/step.json").read_text())
    rule = step["finish"]["rule"]
    assert rule["move_lower"] is True and rule["move_lower_rule"]["name"] == "move-lower-1"
    assert rule["move_lower_rule"]["max_rounds"] == 20
    lower = step["finish"]["move_lower"]
    assert lower["stop"] in ("nothing lower", "stress did not fall", "round limit")
    if not any(r["kept"] for r in lower["rounds"]):  # the synthetic map had nothing lower
        assert _current(made.root) == made.version


def test_a_layout_whose_stress_is_not_its_own_is_refused(made, monkeypatch):
    import dataclasses

    import af.chain.finish as finish

    real = finish.check_source

    def wrong(*a, **k):  # as if the arrays were not the problem the map was made for
        source = real(*a, **k)
        return dataclasses.replace(source, stress=source.stress * 1.01)

    monkeypatch.setattr(finish, "check_source", wrong)
    with pytest.raises(FinishError, match="not the arrays it was made from"):
        _finish(made, "--move-lower")
    assert _current(made.root) == made.version
