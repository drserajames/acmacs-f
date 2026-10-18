"""Drawing every tree's figure and book in parallel processes: the same bytes as one by one."""

import json
from dataclasses import dataclass, replace

import pytest

pytest.importorskip("numpy", reason="numpy not installed: af.tree.draw needs it (pyproject, WS1)")
pytest.importorskip("matplotlib", reason="matplotlib not installed: the renderer needs it")

from af.tree.draw.defaults import load_defaults  # noqa: E402
from af.tree.draw.figure import FigureConfig  # noqa: E402
from af.tree.draw.stage import TreeJob, TreeStageError, draw_trees  # noqa: E402
from af.tree.draw.store import StoreTree  # noqa: E402

from .synthetic import PARENTS, standard_tree  # noqa: E402

CODE: dict[str, object] = {"commit": "c0ffee", "release": None, "python": "3.12", "libraries": {}}


@dataclass(frozen=True)
class Invented:
    """An invented tree as a job's source; module level so a spawned process can load it."""

    fail: bool = False

    def load(self) -> StoreTree:
        if self.fail:
            raise RuntimeError("invented failure")
        return StoreTree(standard_tree(), dict(PARENTS), "v1", {}, {}, {"tree": "x"})

    def identity(self):
        return {"invented": "standard", "fail": self.fail}

    def refs(self):
        return []


def config(title):
    return FigureConfig(
        title=title,
        window_start="2024-10",
        window_end="2026-10",
        select=replace(load_defaults().clades, min_share=0.1, min_window_leaves=10**9),
        leaf_names=True,
    )


def jobs(root, fail=False):
    return [
        TreeJob(
            name,
            Invented(fail and name == "two"),
            config(f"TEST {name}"),
            root / name / "figure.pdf",
            root / name / "book.pdf",
        )
        for name in ("one", "two")
    ]


def drawn_files(root):
    return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())


def without_run_facts(path):
    data = json.loads(path.read_text())
    data.pop("seconds", None)
    data.get("provenance", {}).pop("created", None)
    return data


def test_parallel_parts_draw_the_same_bytes_as_one_by_one(tmp_path):
    serial = draw_trees(jobs(tmp_path / "serial"), workers=1, code=CODE)
    parallel = draw_trees(jobs(tmp_path / "parallel"), workers=4, code=CODE)
    files = drawn_files(tmp_path / "serial")
    assert files == drawn_files(tmp_path / "parallel") and len(files) == 10
    for rel in files:
        a, b = tmp_path / "serial" / rel, tmp_path / "parallel" / rel
        if rel.name.endswith((".i7.json", ".book.json")):  # these record when and how long
            assert without_run_facts(a) == without_run_facts(b), rel
        else:
            assert a.read_bytes() == b.read_bytes(), rel
    assert serial["one"].report["rows"] == parallel["one"].report["rows"] == 50
    assert parallel["two"].report["book"]["pages"] >= 1


def test_every_part_is_reused_from_the_cache(tmp_path):
    first = draw_trees(jobs(tmp_path / "a"), cache=tmp_path / "c", workers=2, code=CODE)
    again = draw_trees(jobs(tmp_path / "b"), cache=tmp_path / "c", workers=2, code=CODE)
    assert [r.hit for res in first.values() for r in res.cache.values()] == [False] * 4
    assert [r.hit for res in again.values() for r in res.cache.values()] == [True] * 4
    for rel in drawn_files(tmp_path / "a"):
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()


def test_a_failed_part_is_named_and_stores_nothing(tmp_path):
    with pytest.raises(TreeStageError, match="tree two figure: RuntimeError: invented failure"):
        draw_trees(jobs(tmp_path / "a", fail=True), cache=tmp_path / "c", workers=2, code=CODE)
    assert (tmp_path / "a" / "one" / "book.pdf").is_file()  # the others still ran
    assert not (tmp_path / "c").exists()  # but no entry is made from a failed stage


@pytest.mark.parametrize("workers", [0, -1])
def test_workers_must_be_positive(tmp_path, workers):
    with pytest.raises(ValueError, match="at least 1"):
        draw_trees(jobs(tmp_path), workers=workers, code=CODE)


def test_job_names_must_not_repeat(tmp_path):
    one = jobs(tmp_path)[0]
    with pytest.raises(ValueError, match="repeat"):
        draw_trees([one, one], code=CODE)
