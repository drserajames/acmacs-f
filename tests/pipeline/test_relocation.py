"""Moving a pipeline's tree (or syncing it to another machine) must not re-run anything.

Records are keyed by role: an input's name, or a path relative to the pipeline root.
Found with real data: a chain whose byte-identical tables moved directory re-ran every
step, because records were keyed by absolute path.
"""

import json
import shutil
from pathlib import Path

import pytest

from af.pipeline import Pipeline, PipelineError, Step, StepContext
from af.run import LocalRunner
from af.util.artefacts import Artefact


def make_tree(root: Path) -> None:
    (root / "tables").mkdir(parents=True)
    (root / "tables" / "t1.txt").write_text("table one\n")
    (root / "tables" / "t2.txt").write_text("table two\n")


def steps(root: Path, ran: list[str], named: bool) -> list[Step]:
    """merge: t1 + t2 -> merged.txt; count: merged.txt -> count.txt."""
    merged = root / "out" / "merged.txt"
    count = root / "out" / "count.txt"

    def merge(context: StepContext) -> None:
        ran.append("merge")
        merged.parent.mkdir(exist_ok=True)
        merged.write_text("".join(p.read_text() for p in sorted((root / "tables").iterdir())))

    def counting(context: StepContext) -> None:
        ran.append("count")
        count.write_text(str(len(merged.read_text())))

    tables = [root / "tables" / "t1.txt", root / "tables" / "t2.txt"]
    inputs = {"first": tables[0], "second": tables[1]} if named else tables
    return [
        Step("merge", merge, outputs=[Artefact(merged)], inputs=inputs),
        Step("count", counting, outputs=[Artefact(count)], inputs=[merged]),
    ]


def run(root: Path, ran: list[str], *, named: bool = False, use_root: bool = True) -> list[str]:
    pipeline = Pipeline(
        steps(root, ran, named),
        state_dir=root / "state",
        runner=LocalRunner(),
        root=root if use_root else None,
    )
    return [outcome.status for outcome in pipeline.run()]


@pytest.mark.parametrize("named", [False, True], ids=["relative-paths", "named-inputs"])
def test_moved_tree_is_up_to_date(tmp_path: Path, named: bool) -> None:
    before = tmp_path / "machine-a" / "work"
    make_tree(before)
    ran: list[str] = []
    assert run(before, ran, named=named) == ["ran", "ran"]

    after = tmp_path / "machine-b" / "elsewhere"
    after.parent.mkdir()
    shutil.move(before, after)
    assert run(after, ran, named=named) == ["skipped", "skipped"]
    assert ran == ["merge", "count"], "nothing re-ran after the move"


def test_changed_table_after_move_still_reruns(tmp_path: Path) -> None:
    before = tmp_path / "a"
    make_tree(before)
    ran: list[str] = []
    run(before, ran)
    after = tmp_path / "b"
    shutil.move(before, after)
    (after / "tables" / "t2.txt").write_text("table two, revised\n")
    assert run(after, ran) == ["ran", "ran"]


def test_without_root_a_move_reruns(tmp_path: Path) -> None:
    """The documented fallback: no root means absolute paths, so a move re-runs."""
    before = tmp_path / "a"
    make_tree(before)
    ran: list[str] = []
    run(before, ran, use_root=False)
    after = tmp_path / "b"
    shutil.move(before, after)
    assert run(after, ran, use_root=False) == ["ran", "ran"]


def test_record_is_keyed_by_role_and_keeps_paths(tmp_path: Path) -> None:
    root = tmp_path / "work"
    make_tree(root)
    run(root, [], named=True)
    record = json.loads((root / "state" / "merge.json").read_text())
    assert record["format"] == 2
    assert sorted(record["inputs"]) == ["first", "second"]
    assert list(record["outputs"]) == ["out/merged.txt"]
    assert record["paths"]["inputs"]["first"] == str(root / "tables" / "t1.txt")
    count = json.loads((root / "state" / "count.json").read_text())
    assert list(count["inputs"]) == ["out/merged.txt"]


def test_old_format_record_reruns(tmp_path: Path) -> None:
    root = tmp_path / "work"
    make_tree(root)
    run(root, [])
    record_path = root / "state" / "merge.json"
    record = json.loads(record_path.read_text())
    del record["format"]
    record_path.write_text(json.dumps(record))
    pipeline = Pipeline(
        steps(root, [], False), state_dir=root / "state", runner=LocalRunner(), root=root
    )
    outcomes = pipeline.run(["merge"])
    assert (outcomes[0].status, outcomes[0].reason) == (
        "ran",
        "record written by an older af (different format)",
    )


def test_duplicate_role_is_an_error(tmp_path: Path) -> None:
    root = tmp_path / "work"
    make_tree(root)
    table = root / "tables" / "t1.txt"
    step = Step("dup", lambda c: None, outputs=[Artefact(root / "x")], inputs=[table, table])
    pipeline = Pipeline([step], state_dir=root / "state", runner=LocalRunner(), root=root)
    with pytest.raises(PipelineError, match="two inputs share the role 'tables/t1.txt'"):
        pipeline.run()
