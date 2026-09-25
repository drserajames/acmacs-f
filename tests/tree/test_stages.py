"""The tree stages as pipeline steps: what re-runs, and what a moved work area does.

CMAPLE is not installed here, so a stub stands in for it: it writes the synthetic fixture tree
wherever CMAPLE would. ASR uses the in-process Fitch backend, so everything after the build runs
for real.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from af.store import Store
from af.store.work import Work
from af.tree import stages
from af.tree.io import i6
from af.tree.io.fasta import write_alignment
from tests.clades.synthetic import build_clone, commit_command, write_clade

from .tree_fixtures import KEYS, LEAF_SEQ, records

TREE_STEPS = ("build", "asr", "populate", "publish")

NEWICK = (
    f"({KEYS['o']}:0.1,(({KEYS['a']}:0.2,{KEYS['b']}:0.3):0.05,"
    f"({KEYS['c']}:0.1,{KEYS['d']}:0.1):0.01):0.02);"
)

STUB = f"""#!{sys.executable}
import sys
if "--help" in sys.argv:
    print("CMAPLE version 0.0-stub")
    sys.exit(0)
prefix = sys.argv[sys.argv.index("--prefix") + 1]
open(prefix + ".treefile", "w").write({NEWICK!r} + "\\n")
"""


def make_project(root: Path) -> Path:
    """A work area, a store, the export's two files, and a config naming them relatively."""
    root.mkdir(parents=True)
    stub = root / "cmaple-stub"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    Work.create(root / "work")
    Store.create(root / "store")
    write_alignment(root / "alignment.fasta", {KEYS[n]: LEAF_SEQ[n] for n in "oabcd"})
    stages.write_leaves(records(), root / "leaves.parquet")
    (root / "trees.toml").write_text(
        f"""
[paths]
store = "store"
work = "work"

[trees]
threads = 1

[trees.cmaple]
executable = "{stub}"

[trees.subtypes.h3]
outgroup = "{KEYS["o"]}"
asr_backend = "parsimony"

[inputs.h3]
alignment = "alignment.fasta"
leaves = "leaves.parquet"
"""
    )
    return root / "trees.toml"


def statuses(config_path: Path) -> dict[str, str]:
    config = stages.load_run_config(config_path)
    return {o.name: o.status for o in stages.run(config)["h3"]}


def test_the_stages_run_in_order_and_publish_an_i6_version(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    assert statuses(config) == dict.fromkeys(TREE_STEPS, "ran")

    store = Store.open(tmp_path / "p" / "store")
    ref = store.current("trees", "h3/weekly")
    version = store.version_dir(ref)
    meta = i6.read_metadata(version)
    assert meta["leaves"] == 5
    assert meta["asr"]["backend"] == "parsimony"
    published = json.loads((tmp_path / "p/work/trees/h3/publish/published.json").read_text())
    assert published == ref.to_json()


def test_nothing_re_runs_when_nothing_changed(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    statuses(config)
    assert statuses(config) == dict.fromkeys(TREE_STEPS, "skipped")


def test_a_moved_work_area_is_still_up_to_date(tmp_path: Path) -> None:
    """The point of task 5.12: a store and work area synced elsewhere must not re-run CMAPLE."""
    config = make_project(tmp_path / "laptop")
    statuses(config)
    shutil.copytree(tmp_path / "laptop", tmp_path / "server")
    shutil.rmtree(tmp_path / "laptop")
    server = tmp_path / "server" / "trees.toml"
    # CMAPLE is installed somewhere else there too; that alone must not rebuild.
    moved = str(tmp_path / "server" / "cmaple-stub")
    server.write_text(server.read_text().replace(str(tmp_path / "laptop" / "cmaple-stub"), moved))
    assert moved in server.read_text()
    assert statuses(server) == dict.fromkeys(TREE_STEPS, "skipped")


def test_leaf_metadata_re_runs_populate_but_not_the_build(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    statuses(config)
    changed = records()
    key = KEYS["a"]
    changed[key] = dataclasses.replace(changed[key], country="ELSEWHERELAND")
    stages.write_leaves(changed, tmp_path / "p" / "leaves.parquet")
    assert statuses(config) == {
        "build": "skipped",
        "asr": "skipped",
        "populate": "ran",
        "publish": "ran",
    }


def test_a_changed_clock_threshold_re_runs_populate_not_asr(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    statuses(config)
    parsed = stages.load_run_config(config)
    h3 = dataclasses.replace(parsed.trees.subtypes["h3"], clock_z_threshold=5.0)
    trees = dataclasses.replace(parsed.trees, subtypes={"h3": h3})
    outcomes = stages.run(dataclasses.replace(parsed, trees=trees))["h3"]
    assert {o.name: o.status for o in outcomes} == {
        "build": "skipped",
        "asr": "skipped",
        "populate": "ran",
        "publish": "ran",
    }


def test_a_first_build_says_why_nothing_was_dropped(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    statuses(config)
    meta = json.loads((tmp_path / "p/work/trees/h3/build/build.json").read_text())
    assert meta["excluded"] == []
    assert meta["prebuild_not_applied"]


def test_leaves_round_trip_and_join_their_sequences(tmp_path: Path) -> None:
    path = tmp_path / "leaves.parquet"
    stages.write_leaves(records(), path)
    back = stages.read_leaves(path, {KEYS[n]: LEAF_SEQ[n] for n in "oabcd"})
    assert back == records()


def test_a_leaf_without_a_sequence_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "leaves.parquet"
    stages.write_leaves(records(), path)
    with pytest.raises(stages.StageError, match="no sequence"):
        stages.read_leaves(path, {KEYS["o"]: LEAF_SEQ["o"]})


def test_incremental_needs_a_previous_tree(tmp_path: Path) -> None:
    with pytest.raises(stages.StageError, match="previous"):
        stages.SubtypeInputs(
            alignment=tmp_path / "a.fasta", leaves=tmp_path / "l.parquet", incremental=True
        )


def test_clades_need_both_the_set_and_the_nomenclature(tmp_path: Path) -> None:
    with pytest.raises(stages.StageError, match="both or neither"):
        stages.SubtypeInputs(
            alignment=tmp_path / "a.fasta", leaves=tmp_path / "l.parquet", clade_set="A(H3N2)"
        )


def test_inputs_for_an_unconfigured_subtype_are_refused(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    text = config.read_text() + '\n[inputs.h1]\nalignment = "alignment.fasta"\n'
    config.write_text(text + 'leaves = "leaves.parquet"\n')
    with pytest.raises(Exception, match="h1"):
        stages.load_run_config(config)


def with_clades(config: Path, pin: str | None = None) -> None:
    """Point the config at a synthetic nomenclature clone (tests/clades/synthetic.py)."""
    build_clone(config.parent / "clones")
    text = config.read_text() + (
        'clade_set = "A(H3N2)"\nnomenclature = "clones"\nnomenclature_repository = "synthetic_HA"\n'
    )
    if pin is not None:
        text += f'clade_pin = "{pin}"\n'
    config.write_text(text)


def test_the_clades_step_publishes_clades_from_the_tree_version(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    with_clades(config)
    assert statuses(config) == dict.fromkeys((*TREE_STEPS, "clades"), "ran")
    store = Store.open(tmp_path / "p" / "store")
    tree_meta = i6.read_metadata(store.version_dir(store.current("trees", "h3/weekly")))
    assert tree_meta["clade_set_version"].startswith("synthetic_HA@")
    assert store.current("clades", "h3") is not None


def test_a_pin_move_re_runs_populate_publish_and_clades_in_order(tmp_path: Path) -> None:
    """WS4's guard refuses a tree labelled at another pin; the pipeline must never reach it.

    The new commit changes no subclade file, so only the pin (a parameter) says anything moved.
    """
    config = make_project(tmp_path / "p")
    with_clades(config)
    statuses(config)
    clone = tmp_path / "p" / "clones" / "synthetic_HA"
    (clone / "README.md").write_text("a commit that changes no clade\n")
    subprocess.run(["git", "-C", str(clone), "add", "README.md"], check=True)
    subprocess.run(commit_command(clone, "readme"), check=True)
    outcomes = stages.run(stages.load_run_config(config))["h3"]
    assert [(o.name, o.status) for o in outcomes] == [
        ("build", "skipped"),
        ("asr", "skipped"),
        ("populate", "ran"),
        ("publish", "ran"),
        ("clades", "ran"),
    ]
    assert "parameters changed: clade_set_version" in outcomes[2].reason


def test_a_pin_the_clone_is_not_at_is_refused(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    with_clades(config, pin="0000000")
    with pytest.raises(Exception, match="not the pinned"):
        stages.run(stages.load_run_config(config))


def test_a_new_clade_file_re_runs_populate(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    with_clades(config)
    statuses(config)
    clone = tmp_path / "p" / "clones" / "synthetic_HA"
    write_clade(
        clone / "subclades",
        "P.9",
        "name: P.9\nparent: P\ndefining_mutations:\n- locus: HA1\n  position: 20\n  state: Y\n",
    )
    subprocess.run(["git", "-C", str(clone), "add", "."], check=True)
    subprocess.run(commit_command(clone, "P.9"), check=True)
    ran = {o.name for o in stages.run(stages.load_run_config(config))["h3"] if o.status == "ran"}
    assert ran == {"populate", "publish", "clades"}
