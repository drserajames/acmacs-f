"""A clade renamed upstream by aliasing is resolved from upstream's own ``unaliased_name``.

Synthetic names only. The nomenclature here has ``X`` aliasing the path ``A.1`` and a
subclade ``X.2.1`` since renamed ``Z`` (same path ``A.1.2.1``), the shape of a real alias.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from af.clades.fallback import FallbackError, assign_from_rows, check_dataset_agrees
from af.clades.nomenclature import CladeSet, load_clade_set

from .synthetic import commit_command, write_clade

MUT = "defining_mutations:\n- locus: HA1\n  position: {pos}\n  state: {state}\nclade: none"


def build(tmp_path: Path, *, twin: bool = False) -> CladeSet:
    root = tmp_path / "alias_HA"
    sub = root / "subclades"
    write_clade(
        sub, "X", f"name: X\nunaliased_name: A.1\nparent: none\n{MUT.format(pos=5, state='K')}"
    )
    write_clade(
        sub, "X.2", f"name: X.2\nunaliased_name: A.1.2\nparent: X\n{MUT.format(pos=7, state='R')}"
    )
    write_clade(
        sub, "Z", f"name: Z\nunaliased_name: A.1.2.1\nparent: X.2\n{MUT.format(pos=9, state='T')}"
    )
    write_clade(
        sub, "Y", f"name: Y\nunaliased_name: B.1\nparent: none\n{MUT.format(pos=11, state='N')}"
    )
    if twin:
        write_clade(
            sub,
            "Z2",
            f"name: Z2\nunaliased_name: A.1.2.1\nparent: X.2\n{MUT.format(pos=12, state='N')}",
        )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(commit_command(root, "alias"), check=True)
    return load_clade_set("A(H3N2)", tmp_path, repository="alias_HA")


def row(name: str, clade: str) -> dict[str, str]:
    return {"seqName": name, "subclade": clade, "qc.overallStatus": "good"}


def test_an_older_name_resolves_through_its_prefix_path(tmp_path: Path) -> None:
    assert build(tmp_path).resolve_alias("X.2.1") == "Z"


def test_a_defined_name_resolves_to_itself(tmp_path: Path) -> None:
    assert build(tmp_path).resolve_alias("X.2") == "X.2"


def test_a_name_whose_path_matches_nothing_is_not_resolved(tmp_path: Path) -> None:
    clades = build(tmp_path)
    assert clades.resolve_alias("X.2.7") is None
    assert clades.resolve_alias("Q.1") is None


def test_two_clades_on_one_path_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="several clades"):
        build(tmp_path, twin=True).resolve_alias("X.2.1")


def test_the_fallback_resolves_and_counts_every_rename(tmp_path: Path) -> None:
    calls, counts = assign_from_rows(
        [row("v1", "X.2.1"), row("v2", "X.2.1"), row("v3", "Y"), row("v4", "Z")], build(tmp_path)
    )
    assert [calls[n].clade for n in ("v1", "v2", "v3", "v4")] == ["Z", "Z", "Y", "Z"]
    assert counts.renamed == (("X.2.1", "Z", 2),)
    assert counts.to_json()["renamed"] == [["X.2.1", "Z", 2]]


def test_an_unresolvable_name_is_still_fatal(tmp_path: Path) -> None:
    with pytest.raises(FallbackError, match="does not define"):
        assign_from_rows([row("v1", "X.2.7")], build(tmp_path))


def test_a_dataset_using_the_older_name_is_accepted_and_one_with_an_unknown_name_is_not(
    tmp_path: Path,
) -> None:
    import json

    def dataset(*names: str) -> Path:
        directory = tmp_path / "ds"
        directory.mkdir(exist_ok=True)
        children = [{"node_attrs": {"subclade": {"value": n}}} for n in names]
        tree = {"tree": {"node_attrs": {"subclade": {"value": "X"}}, "children": children}}
        (directory / "tree.json").write_text(json.dumps(tree))
        return directory

    clades = build(tmp_path)
    assert "Z" in check_dataset_agrees(dataset("X.2.1"), clades)
    with pytest.raises(FallbackError, match="out of step"):
        check_dataset_agrees(dataset("X.2.7"), clades)
