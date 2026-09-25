"""The fallback for sequences with no tree, and the guard nothing else performs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from af.clades.assign import Assignment
from af.clades.fallback import (
    FallbackError,
    assign_from_rows,
    assign_from_tsv,
    check_dataset_agrees,
    clades_the_dataset_cannot_assign,
    dataset_subclades,
    disagreements,
)
from af.clades.nomenclature import CladeSet

from .synthetic import build_clone, load_synthetic

SUBTYPE = "A(H3N2)"


def clade_set(tmp_path: Path) -> CladeSet:
    return load_synthetic(build_clone(tmp_path / "clone").parent)


def write_dataset(tmp_path: Path, *clades: str) -> Path:
    """A Nextclade dataset directory, reduced to the one file this module reads."""
    directory = tmp_path / "dataset"
    directory.mkdir(exist_ok=True)
    children = [{"node_attrs": {"subclade": {"value": name}}} for name in clades]
    tree = {"tree": {"node_attrs": {"subclade": {"value": "P"}}, "children": children}}
    (directory / "tree.json").write_text(json.dumps(tree))
    return directory


def write_tsv(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path = tmp_path / "nextclade.tsv"
    lines = ["seqName\tsubclade\tqc.overallStatus"]
    lines.extend("\t".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n")
    return path


def test_reads_the_clades_a_dataset_can_assign(tmp_path: Path) -> None:
    assert dataset_subclades(write_dataset(tmp_path, "P.1", "P.2")) == {"P", "P.1", "P.2"}


def test_a_dataset_without_a_tree_is_fatal(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FallbackError, match="no tree.json"):
        dataset_subclades(tmp_path / "empty")


def test_a_dataset_built_from_another_nomenclature_is_refused(tmp_path: Path) -> None:
    """The guard this module exists for: the dataset and the nomenclature are pinned
    separately and nothing else checks that they agree. If they drift, the fallback and
    the tree engine name clades from two different vocabularies."""
    with pytest.raises(FallbackError, match="out of step"):
        check_dataset_agrees(write_dataset(tmp_path, "P.1", "NEWER.9"), clade_set(tmp_path))


def test_a_matching_dataset_is_accepted(tmp_path: Path) -> None:
    shared = check_dataset_agrees(write_dataset(tmp_path, "P.1", "P.2"), clade_set(tmp_path))
    assert shared == {"P", "P.1", "P.2"}


def test_a_clade_the_dataset_lacks_is_reported_not_refused(tmp_path: Path) -> None:
    """A clade designated after the dataset was built simply has nothing placed in it
    yet; the tree engine still assigns it."""
    clades = clade_set(tmp_path)
    absent = clades_the_dataset_cannot_assign(write_dataset(tmp_path, "P.1"), clades)
    assert "P.2" in absent
    assert "P.1" not in absent


def test_revoked_clades_are_not_reported_as_absent(tmp_path: Path) -> None:
    """A dataset not offering a withdrawn name is correct; listing them every run would
    bury the one that matters."""
    clades = clade_set(tmp_path)
    assert "P.3" in clades.names and clades["P.3"].revoked
    assert "P.3" not in clades_the_dataset_cannot_assign(write_dataset(tmp_path, "P.1"), clades)


def test_assigns_from_rows(tmp_path: Path) -> None:
    assignments, counts = assign_from_rows(
        [{"seqName": "v1", "subclade": "P.1", "qc.overallStatus": "good"}], clade_set(tmp_path)
    )
    assert assignments["v1"].clade == "P.1"
    assert counts.assigned == 1 and counts.unassigned == 0


def test_unassigned_is_an_answer_not_a_failure(tmp_path: Path) -> None:
    """On the round's H1 tree 10,942 leaves are ancestral to the nomenclature's own root
    clade; Nextclade calls every one of them unassigned, and that is correct."""
    assignments, counts = assign_from_rows(
        [
            {"seqName": "v1", "subclade": "unassigned", "qc.overallStatus": "good"},
            {"seqName": "v2", "subclade": "", "qc.overallStatus": "good"},
        ],
        clade_set(tmp_path),
    )
    assert assignments["v1"].clade is None and assignments["v2"].clade is None
    assert counts.unassigned == 2 and counts.assigned == 0


def test_bad_qc_is_counted_not_dropped(tmp_path: Path) -> None:
    _, counts = assign_from_rows(
        [{"seqName": "v1", "subclade": "P.1", "qc.overallStatus": "bad"}], clade_set(tmp_path)
    )
    assert counts.qc_bad == 1 and counts.assigned == 1


def test_a_clade_the_nomenclature_does_not_define_is_fatal(tmp_path: Path) -> None:
    with pytest.raises(FallbackError, match="not from the checked dataset"):
        assign_from_rows(
            [{"seqName": "v1", "subclade": "NEWER.9", "qc.overallStatus": "good"}],
            clade_set(tmp_path),
        )


def test_a_repeated_sequence_is_fatal(tmp_path: Path) -> None:
    with pytest.raises(FallbackError, match="appears twice"):
        assign_from_rows(
            [
                {"seqName": "v1", "subclade": "P.1", "qc.overallStatus": "good"},
                {"seqName": "v1", "subclade": "P.2", "qc.overallStatus": "good"},
            ],
            clade_set(tmp_path),
        )


def test_output_without_clade_columns_is_fatal(tmp_path: Path) -> None:
    """Nextclade can be asked for columns that leave the clade out; silently returning no
    clades would look like a nomenclature that names nothing."""
    with pytest.raises(FallbackError, match="none of the columns"):
        assign_from_rows([{"seqName": "v1", "qc.overallStatus": "good"}], clade_set(tmp_path))


def test_empty_output_is_fatal(tmp_path: Path) -> None:
    with pytest.raises(FallbackError, match="no rows"):
        assign_from_rows([], clade_set(tmp_path))


def test_identities_map_nextclade_names_to_af_ids(tmp_path: Path) -> None:
    assignments, _ = assign_from_rows(
        [{"seqName": "v1", "subclade": "P.1", "qc.overallStatus": "good"}],
        clade_set(tmp_path),
        identities={"v1": "EPI_ISL_1"},
    )
    assert list(assignments) == ["EPI_ISL_1"]


def test_assign_from_tsv_checks_the_dataset_first(tmp_path: Path) -> None:
    tsv = write_tsv(tmp_path, [("v1", "P.1", "good")])
    dataset = write_dataset(tmp_path, "P.1", "NEWER.9")
    with pytest.raises(FallbackError, match="out of step"):
        assign_from_tsv(tsv, clade_set(tmp_path), dataset_dir=dataset)


def test_assign_from_tsv_reads_the_step_output(tmp_path: Path) -> None:
    tsv = write_tsv(tmp_path, [("v1", "P.1", "good"), ("v2", "unassigned", "mediocre")])
    assignments, counts = assign_from_tsv(
        tsv, clade_set(tmp_path), dataset_dir=write_dataset(tmp_path, "P.1", "P.2")
    )
    assert assignments["v1"].clade == "P.1"
    assert counts.sequences == 2 and counts.unassigned == 1


def test_disagreements_ignore_the_tree_being_more_specific(tmp_path: Path) -> None:
    """The tree engine is often deeper, which is the point of it; only genuinely
    different clades count."""
    clades = clade_set(tmp_path)
    tree = {"v1": Assignment("v1", "P.1.1"), "v2": Assignment("v2", "P.1")}
    fallback = {"v1": Assignment("v1", "P.1"), "v2": Assignment("v2", "P.2")}
    assert disagreements(tree, fallback, clades) == {"v2": ("P.1", "P.2")}


def test_disagreements_skip_sequences_only_one_side_has(tmp_path: Path) -> None:
    clades = clade_set(tmp_path)
    tree = {"v1": Assignment("v1", "P.1")}
    assert disagreements(tree, {}, clades) == {}
