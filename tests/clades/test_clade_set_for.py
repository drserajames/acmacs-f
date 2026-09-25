"""Loading the clade set a published clade table was built with. Synthetic data only."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from af.clades.local import extend_from_file
from af.clades.nomenclature import CladeSet, NomenclatureError
from af.clades.store import CladeRow, CladeStoreError, clade_set_for, publish
from af.store import Store, StoreRef

from .synthetic import build_clone, commit_command, load_synthetic
from .test_store import STARTED, nomenclature_input, sequences_ref

SUBTYPE = "A(H3N2)"
LOCAL = "subtype\tname\tparent\tmutations\tscope\tnote\nA(H3N2)\tP.1.x\tP.1\t20R\tactive\twhy\n"


def published(tmp_path: Path, clades: CladeSet) -> tuple[Store, StoreRef]:
    store = Store.create(tmp_path / "store")
    rows = [CladeRow("EPI_ISL_1", "EPI1", SUBTYPE, "P.1", "fallback")]
    ref = publish(
        store,
        SUBTYPE,
        rows,
        clades,
        labelled=sequences_ref(),
        nomenclature=[nomenclature_input(tmp_path)],
        started=STARTED,
    )
    return store, ref


def clones(tmp_path: Path) -> Path:
    return build_clone(tmp_path / "clone").parent


def test_loads_the_pinned_set_the_table_records(tmp_path: Path) -> None:
    directory = clones(tmp_path)
    clades = load_synthetic(directory)
    store, ref = published(tmp_path, clades)
    loaded = clade_set_for(store, ref, directory)
    assert loaded.version == clades.version
    assert loaded.names == clades.names


def test_with_the_local_layer_the_same_file_must_be_given(tmp_path: Path) -> None:
    directory = clones(tmp_path)
    local = tmp_path / "local.tsv"
    local.write_text(LOCAL)
    clades = extend_from_file(load_synthetic(directory), local)
    store, ref = published(tmp_path, clades)
    assert clade_set_for(store, ref, directory, local=local).is_local("P.1.x")
    with pytest.raises(CladeStoreError, match="give the local.tsv it used"):
        clade_set_for(store, ref, directory)
    local.write_text(LOCAL.replace("20R", "21R"))
    with pytest.raises(CladeStoreError, match="the table records"):
        clade_set_for(store, ref, directory, local=local)


def test_a_local_file_for_a_table_built_without_one_is_an_error(tmp_path: Path) -> None:
    directory = clones(tmp_path)
    store, ref = published(tmp_path, load_synthetic(directory))
    local = tmp_path / "local.tsv"
    local.write_text(LOCAL)
    with pytest.raises(CladeStoreError, match="built without local clades"):
        clade_set_for(store, ref, directory, local=local)


def test_a_clone_that_has_moved_on_is_an_error(tmp_path: Path) -> None:
    directory = clones(tmp_path)
    store, ref = published(tmp_path, load_synthetic(directory))
    clone = directory / "synthetic_HA"
    (clone / "README").write_text("moved\n")
    subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True)
    subprocess.run(commit_command(clone, "move"), check=True)
    with pytest.raises(NomenclatureError):
        clade_set_for(store, ref, directory)


def test_only_a_clades_version(tmp_path: Path) -> None:
    directory = clones(tmp_path)
    store, _ = published(tmp_path, load_synthetic(directory))
    with pytest.raises(CladeStoreError, match="expected a clades version"):
        clade_set_for(store, sequences_ref(), directory)
