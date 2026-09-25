"""Local clades whose signature is read from the source they were carried over from.

The local file then holds only what the source lacks (parent, scope, reason), so the
signature keeps one editable copy. Synthetic nomenclature: ``P`` → ``P.1`` → ``P.1.1``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from af.clades.importer import ImportError_, clades_json_signatures
from af.clades.local import LocalCladeError, extend_from_file, load_local_clades
from af.clades.nomenclature import CladeSet

from .synthetic import build_clone, load_synthetic

HEADER = "subtype\tname\tparent\tscope\tnote\n"
SUBTYPE = "A(H3N2)"


def write_local(tmp_path: Path, *rows: str, header: str = HEADER) -> Path:
    path = tmp_path / "local.tsv"
    path.write_text(header + "".join(row + "\n" for row in rows))
    return path


def synthetic(tmp_path: Path) -> CladeSet:
    return load_synthetic(build_clone(tmp_path).parent)


def parent_tokens(clade_set: CladeSet, parent: str) -> list[str]:
    """The parent's cumulative signature, spelled the way the old source spells one."""
    return [
        f"{position}{state}" if alphabet == "aa" else f"nuc{position}{state}"
        for (alphabet, position), state in clade_set.cumulative(parent).items()
    ]


def test_own_mutations_are_the_signature_minus_the_parents(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    path = write_local(tmp_path, f"{SUBTYPE}\tP.1.x\tP.1\tactive\tcarried over")
    signature = [*parent_tokens(clade_set, "P.1"), "20R", "nuc300G"]
    extended = extend_from_file(clade_set, path, signatures={"P.1.x": signature})
    assert sorted(str(m) for m in extended["P.1.x"].mutations) == ["20R", "nuc 300G"]
    assert extended.is_local("P.1.x")


def test_the_version_follows_the_signatures_used_and_only_those(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    path = write_local(tmp_path, f"{SUBTYPE}\tP.1.x\tP.1\tactive\t")
    base = parent_tokens(clade_set, "P.1")

    def version(signatures: dict[str, list[str]]) -> str:
        return extend_from_file(clade_set, path, signatures=signatures).version

    first = version({"P.1.x": [*base, "20R"], "other": ["1A"]})
    assert first == version({"P.1.x": [*base, "20R"], "other": ["2C"]})
    assert first != version({"P.1.x": [*base, "20K"], "other": ["1A"]})


def test_a_name_the_source_lacks_is_an_error(tmp_path: Path) -> None:
    path = write_local(tmp_path, f"{SUBTYPE}\tP.1.x\tP.1\tactive\t")
    with pytest.raises(LocalCladeError, match="no source signature"):
        extend_from_file(synthetic(tmp_path), path, signatures={})


def test_mutations_here_and_in_the_source_are_two_copies(tmp_path: Path) -> None:
    header = "subtype\tname\tparent\tmutations\tscope\tnote\n"
    path = write_local(tmp_path, f"{SUBTYPE}\tP.1.x\tP.1\t20R\tactive\t", header=header)
    with pytest.raises(LocalCladeError, match="keep one copy"):
        extend_from_file(synthetic(tmp_path), path, signatures={"P.1.x": ["20R"]})


def test_the_parent_must_be_published(tmp_path: Path) -> None:
    path = write_local(tmp_path, f"{SUBTYPE}\tL0\t-\tactive\t")
    with pytest.raises(LocalCladeError, match="must be a published clade"):
        extend_from_file(synthetic(tmp_path), path, signatures={"L0": ["20R"]})


def test_a_signature_adding_nothing_to_its_parent_is_an_error(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    path = write_local(tmp_path, f"{SUBTYPE}\tP.1.x\tP.1\tactive\t")
    with pytest.raises(LocalCladeError, match="adds nothing"):
        extend_from_file(clade_set, path, signatures={"P.1.x": parent_tokens(clade_set, "P.1")})


def test_a_file_without_a_mutations_column_loads(tmp_path: Path) -> None:
    path = write_local(tmp_path, f"{SUBTYPE}\tP.1.x\tP.1\tactive\twhy")
    [row] = load_local_clades(path)[SUBTYPE]
    assert (row.name, row.parent, row.mutations, row.note) == ("P.1.x", "P.1", (), "why")


def test_clades_json_signatures_reads_amino_acids_and_nucleotides(tmp_path: Path) -> None:
    path = tmp_path / "clades.json"
    entries = [{"N": "P.9", "aa": "20V 30K", "nuc": "300G"}, {"?N": ""}, "note"]
    path.write_text(json.dumps({"  version": "v", SUBTYPE: entries}))
    assert clades_json_signatures(path) == {SUBTYPE: {"P.9": ("20V", "30K", "nuc300G")}}


def test_clades_json_signatures_refuses_a_name_defined_twice_differently(tmp_path: Path) -> None:
    path = tmp_path / "clades.json"
    entries = [{"N": "P.9", "aa": "20V"}, {"N": "P.9", "aa": "20K"}]
    path.write_text(json.dumps({SUBTYPE: entries}))
    with pytest.raises(ImportError_, match="twice"):
        clades_json_signatures(path)
