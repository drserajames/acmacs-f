"""The real ``clades/local.tsv`` against the pinned nomenclature and its source.

Until switch-over, the local clades af carries are also defined in ``acmacs-data``'s
``clades.json`` — as whole signatures without a parent — and that file stays the one ae
edits. ``local.tsv`` adds the parent and stores each clade's own mutations: its signature
minus the parent's cumulative signature. This test recomputes that subtraction, so an
edit on either side fails here instead of the two copies quietly disagreeing.

Skips when acmacs-f-data, the nomenclature clones or acmacs-data are absent. No clade
name appears in this file: the rows are read from the data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from af.clades.local import extend_from_file, load_local_clades
from af.clades.nomenclature import load_clade_set

CLONES = Path.home() / "AC/eu/influenza-clade-nomenclature"
ACMACS_DATA = Path.home() / "AC/eu/acmacs-data"


def local_file(af_data: Path) -> Path:
    path = af_data / "clades" / "local.tsv"
    if not path.is_file():
        pytest.skip(f"no local clades at {path}")
    if not CLONES.is_dir():
        pytest.skip(f"nomenclature clones not found at {CLONES}")
    return path


def test_local_clades_load_onto_the_pinned_nomenclature(af_data: Path) -> None:
    path = local_file(af_data)
    for subtype, rows in load_local_clades(path).items():
        extended = extend_from_file(load_clade_set(subtype, CLONES), path)
        assert {row.name for row in rows} <= set(extended.local_names)


def test_local_mutations_match_their_source_signature(af_data: Path) -> None:
    path = local_file(af_data)
    source = ACMACS_DATA / "clades.json"
    if not source.is_file():
        pytest.skip(f"{source} not found")
    old = json.loads(source.read_text())
    checked = 0
    for subtype, rows in load_local_clades(path).items():
        clade_set = load_clade_set(subtype, CLONES)
        signatures = {
            entry["N"]: entry["aa"].split()
            for entry in old.get(subtype, [])
            if isinstance(entry, dict) and "N" in entry and "aa" in entry
        }
        for row in rows:
            assert row.name in signatures, f"{row.name}: not in {source} any more"
            assert row.parent is not None, f"{row.name}: a carried-over clade needs its parent"
            parent = {
                position: state
                for (alphabet, position), state in clade_set.cumulative(row.parent).items()
                if alphabet == "aa"
            }
            own = {t for t in signatures[row.name] if parent.get(int(t[:-1])) != t[-1]}
            stored = {f"{m.position}{m.state}" for m in row.mutations}
            assert stored == own, f"{row.name}: local.tsv {sorted(stored)}, source {sorted(own)}"
            checked += 1
    assert checked
