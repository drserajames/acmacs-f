"""The real ``clades/local.tsv`` against the pinned nomenclature and its source.

Until switch-over, a carried-over local clade's signature stays in ``acmacs-data``'s
``clades.json`` — the one editable copy — and ``local.tsv`` adds only its parent, scope and
reason. This checks the two still fit together: every row finds its signature, and every
signature adds something to its parent's.

Skips when acmacs-f-data, the nomenclature clones or acmacs-data are absent. No clade
name appears in this file: the rows are read from the data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from af.clades.importer import clades_json_signatures
from af.clades.local import extend_from_file, load_local_clades
from af.clades.nomenclature import load_clade_set

CLONES = Path.home() / "AC/eu/influenza-clade-nomenclature"
SOURCE = Path.home() / "AC/eu/acmacs-data/clades.json"


def test_local_clades_load_with_their_source_signatures(af_data: Path) -> None:
    path = af_data / "clades" / "local.tsv"
    for required in (path, CLONES, SOURCE):
        if not required.exists():
            pytest.skip(f"{required} not found")
    signatures = clades_json_signatures(SOURCE)
    checked = 0
    for subtype, rows in load_local_clades(path).items():
        extended = extend_from_file(
            load_clade_set(subtype, CLONES), path, signatures=signatures.get(subtype, {})
        )
        for row in rows:
            assert extended.is_local(row.name)
            assert extended[row.name].mutations, f"{row.name}: no own mutations"
            checked += 1
    assert checked
