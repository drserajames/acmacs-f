"""Loading geo's clade colouring from the clade store and the user's scheme tables."""

from pathlib import Path

import pytest

from af.serology.outputs import SchemeChoice, clade_colouring
from tests.clades.synthetic import load_synthetic
from tests.clades.test_clade_set_for import clones, published

SCHEME = "order\tkey\tlegend\tcolour\n1\tP\tClade P\t#aa0000\n2\tP.1\tClade P.1\t#0000aa\n"


def test_colouring_uses_the_clade_set_the_table_was_labelled_with(tmp_path: Path) -> None:
    directory = clones(tmp_path)
    store, _ = published(tmp_path, load_synthetic(directory))
    schemes = tmp_path / "colours" / "h3"
    schemes.mkdir(parents=True)
    (schemes / "clades.tsv").write_text(SCHEME)
    colouring = clade_colouring(store, directory, {"A(H3N2)": SchemeChoice(schemes, "clades")})
    h3 = colouring["A(H3N2)"]
    assert h3.scheme.keys == ("P", "P.1")
    assert h3.clade_set.version == load_synthetic(directory).version
    assert h3.group_set is None

    with pytest.raises(ValueError, match="no colour scheme 'clades-v9'"):
        clade_colouring(store, directory, {"A(H3N2)": SchemeChoice(schemes, "clades-v9")})
    with pytest.raises(ValueError, match="no clade set for table subtype"):
        clade_colouring(store, directory, {"A(H5N1)": SchemeChoice(schemes, "clades")})
