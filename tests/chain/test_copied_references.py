"""ae's "cheating assay" is called "copied references" (Sarah, 6 Oct 2026); published chains keep
their step parameters, so the rename reruns nothing."""

import dataclasses

import pytest

from af.chain.config import ChainConfigError, MapOptions, load_chain_config, option_parameters

from .test_engine import GROUP


def test_the_option_is_a_parameter_under_its_published_name_in_the_same_place():
    params = option_parameters(MapOptions())
    assert "combine_copied_references" not in params
    assert list(params) == [
        "combine_cheating_assays" if k == "combine_copied_references" else k
        for k in (f.name for f in dataclasses.fields(MapOptions))
        if k in params or k == "combine_copied_references"
    ]
    off = option_parameters(MapOptions(combine_copied_references=False))
    assert off["combine_cheating_assays"] is False


def test_a_chain_file_with_the_old_name_is_told_the_new_one(tmp_path):
    path = tmp_path / "chain.toml"
    path.write_text(
        f'name = "{GROUP}"\nseed = 3\n[tables]\ndirectory = "tables"\ngroup = "{GROUP}"\n'
        '[options]\nminimum_column_basis = "none"\ncombine_cheating_assays = false\n'
    )
    with pytest.raises(ChainConfigError, match="now called combine_copied_references"):
        load_chain_config(path)
