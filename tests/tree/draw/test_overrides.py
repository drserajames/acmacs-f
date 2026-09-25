"""Named overrides read from TOML (invented clade names only)."""

import pytest

pytest.importorskip("numpy", reason="numpy not installed: af.tree.draw needs it")

from af.tree.draw.overrides import OverrideFileError, load_overrides  # noqa: E402
from af.util.config import ConfigError  # noqa: E402

ROW = 'clade = "{c}"\nreason = "{r}"\ndecided = "someone, 1 Jan 2000"\n'


def write(tmp_path, text):
    path = tmp_path / "overrides.toml"
    path.write_text(text)
    return path


def test_rows_become_overrides_with_reasons(tmp_path):
    path = write(tmp_path, "[[subtypes.t1.hide_clades]]\n" + ROW.format(c="X.2", r="repeats X"))
    o = load_overrides(path, "t1")
    assert o.hide_clades == frozenset({"X.2"}) and not o.show_clades
    assert o.reasons == {"hide:X.2": "repeats X (someone, 1 Jan 2000)"}
    assert load_overrides(path, "t2").hide_clades == frozenset()  # no table: no overrides


def test_a_row_without_a_reason_is_rejected(tmp_path):
    path = write(tmp_path, '[[subtypes.t1.hide_clades]]\nclade = "X.2"\ndecided = "someone"\n')
    with pytest.raises(ConfigError, match="reason"):
        load_overrides(path, "t1")


def test_show_and_hide_the_same_clade_is_rejected(tmp_path):
    text = "[[subtypes.t1.hide_clades]]\n" + ROW.format(c="X.2", r="a")
    text += "[[subtypes.t1.show_clades]]\n" + ROW.format(c="X.2", r="b")
    with pytest.raises(OverrideFileError):
        load_overrides(write(tmp_path, text), "t1")
