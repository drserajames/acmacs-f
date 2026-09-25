"""The tree stage's TOML config. Invented subtypes and strain names only."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.tree.config import (
    LONG_BRANCH_BY_SCALE,
    SubtypeSettings,
    TreeConfigError,
    load_tree_config,
)
from af.util.config import ConfigError

EXAMPLE = """
threads = 8

[cmaple]
search = "EXHAUSTIVE"
seed = 7

[subtypes.h3]
outgroup = "EPI_ISL_70001|EPI70001"
branch_scale = "mutations"
long_branch_threshold = 0.006
report_cutoff = "2021-01-01"

[subtypes.h1]
outgroup = "EPI_ISL_70002|EPI70002"
branch_scale = "ml"
asr_backend = "treetime"
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "trees.toml"
    path.write_text(text)
    return path


def test_a_config_loads_per_subtype_settings(tmp_path: Path) -> None:
    settings = load_tree_config(write(tmp_path, EXAMPLE))
    assert sorted(settings.subtypes) == ["h1", "h3"]
    h3 = settings.for_subtype("h3")
    assert h3.branch_scale == "mutations"
    assert h3.long_branch == 0.006
    assert h3.report_cutoff == "2021-01-01"
    assert settings.for_subtype("h1").asr_backend == "treetime"


def test_the_run_thread_count_reaches_cmaple(tmp_path: Path) -> None:
    settings = load_tree_config(write(tmp_path, EXAMPLE))
    assert settings.cmaple.seed == 7
    assert settings.cmaple_for("h3").threads == 8


def test_an_unknown_subtype_is_an_error_not_a_default(tmp_path: Path) -> None:
    settings = load_tree_config(write(tmp_path, EXAMPLE))
    with pytest.raises(TreeConfigError, match="no settings for subtype 'bvic'"):
        settings.for_subtype("bvic")


def test_a_config_with_no_subtypes_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(TreeConfigError, match="nothing to build"):
        load_tree_config(write(tmp_path, "threads = 2\n[subtypes]\n"))


def test_a_typo_is_reported_rather_than_left_at_its_default(tmp_path: Path) -> None:
    text = EXAMPLE.replace('branch_scale = "mutations"', 'branch_scal = "mutations"')
    with pytest.raises(ConfigError, match="branch_scal"):
        load_tree_config(write(tmp_path, text))


def test_a_bad_value_names_the_field(tmp_path: Path) -> None:
    text = EXAMPLE.replace('branch_scale = "mutations"', 'branch_scale = "furlongs"')
    with pytest.raises((ConfigError, TreeConfigError), match="furlongs"):
        load_tree_config(write(tmp_path, text))


def test_an_empty_outgroup_is_refused() -> None:
    with pytest.raises(TreeConfigError, match="cannot root"):
        SubtypeSettings(outgroup="")


def test_an_unknown_asr_backend_is_refused() -> None:
    with pytest.raises(TreeConfigError, match="asr_backend"):
        SubtypeSettings(outgroup="X", asr_backend="haruspicy")


def test_the_ml_scale_has_a_calibrated_default_threshold() -> None:
    settings = SubtypeSettings(outgroup="X", branch_scale="ml")
    assert settings.long_branch == LONG_BRANCH_BY_SCALE["ml"] == 0.01
    assert settings.no_long_branch_reason() is None
    rule = settings.long_branch_rule()
    assert rule is not None and "9 of 9 hand-hidden" in rule.why


def test_the_mutations_scale_has_no_default_and_will_not_invent_one() -> None:
    """Its only would-be threshold has no ground truth; guessing would invent a curation rule."""
    settings = SubtypeSettings(outgroup="X", branch_scale="mutations")
    assert settings.long_branch is None
    assert settings.long_branch_rule() is None
    assert "will not invent one" in str(settings.no_long_branch_reason())


def test_setting_the_threshold_explicitly_enables_the_drop_on_either_scale() -> None:
    settings = SubtypeSettings(outgroup="X", branch_scale="mutations", long_branch_threshold=0.006)
    rule = settings.long_branch_rule()
    assert rule is not None and rule.threshold == 0.006
    assert "Set explicitly in config" in rule.why
    assert settings.no_long_branch_reason() is None


def test_turning_the_drop_off_is_reported_as_a_reason() -> None:
    settings = SubtypeSettings(outgroup="X", drop_long_branches=False)
    assert settings.no_long_branch_reason() == "drop_long_branches is off in config"


def test_the_clock_still_reports_on_a_scale_with_no_drop_threshold() -> None:
    """No drop does not mean no report: the check runs at ae's number so nothing is silent."""
    settings = SubtypeSettings(outgroup="X", branch_scale="mutations")
    assert settings.clock_settings().branch_threshold == 0.01


def test_the_typed_scale_accessor_matches_what_populate_takes() -> None:
    from af.tree.populate import BRANCH_SCALES

    for scale in BRANCH_SCALES:
        assert SubtypeSettings(outgroup="X", branch_scale=scale).scale == scale
