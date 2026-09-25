"""The tree stage's TOML config. Invented subtypes and strain names only."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.tree.config import (
    LongBranchSetting,
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
long_branch_reason = "an invented override for this test"
report_cutoff = "2021-01-01"

[subtypes.h1]
outgroup = "EPI_ISL_70002|EPI70002"
branch_scale = "ml"
asr_backend = "treetime"

[long_branch.ml]
threshold = 0.01
reason = "an invented calibration for this test"
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


def test_a_scale_row_applies_to_every_subtype_on_that_scale(tmp_path: Path) -> None:
    settings = load_tree_config(write(tmp_path, EXAMPLE))
    h1 = settings.for_subtype("h1")
    assert h1.long_branch == 0.01
    rule = h1.long_branch_rule()
    assert rule is not None and "[long_branch.ml] an invented calibration" in rule.why


def test_a_subtype_override_beats_its_scale_row_and_carries_its_own_reason(tmp_path: Path) -> None:
    text = EXAMPLE + '\n[long_branch.mutations]\nthreshold = 0.004\nreason = "invented"\n'
    h3 = load_tree_config(write(tmp_path, text)).for_subtype("h3")
    rule = h3.long_branch_rule()
    assert rule is not None and rule.threshold == 0.006
    assert "an invented override" in rule.why


def test_a_scale_with_no_row_drops_nothing_and_says_why() -> None:
    """No threshold in code: a scale config does not calibrate is never given an invented one."""
    settings = SubtypeSettings(outgroup="X", branch_scale="mutations")
    assert settings.long_branch is None
    assert settings.long_branch_rule() is None
    assert "no [long_branch.mutations] row" in str(settings.no_long_branch_reason())


def test_a_threshold_without_its_reason_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TreeConfigError, match="reason"):
        SubtypeSettings(outgroup="X", long_branch_threshold=0.006)
    with pytest.raises(TreeConfigError, match="reason"):
        LongBranchSetting(threshold=0.01, reason=" ")


def test_a_row_for_an_unknown_scale_is_refused(tmp_path: Path) -> None:
    text = EXAMPLE + '\n[long_branch.furlongs]\nthreshold = 1.0\nreason = "x"\n'
    with pytest.raises((ConfigError, TreeConfigError), match="furlongs"):
        load_tree_config(write(tmp_path, text))


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
