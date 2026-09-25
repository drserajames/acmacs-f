"""Reading a round's maps.toml: what it accepts, and what it refuses."""

import datetime as dt
from pathlib import Path

import pytest

from af.map.roundconfig import ConfigError, load_maps_config

GOOD = """
vaccine_list = "vaccines.py"
vaccine_defaults = "shared/vaccine-defaults.toml"

[defaults]
must_show_since = 2025-09-01
previous_round = "../previous"

[[defaults.windows]]
name = "all"
[[defaults.windows]]
name = "12m"
since = 2025-09-01

[[frames]]
subtype = "A(H3N2)"
size = 25
[[frames]]
subtype = "B"
assay = "HI"
size = 15

[[maps]]
folder = "example-lab"
clade_scheme = "clades-v2"
layout_stand_in = "charts/example.ace"

  [[maps.moves]]
  name = "pull strays in"
  reason = "decided at the meeting"
  decided = 2026-09-22
  movers = ["ONE", "TWO"]
  target_legend = "a clade"
  max_stress_rise = 5.0
  max_from_target = 2.5

  [[maps.hides]]
  name = "H-1"
  reason = "outliers that did not hold"
  decided = 2026-09-23
  designations = ["ONE"]
"""


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "maps.toml"
    p.write_text(text)
    return p


def test_loads_and_resolves_paths_against_the_file(tmp_path: Path) -> None:
    config, vaccine_list = load_maps_config(write(tmp_path, GOOD))
    assert vaccine_list == (tmp_path / "vaccines.py").resolve()
    assert config.defaults.must_show_since == dt.date(2025, 9, 1)
    assert [w.name for w in config.defaults.windows] == ["all", "12m"]
    assert config.defaults.windows[0].since is None
    assert config.frame_size("A(H3N2)", "HINT") == 25  # subtype rule covers every assay
    assert config.frame_size("B", "HI") == 15
    one = config.maps[0]
    assert one.moves[0].decided == dt.date(2026, 9, 22)
    assert one.hides[0].load() == ("ONE",)
    assert one.layout_stand_in == (tmp_path / "charts/example.ace").resolve()


def test_unknown_key_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        load_maps_config(write(tmp_path, GOOD.replace("size = 25", "size = 25\nsized = 30")))


def test_frame_size_must_exist_for_the_subtype(tmp_path: Path) -> None:
    config, _ = load_maps_config(write(tmp_path, GOOD))
    with pytest.raises(ValueError, match="no frame size"):
        config.frame_size("A(H1N1)", "HI")


def test_a_map_needs_a_chain_or_a_stand_in(tmp_path: Path) -> None:
    text = GOOD.replace('layout_stand_in = "charts/example.ace"', "")
    with pytest.raises(ValueError, match="chain or a layout_stand_in"):
        load_maps_config(write(tmp_path, text))


def test_hide_designations_from_a_file(tmp_path: Path) -> None:
    (tmp_path / "hides.txt").write_text("# why\nONE\n\nTWO\n")
    text = GOOD.replace('designations = ["ONE"]', 'designations_file = "hides.txt"')
    config, _ = load_maps_config(write(tmp_path, text))
    assert config.maps[0].hides[0].load() == ("ONE", "TWO")


def test_windows_are_required(tmp_path: Path) -> None:
    text = "\n".join(
        ln
        for ln in GOOD.splitlines()
        if "windows" not in ln
        and ln.strip() not in ('name = "all"', 'name = "12m"', "since = 2025-09-01")
    )
    with pytest.raises(ConfigError, match="at least one window|must_show_since"):
        load_maps_config(write(tmp_path, text))
