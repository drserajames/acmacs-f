"""Reading a round's maps.toml: what it accepts, and what it refuses."""

import datetime as dt
from pathlib import Path

import pytest

from af.map.roundconfig import ConfigError, load_maps_config

GOOD = """
vaccine_list = "vaccines.py"
af_data = "shared/acmacs-f-data"
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


# Invented names, built rather than written out (see tools/WHO-DATA-GATE.md).
OLD_ONE = "/".join(("OLDTOWN", "1", "2009"))
OLD_TWO = "/".join(("OLDTOWN", "2", "2010"))
VACCINE_RULES = f"""
  [[maps.vaccine_disable]]
  name = "{OLD_ONE}"
  reason = "superseded"
  decided = 2026-09-20

  [[maps.vaccine_choose]]
  name = "{OLD_TWO}"
  reason = "the reference preparation"
  passage_class = "cell"
  passage = "MDCK1"
"""


def test_vaccine_rules_may_say_when_they_were_decided(tmp_path: Path) -> None:
    config, _ = load_maps_config(write(tmp_path, GOOD + VACCINE_RULES))
    (m,) = config.maps
    assert m.vaccine_disable[0].decided == dt.date(2026, 9, 20)
    assert m.vaccine_choose[0].decided is None  # optional: older rules carry no date
    bad = GOOD + VACCINE_RULES.replace("2026-09-20", '"20 Sep"')
    with pytest.raises(ConfigError, match="vaccine_disable"):
        load_maps_config(write(tmp_path, bad))


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


STORE_COLOURING = """
[colouring]
source = "store"
acmacs_data = "shared/acmacs-data"
nomenclature = "shared/nomenclature"
"""


def test_colouring_defaults_to_the_stand_in(tmp_path: Path) -> None:
    config, _ = load_maps_config(write(tmp_path, GOOD))
    assert config.colouring.source == "stand-in"


def test_store_colouring_resolves_its_paths(tmp_path: Path) -> None:
    config, _ = load_maps_config(write(tmp_path, GOOD + STORE_COLOURING))
    c = config.colouring
    assert c.source == "store"
    assert c.acmacs_data == (tmp_path / "shared/acmacs-data").resolve()
    assert config.af_data == (tmp_path / "shared/acmacs-f-data").resolve()


def test_colouring_source_must_be_said(tmp_path: Path) -> None:
    text = GOOD + STORE_COLOURING.replace('source = "store"\n', "")
    with pytest.raises(ConfigError, match="source is required"):
        load_maps_config(write(tmp_path, text))


def test_store_colouring_needs_its_inputs(tmp_path: Path) -> None:
    text = GOOD + STORE_COLOURING.replace('nomenclature = "shared/nomenclature"\n', "")
    with pytest.raises(ConfigError, match="needs colouring.nomenclature"):
        load_maps_config(write(tmp_path, text))


def test_store_colouring_refuses_a_scheme_stand_in(tmp_path: Path) -> None:
    """Half a round on each source would draw one virus in two colours on facing pages."""
    text = (
        GOOD.replace(
            'layout_stand_in = "charts/example.ace"',
            'layout_stand_in = "charts/example.ace"\nscheme_stand_in = "charts/styled.ace"',
        )
        + STORE_COLOURING
    )
    with pytest.raises(ConfigError, match="still set scheme_stand_in: example-lab"):
        load_maps_config(write(tmp_path, text))


def test_unknown_colouring_source(tmp_path: Path) -> None:
    text = GOOD + STORE_COLOURING.replace('"store"', '"chart"')
    with pytest.raises(ConfigError, match="expected one of"):
        load_maps_config(write(tmp_path, text))


STAND_IN_DEFAULT = STORE_COLOURING.replace('source = "store"', 'source = "stand-in"')


def map_colouring(source: str) -> str:
    """GOOD with the example map's own colouring set."""
    scheme = 'clade_scheme = "clades-v2"'
    return GOOD.replace(scheme, f'{scheme}\ncolouring = "{source}"')


def test_one_map_can_switch_to_the_store(tmp_path: Path) -> None:
    """Sarah, 29 Sep: maps switch one at a time, each once its loss is within the limit."""
    text = map_colouring("store") + STAND_IN_DEFAULT
    config, _ = load_maps_config(write(tmp_path, text))
    assert config.colouring.source == "stand-in"
    assert config.colour_source(config.maps[0]) == "store"


def test_a_store_map_needs_the_store_inputs_even_under_a_stand_in_default(tmp_path: Path) -> None:
    text = map_colouring("store")
    with pytest.raises(ConfigError, match=r"store colouring \(example-lab\) needs"):
        load_maps_config(write(tmp_path, text))


def test_a_map_can_stay_on_the_stand_in_under_a_store_default(tmp_path: Path) -> None:
    text = (
        GOOD.replace(
            'layout_stand_in = "charts/example.ace"',
            'layout_stand_in = "charts/example.ace"\nscheme_stand_in = "charts/styled.ace"'
            '\ncolouring = "stand-in"',
        )
        + STORE_COLOURING
    )
    config, _ = load_maps_config(write(tmp_path, text))
    assert config.colour_source(config.maps[0]) == "stand-in"


def test_unknown_map_colouring(tmp_path: Path) -> None:
    text = map_colouring("chart")
    with pytest.raises((ConfigError, ValueError), match="expected one of"):
        load_maps_config(write(tmp_path, text))


def test_af_data_is_required(tmp_path: Path) -> None:
    """One root for acmacs-f-data, required: the sera markers are read for every map."""
    text = GOOD.replace('af_data = "shared/acmacs-f-data"\n', "")
    with pytest.raises(ConfigError, match="af_data .* is required"):
        load_maps_config(write(tmp_path, text))


def test_colouring_may_not_carry_its_own_af_data(tmp_path: Path) -> None:
    text = GOOD + STORE_COLOURING + 'af_data = "elsewhere"\n'
    with pytest.raises(ConfigError, match="moved to the top-level af_data"):
        load_maps_config(write(tmp_path, text))


BLOCK = """
  [[maps.blocks]]
  name = "outliers to the clade"
  reason = "decided at the meeting"
  decided = 2026-10-07
  movers = ["ONE", "TWO"]
  target_legend = "a clade"
  max_stress_rise = 60.0
  settled_within = 2.0
  min_settled = 2
  max_other_move = 4.0
"""


def test_a_block_takes_a_fixed_shift_or_a_target(tmp_path: Path) -> None:
    fixed = BLOCK + 'shift = [1.0, -2.0]\nderived_from = "the old layout"\n'
    config, _ = load_maps_config(write(tmp_path, GOOD + fixed))
    (b,) = config.maps[0].blocks
    assert b.shift == (1.0, -2.0) and b.to is None
    config, _ = load_maps_config(write(tmp_path, GOOD + BLOCK + 'to = "target-median"\n'))
    (b,) = config.maps[0].blocks
    assert b.to == "target-median" and b.shift is None and b.derived_from is None


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ('to = "target-median"\nshift = [1.0, 2.0]\n', "computes the shift"),
        ('to = "somewhere"\n', "to must be"),
        ("shift = [1.0, 2.0]\n", "needs derived_from"),
        ("", "shift must be"),
    ],
)
def test_a_block_placement_is_checked_on_load(tmp_path: Path, extra: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_maps_config(write(tmp_path, GOOD + BLOCK + extra))
