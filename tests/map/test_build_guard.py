"""A store-coloured map build refuses a serology store that is behind the tables store."""

import datetime as dt
from pathlib import Path

import pytest

from af.map.build import BuildError, build
from af.map.config import (
    ColouringConfig,
    Defaults,
    FrameConfig,
    MapConfig,
    MapsConfig,
    WindowConfig,
)
from af.store import Store


def test_stale_serology_stops_the_build_before_any_map(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    config = MapsConfig(
        defaults=Defaults(dt.date(2025, 9, 1), (WindowConfig("all", None),)),
        frames=(FrameConfig("B", 15.0, "HI"),),
        maps=(MapConfig("example-lab", "test", chain="lab/example/main"),),
        colouring=ColouringConfig(
            "store",
            acmacs_data=tmp_path / "acmacs-data",
            nomenclature=tmp_path / "clones",
        ),
        af_data=tmp_path / "af-data",
    )
    built: list[str] = []
    with pytest.raises(BuildError, match="serology"):
        build(
            config,
            store_root=store.root,
            out_root=tmp_path / "out",
            vaccine_list=[],
            vaccine_defaults={},
            log=built.append,
        )
    assert built == []  # nothing was drawn from the stale store
    assert not (tmp_path / "out").exists()
