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


def _store_config(tmp_path: Path) -> MapsConfig:
    return MapsConfig(
        defaults=Defaults(dt.date(2025, 9, 1), (WindowConfig("all", None),)),
        frames=(FrameConfig("B", 15.0, "HI"),),
        maps=(MapConfig("example-lab", "test", chain="lab/example/main"),),
        colouring=ColouringConfig(
            "store", acmacs_data=tmp_path / "acmacs-data", nomenclature=tmp_path / "clones"
        ),
        af_data=tmp_path / "af-data",
    )


def test_no_build_while_a_batch_is_writing_the_store(tmp_path: Path) -> None:
    """A build reads the store under one guard (af.store.busy): while a batch holds the store,
    it refuses before reading anything, unless --ignore-busy says to read anyway."""
    from af.store.busy import StoreBusy

    store = Store.create(tmp_path / "store")
    config = _store_config(tmp_path)
    built: list[str] = []
    with store.batch("sequences-sweep", ["sequences/h3"]):
        with pytest.raises(StoreBusy, match="sequences-sweep"):
            build(
                config, store_root=store.root, out_root=tmp_path / "out", vaccine_list=[],
                vaccine_defaults={}, log=built.append,
            )  # fmt: skip
        assert built == [] and not (tmp_path / "out").exists()
        # --ignore-busy gets past the guard, to the next check (here: the serology store)
        with pytest.raises(BuildError, match="serology"):
            build(
                config, store_root=store.root, out_root=tmp_path / "out", vaccine_list=[],
                vaccine_defaults={}, log=built.append, ignore_busy=True,
            )  # fmt: skip


def test_store_colours_refuses_a_busy_store_before_reading(tmp_path: Path) -> None:
    from af.map.colouring import StoreColours
    from af.seq.matching_rules import matching_rules
    from af.store.busy import StoreBusy
    from tests.seq.test_matching_rules import write_af_data

    store = Store.create(tmp_path / "store")
    rules = matching_rules(write_af_data(tmp_path / "af-data"))
    with store.batch("clades-refresh", ["clades/h3"]), pytest.raises(StoreBusy, match="clades"):
        StoreColours(store, _store_config(tmp_path).colouring, rules)


def test_the_command_line_says_when_the_store_is_busy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A busy store is a clean error from the command line, not a traceback."""
    from af.map.build import main
    from tests.map.test_roundconfig import GOOD

    store = Store.create(tmp_path / "store")
    (tmp_path / "maps.toml").write_text(GOOD)  # a valid round config, so the build gets going
    (tmp_path / "vaccines.py").write_text("")
    # the curated list's own format is not what this test is about
    monkeypatch.setattr("af.map.roundconfig.load_vaccine_list", lambda path: {})
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "vaccine-defaults.toml").write_text("")
    with store.batch("sweep", ["sequences/h3"]):
        code = main(["--config", str(tmp_path / "maps.toml"), "--out", str(tmp_path / "out"),
                     "--store", str(store.root)])  # fmt: skip
    assert code == 1
    assert "being written by" in capsys.readouterr().err  # the guard's message, not a traceback
