"""A map drawn from a chain as it stood after a named table (reproducing a past round)."""

import datetime as dt
import json
from pathlib import Path
from typing import Any, cast

import pytest

from af.map.build import BuildError, chain_chart, chain_version_chart
from af.map.roundconfig import ConfigError, load_maps_config
from af.store import Provenance
from af.store.store import Store

from .test_roundconfig import GOOD, write


class _Ref:
    version = "v1"


class _Store:
    """The two calls chain_chart makes, over a chain written in a temporary directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def current(self, kind: str, dataset: str) -> Any:
        return _Ref()

    def resolve(self, ref: Any) -> Path:
        return self.root


def chain(tmp_path: Path, tables: list[str]) -> Store:
    steps = []
    for i, table in enumerate(tables):
        d = tmp_path / f"steps/{i:04d}"
        d.mkdir(parents=True)
        name = "/".join(("B", "OLDTOWN", str(i + 1), "2020"))
        chart = {
            "c": {"i": {"V": "B"}, "a": [{"N": name}], "s": [{"N": name}], "t": {"l": [["40"]]}}
        }
        (d / "chosen.ace").write_text(json.dumps(chart))
        directory = f"steps/{i:04d}"
        steps.append({"table_id": table, "directory": directory, "chosen_file": "chosen.ace"})
    (tmp_path / "chain.json").write_text(json.dumps({"steps": steps}))
    return cast(Store, _Store(tmp_path))


def test_until_takes_the_named_tables_step(tmp_path: Path) -> None:
    store = chain(tmp_path, ["lab-a-20260801", "lab-a-20260825", "lab-a-20260904"])
    last, _, _ = chain_chart(store, "lab/a/main")
    cut, _, ace = chain_chart(store, "lab/a/main", "lab-a-20260825")
    assert ace == tmp_path / "steps/0001/chosen.ace"
    assert cut.antigens[0].name != last.antigens[0].name


@pytest.mark.parametrize("tables", [["lab-a-1", "lab-a-2"], ["lab-a-2", "lab-a-9", "lab-a-9"]])
def test_until_must_match_exactly_one_step(tmp_path: Path, tables: list[str]) -> None:
    store = chain(tmp_path, tables)
    with pytest.raises(BuildError, match="matches [02] steps"):
        chain_chart(store, "lab/a/main", "lab-a-9" if "lab-a-9" in tables else "lab-a-7")


UNTIL = """
  [maps.chain_until]
  table = "lab-a-20260825"
  reason = "round built before the next table"
  decided = 2026-09-29
"""


def test_chain_until_loads_with_its_reason(tmp_path: Path) -> None:
    text = GOOD.replace('layout_stand_in = "charts/example.ace"', 'chain = "lab/a/main"' + UNTIL)
    config, _ = load_maps_config(write(tmp_path, text))
    until = config.maps[0].chain_until
    assert until is not None and until.table == "lab-a-20260825"
    assert until.reason == "round built before the next table"


def test_chain_until_needs_a_chain(tmp_path: Path) -> None:
    text = GOOD.replace(
        'layout_stand_in = "charts/example.ace"', 'layout_stand_in = "charts/example.ace"' + UNTIL
    )
    with pytest.raises((ConfigError, ValueError), match="chain_until needs a chain"):
        load_maps_config(write(tmp_path, text))


def test_a_pinned_version_is_read_after_current_moves_on(tmp_path: Path) -> None:
    """The previous report's map (af.map.new_antigens) is the version that report drew, not the
    chain's CURRENT: a later table must not change what counted as already shown."""
    store = Store.create(tmp_path / "store")
    t0 = dt.datetime(2026, 10, 2, tzinfo=dt.UTC)
    refs = []
    for i in range(2):
        name = "/".join(("B", "OLDTOWN", str(i + 1), "2020"))
        chart = {"c": {"i": {"V": "B"}, "a": [{"N": name}], "s": [{"N": name}],
                       "t": {"l": [["40"]]}}}  # fmt: skip
        with store.build("chains", "lab/a/main") as build:
            (build.path / "steps/0000").mkdir(parents=True)
            (build.path / "steps/0000/chosen.ace").write_text(json.dumps(chart))
            steps = [
                {"table_id": f"lab-a-{i}", "directory": "steps/0000", "chosen_file": "chosen.ace"}
            ]
            (build.path / "chain.json").write_text(json.dumps({"steps": steps}))
            refs.append(build.publish(Provenance("af.chain", (), {"i": i}, t0, t0)))
    old, ref, _ = chain_version_chart(store, "lab/a/main", refs[0].version)
    current, _, _ = chain_chart(store, "lab/a/main")
    assert ref == refs[0] and old.antigens[0].name.endswith("/1/2020")
    assert current.antigens[0].name.endswith("/2/2020")
    with pytest.raises(BuildError, match="no chain version chains/lab/a/main@0123456789abcdef"):
        chain_version_chart(store, "lab/a/main", "0123456789abcdef")
