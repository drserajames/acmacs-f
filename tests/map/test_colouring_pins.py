"""StoreColours version pins: what ``versions=`` accepts and what it refuses (design rule 1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.map.colouring import MapColouringError, Pins, pinned_refs
from af.store import Store
from tests.serology.test_outputs import _clades, _sequences

DATASETS = ("bvic", "byam", "h1", "h3")


def _store(tmp_path: Path) -> Store:
    """Sequences for every dataset, and two versions of clades/h3."""
    store = Store.create(tmp_path / "store")
    _sequences(store, tmp_path)
    _clades(store, tmp_path)
    _clades(store, tmp_path, extra=((2, "P.2"),))
    return store


def _all_sequences(store: Store) -> dict[str, str]:
    return {f"sequences/{d}": store.current("sequences", d).version for d in DATASETS}


def test_no_versions_pins_nothing(tmp_path: Path) -> None:
    pins = pinned_refs(_store(tmp_path), {})
    assert pins == Pins() and pins.to_json() == {}


def test_pins_name_the_versions_given_not_current(tmp_path: Path) -> None:
    store = _store(tmp_path)
    older = store.history("clades", "h3")[0]["version"]
    assert older != store.current("clades", "h3").version
    pins = pinned_refs(store, {**_all_sequences(store), "clades/h3": older})
    assert pins.clades is not None and [r.version for r in pins.clades] == [older]
    assert pins.sequences is not None and sorted(pins.sequences) == list(DATASETS)
    assert pins.serology is None  # not pinned: read CURRENT
    store.resolve(pins.clades[0])  # a ref the store accepts (manifest hash checked)
    assert pins.to_json()["clades/h3"] == older


@pytest.mark.parametrize(
    "key",
    ["tables/h3", "serology/h3", "sequences/nosuch", "clades/h1", "h3"],
)
def test_a_key_storecolours_does_not_read_is_refused(tmp_path: Path, key: str) -> None:
    store = _store(tmp_path)
    version = store.current("clades", "h3").version
    with pytest.raises(MapColouringError, match="does not read"):
        pinned_refs(store, {key: version})


@pytest.mark.parametrize("version", ["0123456789abcdef", "../../CURRENT", "short"])
def test_a_version_the_store_does_not_hold_is_refused(tmp_path: Path, version: str) -> None:
    with pytest.raises(MapColouringError, match="not in the store"):
        pinned_refs(_store(tmp_path), {"clades/h3": version})


def test_sequences_are_pinned_all_or_none(tmp_path: Path) -> None:
    store = _store(tmp_path)
    one = {"sequences/h3": store.current("sequences", "h3").version}
    with pytest.raises(MapColouringError, match="all or none; not pinned: sequences/bvic"):
        pinned_refs(store, one)


def test_pinned_clades_labelled_from_other_sequences_are_refused() -> None:
    from af.store import StoreRef

    sha = "ab" * 32
    clades = StoreRef("clades", "h3", sha[:16], sha)
    pins = Pins(clades=(clades,))
    pins.check_clades_labelled({})  # labelled from the sequences read: fine
    pins.check_clades_labelled({"h1": ("1" * 16, "2" * 16)})  # another dataset, not pinned
    with pytest.raises(MapColouringError, match="labelled from sequences/h3@1111111111111111"):
        pins.check_clades_labelled({"h3": ("1" * 16, "2" * 16)})
    with pytest.raises(MapColouringError, match="no single sequences version"):
        pins.check_clades_labelled({"h3": (None, "2" * 16)})
    Pins().check_clades_labelled({"h3": ("1" * 16, "2" * 16)})  # unpinned: reported, not refused


def test_provenance_records_the_pins_and_the_skipped_guard(tmp_path: Path) -> None:
    """A pinned serology skips the staleness guard (it compares CURRENTs); the figure says so.
    Unpinned, the pins are empty and no guard line is written (the build applied the guard)."""
    from af.clades.colours import ColourEntry
    from af.clades.colours import ColourScheme as CladeColourScheme
    from af.store import StoreRef
    from tests.map.test_colouring import caller_setup

    colours, chart, clade_subtype = caller_setup(tmp_path)
    own = CladeColourScheme(
        clade_subtype, "caller", (ColourEntry(1, "P", "Clade P", "#aa0000", False),)
    )
    unpinned = colours.for_chart(chart, own).provenance
    assert unpinned["pins"] == {} and "serology_guard" not in unpinned
    assert unpinned["clades_same_content"] == {}  # nothing read: no version pairs to record

    sha = "cd" * 32
    colours.pins = Pins(serology=StoreRef("serology", "all", sha[:16], sha))
    pinned = colours.for_chart(chart, own).provenance
    assert pinned["pins"] == {"serology/all": sha[:16]}
    assert pinned["serology_guard"] == "pinned: guard not applied"


def test_a_pinned_behind_table_refuses_only_charts_that_read_it() -> None:
    from af.store import StoreRef

    sha = "ef" * 32
    pins = Pins(clades=(StoreRef("clades", "h1", sha[:16], sha),))
    behind = {"h1": ("1" * 16, "2" * 16)}
    pins.check_clades_labelled(behind, {"bvic"})  # a B/Vic chart: not its table
    with pytest.raises(MapColouringError, match="clades/h1"):
        pins.check_clades_labelled(behind, {"h1"})


def test_a_chart_cites_only_the_datasets_its_colours_come_from(tmp_path: Path) -> None:
    """Its own row, plus any dataset an antigen matched in; the rest of the run's join is not
    the figure's input (7 Oct 2026), so a behind table elsewhere does not mark it stale."""
    import dataclasses

    from af.clades.colours import ColourEntry
    from af.clades.colours import ColourScheme as CladeColourScheme
    from af.serology.joins import LinkCounts
    from af.store import StoreRef
    from tests.map.test_colouring import caller_setup

    colours, chart, clade_subtype = caller_setup(tmp_path)
    key = next(iter(colours._sequences))
    colours._sequences[key] = dataclasses.replace(
        colours._sequences[key], datasets=frozenset({"h3", "other"})
    )
    colours.links = LinkCounts(
        clades_behind={"h1": ("1" * 16, "2" * 16), "other": ("3" * 16, "4" * 16)},
        refs={
            "sequences": {d: {"kind": "sequences", "dataset": d} for d in ("h1", "h3", "other")},
            "clades": [{"kind": "clades", "dataset": d} for d in ("h1", "h3")],
        },
    )
    sha = "ab" * 32
    colours.serology = StoreRef("serology", "all", sha[:16], sha)
    own = CladeColourScheme(
        clade_subtype, "caller", (ColourEntry(1, "P", "Clade P", "#aa0000", False),)
    )
    p = colours.for_chart(chart, own).provenance
    assert p["datasets"] == ["h3", "other"]  # its row, and where one antigen matched
    assert p["clades_behind"] == {"other": ["3" * 16, "4" * 16]}  # h1 is not this chart's
    refs = colours.store_refs(p["datasets"])
    assert [(r["kind"], r["dataset"]) for r in refs] == [
        ("serology", "all"), ("sequences", "h3"), ("sequences", "other"), ("clades", "h3")
    ]  # fmt: skip
    assert len(colours.store_refs()) == 6  # no datasets: the whole run, as before
