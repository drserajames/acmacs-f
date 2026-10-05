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
