"""The map's side of the shared clade colouring: scheme -> legend rows, chosen style -> label."""

import pytest

from af.clades.colours import ColourEntry
from af.clades.colours import ColourScheme as CladeColourScheme
from af.map.colouring import MapColouringError, key_for_legend, labels_for, map_scheme


def scheme(*entries: ColourEntry) -> CladeColourScheme:
    return CladeColourScheme("SUB", "test", entries)


PARENT = ColourEntry(1, "X", "X (parent)", "#111111", False)
CHILD = ColourEntry(2, "X.1", "X.1", "#222222", False)
GROUP = ColourEntry(3, "X 10K", "X 10K", "#333333", True)


def test_rows_follow_scheme_order_one_key_each() -> None:
    rows = map_scheme(scheme(GROUP, PARENT, CHILD)).rows
    assert [r.legend for r in rows] == ["X (parent)", "X.1", "X 10K"]
    assert [r.labels for r in rows] == [frozenset({"X"}), frozenset({"X.1"}), frozenset({"X 10K"})]


def test_paint_finds_exactly_the_chosen_entry() -> None:
    """The shared path chose the parent; row order must not let a later row overrule it."""
    s = scheme(PARENT, CHILD, GROUP)
    row = map_scheme(s).paint(labels_for("X (parent)", key_for_legend(s)))
    assert row is not None and row.colour == "#111111"


def test_uncoloured_carries_no_label() -> None:
    s = scheme(PARENT)
    assert labels_for("", key_for_legend(s)) == frozenset()
    assert map_scheme(s).paint(frozenset()) is None


def test_shared_legend_text_is_an_error() -> None:
    twin = ColourEntry(2, "Y", "X (parent)", "#444444", False)
    with pytest.raises(MapColouringError, match="used by both"):
        map_scheme(scheme(PARENT, twin))


def test_unknown_chosen_legend_is_an_error() -> None:
    with pytest.raises(MapColouringError, match="not a row"):
        labels_for("Z", key_for_legend(scheme(PARENT)))
