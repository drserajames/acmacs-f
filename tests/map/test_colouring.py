"""The map's side of the shared clade colouring: scheme -> legend rows, chosen style -> label."""

from pathlib import Path

import pytest

from af.chart.model import Antigen, Chart, Titres
from af.clades.colours import ColourEntry
from af.clades.colours import ColourScheme as CladeColourScheme
from af.clades.sequence import AlignedSequence
from af.map.colouring import (
    MapColouringError,
    StoreColours,
    key_for_legend,
    labels_for,
    map_scheme,
)
from af.seq.matching_rules import matching_rules
from af.serology.joins import PreparationSequence
from af.serology.outputs import AlignedSequences, SubtypeColouring
from tests.clades.synthetic import build_clone, load_synthetic
from tests.seq.test_matching_rules import write_af_data


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


# ---------------------------------------------------------------- one chart through the shared path


def test_chart_antigens_take_the_shared_choice(tmp_path: Path) -> None:
    """Each antigen is looked up by preparation and gets the entry dot_styles chose; an
    antigen with no sequence is unpainted and counted, never guessed."""
    sub = "A(H3N2)"
    names = ["/".join(("A", "PLACE", str(n), "2021")) for n in (1, 2, 3)]
    antigens = [Antigen(n, passage="E3") for n in names]
    chart = Chart({"V": sub}, antigens, [], Titres([[] for _ in antigens]))
    clade_set = load_synthetic(build_clone(tmp_path / "clone").parent)
    s = CladeColourScheme(
        sub,
        "test",
        (
            ColourEntry(1, "P", "Clade P", "#aa0000", False),
            ColourEntry(2, "P.1", "Clade P.1", "#0000aa", False),
        ),
    )
    # Built without its constructor: the stores it would read are replaced by these tables.
    colours = object.__new__(StoreColours)
    colours._sequences = {
        (sub, names[0], "", (), "E3"): PreparationSequence("EPI_1", "A1", "P.1.1", "exact", False),
        (sub, names[1], "", (), "E3"): PreparationSequence("EPI_2", "A2", "P.2", "exact", False),
    }
    colours._aligned = AlignedSequences(
        {("EPI_1", "A1"): AlignedSequence("M"), ("EPI_2", "A2"): AlignedSequence("M")}
    )
    colours._colourings = {(sub, "test"): SubtypeColouring(s, clade_set)}
    colours.rules = matching_rules(write_af_data(tmp_path / "af-data"))

    got = colours.for_chart(chart, "test")
    assert got.labels == (frozenset({"P.1"}), frozenset({"P"}), frozenset())
    assert got.sequenced == (True, True, False)
    painted = [got.scheme.paint(lb) for lb in got.labels]
    assert [row.colour if row else None for row in painted] == ["#0000aa", "#aa0000", None]
    assert got.provenance["uncoloured"] == {"no sequence": 1}
    assert got.provenance["shadowed_rows"] == []
    assert len(got.provenance["matching_rules"]) == 6  # every rule table, with its hash
