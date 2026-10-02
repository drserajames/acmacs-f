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
    colours._colourings = {("h3", "test"): SubtypeColouring(s, clade_set)}  # by row
    colours.rules = matching_rules(write_af_data(tmp_path / "af-data"))

    got = colours.for_chart(chart, "test")
    assert got.labels == (frozenset({"P.1"}), frozenset({"P"}), frozenset())
    assert got.sequenced == (True, True, False)
    painted = [got.scheme.paint(lb) for lb in got.labels]
    assert [row.colour if row else None for row in painted] == ["#0000aa", "#aa0000", None]
    assert got.provenance["uncoloured"] == {"no sequence": 1}
    assert got.provenance["shadowed_rows"] == []
    assert got.provenance["ties"] == {} and got.provenance["doubtful"] == {}  # none in this chart
    assert "rows" in got.provenance
    assert len(got.provenance["matching_rules"]) == 6  # every rule table, with its hash
    assert "scheme_origin" not in got.provenance  # a named scheme: its tables are its origin
    assert got.provenance["lineage_minority"] == {}  # every antigen is the map's lineage


# ---------------------------------------------------------------- a scheme the caller built


def caller_setup(tmp_path: Path) -> tuple[StoreColours, Chart, str]:
    """Two sequenced antigens (one P.1.1, one P.2) and one unsequenced, with no named schemes:
    a caller's scheme is judged against the store's clade set, injected here."""
    sub = "A(H3N2)"
    names = ["/".join(("A", "PLACE", str(n), "2021")) for n in (1, 2, 3)]
    antigens = [Antigen(n, passage="E3") for n in names]
    chart = Chart({"V": sub}, antigens, [], Titres([[] for _ in antigens]))
    colours = object.__new__(StoreColours)
    colours._sequences = {
        (sub, names[0], "", (), "E3"): PreparationSequence("EPI_1", "A1", "P.1.1", "exact", False),
        (sub, names[1], "", (), "E3"): PreparationSequence("EPI_2", "A2", "P.2", "exact", False),
    }
    colours._aligned = AlignedSequences(
        {("EPI_1", "A1"): AlignedSequence("M"), ("EPI_2", "A2"): AlignedSequence("M")}
    )
    colours._colourings = {}
    clade_set = load_synthetic(build_clone(tmp_path / "clone").parent)
    colours._sets = {"h3": (clade_set, None)}  # keyed by subtype row
    colours.rules = matching_rules(write_af_data(tmp_path / "af-data"))
    return colours, chart, clade_set.subtype


def test_a_callers_scheme_colours_as_a_named_one_would(tmp_path: Path) -> None:
    colours, chart, clade_subtype = caller_setup(tmp_path)
    own = CladeColourScheme(
        clade_subtype,
        "caller",
        (
            ColourEntry(1, "P", "Clade P", "#AA0000", False),
            ColourEntry(2, "P.1", "Clade P.1", "#0000aa", False),
        ),
        source=Path("caller-rows.toml"),
    )
    got = colours.for_chart(chart, own)
    assert got.labels == (frozenset({"P.1"}), frozenset({"P"}), frozenset())
    painted = [got.scheme.paint(lb) for lb in got.labels]
    assert [row.colour if row else None for row in painted] == ["#0000aa", "#aa0000", None]
    p = got.provenance
    assert p["source"] == "store" and p["scheme"] == f"{clade_subtype} caller"
    assert p["scheme_origin"] == "caller-supplied" and p["scheme_file"] == "caller-rows.toml"
    assert len(p["scheme_sha256"]) == 64
    assert p["user_tables"] == {}  # no group rows, so nothing was read from the user's tables
    # The hash follows the rows: a different colour is a different scheme.
    recoloured = CladeColourScheme(
        clade_subtype, "caller", (ColourEntry(1, "P", "Clade P", "#aa0001", False),)
    )
    assert colours.for_chart(chart, recoloured).provenance["scheme_sha256"] != p["scheme_sha256"]


def test_a_callers_row_naming_no_clade_is_an_error(tmp_path: Path) -> None:
    colours, chart, clade_subtype = caller_setup(tmp_path)
    own = CladeColourScheme(
        clade_subtype,
        "caller",
        (
            ColourEntry(1, "P", "Clade P", "#aa0000", False),
            ColourEntry(2, "Q.9", "Clade Q.9", "#0000aa", False),
        ),
    )
    with pytest.raises(MapColouringError, match="'Q.9' is neither a clade"):
        colours.for_chart(chart, own)


def test_a_callers_scheme_for_another_subtype_is_an_error(tmp_path: Path) -> None:
    colours, chart, _ = caller_setup(tmp_path)
    own = CladeColourScheme("B/Vic", "caller", (ColourEntry(1, "P", "P", "#aa0000", False),))
    with pytest.raises(MapColouringError, match="is for B/Vic"):
        colours.for_chart(chart, own)


def invented(prefix: str, n: int) -> str:
    """An invented virus name, assembled so no strain-shaped text is committed."""
    return "/".join((prefix, "PLACE", str(n), "2021"))


def test_a_chart_is_coloured_by_its_subtype_row() -> None:
    """The row comes from the chart's subtype and its antigens' majority lineage code (one
    rule with af.map.build.chart_subtype); lineages carried equally, or a lineage the subtype
    table does not list, are errors."""
    from af.map.colouring import chart_row

    def b_chart(*codes: str) -> Chart:
        antigens = [Antigen(invented("B", n), extra={"L": c}) for n, c in enumerate(codes)]
        return Chart({"V": "B"}, antigens, [], Titres([[] for _ in antigens]))

    h3 = Chart({"V": "A(H3N2)"}, [Antigen(invented("A", 1))], [], Titres([[]]))
    assert chart_row(h3) == "h3"
    assert chart_row(b_chart("V", "V")) == "bvic"
    assert chart_row(b_chart("Y")) == "byam"
    # the majority lineage, as the map build decides it: a few B/Yam antigens on a B/Vic map
    assert chart_row(b_chart("V", "V", "Y")) == "bvic"
    with pytest.raises(MapColouringError, match="lineages equally"):
        chart_row(b_chart("V", "Y"))
    with pytest.raises(MapColouringError, match="no .ace lineage code ''"):
        chart_row(b_chart(""))


def test_a_callers_scheme_for_a_lineage_without_clade_labels_is_an_error(tmp_path: Path) -> None:
    """B/Yam has no clade labels (the subtype table says why): never an empty colouring."""
    from af.clades.subtypes import CladeSubtypeError

    colours, _, _ = caller_setup(tmp_path)
    colours._store, colours._clones = None, None  # type: ignore[assignment]  # refused first
    chart = Chart({"V": "B"}, [Antigen(invented("B", 1), extra={"L": "Y"})], [], Titres([[]]))
    own = CladeColourScheme("B/Yam", "caller", (ColourEntry(1, "P", "P", "#aa0000", False),))
    with pytest.raises(CladeSubtypeError, match="B/Yam has no clade labels"):
        colours.for_chart(chart, own)
