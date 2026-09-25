"""Dot colours from a colour scheme, with every uncoloured dot counted by reason."""

import datetime
from pathlib import Path

from af.clades.colours import ColourEntry, ColourScheme
from af.clades.sequence import AlignedSequence
from af.geo.colours import UNCOLOURED, DotStyle, dot_styles
from af.geo.records import Month, geo_counts, to_i7
from af.serology.joins import PreparationSequence
from af.serology.query import Preparation
from tests.clades.synthetic import build_clone, load_synthetic

SUBTYPE = "A(H3N2)"


def prep(name: str) -> Preparation:
    day = datetime.date(2021, 1, 9)
    return Preparation(SUBTYPE, "", name, "", (), "E3", day, "LABX", day)


def key(p: Preparation) -> tuple[str, str, str, tuple[str, ...], str]:
    return (p.subtype, p.name, p.reassortant, p.annotations, p.passage)


def linked(
    clade: str | None, pairing: str = "exact", conflict: bool = False
) -> PreparationSequence:
    if conflict:
        return PreparationSequence(None, None, None, pairing, conflict=True)
    return PreparationSequence("EPI_ISL_1", "ACC1", clade, pairing, conflict=False)


SCHEME = ColourScheme(
    subtype=SUBTYPE,
    name="test",
    entries=(
        ColourEntry(order=1, key="P", legend="Clade P", colour="#aa0000", is_group=False),
        ColourEntry(order=2, key="P.1", legend="Clade P.1", colour="#0000aa", is_group=False),
    ),
)


def test_styles_and_reasons(tmp_path: Path) -> None:
    clade_set = load_synthetic(build_clone(tmp_path / "clone").parent)
    preps = {n: prep(n) for n in ("A-1", "B-2", "C-3", "D-4", "E-5", "F-6", "G-7", "H-8")}
    links = {
        key(preps["A-1"]): linked("P.1.1"),  # deepest entry it lies within: P.1
        key(preps["B-2"]): linked("P.2"),  # only within P
        key(preps["C-3"]): linked(""),  # nomenclature names no clade
        key(preps["E-5"]): linked(None, conflict=True),
        key(preps["F-6"]): linked("P.1", pairing="proxy"),
        key(preps["G-7"]): linked(None),  # sequence, but no clade row
        key(preps["H-8"]): linked("P.1"),  # no aligned sequence (below)
    }  # D-4 has no sequence at all

    def aligned(epi: str, acc: str) -> AlignedSequence | None:
        return None if style_calls["H-8"] else AlignedSequence("K" * 30)

    style_calls = {"H-8": False}
    style, counts = dot_styles(links, aligned, SCHEME, clade_set, use_proxies=False)
    got = {}
    for name, p in preps.items():
        style_calls["H-8"] = name == "H-8"
        got[name] = style(p)
    assert got["A-1"] == DotStyle("Clade P.1", "#0000aa")
    assert got["B-2"] == DotStyle("Clade P", "#aa0000")
    assert all(got[n] == UNCOLOURED for n in ("C-3", "D-4", "E-5", "F-6", "G-7", "H-8"))
    assert counts.coloured == {"Clade P.1": 1, "Clade P": 1}
    assert counts.uncoloured == {
        "nomenclature names no clade": 1,
        "no sequence": 1,
        "rows name different sequences": 1,
        "proxy pairing not used": 1,
        "no clade assignment": 1,
        "no aligned sequence": 1,
    }
    # styled dots reach the I7 document with their legend label as the clade
    geo = geo_counts(
        list(preps.values()), Month(2021, 1), Month(2021, 1), lambda n: "Place", style_of=style
    )
    points = to_i7(geo, SUBTYPE)["periods"][0]["locations"][0]["points"]
    assert points[0] == {"color": "transparent", "count": 6}
    assert {"color": "#0000aa", "count": 1, "clade": "Clade P.1"} in points
