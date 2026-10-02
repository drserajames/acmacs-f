"""Dot colours from a colour scheme, with every uncoloured dot counted by reason."""

import datetime
from pathlib import Path

from af.clades.colours import ColourEntry, ColourScheme
from af.clades.sequence import AlignedSequence
from af.geo.colours import BASIS_CLADE, BASIS_GROUP_NO_CLADE, UNCOLOURED, DotStyle, dot_styles
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
    assert got["A-1"] == DotStyle("Clade P.1", "#0000aa", BASIS_CLADE)
    assert got["B-2"] == DotStyle("Clade P", "#aa0000", BASIS_CLADE)
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


def test_a_refused_tie_is_coloured_when_its_sequences_agree_else_by_aes_rank(
    tmp_path: Path,
) -> None:
    """Sarah, Q81: "Agree + ae's rank for splits". No sequence is chosen for an agreeing tie."""
    from af.seq.matching import TIE_AGREES, TIE_RANKED
    from af.serology.joins import TiedSequence

    clade_set = load_synthetic(build_clone(tmp_path / "clone").parent)
    one, two, three = (TiedSequence(f"EPI_ISL_{n}", f"ACC{n}", c)
                       for n, c in ((1, "P.1"), (2, "P.1.1"), (3, "P.2")))  # fmt: skip

    def tie(*seqs: TiedSequence, ranked: TiedSequence | None) -> PreparationSequence:
        return PreparationSequence(None, None, None, "", conflict=False, tied=seqs, ranked=ranked)

    preps = {n: prep(n) for n in ("agree", "split", "split-unranked")}
    links = {
        key(preps["agree"]): tie(one, two, ranked=two),  # P.1 and P.1.1 both colour as P.1
        key(preps["split"]): tie(one, three, ranked=three),  # P.1 vs P: ae's pick decides
        key(preps["split-unranked"]): tie(one, three, ranked=None),
    }
    style, counts = dot_styles(links, lambda e, a: AlignedSequence("K" * 30), SCHEME, clade_set)
    assert style(preps["agree"]) == DotStyle("Clade P.1", "#0000aa", BASIS_CLADE)
    assert style(preps["split"]) == DotStyle("Clade P", "#aa0000", BASIS_CLADE)
    assert style(preps["split-unranked"]) == UNCOLOURED
    assert counts.ties == {TIE_AGREES: 1, TIE_RANKED: 1}
    assert counts.uncoloured == {"tie with no ranked sequence": 1}
    assert counts.coloured == {"Clade P.1": 1, "Clade P": 1}


def test_dots_coloured_through_a_doubtful_match_are_counted_per_doubt(tmp_path: Path) -> None:
    """Sarah, Q81 D: flagged + counted. An uncoloured dot counts no doubt: it was not used."""
    clade_set = load_synthetic(build_clone(tmp_path / "clone").parent)
    egg, both, lost = prep("egg"), prep("both"), prep("lost")
    links = {
        key(egg): PreparationSequence("EPI_ISL_1", "ACC1", "P.1", "", conflict=False,
                                      doubts=("match.egg-antigen-non-egg-sequence",)),
        key(both): PreparationSequence("EPI_ISL_2", "ACC2", "P.2", "", conflict=False,
                                       doubts=("match.epi-name-differs", "match.reassortant")),
        key(lost): PreparationSequence("EPI_ISL_3", "ACC3", "", "", conflict=False,
                                       doubts=("match.reassortant",)),  # no clade: uncoloured
    }  # fmt: skip
    style, counts = dot_styles(links, lambda e, a: AlignedSequence("K" * 30), SCHEME, clade_set)
    assert [style(p).label for p in (egg, both, lost)] == ["Clade P.1", "Clade P", ""]
    assert counts.doubtful == {
        "match.egg-antigen-non-egg-sequence": 1, "match.epi-name-differs": 1,
        "match.reassortant": 1,
    }  # fmt: skip


def test_rows_naming_two_records_colour_by_the_passage_matched_one_else_by_agreement(
    tmp_path: Path,
) -> None:
    """Sarah, Q81 (30 Sep): the passage-matched record colours; with none or several matching,
    the records colour only if they agree. Counted in ColourCounts.rows."""
    from af.serology.joins import (
        ROWS_NONE_MATCH,
        ROWS_PASSAGE_MATCHED,
        ROWS_SEVERAL_MATCH,
        TiedSequence,
    )

    clade_set = load_synthetic(build_clone(tmp_path / "clone").parent)
    p1, p11, p2 = (TiedSequence(f"EPI_ISL_{n}", f"ACC{n}", c)
                   for n, c in ((1, "P.1"), (2, "P.1.1"), (3, "P.2")))  # fmt: skip

    def rows(
        chosen: TiedSequence | None, resolution: str, *alts: TiedSequence
    ) -> PreparationSequence:
        return PreparationSequence(
            chosen.epi_isl if chosen else None, chosen.accession if chosen else None,
            chosen.clade if chosen else None, "", conflict=True, alternatives=alts,
            resolution=resolution,
        )  # fmt: skip

    preps = {n: prep(n) for n in ("matched", "agree", "disagree")}
    links = {
        key(preps["matched"]): rows(p2, ROWS_PASSAGE_MATCHED, p1, p2),  # P, not P.1
        key(preps["agree"]): rows(None, ROWS_SEVERAL_MATCH, p1, p11),  # both colour as P.1
        key(preps["disagree"]): rows(None, ROWS_NONE_MATCH, p1, p2),
    }
    style, counts = dot_styles(links, lambda e, a: AlignedSequence("K" * 30), SCHEME, clade_set)
    assert style(preps["matched"]) == DotStyle("Clade P", "#aa0000", BASIS_CLADE)
    assert style(preps["agree"]) == DotStyle("Clade P.1", "#0000aa", BASIS_CLADE)
    assert style(preps["disagree"]) == UNCOLOURED
    assert counts.rows == {ROWS_PASSAGE_MATCHED: 1, ROWS_SEVERAL_MATCH: 1}
    assert counts.uncoloured == {"rows name different sequences": 1}


def test_a_group_needing_no_clade_colours_a_virus_with_none(tmp_path: Path) -> None:
    """A sequence the nomenclature names no clade for is tested against groups with no anchor
    (Sarah, 2 Oct 2026). Row order still decides (Q80): a group row placed first is the lowest
    precedence, so a virus WITH a clade keeps its clade's colour."""
    from af.clades.groups import Group, GroupSet, Substitution

    clade_set = load_synthetic(build_clone(tmp_path / "clone").parent)
    groups = GroupSet(
        SUBTYPE,
        (
            Group(SUBTYPE, "5K", None, (Substitution(5, "K"),)),
            Group(SUBTYPE, "P 5K", "P", (Substitution(5, "K"),)),
        ),
    )
    scheme = ColourScheme(
        SUBTYPE,
        "test",
        (
            ColourEntry(1, "5K", "5K, no clade", "#00aa00", True),  # first: lowest precedence
            ColourEntry(2, "P", "Clade P", "#aa0000", False),
            ColourEntry(3, "P 5K", "P with 5K", "#aa00aa", True),
        ),
    )
    preps = {n: prep(n) for n in ("carries", "lacks", "clade")}
    links = {
        key(preps["carries"]): PreparationSequence("EPI_ISL_1", "C1", "", "exact", False),
        key(preps["lacks"]): PreparationSequence("EPI_ISL_2", "C2", "", "exact", False),
        key(preps["clade"]): PreparationSequence("EPI_ISL_3", "C3", "P.2", "exact", False),
    }
    sequences = {"C1": "K" * 30, "C2": "A" * 30, "C3": "K" * 30}
    style, counts = dot_styles(
        links, lambda epi, acc: AlignedSequence(sequences[acc]), scheme, clade_set, groups
    )
    assert style(preps["carries"]) == DotStyle("5K, no clade", "#00aa00", BASIS_GROUP_NO_CLADE)
    # the anchored "P 5K" needs a clade, so it never colours a virus with none
    assert style(preps["lacks"]) == UNCOLOURED  # no group matches: the old reason, unchanged
    assert style(preps["clade"]) == DotStyle("P with 5K", "#aa00aa", BASIS_CLADE)
    assert counts.uncoloured == {"nomenclature names no clade": 1}
    assert counts.basis == {BASIS_GROUP_NO_CLADE: 1, BASIS_CLADE: 1}
