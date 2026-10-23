"""The serum-coverage addenda as [[coverage]] blocks of the addenda pass (af.serology.coverage),
on 08's invented maps (tests.map.test_addenda), with the figure renderer stood in for."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from af.chart.model import Antigen, Chart, Serum, Titres
from af.chart.serum_circles import serum_circles
from af.chart.titre import Titre
from af.report.howmade import load_declarations
from af.serology.coverage import DESCRIPTION, CoverageError, Shown, write_fragment

pytest.importorskip("matplotlib")

from af.map import addenda  # noqa: E402

from ..map.test_addenda import context, page  # noqa: E402


def coverage_block(**kw: Any) -> dict[str, Any]:
    return {
        "addendum": 2,
        "subtype": "h3",
        "title": "Addendum 2 (example serum coverage)",
        "window": "all",
        "maps": [{"chart": "lab-a-hi"}, {"chart": "lab-a-neut"}],
        "sections": [{"label": "X", "clade": "X", "reason": "the test clade"}],
        **kw,
    }


@pytest.fixture
def drawn(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """render_coverage stood in for: each call recorded, an I7 written where it would be."""
    calls: list[dict[str, Any]] = []

    def render(page: Any, record: Any, **kw: Any) -> Path:
        calls.append({"map": page.folder, "serum": record.serum_id, **kw})
        i7 = Path(kw["out_pdf"]).with_name("figure.i7.json")
        i7.parent.mkdir(parents=True, exist_ok=True)
        i7.write_text(json.dumps({"map": {"addendum": kw["addendum"]}}))
        return i7

    monkeypatch.setattr(addenda, "render_coverage", render)
    return calls


def test_a_coverage_block_draws_a_figure_per_serum_and_plans_its_pages(
    tmp_path: Path, drawn: list[dict[str, Any]]
) -> None:
    """Both maps are one centre's: its cell sera on the top row over the egg sera of the same
    strain, map by map, after the description page. The block's hand declaration lands on
    its map's chain version."""
    pages = {f: page(f) for f in ("lab-a-hi", "lab-a-neut")}
    hand = {"map": "lab-a-neut", "ruling": "test ruling", "text": "a hand declaration"}
    config = addenda.AddendaConfig(
        addenda.dt.date(2026, 9, 21), 12, {"coverage": [coverage_block(declarations=[hand])]}
    )
    (result,) = addenda.run_addenda(config, pages, context(tmp_path))
    out = tmp_path / "addenda" / "addendum-2-h3"
    assert result.directory == out
    assert [(c["map"], c["serum"]) for c in drawn] == [
        ("lab-a-hi", "LABA F1/25"), ("lab-a-hi", "LABA F2/25"),
        ("lab-a-neut", "LABA F1/25"), ("lab-a-neut", "LABA F2/25"),
    ]  # fmt: skip
    assert {(c["radius"], c["clade_line"]) for c in drawn} == {("empirical", "X")}
    first = drawn[0]["addendum"]
    assert (first["section"], first["reason"], first["rules"]) == (
        "X", "clade X", ["best-match", "best-match"]
    )  # fmt: skip
    manifest = json.loads((out / "addendum.json").read_text())
    assert manifest["kind"] == "coverage" and manifest["addendum"] == 2
    text, grid = manifest["pages"]
    assert text["text"] == list(DESCRIPTION) and text["margins"]["right"] == "10em"
    assert grid["caption"] == "Ferret serum coverage maps for X sera."
    assert grid["cells"] == [
        "lab-a-hi/sr0/v1/figure.i7.json", "lab-a-neut/sr0/v1/figure.i7.json", None,
        "lab-a-hi/sr1/v1/figure.i7.json", "lab-a-neut/sr1/v1/figure.i7.json", None,
    ]  # fmt: skip
    assert manifest["selection"]["reasons"] == {"X": "the test clade"}
    assert manifest["disagreements"]["shown"] == 4 and manifest["disagreements"]["hand"] == 1
    shown = json.loads((out / "shown.json").read_text())
    assert [s["map"] for s in shown] == ["lab-a-hi", "lab-a-hi", "lab-a-neut", "lab-a-neut"]
    assert shown[0]["chain"]["dataset"] == "laba/lab-a-hi/merged"
    assert shown[0]["record"]["coverage"]["rule"] == "best-match"
    toml = out / manifest["declarations"][0]
    (entry,) = tomllib.loads(toml.read_text())["declaration"]  # the rules agree: hand only
    assert (entry["dataset"], entry["ruling"]) == ("laba/lab-a-neut/merged", "test ruling")
    assert entry["file"] == "declarations/addendum-2-h3-homologous-rules.md"
    assert len(load_declarations(toml)) == 1  # the round's loader takes the fragment


def test_a_coverage_block_refuses_what_it_cannot_draw(
    tmp_path: Path, drawn: list[dict[str, Any]]
) -> None:
    pages = {f: page(f) for f in ("lab-a-hi", "lab-a-neut")}
    ctx = context(tmp_path)
    run = addenda.handlers()["coverage"]
    with pytest.raises(CoverageError, match="draws the empirical radius at fold 2.0"):
        run(coverage_block(radius="theoretical"), pages, ctx)
    with pytest.raises(CoverageError, match="no empty map slots"):
        run(coverage_block(maps=[{"chart": "lab-a-hi"}, {"empty": True}]), pages, ctx)
    with pytest.raises(CoverageError, match="not one of its maps"):
        run(coverage_block(declarations=[{"map": "lab-b-hi", "ruling": "r", "text": "t"}]),
            pages, ctx)  # fmt: skip


NAME = "A(H3N2)/EXAMPLETOWN/1/2025"
OTHER = "A(H3N2)/EXAMPLEVILLE/2/2025"
CHAIN = {"dataset": "lab-a/h3-hi-lab-a/merged", "version": "0123456789abcdef"}


def test_a_serum_whose_rules_draw_different_maps_is_declared(tmp_path: Path) -> None:
    """Serum 0 has two homologous antigens: the best match (exact passage, 320) and a smaller
    radius (SIAT1, 640), so its two rules draw different maps. Serum 1 has one: they agree."""
    antigens = [Antigen(NAME, "MDCK1"), Antigen(NAME, "SIAT1"),
                *(Antigen(OTHER, f"MDCK{i}") for i in (1, 2, 3))]  # fmt: skip
    sera = [Serum(NAME, "F1/25", "MDCK1"), Serum(OTHER, "F2/25", "MDCK1")]
    columns = [["320", "640", "80", "40", "20"], ["40", "40", "640", "320", "40"]]
    t = [[Titre.parse(columns[s][a]) for s in range(2)] for a in range(5)]
    chart = Chart({"D": "20260110"}, antigens, sera, Titres(t))
    layout = np.array([[1.5, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 0], [2, 0.5]])
    records = serum_circles(chart, layout, 2.0)
    shown = [Shown(2, "vaccine", "r", "h3-hi-lab-a", CHAIN, r) for r in records]
    toml, md = tmp_path / "frag.toml", tmp_path / "frag.md"
    counts = write_fragment(shown, [], toml, md, "declarations/x.md", {"addendum": "test"})
    assert counts["disagree"] == 1 and counts["by_kind"] == {"visible": 1}
    (entry,) = tomllib.loads(toml.read_text())["declaration"]
    assert (entry["ruling"], entry["dataset"], entry["version"]) == ("Q228", *CHAIN.values())
    assert entry["text"].startswith(f"serum coverage addendum 2, vaccine: {NAME} F1/25 (visible)")
    assert "titre 320, empirical radius 2.50" in entry["text"]  # drawn: the best match
    assert "titre 640, empirical radius 2.00" in entry["text"]  # ae's circle rule
    assert "1 of 2 shown sera disagree: 1 visible" in md.read_text()


def test_the_one_writer_declares_circle_panels_by_their_theoretical_radius(tmp_path: Path) -> None:
    """A serum-circle panel draws the theoretical radius and no colouring: the same writer,
    told so, judges and quotes that radius (fold 3: 3 + 6 - 5 = 4 for the best match's 320,
    3 + 6 - 6 = 3 for ae's 640) and no within/outside sets."""
    antigens = [Antigen(NAME, "MDCK1"), Antigen(NAME, "SIAT1"),
                *(Antigen(OTHER, f"MDCK{i}") for i in (1, 2, 3))]  # fmt: skip
    sera = [Serum(NAME, "F1/25", "MDCK1")]
    t = [[Titre.parse(v)] for v in ("320", "640", "80", "40", "20")]
    chart = Chart({"D": "20260110"}, antigens, sera, Titres(t))
    layout = np.array([[1.5, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 0]])
    (rec,) = serum_circles(chart, layout, 3.0)
    shown = [Shown(5, "X", "r", "h3-hi-lab-a", CHAIN, rec)]
    toml, md = tmp_path / "circles.toml", tmp_path / "circles.md"
    counts = write_fragment(shown, [], toml, md, "declarations/c.md", {"addendum": "test"},
                            radius="theoretical", kind="serum circle", coverage=False)  # fmt: skip
    assert counts["by_kind"] == {"visible": 1}
    (entry,) = tomllib.loads(toml.read_text())["declaration"]
    assert entry["text"].startswith("serum circle addendum 5, X:")
    assert (
        "titre 320, theoretical radius 4.00); ae's circle rule gives titre 640, " in entry["text"]
    )
    assert "within" not in entry["text"]
    assert md.read_text().startswith("# Serum circle maps whose two homologous rules disagree")


def test_a_hand_declaration_is_made_on_its_maps_chain_version() -> None:
    from af.serology.coverage import hand_declaration

    pages = {"lab-a-hi": page("lab-a-hi")}
    entry = {"map": "lab-a-hi", "ruling": "Q74, Q123", "text": "ids print normalised"}
    assert hand_declaration("addendum-5-h3", entry, pages, ["lab-a-hi"]) == {
        "ruling": "Q74, Q123", "dataset": "laba/lab-a-hi/merged", "version": "0" * 16,
        "text": "ids print normalised", "file": "declarations/addendum-5-h3-homologous-rules.md",
    }  # fmt: skip
    with pytest.raises(CoverageError, match="needs map, ruling and text"):
        hand_declaration("addendum-5-h3", {"map": "lab-a-hi", "text": "t"}, pages, ["lab-a-hi"])
