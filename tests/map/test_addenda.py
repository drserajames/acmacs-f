"""af.map.addenda: the pass that draws a round's addenda from its built maps (circle blocks)."""

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from af.chart.model import Antigen, Chart, Projection, Serum, Titres
from af.chart.serum_selection import SelectionError
from af.chart.titre import Titre
from af.map.addenda import (
    AddendaConfig,
    AddendaContext,
    AddendaError,
    MapPage,
    load_addenda,
    run_addenda,
)
from af.map.colouring import SerumStrain
from af.map.render import DEFAULT_LOOK
from af.map.style import ColourRow, ColourScheme, Window
from af.map.viewport import Frame
from af.tables.labs import Lab

pytest.importorskip("matplotlib")

from af.map.figure import chart_points  # noqa: E402

SCHEME = ColourScheme("test", (ColourRow("Clade X", "#2ca02c", frozenset({"X"})),))
STRAIN = "A(H3N2)/EXAMPLETOWN/1/2025"
LAB = Lab("LABA", "Lab A", "none yet", "bare")


def page(folder: str, hidden: dict[int, str] | None = None) -> MapPage:
    """A lab's map: the strain's cell and egg preparations, two other antigens, and a cell and
    an egg serum of the strain, titrated in January 2026."""
    t = Titre.parse
    antigens = [
        Antigen(STRAIN, "SIAT2", date="2025-03-01"),
        Antigen(STRAIN, "E4", date="2025-03-01"),
        Antigen("A(H3N2)/EXAMPLEVILLE/2/2025", "SIAT1", date="2025-06-01"),
        Antigen("A(H3N2)/EXAMPLEVILLE/3/2025", "SIAT1", date="2025-07-01"),
    ]
    sera = [Serum(STRAIN, "LABA F1/25", "SIAT1"), Serum(STRAIN, "LABA F2/25", "E3")]
    rows = [["1280", "80"], ["160", "640"], ["320", "40"], ["40", "20"]]
    chart = Chart(
        {"V": "A(H3N2)", "l": "LABA", "D": "20260110"},
        antigens,
        sera,
        Titres([[t(v) for v in row] for row in rows]),
    )
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [1.0, 1.0], [-2.0, 1.0], [0.5, -0.5], [2.5, 0.5]])
    labels = [frozenset({"X"})] * 4
    points = chart_points(chart, xy, labels=labels, sequenced=[True] * 4, hidden=hidden)
    return MapPage(
        folder=folder,
        centre="laba",
        chain_ref={"dataset": f"laba/{folder}/merged", "version": "0" * 16},
        chart=chart,
        projection=Projection(xy),
        points=tuple(points),
        scheme=SCHEME,
        frame=Frame(-5.0, 5.0, 10.0),
        look=DEFAULT_LOOK,
        windows=((Window("all", None), f"{folder} by clade"),),
        must_show_since=dt.date(2025, 9, 1),
        provenance={"inputs": {"chain": {"name": folder, "sha256": "0" * 64}}},
    )


class Strains:
    """The run's store join, as an addendum reads it: each serum's strain is clade X."""

    def for_sera(self, chart: Chart, residues: Any = ()) -> list[SerumStrain]:
        return [
            SerumStrain(sr, "EPI_1", "A1", s.passage, "X", ("X",), {}, "test")
            for sr, s in enumerate(chart.sera)
        ]


def block(**kw: Any) -> dict[str, Any]:
    return {
        "addendum": 5,
        "subtype": "h3",
        "title": "Addendum 5 (example circles)",
        "window": "all",
        "fold": 2.0,
        "radius": "theoretical",
        "passage_classes": ["cell"],
        "maps": [{"chart": "lab-a-hi"}, {"empty": True}, {"chart": "lab-a-neut"}],
        "sections": [{"label": "X", "clade": "X", "reason": "the example clade"}],
        "declarations": [
            {"map": "lab-a-neut", "ruling": "Q74, Q123", "text": "serum ids print normalised"}
        ],
        **kw,
    }


def context(tmp_path: Path) -> AddendaContext:
    return AddendaContext(
        colours=Strains(),
        since=dt.date(2025, 9, 1),
        out_root=tmp_path,
        created=dt.datetime(2026, 10, 21, tzinfo=dt.UTC),
        labs={"LABA": LAB},
    )


def test_a_circle_block_draws_three_pages_per_map_and_writes_its_plan(tmp_path: Path) -> None:
    pages = {f: page(f) for f in ("lab-a-hi", "lab-a-neut")}
    config = AddendaConfig(dt.date(2026, 9, 21), 12, {"circles": [block()]})
    (result,) = run_addenda(config, pages, context(tmp_path))
    out = tmp_path / "addenda" / "addendum-5-h3"
    assert result.directory == out
    manifest = json.loads((out / "addendum.json").read_text())
    assert manifest["kind"] == "circles" and manifest["addendum"] == 5
    # plain, circles, names pages; the empty slot kept where the block put it
    assert [p["cells"][:3] for p in manifest["pages"]] == [
        [f"lab-a-hi/{k}/v1/figure.i7.json", None, f"lab-a-neut/{k}/v1/figure.i7.json"]
        for k in ("plain", "circles", "names")
    ]
    assert all(len(p["cells"]) == 6 and p["caption"] is None for p in manifest["pages"])
    # only the cell serum: the egg one is left out by passage class, and counted
    assert manifest["selection"]["sections"] == {"X": 2}
    # Q228 rows go through the one writer, judged on the radius a circle panel draws
    assert manifest["declarations"] == [
        "addendum-5-h3-homologous-rules.toml", "addendum-5-h3-homologous-rules.md",
    ]  # fmt: skip
    assert all((out / d).is_file() for d in manifest["declarations"])
    assert manifest["disagreements"]["shown"] == 2
    assert manifest["disagreements"]["hand"] == 1
    rows = (out / manifest["declarations"][0]).read_text()
    assert 'ruling = "Q74, Q123"' in rows and 'dataset = "laba/lab-a-neut/merged"' in rows
    assert manifest["selection"]["excluded"] == {"passage class egg": 2}
    shown = json.loads((out / "shown.json").read_text())
    assert [(s["map"], s["section"]) for s in shown] == [("lab-a-hi", "X"), ("lab-a-neut", "X")]
    assert shown[0]["record"]["serum_id"] == "LABA F1/25"
    i7 = {k: json.loads((out / "lab-a-hi" / k / "v1" / "figure.i7.json").read_text())["map"]
          for k in ("plain", "circles", "names")}  # fmt: skip
    assert "circles" not in i7["plain"] and len(i7["circles"]["circles"]) == 1
    assert i7["names"]["caption"] == [f"{STRAIN} F1/25 SIAT1"]  # the id as the lab prints it
    assert i7["circles"]["addendum"]["sera"][0]["section"] == "X"
    assert i7["circles"]["circles"][0]["fill"] == "#2ca02c"  # the homologous antigen's colour


def test_a_section_matching_no_serum_in_the_whole_addendum_is_an_error(tmp_path: Path) -> None:
    pages = {f: page(f) for f in ("lab-a-hi", "lab-a-neut")}
    none = block(sections=[{"label": "Y", "clade": "Y", "reason": "a clade no serum has"}])
    with pytest.raises(SelectionError, match="matches no serum"):
        run_addenda(AddendaConfig(dt.date(2026, 9, 21), 12, {"circles": [none]}), pages,
                    context(tmp_path))  # fmt: skip
    missing = block(maps=[{"chart": "lab-a-hi"}, {"chart": "lab-z"}])
    with pytest.raises(AddendaError, match="maps not built"):
        run_addenda(AddendaConfig(dt.date(2026, 9, 21), 12, {"circles": [missing]}), pages,
                    context(tmp_path))  # fmt: skip


def test_the_layout_keeps_hidden_points_where_they_are_drawn() -> None:
    p = page("lab-a-hi", hidden={2: "rule-1"})
    assert np.isfinite(p.layout()[2]).all()  # radii and coverage count every titrated antigen
    assert p.layout().shape == (6, 2)


def test_recency_is_counted_back_from_the_meetings_month(tmp_path: Path) -> None:
    assert AddendaConfig(dt.date(2026, 9, 21), 12, {}).since == dt.date(2025, 9, 1)
    assert AddendaConfig(dt.date(2026, 1, 15), 6, {}).since == dt.date(2025, 7, 1)
    path = tmp_path / "addenda.toml"
    path.write_text("[addenda]\nmeeting = 2026-09-21\n\n[[circles]]\naddendum = 5\n")
    assert load_addenda(path).since == dt.date(2025, 9, 1)
    path.write_text("[addenda]\nmeeting = 2026-09-21\n\n[[maps]]\naddendum = 5\n")
    with pytest.raises(AddendaError, match="unknown addendum kinds"):
        load_addenda(path)
    path.write_text("[[circles]]\naddendum = 5\n")
    with pytest.raises(AddendaError, match="meeting"):
        load_addenda(path)


def test_a_coverage_figure_has_the_three_line_title_and_its_record(tmp_path: Path) -> None:
    from af.chart.serum_circles import serum_circles
    from af.map.addenda import render_coverage

    p = page("lab-a-hi")
    record = serum_circles(p.chart, p.layout(), 2.0, projection=p.projection)[0]
    i7 = render_coverage(
        p, record, radius="empirical", clade_line="X", out_pdf=tmp_path / "c.pdf",
        created=dt.datetime(2026, 10, 22, tzinfo=dt.UTC), addendum={"section": "X"}, lab=LAB,
    )  # fmt: skip
    doc = json.loads(i7.read_text())
    assert doc["title"] == f"lab-a-hi by clade\n{STRAIN} F1/25\nX"
    assert doc["map"]["addendum"] == {"section": "X"}
    (circle,) = doc["map"]["circles"]
    assert circle["point"] == "sr0" and circle["fill"] == "#0000ff"  # a cell serum: blue
    styled = {a["id"]: a.get("style", {}) for a in doc["map"]["antigens"] + doc["map"]["sera"]}
    assert styled["sr0"]["fill"] == "black" and styled["ag0"]["outline_width"] == 3.0


def test_a_figure_keeps_its_maps_frame_when_its_taller_title_covers_a_recent_antigen(
    tmp_path: Path,
) -> None:
    """A recent antigen 1 unit below the frame's top is clear of a map's one-line title and
    under a coverage figure's three-line one: the map page would refuse the frame; the figure
    keeps its map's zoom and records what its title hides."""
    from dataclasses import replace

    from af.chart.serum_circles import serum_circles
    from af.map.addenda import render_coverage
    from af.map.finish import finish_map
    from af.map.viewport import FrameError

    p = page("lab-a-hi")
    moved = [replace(q, xy=(-4.0, 4.0)) if q.id == "ag3" else q for q in p.points]
    p = replace(p, points=tuple(moved), must_show_since=dt.date(2025, 1, 1))
    record = serum_circles(p.chart, p.layout(), 2.0, projection=p.projection)[0]
    created = dt.datetime(2026, 10, 22, tzinfo=dt.UTC)
    one_line: dict[str, Any] = dict(
        chart=p.folder, scheme=p.scheme, window=p.windows[0][0], frame_size=10.0, frame=p.frame,
        must_show_since=p.must_show_since, vaccine_labels={}, created=created,
        provenance=p.provenance,
    )  # fmt: skip
    finish_map(list(p.points), title="Map", out_pdf=tmp_path / "m.pdf", **one_line)  # clear
    with pytest.raises(FrameError, match="hides points"):
        finish_map(list(p.points), title="Map\nA long serum line\nX", out_pdf=tmp_path / "x.pdf",
                   **one_line)  # fmt: skip
    i7 = render_coverage(
        p, record, radius="empirical", clade_line="X", out_pdf=tmp_path / "c.pdf",
        created=created, addendum={}, lab=LAB,
    )  # fmt: skip
    doc = json.loads(i7.read_text())["map"]
    assert doc["viewport"][:3] == [p.frame.x, p.frame.y, p.frame.size]  # the map's zoom
    assert doc["hidden"]["recent antigens"] == 1
