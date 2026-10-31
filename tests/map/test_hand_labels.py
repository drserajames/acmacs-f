"""Map labels placed by hand (af.map.hand): the file, its refusals, and the placer holding them."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from af.map.hand import HandLabelsError, load_hand_labels, resolve
from af.map.labels import HandLabelClash, place_labels
from af.map.style import Window
from af.map.viewport import Box

from .test_finish import SCHEME, SINCE, synthetic_points

HEAD = 'map = "x"\nchain = "lab/x/merged@0123456789abcdef"\n'


def pin(point: str, text: str, dx: float, dy: float, **extra: str) -> str:
    row = {"reason": "clear", "by": "AB", "date": "2026-10-31"} | extra
    rest = "".join(f'{k} = "{v}"\n' for k, v in row.items())
    return f'[[label_pins]]\npoint = "{point}"\ntext = "{text}"\ndx = {dx}\ndy = {dy}\n{rest}'


def hide(point: str, text: str) -> str:
    return (f'[[hide_labels]]\npoint = "{point}"\ntext = "{text}"\nreason = "said beside it"\n'
            'by = "AB"\ndate = "2026-10-31"\n')  # fmt: skip


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "labels-x.toml"
    path.write_text(HEAD + body)
    return path


def test_a_file_of_comments_only_has_no_rows(tmp_path: Path) -> None:
    path = tmp_path / "labels-x.toml"
    path.write_text("# no labels moved by hand yet\n")
    rows = load_hand_labels(path)
    assert rows.label_pins == [] and rows.hide_labels == []
    with pytest.raises(FileNotFoundError):
        load_hand_labels(tmp_path / "labels-missing.toml")


def test_a_row_without_why_who_or_a_real_date_refuses_the_file(tmp_path: Path) -> None:
    for bad, said in ((dict(reason=" "), "no reason"), (dict(by=""), "no by"),
                      (dict(date="31/10/2026"), "not YYYY-MM-DD")):  # fmt: skip
        with pytest.raises(HandLabelsError, match=said):
            load_hand_labels(write(tmp_path, pin("ag100", "Te/25-cell", 0.01, 0.02, **bad)))
    with pytest.raises(HandLabelsError, match="both pinned and hidden"):
        load_hand_labels(write(tmp_path, pin("ag100", "T", 0.0, 0.0) + hide("ag100", "T")))


def test_a_row_matches_exactly_one_labelled_point_and_its_text() -> None:
    from af.map.hand import HandHide, HandLabels, HandPin

    labelled = [("ag1", "pt1 E3", "X/25-egg"), ("ag2", "pt1 MDCK1", "X/25-cell")]
    good = HandLabels(label_pins=[HandPin("pt1 E3", "X/25-egg", 0.01, -0.02, "r", "AB",
                                          "2026-10-31")],
                      hide_labels=[HandHide("pt1 MDCK1", "X/25-cell", "r", "AB",
                                            "2026-10-31")])  # fmt: skip
    assert resolve(good, labelled) == ({"ag1": (0.01, -0.02)}, {"ag2"})
    wrong_text = HandLabels(label_pins=[HandPin("pt1 E3", "X/25-cell", 0, 0, "r", "AB", "d")])
    with pytest.raises(HandLabelsError, match="the label prints 'X/25-egg'"):
        resolve(wrong_text, labelled)
    missing = HandLabels(hide_labels=[HandHide("pt2 E3", "Y", "r", "AB", "d")])
    with pytest.raises(HandLabelsError, match="0 labelled points"):
        resolve(missing, labelled)


def test_the_placer_holds_a_hand_label_and_places_the_others_round_it() -> None:
    anchors = np.array([[0.5, 0.5], [0.52, 0.5]])
    texts, sizes = ["first", "second"], [(0.08, 0.02), (0.08, 0.02)]
    free = place_labels(anchors, texts, anchors, [], sizes=sizes)
    held = place_labels(anchors, texts, anchors, [], sizes=sizes, pinned={0: (0.03, -0.05)})
    assert held[0].hand and held[0].box == pytest.approx((0.53, 0.45, 0.61, 0.47))
    assert not held[1].hand and held[1].overlaps == 0
    assert free[0].box != held[0].box


def test_a_hand_label_the_placer_would_move_is_refused_by_name() -> None:
    anchors = np.array([[0.5, 0.5], [0.9, 0.1]])
    texts, sizes = ["first", "second"], [(0.08, 0.02), (0.08, 0.02)]
    legend = Box("legend", 0.0, 0.0, 0.3, 0.3)
    with pytest.raises(HandLabelClash, match="'first' runs off the page"):
        place_labels(anchors, texts, anchors, [], sizes=sizes, pinned={0: (0.6, 0.0)})
    with pytest.raises(HandLabelClash, match="'first' is on the legend"):
        place_labels(anchors, texts, anchors, [legend], sizes=sizes, pinned={0: (-0.4, -0.4)})
    with pytest.raises(HandLabelClash, match="'first' is on the label 'second'"):
        place_labels(anchors, texts, anchors, [], sizes=sizes,
                     pinned={0: (0.4, -0.4), 1: (0.0, 0.0)})  # fmt: skip


def test_a_maps_hand_labels_reach_its_figure_and_record(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from af.map.finish import finish_map

    path = write(tmp_path, pin("ag100", "Te/25-cell", 0.03, -0.06) + hide("ag101", "Te/25-egg"))
    result = finish_map(
        synthetic_points(),
        chart="synthetic",
        scheme=SCHEME,
        window=Window("12m", SINCE),
        title="Synthetic by group",
        frame_size=14.0,
        must_show_since=SINCE,
        vaccine_labels={"ag100": "Te/25-cell", "ag101": "Te/25-egg"},
        out_pdf=tmp_path / "map" / "figure.pdf",
        created=dt.datetime(2026, 10, 31, 12, 0, tzinfo=dt.UTC),
        provenance={"inputs": {"chain": {"name": "synthetic", "sha256": "0" * 64}}},
        hand=load_hand_labels(path),
    )
    assert result.labels["ag100"].hand and "ag101" not in result.labels
    anchor, box = result.labels["ag100"].anchor, result.labels["ag100"].box
    assert box[:2] == pytest.approx((anchor[0] + 0.03, anchor[1] - 0.06))
    points = {p["id"]: p for p in json.loads(result.i7.read_text())["map"]["antigens"]}
    assert points["ag100"]["label_hand"] is True and "label" not in points["ag101"]


def test_a_map_names_its_hand_labels_file_beside_the_config(tmp_path: Path) -> None:
    from af.map.roundconfig import load_maps_config

    from .test_roundconfig import GOOD
    from .test_roundconfig import write as write_config

    text = GOOD.replace(
        'folder = "example-lab"\n',
        'folder = "example-lab"\nhand_labels = "labels-example-lab.toml"\n',
        1,
    )
    config, _ = load_maps_config(write_config(tmp_path, text))
    (m,) = [m for m in config.maps if m.folder == "example-lab"]
    assert m.hand_labels == (tmp_path / "labels-example-lab.toml").resolve()
