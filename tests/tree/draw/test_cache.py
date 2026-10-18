"""Reusing a drawn figure: same inputs give the same bytes without a redraw; anything else draws."""

import json
from dataclasses import replace

import pytest

pytest.importorskip("numpy", reason="numpy not installed: af.tree.draw needs it (pyproject, WS1)")
pytest.importorskip("matplotlib", reason="matplotlib not installed: the renderer needs it")

from af.tree.draw.cache import (  # noqa: E402
    DrawCacheError,
    canonical,
    entry_key,
    reuse_or_draw,
)
from af.tree.draw.defaults import load_defaults  # noqa: E402
from af.tree.draw.figure import FigureConfig, make_figure  # noqa: E402
from af.util.artefacts import ArtefactError  # noqa: E402

from .synthetic import PARENTS, standard_tree  # noqa: E402

CODE = {"commit": "c0ffee", "release": None, "python": "3.12", "libraries": {"matplotlib": "x"}}


class Drawer:
    """A stand-in draw: writes each output with a run-numbered body, and counts its runs."""

    def __init__(self, outputs):
        self.outputs, self.runs = outputs, 0

    def __call__(self):
        self.runs += 1
        for name, path in self.outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{name} run {self.runs}")


def outputs(root):
    return {"figure.pdf": root / "report" / "figure.pdf", "book.pdf": root / "book" / "book.pdf"}


def test_same_inputs_reuse_the_entry_into_a_fresh_folder(tmp_path):
    first, second = outputs(tmp_path / "a"), outputs(tmp_path / "b")
    draw_a, draw_b = Drawer(first), Drawer(second)
    a = reuse_or_draw(tmp_path / "cache", {"tree": "t@1"}, CODE, first, draw_a, "tree h0")
    b = reuse_or_draw(tmp_path / "cache", {"tree": "t@1"}, CODE, second, draw_b, "tree h0")
    assert (a.hit, b.hit) == (False, True)
    assert a.key == b.key
    assert (draw_a.runs, draw_b.runs) == (1, 0)
    for name in first:
        assert second[name].read_bytes() == first[name].read_bytes()


@pytest.mark.parametrize(
    "parts, code",
    [({"tree": "t@2"}, CODE), ({"tree": "t@1"}, CODE | {"commit": "beef"})],
    ids=["another tree", "other code"],
)
def test_a_changed_input_or_code_draws_again(tmp_path, parts, code):
    first = outputs(tmp_path)
    reuse_or_draw(tmp_path / "c", {"tree": "t@1"}, CODE, first, Drawer(first), "t")
    out = outputs(tmp_path / "new")
    draw = Drawer(out)
    result = reuse_or_draw(tmp_path / "c", parts, code, out, draw, "t")
    assert not result.hit and draw.runs == 1


def test_no_cache_or_unknown_code_always_draws_and_stores_nothing(tmp_path):
    out = outputs(tmp_path)
    draw = Drawer(out)
    assert "no cache" in reuse_or_draw(None, {"tree": "t"}, CODE, out, draw, "t").reason
    unknown = reuse_or_draw(tmp_path / "c", {"tree": "t"}, None, out, draw, "t")
    assert "not identified" in unknown.reason and unknown.key is None
    assert draw.runs == 2
    assert not (tmp_path / "c").exists()


def test_a_damaged_entry_is_an_error_not_a_redraw(tmp_path):
    out = outputs(tmp_path / "a")
    key = reuse_or_draw(tmp_path / "c", {"tree": "t"}, CODE, out, Drawer(out), "t").key
    (tmp_path / "c" / key / "book.pdf").write_text("changed")
    with pytest.raises(DrawCacheError, match="book.pdf does not match"):
        reuse_or_draw(tmp_path / "c", {"tree": "t"}, CODE, outputs(tmp_path / "b"), Drawer({}), "t")


def test_a_draw_that_writes_nothing_leaves_no_entry(tmp_path):
    with pytest.raises(ArtefactError, match="missing"):
        reuse_or_draw(tmp_path / "c", {"tree": "t"}, CODE, outputs(tmp_path), Drawer({}), "t")
    assert not (tmp_path / "c").exists()


def test_output_names_must_be_plain_file_names(tmp_path):
    with pytest.raises(ValueError, match="plain file name"):
        reuse_or_draw(tmp_path, {}, CODE, {"a/b.pdf": tmp_path / "x"}, Drawer({}), "t")


def config(**kw):
    base = FigureConfig(
        title="TEST tree",
        window_start="2024-10",
        window_end="2026-10",
        select=replace(load_defaults().clades, min_share=0.1, min_window_leaves=10**9),
    )
    return replace(base, **kw)


def test_the_key_follows_every_config_field_and_not_set_order():
    a = canonical(config(marked_ids=frozenset({"EPI_ISL_0000002", "EPI_ISL_0000001"})))
    b = canonical(config(marked_ids=frozenset({"EPI_ISL_0000001", "EPI_ISL_0000002"})))
    assert entry_key({"figure": a}, CODE) == entry_key({"figure": b}, CODE)
    labels = replace(load_defaults().labels, max_share=0.5)
    c = canonical(config(labels=labels))
    assert entry_key({"figure": c}, CODE) != entry_key({"figure": a}, CODE)
    with pytest.raises(TypeError, match="no canonical form"):
        canonical({"x": object()})


def test_a_reused_figure_is_the_drawn_figure_byte_for_byte(tmp_path):
    def draw_into(root):
        out = {
            "figure.pdf": root / "figure.pdf",
            "figure.i7.json": root / "figure.i7.json",
            "figure.draw.json": root / "figure.draw.json",
        }
        draw = Drawer(out)

        def real():
            draw.runs += 1
            root.mkdir(parents=True, exist_ok=True)
            make_figure(standard_tree(), PARENTS, config(), out["figure.pdf"], {"tree": "x"})

        result = reuse_or_draw(
            tmp_path / "c", {"figure": canonical(config())}, CODE, out, real, "tree"
        )
        return out, result, draw.runs

    first, a, ran_a = draw_into(tmp_path / "a")
    second, b, ran_b = draw_into(tmp_path / "b")
    assert (a.hit, b.hit, ran_a, ran_b) == (False, True, 1, 0)
    for name in first:
        assert second[name].read_bytes() == first[name].read_bytes()
    assert json.loads(second["figure.draw.json"].read_text())["rows"] == 50
