"""The leaf book: every drawn leaf on a readable row, paginated, indexed, searchable."""

import json
import re
from dataclasses import replace

import pytest

pytest.importorskip("numpy", reason="numpy not installed: af.tree.draw needs it (pyproject, WS1)")
pytest.importorskip("matplotlib", reason="matplotlib not installed: the book renderer needs it")

from af.tree.draw.book import (  # noqa: E402
    BookGeometry,
    BookLayoutError,
    make_book,
    page_ranges,
)
from af.tree.draw.defaults import load_defaults  # noqa: E402
from af.tree.draw.figure import FigureConfig, arrange  # noqa: E402

from .synthetic import PARENTS, standard_tree  # noqa: E402


def config():
    return FigureConfig(
        title="TEST tree",
        window_start="2024-10",
        window_end="2026-10",
        select=replace(load_defaults().clades, min_share=0.1, min_window_leaves=10**9),
    )


def test_page_ranges_cover_every_row_once():
    assert page_ranges(10, 4) == [(0, 3), (4, 7), (8, 9)]
    assert page_ranges(8, 4) == [(0, 3), (4, 7)]


def test_book_has_index_then_every_row(tmp_path):
    pdf = tmp_path / "book.pdf"
    g = BookGeometry(height=200.0)  # few rows per page, so the synthetic tree spans pages
    report = make_book(standard_tree(), PARENTS, config(), pdf, {"tree": "x"}, geometry=g)
    layout, _, _, hz = arrange(standard_tree(), PARENTS, config())
    assert report["rows"] == layout.n_rows == 50
    assert report["tree_pages"] == -(-50 // g.rows_per_page)
    assert report["pages"] == report["index_pages"] + report["tree_pages"]
    assert [b["letter"] for b in report["bands"]] == [h.letter for h in hz]
    first_tree_page = report["index_pages"] + 1
    assert report["bands"][0]["first_page"] == first_tree_page
    assert sum(c["leaves"] for c in report["clades"]) == 50
    assert json.loads((tmp_path / "book.book.json").read_text())["pages"] == report["pages"]
    raw = pdf.read_bytes()
    assert b"/FontFile2" in raw  # Type 42: searchable
    assert len(re.findall(rb"/Type /Page\b", raw)) == report["pages"]


def test_book_refuses_rows_too_wide_for_the_page(tmp_path):
    g = BookGeometry(font=40.0)
    with pytest.raises(BookLayoutError, match="for the tree"):
        make_book(standard_tree(), PARENTS, config(), tmp_path / "b.pdf", {}, geometry=g)
