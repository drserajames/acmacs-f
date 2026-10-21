"""The addenda's page frame: one LaTeX document per addendum, built from a list of pages.

Every addendum (signature pages, serum coverage, serum circles) is A4 landscape with the same
cover and closing line, so one module sets them (Q227 b; 09 owns it, 08 and 10 use it). It
writes the document the way af.report.build writes the report: each placed PDF copied into the
build directory under its content hash (no absolute paths in the ``.tex``), LaTeX run until its
cross-references settle (``af.report.build.run_latex``), and the output checked: every placed
PDF typeset (its ``\\zlabel`` page in the ``.aux``) and the page count what the list says.

Pages:

- :class:`Cover`: the WHO consultation title, "Addendum N (kind)" with an optional suffix (a
  bespoke variant, " -- VIDRL"), the date and the centre's lines, as the delivered addenda.
- :class:`SigPage`: a signature page, its tree panel and its map grid placed at given boxes
  (points from the page's top-left), each scaled to fit its box keeping its aspect.

To come, for serum coverage and the circle addenda (10, 20 Oct 2026): a text page with its own
margins, and a grid page (cells that may be empty, a caption pinned to the page's foot).
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from af.report.build import BuildError, _label, run_latex, tex_escape
from af.run import LocalRunner

PAGE = (841.89, 595.28)  # A4 landscape, points

Box = tuple[float, float, float, float]  # left, top, width, height: points from top-left


@dataclass(frozen=True)
class Cover:
    title: Sequence[str]  # e.g. the WHO consultation's title, in lines
    addendum: str  # "Addendum 4 (integrated genetic-antigenic analyses)"
    date: str  # as printed: "21 September 2026"
    centre: Sequence[str]  # the centre's lines
    suffix: str = ""


@dataclass(frozen=True)
class SigPage:
    name: str  # names the page in the build record and its label
    tree: Path
    maps: Path
    tree_box: Box
    maps_box: Box


Page = Cover | SigPage


def _cover(c: Cover) -> list[str]:
    """The delivered addendum 4's cover (its addendum-4.tex), on a page of its own."""
    return [
        r"\begin{titlepage}\begin{quotation}\vspace*{10pt}",
        r"{\fontsize{22}{26}\selectfont\noindent\textbf{"
        + r"\\ ".join(tex_escape(t) for t in c.title)
        + r"}\par}",
        r"\vspace{90pt}{\fontsize{19}{24}\selectfont\noindent",
        rf"\vspace{{10pt}}\noindent {tex_escape(c.addendum + c.suffix)}",
        rf"\vspace{{50pt}}\noindent {tex_escape(c.date)}\par}}",
        r"\vspace{100pt}{\large\noindent "
        + r"\par\vspace{10pt}\noindent ".join(tex_escape(t) for t in c.centre)
        + r"\par}",
        r"\end{quotation}\end{titlepage}",
    ]


def _node(box: Box, path: str) -> str:
    left, top, width, height = box
    return (
        r"\node[anchor=north west,inner sep=0pt] at "
        rf"([xshift={left:.2f}pt,yshift=-{top:.2f}pt]current page.north west) "
        rf"{{\includegraphics[width={width:.2f}pt,height={height:.2f}pt,keepaspectratio]{{{path}}}}};"
    )


def _sig_page(p: SigPage, tree: str, maps: str) -> list[str]:
    return [
        r"\newpage\thispagestyle{empty}\null",
        r"\begin{tikzpicture}[remember picture,overlay]",
        _node(p.tree_box, tree),
        _node(p.maps_box, maps),
        r"\end{tikzpicture}",
        rf"\zlabel{{{_label(p.name)}}}",
    ]


def _copy(path: Path, figures: Path) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        raise BuildError(f"{path}: missing or empty")
    name = f"{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}.pdf"
    shutil.copyfile(path, figures / name)
    return f"figures/{name}"


def build(pages: Sequence[Page], out: Path, generated: str | None = None) -> Path:
    """Write ``out/<name>.tex`` for ``pages``, run LaTeX, check it; return the PDF.

    ``generated``: the "Report generated: …" line at the foot of a closing page, or None (the
    delivered addenda 1-3 have none)."""
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    lines = [
        r"\documentclass[a4paper,12pt,landscape]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{times}",
        r"\usepackage[landscape,top=3em,bottom=5em,left=2.5em,right=2.5em]{geometry}",
        r"\usepackage{graphicx,tikz,zref-user,zref-abspage}",
        r"\makeatletter\zref@addprop{main}{abspage}\makeatother",
        r"\setlength{\parindent}{0pt}",
        r"\begin{document}",
    ]
    names = []
    for page in pages:
        if isinstance(page, Cover):
            lines += _cover(page)
        else:
            lines += _sig_page(page, _copy(page.tree, figures), _copy(page.maps, figures))
            names.append(page.name)
    if generated is not None:
        lines += [r"\newpage\null\vfill", rf"{{\tiny Report generated: {tex_escape(generated)}}}"]
    lines.append(r"\end{document}")
    tex = out / "addendum.tex"
    tex.write_text("\n".join(lines) + "\n")
    run_latex(tex, LocalRunner())
    _check(tex, names, len(pages) + (generated is not None))
    return tex.with_suffix(".pdf")


def _check(tex: Path, names: Sequence[str], expected_pages: int) -> None:
    from af.report.build import figure_pages, pdf_pages

    placed = figure_pages(tex.with_suffix(".aux"))
    missing = [n for n in names if _label(n) not in placed]
    if missing:
        raise BuildError(f"{len(missing)} page(s) not typeset: {missing[:5]}")
    n = pdf_pages(tex.with_suffix(".log"))
    if n != expected_pages:
        raise BuildError(f"{tex.stem}.pdf: {n} pages, the list has {expected_pages}")
