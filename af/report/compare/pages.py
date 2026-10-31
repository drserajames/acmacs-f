"""Pages at their own sizes, assembled by pdflatex: what the side-by-side documents are made of.

Each page is a TeX box shipped out at its own size (Sarah, 30 Oct 2026: "Keep the original page
sizes in side by side"), so a page of two 800-point maps is 1,600 points wide and nothing is
scaled or cropped. Inputs are linked into the work directory under plain names, so no path
reaches TeX unescaped.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from af.report.build import LATEX, pdf_pages, tex_escape

# In PDF points (TeX's bp), the unit page sizes are given in: TeX's pt is 1/72.27 inch.
MARGIN = 18.0  # round every page
GAP = 14.0  # between the pictures of a page


class PagesError(RuntimeError):
    """The document could not be assembled."""


def page_sizes(pdf: Path) -> list[tuple[float, float]]:
    """Every page's size as displayed (width, height in pt): a page rotated by 90 or 270 is
    reported turned, as a reader and pdftoppm see it."""
    out = subprocess.run(
        ["pdfinfo", "-f", "1", "-l", "100000", str(pdf)], capture_output=True, text=True
    )
    if out.returncode != 0:
        raise PagesError(f"{pdf}: pdfinfo failed: {out.stderr.strip()}")
    sizes = []
    pattern = r"Page\s+(\d+) size:\s+([\d.]+) x ([\d.]+).*\n.*Page\s+\1 rot:\s+(\d+)"
    for m in re.finditer(pattern, out.stdout):
        w, h, rot = float(m.group(2)), float(m.group(3)), int(m.group(4))
        sizes.append((h, w) if rot in (90, 270) else (w, h))
    if not sizes:
        raise PagesError(f"{pdf}: pdfinfo reported no pages")
    return sizes


@dataclass
class Document:
    """Pages of TeX, each shipped at its own size; :meth:`graphic` names an input PDF."""

    pages: list[str] = field(default_factory=list)
    inputs: dict[Path, str] = field(default_factory=dict)
    banner: str = ""  # plain text printed in red at the top of every page

    def graphic(self, pdf: Path) -> str:
        """The name TeX includes ``pdf`` by (linked into the work directory)."""
        pdf = pdf.resolve()
        if pdf not in self.inputs:
            self.inputs[pdf] = f"in{len(self.inputs)}.pdf"
        return self.inputs[pdf]

    def picture(self, pdf: Path, page: int = 1, overlay: Path | None = None) -> str:
        """A page of ``pdf`` at its own size, with ``overlay`` (same size) drawn over it."""
        base = rf"\includegraphics[page={page}]{{{self.graphic(pdf)}}}"
        if overlay is None:
            return base
        return rf"\makebox[0pt][l]{{{base}}}\includegraphics{{{self.graphic(overlay)}}}"

    def add(self, body: str) -> None:
        """One page: ``body`` is vertical material; the page is its size plus the margin."""
        self.pages.append(self.page(body))

    def page(self, body: str) -> str:
        """A page's TeX, not yet added (:meth:`add` appends; a cover can be inserted first)."""
        m = MARGIN
        if self.banner:
            body = rf"\hbox{{\color{{red}}\small {tex_escape(self.banner)}}}\kern4pt" + body
        return (
            rf"\setbox0=\vbox{{\kern{m}bp\hbox{{\kern{m}bp\vtop{{{body}}}\kern{m}bp}}\kern{m}bp}}"
            r"\pdfpagewidth=\wd0 \pdfpageheight=\ht0 \advance\pdfpageheight by \dp0 "
            r"\shipout\box0"
        )

    def write(self, out_pdf: Path) -> int:
        """Typeset to ``out_pdf``; returns its page count, checked against the pages added."""
        if not self.pages:
            raise PagesError(f"{out_pdf}: no pages")
        if shutil.which(LATEX) is None:
            raise PagesError(f"{LATEX} is not installed: the side-by-side documents need it")
        tex = "\n".join([
            r"\documentclass{article}",
            r"\usepackage[T1]{fontenc}\usepackage{helvet,graphicx,xcolor}",
            r"\renewcommand{\familydefault}{\sfdefault}\setlength{\parindent}{0pt}",
            r"\hoffset=-1in \voffset=-1in",
            r"\begin{document}",
            *self.pages,
            r"\end{document}",
        ])  # fmt: skip
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for source, name in self.inputs.items():
                if not source.is_file() or source.stat().st_size == 0:
                    raise PagesError(f"input missing or empty: {source}")
                (work / name).symlink_to(source)
            (work / "doc.tex").write_text(tex)
            run = subprocess.run(
                [LATEX, "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
                cwd=work, capture_output=True, text=True,
            )  # fmt: skip
            pdf = work / "doc.pdf"
            if run.returncode != 0 or not pdf.is_file():
                tail = (work / "doc.log").read_text(errors="replace")[-2000:]
                raise PagesError(f"{LATEX} failed on {out_pdf.name}:\n{tail}")
            n = pdf_pages(work / "doc.log")
            if n != len(self.pages):
                raise PagesError(f"{out_pdf.name}: {n} pages typeset, {len(self.pages)} added")
            out_pdf.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pdf, out_pdf)
        return n


def text_block(width: float, lines: list[str], size: str = r"\small") -> str:
    """A ragged-right block ``width`` pt wide; ``lines`` are TeX (escape plain text first)."""
    body = r"\par ".join(lines)
    return rf"\vtop{{\hsize={width:.1f}bp\raggedright{size} {body}\par}}"


def side_by_side(left: str, right: str, labels: tuple[str, str], stacked: bool = False) -> str:
    """Two pictures with their labels above them, beside each other (or one above the other)."""
    a = rf"\vtop{{\hbox{{\strut\footnotesize {tex_escape(labels[0])}}}\hbox{{{left}}}}}"
    b = rf"\vtop{{\hbox{{\strut\footnotesize {tex_escape(labels[1])}}}\hbox{{{right}}}}}"
    if stacked:
        return rf"\vtop{{\hbox{{{a}}}\kern{GAP}bp\hbox{{{b}}}}}"
    return rf"\hbox{{{a}\kern{GAP}bp{b}}}"
