"""Build a report PDF from its config: resolve figures, write LaTeX, run it, check it, record it.

Why LaTeX: today's reports are LaTeX; the figures are PDFs that LaTeX places losslessly; it
gives the contents page and page numbers for free; and TeX Live is on the Mac and on typical
HPC systems. What changes from today (B-report-layer §4.1): every failure is an error; figures
are copied into the build directory under their content hash, so the ``.tex`` has no absolute
paths and the build folder can move; and the output is checked before success is reported.
The checks are that every figure was typeset (its physical page is in the ``.aux``) and that the
PDF ends on the last figure's page.

Run: ``python -m af.report.build <config.toml> --figures <root> --out <dir>``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from af.report.config import ReportConfig, Section, load, period_first, period_last
from af.report.figures import FigureError, Resolved, resolve
from af.run import Job, JobFailed, LocalRunner
from af.util.artefacts import Artefact, sha256_path
from af.util.config import ConfigError

LATEX = "pdflatex"
MAX_PASSES = 4
MANIFEST_VERSION = 0  # PROVISIONAL: replace with the store's manifest format (I9)


class BuildError(RuntimeError):
    """The report could not be built; the message lists every reason found."""


def tex_escape(text: str) -> str:
    table = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}",
    }  # fmt: skip
    return "".join(table.get(c, c) for c in text)


def resolve_all(cfg: ReportConfig, root: Path) -> dict[str, Resolved]:
    """Resolve every slot, collecting all failures so one run lists every missing figure."""
    out: dict[str, Resolved] = {}
    errors: list[str] = []
    for slot in cfg.all_slots():
        try:
            out[slot] = resolve(root, slot, cfg.figures.pins.get(slot))
        except FigureError as err:
            errors.append(str(err))
    if errors:
        raise BuildError(f"{len(errors)} figure(s) not usable:\n  " + "\n  ".join(errors))
    placeholders = [slot for slot, r in out.items() if r.figure.placeholder]
    if placeholders and not cfg.figures.allow_placeholders:
        raise BuildError(
            f"{len(placeholders)} placeholder figure(s) and allow_placeholders is off: "
            + ", ".join(placeholders)
        )
    return out


def _label(slot: str) -> str:
    return "fig:" + re.sub(r"[^A-Za-z0-9]+", "-", slot)


def _caption(r: Resolved) -> str:
    when = r.created.strftime("%d %b %Y %H:%M %Z")
    tag = r"\textcolor{red}{PLACEHOLDER} " if r.figure.placeholder else ""
    return tag + tex_escape(f"{r.figure.title} ({r.version}, {when})")


def _cover(cfg: ReportConfig, n_placeholders: int, built_at: dt.datetime) -> list[str]:
    r = cfg.report
    first, last = period_first(cfg), period_last(cfg)
    period = first.strftime("%B %Y")
    if last != first:
        period += " -- " + last.strftime("%B %Y")
    lines = [
        r"\begin{titlepage}\centering\vspace*{60mm}",
        rf"{{\Huge {tex_escape(r.title)}\par}}\vspace{{12mm}}",
        rf"{{\Large {tex_escape(r.centre)}\par}}\vspace{{8mm}}",
        rf"{{\Large {period}\par}}\vspace{{20mm}}",
        rf"Data up to {r.data_cutoff.day} {r.data_cutoff.strftime('%B %Y')}\par",
        rf"Built {built_at.strftime('%d %B %Y %H:%M %Z')}\par",
    ]
    if n_placeholders:
        lines.append(
            rf"\vspace{{10mm}}{{\Large\color{{red}} DRAFT: {n_placeholders} placeholder "
            r"figure(s)\par}"
        )
    return [*lines, r"\end{titlepage}", r"\tableofcontents", r"\newpage"]


def _tree_pages(section: Section, figs: dict[str, Resolved], rel: dict[str, str]) -> list[str]:
    lines: list[str] = []
    for slot in section.slots:
        r = figs[slot]
        lines += [
            rf"\section{{{tex_escape(section.title)}: {tex_escape(r.figure.title)}}}",
            r"\begin{center}",
            r"\includegraphics[width=\textwidth,height=0.88\textheight,keepaspectratio]"
            rf"{{{rel[slot]}}}",
            rf"\\ \small {_caption(r)}\zlabel{{{_label(slot)}}}",
            r"\end{center}",
            r"\newpage",
        ]
    return lines


def _map_pages(section: Section, figs: dict[str, Resolved], rel: dict[str, str]) -> list[str]:
    cols, rows = section.grid
    per_page = cols * rows
    width = f"{0.98 / cols:.3f}\\linewidth"
    height = f"{0.8 / rows:.3f}\\textheight"
    lines: list[str] = []
    for window in section.windows:
        slots = [f"{s}/{window.name}" for s in section.slots]
        for start in range(0, len(slots), per_page):
            if start == 0:
                lines.append(rf"\section{{{tex_escape(section.title)} {tex_escape(window.title)}}}")
            lines.append(r"\begin{center}")
            for i, slot in enumerate(slots[start : start + per_page]):
                lines.append(
                    rf"\begin{{minipage}}[t]{{{width}}}\centering"
                    rf"\includegraphics[width=\linewidth,height={height},keepaspectratio]"
                    rf"{{{rel[slot]}}}\\ \scriptsize {_caption(figs[slot])}"
                    rf"\zlabel{{{_label(slot)}}}\end{{minipage}}"
                )
                lines.append(r"\par\medskip" if (i + 1) % cols == 0 else r"\hfill")
            lines += [r"\end{center}", r"\newpage"]
    return lines


def write_tex(
    cfg: ReportConfig, figs: dict[str, Resolved], build: Path, built_at: dt.datetime
) -> Path:
    """Copy figures in by content hash and write ``report.tex`` (relative paths only)."""
    fig_dir = build / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rel: dict[str, str] = {}
    for slot, r in figs.items():
        name = f"{r.figure.doc['figure']['sha256'][:16]}.pdf"
        shutil.copyfile(r.figure.pdf, fig_dir / name)
        rel[slot] = f"figures/{name}"
    lines = [
        r"\documentclass[a4paper,11pt]{article}",
        r"\usepackage[margin=15mm]{geometry}",
        r"\usepackage{graphicx,xcolor,zref-user,zref-abspage}",
        # Physical page on every \zlabel: \label pages are logical and the title page resets them.
        r"\makeatletter\zref@addprop{main}{abspage}\makeatother",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\parindent}{0pt}",
        r"\begin{document}",
        *_cover(cfg, sum(r.figure.placeholder for r in figs.values()), built_at),
    ]
    for section in cfg.sections:
        pages = _tree_pages if section.kind == "trees" else _map_pages
        lines += pages(section, figs, rel)
    lines.append(r"\end{document}")
    tex = build / "report.tex"
    tex.write_text("\n".join(lines) + "\n")
    return tex


def run_latex(tex: Path, runner: LocalRunner) -> int:
    """Run LaTeX until cross-references settle; any failure is fatal. Returns passes used."""
    pdf = tex.with_suffix(".pdf")
    for n in range(1, MAX_PASSES + 1):
        job = Job(
            name=f"latex-pass-{n}",
            command=[LATEX, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
                     tex.name],
            cwd=tex.parent,
            log=tex.parent / f"latex-pass-{n}.out",
            outputs=[Artefact(pdf)],
        )  # fmt: skip
        try:
            runner.run(job)
        except JobFailed as err:
            raise BuildError(f"{LATEX} pass {n} failed:\n{err}") from err
        log = tex.with_suffix(".log").read_text(errors="replace")
        if n >= 2 and "Rerun to get" not in log and "Label(s) may have changed" not in log:
            return n
    raise BuildError(f"cross-references still changing after {MAX_PASSES} {LATEX} passes")


def figure_pages(aux: Path) -> dict[str, int]:
    """Physical page of every figure label, from the .aux written by zref-abspage."""
    pattern = r"\\zref@newlabel\{(fig:[^}]*)\}\{.*?\\abspage\{(\d+)\}"
    return {m.group(1): int(m.group(2)) for m in re.finditer(pattern, aux.read_text())}


def pdf_pages(log: Path) -> int:
    """Page count as LaTeX reports it ("Output written on x.pdf (N pages, ...)")."""
    match = re.search(r"Output written on .*?\((\d+) pages?", log.read_text(errors="replace"))
    if not match:
        raise BuildError(f"{log}: no 'Output written' line; LaTeX produced no PDF")
    return int(match.group(1))


def check_output(tex: Path, slots: list[str]) -> int:
    """Every figure typeset, and the PDF ends on the last figure's page. Returns page count."""
    pages = figure_pages(tex.with_suffix(".aux"))
    missing = [slot for slot in slots if _label(slot) not in pages]
    if missing:
        raise BuildError(f"{len(missing)} figure(s) not typeset: {missing[:5]}")
    n = pdf_pages(tex.with_suffix(".log"))
    last = max(pages.values())
    if n != last:
        raise BuildError(f"{tex.stem}.pdf: {n} pages but the last figure is on page {last}")
    return n


def manifest(
    cfg: ReportConfig, config_path: Path, figs: dict[str, Resolved], final: Path,
    pages: int, passes: int, built_at: dt.datetime,
) -> dict[str, Any]:  # fmt: skip
    return {
        "manifest_version": MANIFEST_VERSION,
        "report": cfg.report.id,
        "kind": cfg.report.kind,
        "built": built_at.isoformat(),
        "config": {"path": str(config_path), "sha256": sha256_path(config_path)},
        "latex": {"engine": LATEX, "passes": passes},
        "output": {"pdf": final.name, "sha256": sha256_path(final), "pages": pages},
        "placeholders": sum(r.figure.placeholder for r in figs.values()),
        "figures": [
            {
                "slot": slot,
                "version": r.version,
                "pinned": r.pinned,
                "placeholder": r.figure.placeholder,
                "created": r.created.isoformat(),
                "i7": str(r.figure.i7_path),
                "i7_sha256": sha256_path(r.figure.i7_path),
                "pdf_sha256": r.figure.doc["figure"]["sha256"],
                "provenance": r.figure.doc["provenance"],
            }
            for slot, r in figs.items()
        ],
    }


def build(config_path: Path, figures_root: Path, out_dir: Path) -> Path:
    """Build the report; return the PDF path. Raises on any problem, leaving no final PDF."""
    cfg = load(config_path)
    built_at = dt.datetime.now(dt.UTC)  # shown on the cover and recorded; decides nothing
    figs = resolve_all(cfg, figures_root)
    build_dir = out_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)  # never reuse a stale .aux/.toc or an old figure
    build_dir.mkdir(parents=True)
    tex = write_tex(cfg, figs, build_dir, built_at)
    passes = run_latex(tex, LocalRunner())
    pages = check_output(tex, cfg.all_slots())
    final = out_dir / f"{cfg.report.id}.pdf"
    shutil.copyfile(tex.with_suffix(".pdf"), final)
    record = manifest(cfg, config_path, figs, final, pages, passes, built_at)
    (out_dir / f"{cfg.report.id}.manifest.json").write_text(json.dumps(record, indent=1))
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a report PDF from its TOML config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--figures", type=Path, required=True, help="figure root (provisional)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        pdf = build(args.config, args.figures, args.out)
    except (BuildError, ConfigError, FigureError) as err:
        print(f"report build FAILED: {err}", file=sys.stderr)
        return 1
    print(pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
