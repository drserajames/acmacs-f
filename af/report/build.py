"""Build a report PDF from its config: resolve figures, write LaTeX, run it, check it, record it.

Why LaTeX: today's reports are LaTeX; the figures are PDFs that LaTeX places losslessly; it
gives the contents page and page numbers for free; and TeX Live is on the Mac and on typical
HPC systems. What changes from today (B-report-layer §4.1): every failure is an error; figures
are copied into the build directory under their content hash, so the ``.tex`` has no absolute
paths and the build folder can move; and the output is checked before success is reported.
The checks are that every figure was typeset (its physical page is in the ``.aux``) and that the
PDF ends on the last figure's page.

Before any LaTeX runs, the figures' store provenance is checked (:mod:`af.report.provenance`):
the refs are consistent, they resolve in the store, and no unpinned figure is stale. After a
successful build, the report manifest (store refs, in the store's snapshot format) is written
where ``--manifest`` says, normally ``acmacs-f-data/reports/<id>/manifest.json``. A build
record with the figure hashes, slots and LaTeX details sits beside the PDF.

Run: ``python -m af.report.build <config.toml> --figures <root> --store <root>
--manifest <path> --out <dir>``.
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

from af.report.config import Meeting, ReportConfig, Section, load, period_first, period_last
from af.report.figures import FigureError, Resolved, resolve
from af.report.provenance import ProvenanceError, StoreUse, check_against_store, collect
from af.run import Job, JobFailed, LocalRunner
from af.store import Store, StoreError, write_manifest
from af.util.artefacts import Artefact, sha256_path
from af.util.config import ConfigError

LATEX = "pdflatex"
MAX_PASSES = 4
BUILD_RECORD_VERSION = 1


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
    return out


def check_bring_up(cfg: ReportConfig, figs: dict[str, Resolved], use: StoreUse) -> None:
    """Placeholders and figures not drawn from the store need bring-up mode (allow_placeholders)."""
    placeholders = [slot for slot, r in figs.items() if r.figure.placeholder]
    if (placeholders or use.not_from_store) and not cfg.figures.allow_placeholders:
        parts = []
        if placeholders:
            parts.append(f"{len(placeholders)} placeholder(s): {', '.join(placeholders)}")
        if use.not_from_store:
            parts.append(
                f"{len(use.not_from_store)} not drawn from the store: "
                + ", ".join(use.not_from_store)
            )
        raise BuildError("allow_placeholders is off and there are " + "; ".join(parts))


def _label(slot: str) -> str:
    return "fig:" + re.sub(r"[^A-Za-z0-9]+", "-", slot)


def _caption(r: Resolved) -> str:
    when = r.created.strftime("%d %b %Y %H:%M %Z")
    tag = r"\textcolor{red}{PLACEHOLDER} " if r.figure.placeholder else ""
    return tag + tex_escape(f"{r.figure.title} ({r.version}, {when})")


def _meeting(meeting: Meeting) -> str:
    """ "21-24 September 2026" (one month), "30 September - 3 October 2026" (two), from config."""
    a, b = meeting.start, meeting.end
    if a == b:
        return f"{a.day} {a.strftime('%B %Y')}"
    if (a.year, a.month) == (b.year, b.month):
        return f"{a.day}--{b.day} {b.strftime('%B %Y')}"
    return f"{a.day} {a.strftime('%B')} -- {b.day} {b.strftime('%B %Y')}"


def _cover(
    cfg: ReportConfig, n_placeholders: int, n_not_from_store: int, built_at: dt.datetime
) -> list[str]:
    r = cfg.report
    first, last = period_first(cfg), period_last(cfg)
    period = first.strftime("%B %Y")
    if last != first:
        period += " -- " + last.strftime("%B %Y")
    lines = [
        r"\begin{titlepage}\centering\vspace*{60mm}",
        rf"{{\Huge {tex_escape(r.title)}\par}}\vspace{{12mm}}",
        rf"{{\Large {tex_escape(r.centre)}\par}}\vspace{{8mm}}",
        rf"{{\Large {tex_escape(r.subtitle)}\par}}\vspace{{6mm}}" if r.subtitle else "",
        rf"{{\Large {_meeting(r.meeting)}\par}}\vspace{{6mm}}" if r.meeting else "",
        rf"{{\Large {period}\par}}\vspace{{20mm}}",
        rf"Data up to {r.data_cutoff.day} {r.data_cutoff.strftime('%B %Y')}\par",
        rf"Built {built_at.strftime('%d %B %Y %H:%M %Z')}\par",
    ]
    if n_placeholders or n_not_from_store:
        lines.append(
            rf"\vspace{{10mm}}{{\Large\color{{red}} DRAFT: {n_placeholders} placeholder "
            rf"figure(s), {n_not_from_store} not drawn from the store\par}}"
        )
    return [*lines, r"\end{titlepage}", r"\tableofcontents", r"\newpage"]


def _tree_pages(section: Section, figs: dict[str, Resolved], rel: dict[str, str]) -> list[str]:
    lines: list[str] = []
    for slot in section.slots:
        r = figs[slot]
        # One tree: the section title is its heading. Several: say which tree each page is.
        title = section.title if len(section.slots) == 1 else f"{section.title}: {r.figure.title}"
        lines += [
            rf"\{_level(section)}{{{tex_escape(title)}}}",
            r"\begin{center}",
            r"\includegraphics[width=\textwidth,height=0.88\textheight,keepaspectratio]"
            rf"{{{rel[slot]}}}",
            rf"\\ \small {_caption(r)}\zlabel{{{_label(slot)}}}",
            r"\end{center}",
            r"\newpage",
        ]
    return lines


def _level(section: Section) -> str:
    """Sections under a group are subsections; ungrouped ones are sections."""
    return "subsection" if section.group else "section"


def _grid_pages(
    section: Section, months: list[str], figs: dict[str, Resolved], rel: dict[str, str]
) -> list[str]:
    """Map and geo pages: a columns x rows grid per page, blank cells left empty.

    Landscape sections turn the page (pdflscape): the report's map grids are 3 x 2 across.
    """
    cols, rows = section.grid
    width = f"{0.98 / cols:.3f}\\linewidth"
    # Fixed gaps, not \hfill: glue inside a centred paragraph shifts a row whose last cell is
    # blank off the column grid (seen on the H3 map pages).
    gap = f"{0.02 / (cols - 1):.4f}\\linewidth" if cols > 1 else "0pt"
    height = f"{0.8 / rows:.3f}\\textheight"
    lines: list[str] = []
    for n, (heading, cells) in enumerate(section.pages(months)):
        if section.landscape:
            lines.append(r"\begin{landscape}")
        if n == 0 or heading:  # the section's first page, and the first page of each map window
            lines.append(
                rf"\{_level(section)}{{{tex_escape(f'{section.title} {heading}'.strip())}}}"
            )
        for start in range(0, len(cells), cols):
            row = []
            for slot in cells[start : start + cols]:
                if slot is None:
                    row.append(rf"\makebox[{width}]{{}}")  # a blank cell keeps its width
                    continue
                row.append(
                    rf"\begin{{minipage}}[t]{{{width}}}\centering"
                    rf"\includegraphics[width=\linewidth,height={height},keepaspectratio]"
                    rf"{{{rel[slot]}}}\\ \scriptsize {_caption(figs[slot])}"
                    rf"\zlabel{{{_label(slot)}}}\end{{minipage}}"
                )
            lines.append(r"\noindent" + rf"\hspace{{{gap}}}".join(row) + r"\par\medskip")
        if section.landscape:
            lines.append(r"\end{landscape}")
        else:
            lines.append(r"\newpage")
    return lines


def write_tex(
    cfg: ReportConfig, figs: dict[str, Resolved], build: Path, built_at: dt.datetime,
    n_not_from_store: int = 0,
) -> Path:  # fmt: skip
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
        r"\usepackage{graphicx,xcolor,pdflscape,zref-user,zref-abspage}",
        # Physical page on every \zlabel: \label pages are logical and the title page resets them.
        r"\makeatletter\zref@addprop{main}{abspage}\makeatother",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\parindent}{0pt}",
        # Grouped reports (subtype > figure) are unnumbered, like the delivered VCM report.
        r"\setcounter{secnumdepth}{0}" if any(sec.group for sec in cfg.sections) else "",
        r"\begin{document}",
        *_cover(cfg, sum(r.figure.placeholder for r in figs.values()), n_not_from_store, built_at),
    ]
    group = None
    for section in cfg.sections:
        if section.group and section.group != group:
            lines.append(rf"\section*{{{tex_escape(section.group)}}}")
            lines.append(rf"\addcontentsline{{toc}}{{section}}{{{tex_escape(section.group)}}}")
        group = section.group
        if section.kind == "trees":
            lines += _tree_pages(section, figs, rel)
        else:
            lines += _grid_pages(section, cfg.months(), figs, rel)
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


def af_source() -> dict[str, Any]:
    """Which af code built this: version, and the git commit of the source actually imported.

    A report must say what produced it. When af runs from a git checkout (an editable install)
    the commit and whether it had uncommitted changes are recorded; from an installed wheel there
    is no checkout, and ``commit`` is null rather than guessed.
    """
    import subprocess

    import af

    source = Path(af.__file__).resolve().parent
    record: dict[str, Any] = {"version": af.__version__, "source": str(source),
                              "commit": None, "dirty": None}  # fmt: skip

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(source), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != source.parent:
            return record  # af sits inside some other repository (e.g. a home-rooted one)
        record["commit"] = git("rev-parse", "HEAD")
        record["dirty"] = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        pass  # not a git checkout: commit stays null, which the record says plainly
    return record


def build_record(
    cfg: ReportConfig, config_path: Path, figs: dict[str, Resolved], use: StoreUse,
    final: Path, pages: int, passes: int, built_at: dt.datetime, manifest_path: Path | None,
) -> dict[str, Any]:  # fmt: skip
    """Everything about this build beyond the store refs: slots, figure hashes, LaTeX."""
    return {
        "build_record_version": BUILD_RECORD_VERSION,
        "af": af_source(),
        "report": cfg.report.id,
        "kind": cfg.report.kind,
        "built": built_at.isoformat(),
        "config": {"path": str(config_path), "sha256": sha256_path(config_path)},
        "latex": {"engine": LATEX, "passes": passes},
        "output": {"pdf": final.name, "sha256": sha256_path(final), "pages": pages},
        "report_manifest": (
            {"path": str(manifest_path), "sha256": sha256_path(manifest_path)}
            if manifest_path else None
        ),
        "store": use.to_json(),
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
    }  # fmt: skip


def check_provenance(
    cfg: ReportConfig, figs: dict[str, Resolved], store_root: Path | None,
    manifest_path: Path | None, *, deep: bool,
) -> StoreUse:  # fmt: skip
    """All provenance checks, before any LaTeX: a problem found after typesetting is wasted work."""
    use = collect(figs)
    check_bring_up(cfg, figs, use)
    if use.refs:
        if store_root is None or manifest_path is None:
            raise BuildError(
                f"figures name {len(use.refs)} store version(s): --store and --manifest are "
                "required to check them and record them"
            )
        check_against_store(use, figs, Store.open(store_root), deep=deep)
    elif manifest_path is not None:
        raise BuildError(
            "no figure was drawn from the store, so there is no report manifest to write; "
            "this is a bring-up build (drop --manifest)"
        )
    return use


def build(
    config_path: Path, figures_root: Path, out_dir: Path, *,
    store_root: Path | None = None, manifest_path: Path | None = None, deep: bool = False,
) -> Path:  # fmt: skip
    """Build the report; return the PDF path. Raises on any problem, leaving no final PDF."""
    cfg = load(config_path)
    built_at = dt.datetime.now(dt.UTC)  # shown on the cover and recorded; decides nothing
    figs = resolve_all(cfg, figures_root)
    use = check_provenance(cfg, figs, store_root, manifest_path, deep=deep)
    build_dir = out_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)  # never reuse a stale .aux/.toc or an old figure
    build_dir.mkdir(parents=True)
    tex = write_tex(cfg, figs, build_dir, built_at, len(use.not_from_store))
    passes = run_latex(tex, LocalRunner())
    pages = check_output(tex, cfg.all_slots())
    final = out_dir / f"{cfg.report.id}.pdf"
    shutil.copyfile(tex.with_suffix(".pdf"), final)
    if manifest_path is not None:
        description = f"report {cfg.report.id}: {final.name} sha256 {sha256_path(final)}"
        write_manifest(manifest_path, use.refs, description)
    record = build_record(
        cfg, config_path, figs, use, final, pages, passes, built_at, manifest_path
    )
    (out_dir / f"{cfg.report.id}.build.json").write_text(json.dumps(record, indent=1))
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a report PDF from its TOML config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--figures", type=Path, required=True, help="figure root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--store", type=Path, help="store root; required when figures name refs")
    parser.add_argument("--manifest", type=Path, help="where to write the report manifest")
    parser.add_argument("--deep", action="store_true", help="re-hash every store file")
    args = parser.parse_args(argv)
    try:
        pdf = build(
            args.config, args.figures, args.out,
            store_root=args.store, manifest_path=args.manifest, deep=args.deep,
        )  # fmt: skip
    except (BuildError, ConfigError, FigureError, ProvenanceError, StoreError) as err:
        print(f"report build FAILED: {err}", file=sys.stderr)
        return 1
    print(pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
