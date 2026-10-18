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
import zoneinfo
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NamedTuple

from af.geo.records import CLADE_NOT_DEFINED, CLADE_NOT_IN_SCHEME
from af.report import text
from af.report.config import (
    Meeting,
    ReportConfig,
    Section,
    lines,
    load,
)
from af.report.figures import FigureError, Resolved, resolve
from af.report.howmade import figure_warnings
from af.report.provenance import (
    ProvenanceError,
    StoreUse,
    check_against_store,
    collect,
    write_report_manifest,
)
from af.run import Job, JobFailed, LocalRunner
from af.store import Store, StoreError
from af.util.artefacts import Artefact, sha256_path
from af.util.config import ConfigError

LATEX = "pdflatex"
GEO_CLADE_KEY = Path("geo") / "clade-key.json"  # written by af.geo.build beside its figures
HEX = re.compile(r"#[0-9a-fA-F]{6}")
MAX_PASSES = 4
BUILD_RECORD_VERSION = 1


GEOMETRY = r"\usepackage[top=3em,bottom=5em,left=2.5em,right=2.5em]{geometry}"
UNRECORDED = "NOT RECORDED"  # what a gap once printed; never on a report page (00, 18 Oct 2026)


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


def _figure_mark(r: Resolved, slot: str, size: str) -> str:
    """What goes under a figure: its page label, and a red PLACEHOLDER line for a stand-in.

    No caption and no "(version, time)" stamp under a real figure (Sarah, Q201 a, 16 Oct 2026):
    the delivered report has none, and a figure's provenance is in its I7, the build record, the
    report manifest, the how-made notes and the PDF metadata. The ``\\zlabel`` stays: it is how
    :func:`check_output` proves every figure was typeset.
    """
    label = rf"\zlabel{{{_label(slot)}}}"
    if not r.figure.placeholder:
        return label
    return rf"\\ {size} \textcolor{{red}}{{PLACEHOLDER {tex_escape(r.figure.title)}}}" + label


def _pdf_metadata(cfg: ReportConfig, built_at: dt.datetime) -> str:
    """Provenance in the PDF's own metadata (Q201 a: it leaves the pages, it stays in the PDF):
    title, the report id and period, and which af built it when. The report manifest is written
    after the PDF (it records the PDF's hash), so the PDF cannot carry the manifest's."""
    r = cfg.report
    source = af_source()
    commit = (source.get("commit") or "unknown")[:12]
    release = ((source.get("release") or {}).get("commit") or "none")[:12]
    period = f"{r.period.first} to {r.period.last}"
    fields = {
        "pdftitle": " ".join([*lines(r.title), *lines(r.subtitle)]),
        "pdfsubject": f"{r.id} ({r.kind}), period {period}, data up to {r.data_cutoff.isoformat()}",
        "pdfkeywords": f"af {commit}; release {release}; built {built_at.isoformat()}",
        "pdfcreator": "af.report.build",
    }
    body = ", ".join(f"{k}={{{tex_escape(v)}}}" for k, v in fields.items())
    return rf"\hypersetup{{{body}}}"


def _meeting(meeting: Meeting) -> str:
    """ "21-24 September 2026" (one month), "30 September -- 3 October 2026" (two), from config.

    One month uses a hyphen, as the delivered report prints it."""
    a, b = meeting.start, meeting.end
    if a == b:
        return f"{a.day} {a.strftime('%B %Y')}"
    if (a.year, a.month) == (b.year, b.month):
        return f"{a.day}-{b.day} {b.strftime('%B %Y')}"
    return f"{a.day} {a.strftime('%B')} -- {b.day} {b.strftime('%B %Y')}"


def _cover(
    cfg: ReportConfig, n_placeholders: int, n_not_from_store: int, built_at: dt.datetime
) -> list[str]:
    """The cover as the delivered report sets it (its report.tex): bold title lines with the
    subtitle as the last, the meeting, then the centre's lines; page 1, unnumbered.

    The period, data cutoff and build time are not on the cover (Sarah, Q203 a, 16 Oct 2026):
    the build record and the PDF metadata carry them, and the closing page the build time."""
    r = cfg.report
    title = [tex_escape(t) for t in [*lines(r.title), *lines(r.subtitle)]]
    out = [
        r"\thispagestyle{empty}",
        r"\begin{quotation}",
        r"\vspace*{130pt}",
        r"{\fontsize{22}{26}\selectfont\noindent\textbf{" + r"\\ ".join(title) + r"}\par}",
        r"\vspace{90pt}{\fontsize{19}{24}\selectfont\vspace{60pt}",
        rf"{_meeting(r.meeting)}\par}}" if r.meeting else r"\par}",
        r"\vspace{180pt}{\large",
        r"\par\vspace{10pt}".join(tex_escape(c) for c in lines(r.centre)) + r"\par}",
    ]
    if n_placeholders or n_not_from_store:
        out.append(
            rf"\vspace{{10mm}}{{\Large\color{{red}} DRAFT: {n_placeholders} placeholder "
            rf"figure(s), {n_not_from_store} not drawn from the store\par}}"
        )
    return [*out, r"\end{quotation}", r"\newpage", r"\tableofcontents", r"\newpage"]


Intro = tuple[list[str], list[str]]  # (paragraphs, gaps: facts no artefact records)


class KeyRow(NamedTuple):
    """A clade key's row as printed: a clade (fill as 6 hex digits, the delivered black ring), or
    an uncoloured class (Q215) in the swatch its dots are drawn in: fill "" for none, its ring."""

    label: str
    fill: str
    edge: str = ""
    edge_width: float | None = None


Key = list[KeyRow]


def clade_keys(
    cfg: ReportConfig, figs: dict[str, Resolved], root: Path
) -> tuple[dict[int, Key], dict[str, Any] | None]:
    """The clade colour key of each geo section that shows one, and what the build record keeps.

    The key is the geo build's (one file beside its figures), so it is checked against the
    figures the report places: every coloured dot's clade must be in the key in the colour it
    is drawn in, and every uncoloured dot's class in the swatch it is drawn in, or the key
    belongs to another geo build. Rows are printed newest first, as the delivered report lists
    them (the key file is in scheme order), then the uncoloured classes in the key's order, each
    in the section's words (Sarah, Q215: "mark unsequenced & clade not defined on different
    lines"). A class the figures draw or the key lists needs words; a dot whose clade is not
    defined for a reason the intro does not explain (clade_not_defined_states) is refused."""
    shown = {i: section for i, section in enumerate(cfg.sections) if section.clade_key}
    if not shown:
        return {}, None
    path = root / GEO_CLADE_KEY
    if not path.is_file():
        raise BuildError(f"{path}: missing, and {len(shown)} geo section(s) show its clade key")
    doc = json.loads(path.read_text())
    keys: dict[int, Key] = {}
    counts: dict[str, dict[str, int]] = {}
    errors: list[str] = []
    for i, section in shown.items():
        where = f"sections[{i}] ({section.title})"
        if section.clade_key not in doc:
            errors.append(f"{where}: {section.clade_key!r} is not in {path} (has {sorted(doc)})")
            continue
        entries = doc[section.clade_key]["entries"]
        colours = {e["legend"]: e["colour"] for e in entries}
        errors += [f"{where}: {e['legend']!r} has colour {e['colour']!r}, not #rrggbb"
                   for e in entries if not HEX.fullmatch(e["colour"])]  # fmt: skip
        classes = {c["class"]: c for c in doc[section.clade_key].get("uncoloured_classes", [])}
        errors += _class_problems(where, section, classes)
        for _, cells in section.pages(cfg.months()):
            for slot in (c for c in cells if c is not None):
                for loc in figs[slot].figure.doc.get("geo", {}).get("locations", []):
                    errors += _dot_problems(slot, loc["points"], colours, classes, section)
        rows = [KeyRow(e["legend"], e["colour"][1:]) for e in reversed(entries)]
        # A class no dot in the window is drawn in has no row (00, 18 Oct 2026: a key row for
        # nothing, as "clade not defined (0 viruses)" on B/Vic, says nothing).
        rows += [_class_row(section.clade_key_classes[cls], c) for cls, c in classes.items()
                 if cls in section.clade_key_classes and c["window"] > 0]  # fmt: skip
        keys[i] = rows
        counts[section.title] = {cls: c["window"] for cls, c in classes.items()}
    if errors:
        found = sorted(set(errors))
        raise BuildError(f"{len(found)} clade key problem(s):\n  " + "\n  ".join(found))
    record = {
        "path": str(path), "sha256": sha256_path(path),
        "sections": {cfg.sections[i].title: cfg.sections[i].clade_key for i in shown},
        "uncoloured_classes": counts,  # dots per class over the window, per section
    }  # fmt: skip
    return keys, record


def _class_problems(where: str, section: Section, classes: dict[str, Any]) -> list[str]:
    """Words for every class the key lists, a listed class for every one worded (the key
    always lists not_sequenced and clade_not_defined, clade_not_in_scheme only when drawn), a
    drawable swatch, and every recorded reason for a clade not defined explained."""
    errors = []
    for cls in classes:
        if cls not in section.clade_key_classes:
            errors.append(f"{where}: the key lists uncoloured class {cls!r}; clade_key_classes "
                          "has no words for it")  # fmt: skip
    for cls in section.clade_key_classes:
        if cls not in classes and cls != CLADE_NOT_IN_SCHEME:
            errors.append(f"{where}: clade_key_classes has {cls!r}, the key does not list it "
                          "(a geo build before Q215?)")  # fmt: skip
    for cls, c in classes.items():
        fill_ok = c.get("fill") == "transparent" or HEX.fullmatch(str(c.get("fill")))
        width = c.get("edge_width")
        if not (fill_ok and HEX.fullmatch(str(c.get("edge")))
                and isinstance(width, int | float) and width >= 0):  # fmt: skip
            errors.append(
                f"{where}: uncoloured class {cls!r} swatch {c.get('fill')!r}, "
                f"{c.get('edge')!r}, {width!r} is not #rrggbb/transparent, #rrggbb, >= 0"
            )
    states = classes.get(CLADE_NOT_DEFINED, {}).get("states", {})
    explained = section.clade_not_defined_states
    unexplained = {s: n for s, n in states.items() if n and s not in explained}
    if unexplained and CLADE_NOT_DEFINED in section.clade_key_classes:
        errors.append(
            f"{where}: viruses whose clade is not defined for a reason the intro does not "
            f"explain {unexplained} (clade_not_defined_states: {explained})"
        )
    return errors


def _dot_problems(
    slot: str, points: list[dict[str, Any]], colours: dict[str, str], classes: dict[str, Any],
    section: Section,
) -> list[str]:  # fmt: skip
    """A figure's dots against the key: a clade in its colour, an uncoloured dot by its class
    in that class's swatch."""
    errors = []
    for dot in points:
        clade, colour = dot.get("clade"), dot["color"]
        if clade is not None:
            if colours.get(clade) != colour:
                errors.append(f"{slot}: {clade!r} drawn {colour}, key has "
                              f"{colours.get(clade)} (a key from another geo build?)")  # fmt: skip
            continue
        cls = dot.get("uncoloured")
        if cls is None:
            if section.clade_key_classes:
                errors.append(f"{slot}: an uncoloured dot with no class (a geo build before Q215?)")
            continue
        drawn = (colour, dot.get("edge"), dot.get("edge_width"))
        if cls not in classes:
            errors.append(f"{slot}: uncoloured class {cls!r} drawn, the key does not list it")
        elif drawn != tuple(classes[cls].get(k) for k in ("fill", "edge", "edge_width")):
            keyed = tuple(classes[cls].get(k) for k in ("fill", "edge", "edge_width"))
            errors.append(f"{slot}: {cls!r} drawn {drawn}, key has {keyed} "
                          "(a key from another geo build?)")  # fmt: skip
    return errors


def _class_row(label: str, swatch: dict[str, Any]) -> KeyRow:
    fill = "" if swatch["fill"] == "transparent" else swatch["fill"][1:]
    return KeyRow(label, fill, swatch["edge"][1:], float(swatch["edge_width"]))


def _key_table(key: Key, n: int) -> list[str]:
    """A clade key as the delivered report draws it: a ring-and-fill swatch, then the label.
    Colours get generated names (labels hold spaces and parentheses). A clade's swatch has the
    delivered black ring; an uncoloured class's has its dots' fill (or none) and ring, the ring
    as wide in points as on the geo figure."""
    out = []
    for j, row in enumerate(key):
        if row.fill:
            out.append(rf"\definecolor{{afkey{n}x{j}}}{{HTML}}{{{row.fill}}}")
        if row.edge:
            out.append(rf"\definecolor{{afkey{n}x{j}e}}{{HTML}}{{{row.edge}}}")
    out.append(r"\begin{tabular}{l l}")
    for j, row in enumerate(key):
        if row.edge:
            fill = f"afkey{n}x{j}" if row.fill else "none"
            box = rf"\ubox{{{fill}}}{{afkey{n}x{j}e}}{{{row.edge_width:g}}}"
        else:
            box = rf"\cbox[afkey{n}x{j}]"
        out.append(rf"{box} & {tex_escape(row.label)} \\")
    return [*out, r"\end{tabular}"]


def intro_texts(
    cfg: ReportConfig, figs: dict[str, Resolved], store: Store | None,
    key_record: dict[str, Any] | None = None,
) -> dict[int, Intro]:  # fmt: skip
    """Each section's intro, placeholders filled from recorded artefacts (af.report.text); a
    clade key's counts come from its record (:func:`clade_keys`)."""
    out: dict[int, Intro] = {}
    classes = (key_record or {}).get("uncoloured_classes", {})
    for i, section in enumerate(cfg.sections):
        if not section.intro:
            continue
        values = {"period": text.period_text(cfg.months())}
        gaps: list[str] = []
        intro = section.intro
        if CLADE_NOT_DEFINED in text.placeholders(intro):
            count = classes.get(section.title, {}).get(CLADE_NOT_DEFINED)
            if count is None:
                gaps.append(f"{section.title}: the clade-not-defined count (no clade key lists it)")
            elif count == 0:  # no such virus: its paragraph goes, as its key row does
                intro = text.without_paragraphs_using(intro, CLADE_NOT_DEFINED)
            unit = "virus" if count == 1 else "viruses"  # a geo dot is a virus
            values[CLADE_NOT_DEFINED] = "?" if count is None else f"{count:,} {unit}"
        if "tree_method" in text.placeholders(section.intro):
            methods = []
            for slot in section.slots:
                if store is None:
                    gaps.append(f"{slot}: the tree's method (no --store to read it from)")
                    continue
                ref, tree = text.tree_json(store, figs[slot].figure.doc)
                method, missing = text.tree_method(tree, text.tree_caveats(store, ref))
                if missing:  # one line per tree: the version, then every field it lacks
                    gaps.append(f"{slot} ({ref}): " + ", ".join(missing))
                methods.append(method)
            values["tree_method"] = " ".join(m for m in methods if m)
        filled = text.fill(intro, values)
        out[i] = ([p.strip() for p in filled.split("\n\n") if p.strip()], gaps)
    return out


def _intro_page(section: Section, intro: Intro, key: Key | None = None, n: int = 0) -> list[str]:
    """The section heading and its paragraphs; then a new page. An unrecorded fact is refused.

    Spaced as the delivered report: 3em under the heading, 1em between paragraphs. A tree
    intro page has no page number there (the tree follows as a whole page), a geo one has.
    A clade key sits after the first paragraph (the period sentence), 1em above and below."""
    paragraphs, gaps = intro
    out = [rf"\{_level(section)}{{{tex_escape(section.title)}}}", r"\vspace*{3em}"]
    if section.kind == "trees":
        out.append(r"\thispagestyle{empty}")
    if key and paragraphs:
        out.append(tex_escape(paragraphs[0]) + r"\par\vspace*{1em}")
        out += [*_key_table(key, n), r"\par\vspace*{1em}"]
        paragraphs = paragraphs[1:]
    if paragraphs:
        out.append(r"\par\vspace{1em}".join(tex_escape(p) for p in paragraphs) + r"\par")
    if gaps:  # build() refuses these before typesetting: a gap never reaches the page
        raise BuildError(f"{section.title}: unrecorded fact(s) in the intro: {gaps}")
    return [*out, r"\newpage"]


def _tree_pages(
    section: Section, figs: dict[str, Resolved], rel: dict[str, str], headed: bool = True
) -> list[str]:
    lines: list[str] = []
    for slot in section.slots:
        r = figs[slot]
        # One tree: the section title is its heading. Several: say which tree each page is.
        title = section.title if len(section.slots) == 1 else f"{section.title}: {r.figure.title}"
        if not headed and len(section.slots) == 1 and not r.figure.placeholder:
            # Under an intro page, as delivered: the tree is the whole page (pdfpages, scaled
            # to fit and centred), unnumbered, with its page label on that page.
            lines.append(
                rf"\includepdf[pages={{1}},pagecommand={{\thispagestyle{{empty}}"
                rf"\zlabel{{{_label(slot)}}}}}]{{{rel[slot]}}}"
            )
            continue
        if headed or len(section.slots) > 1:
            lines.append(rf"\{_level(section)}{{{tex_escape(title)}}}")
        lines += [
            r"\thispagestyle{empty}",  # a tree page is unnumbered, as delivered
            r"\begin{center}",
            r"\includegraphics[width=\textwidth,height=0.88\textheight,keepaspectratio]"
            rf"{{{rel[slot]}}}" + _figure_mark(r, slot, r"\small"),
            r"\end{center}",
            r"\newpage",
        ]
    return lines


def _level(section: Section) -> str:
    """Sections under a group are subsections; ungrouped ones are sections."""
    return "subsection" if section.group else "section"


def _grid_pages(
    section: Section, months: list[str], figs: dict[str, Resolved], rel: dict[str, str],
    headed: bool = True,
) -> list[str]:  # fmt: skip
    """Map and geo pages: a columns x rows grid per page, blank cells left empty.

    Placed as the delivered report's report.tex places them, each panel framed (0.4pt):
    - landscape map grids (pdflscape; 3 x 2 across): square panels (\\textheight - 70pt)/rows
      wide, 3pt column separation, 4pt between rows;
    - portrait geo columns: panels \\textheight x 18/30 wide for 3 rows (scaled for others),
      centred, rows 2.5 x stretched.
    A blank cell keeps its width. A placeholder figure keeps a red PLACEHOLDER line under it.
    """
    cols, rows = section.grid
    if section.landscape:
        width = rf"\dimexpr(\textheight-70pt)/{rows}\relax"
        table = [r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.0}"]
        row_end = r"\\[4pt]"
    else:
        width = rf"\dimexpr\textheight*18/{10 * rows}\relax"  # 18/30 for the delivered 3 rows
        table = [r"\renewcommand{\arraystretch}{2.5}"]
        row_end = r"\\"
    lines: list[str] = []
    pages, starts = section.pages(months), section.run_starts(months)
    for n, (heading, cells) in enumerate(pages):
        if section.landscape and starts[n]:
            lines.append(r"\begin{landscape}")
        elif section.landscape:  # a window's next page, in the same landscape as delivered:
            lines.append(r"\newpage\vspace*{-2pt}")  # 24pt lower than a fresh landscape
        if (n == 0 and headed) or heading:  # the first page (unless an intro page has the
            # heading), and the first page of each map window
            lines.append(
                rf"\{_level(section)}{{{tex_escape(f'{section.title} {heading}'.strip())}}}"
            )
        lines += [r"\begin{center}", *table, r"\begin{tabular}{" + " ".join("c" * cols) + "}"]
        for start in range(0, len(cells), cols):
            row = []
            for slot in cells[start : start + cols]:
                if slot is None:
                    row.append(rf"\hspace*{{{width}}}")  # a blank cell keeps its width
                    continue
                image = rf"\includegraphics[width={width},frame]{{{rel[slot]}}}"
                if figs[slot].figure.placeholder:
                    image = (rf"\parbox[t]{{{width}}}{{\centering{image}"
                             + _figure_mark(figs[slot], slot, r"\scriptsize") + "}")  # fmt: skip
                else:
                    image += _figure_mark(figs[slot], slot, r"\scriptsize")
                row.append(image)
            lines.append(" & ".join(row) + row_end)
        lines += [r"\end{tabular}", r"\end{center}"]
        if section.landscape:
            if n + 1 == len(pages) or starts[n + 1]:  # the window's last page
                lines.append(r"\end{landscape}")
        else:
            lines.append(r"\newpage")
    return lines


def write_tex(
    cfg: ReportConfig, figs: dict[str, Resolved], build: Path, built_at: dt.datetime,
    n_not_from_store: int = 0, intros: dict[int, Intro] | None = None,
    keys: dict[int, Key] | None = None,
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
        # The delivered report's page (its report.tex): 12pt Times, T1, 2.5em side margins, 3em
        # top, 5em bottom; contents set with titletoc, groups bold and unnumbered, entries with
        # their page at a 120pt contents margin; hyperref's default link boxes (screen only).
        r"\documentclass[a4paper,12pt]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{times}",
        GEOMETRY,
        r"\usepackage{graphicx,xcolor,pdflscape,pdfpages,zref-user,zref-abspage}",
        r"\usepackage[export]{adjustbox}",  # the frame key on \includegraphics, as delivered
        r"\usepackage{tikz}",  # the clade key's swatches, drawn as the delivered report's \cbox
        r"\newcommand\cbox[1][black]{\tikz[baseline=-0.5ex]"
        r"\draw[black,fill=#1,radius=0.3em] (0,0) circle ;}",
        # an uncoloured class's swatch (Q215): fill (or none), ring colour, ring width in pt
        r"\newcommand\ubox[3]{\tikz[baseline=-0.5ex]"
        r"\draw[draw=#2,fill=#1,line width=#3pt,radius=0.3em] (0,0) circle ;}",
        # Physical page on every \zlabel: \label pages are logical.
        r"\makeatletter\zref@addprop{main}{abspage}\makeatother",
        r"\usepackage{titletoc}",
        r"\titlecontents{section}[0cm]{\bfseries}{\\}{\\}{}",
        r"\titlecontents{subsection}[1em]{}{}{}{\titlerule*[5pc]{}\vspace{0.8ex}\thecontentspage}",
        r"\contentsmargin{120pt}",
        r"\usepackage{hyperref}",
        _pdf_metadata(cfg, built_at),
        r"\setlength{\parindent}{0pt}",
        # Grouped reports (subtype > figure) are unnumbered, like the delivered VCM report.
        r"\setcounter{secnumdepth}{0}" if any(sec.group for sec in cfg.sections) else "",
        r"\begin{document}",
        # Set again as the delivered report does, after \begin{document}: there an em is the
        # body font's (Times, 12pt; in the preamble it is Computer Modern's, 11.75pt, which
        # put every page 0.6pt left and 0.7pt up), and the node it leaves keeps the cover's
        # opening space, without which the cover sat 11.5pt high (review, 18 Oct 2026).
        GEOMETRY.replace("\\usepackage[", "\\newgeometry{").replace("]{geometry}", "}"),
        *_cover(cfg, sum(r.figure.placeholder for r in figs.values()), n_not_from_store, built_at),
    ]
    group = None
    intros, keys = intros or {}, keys or {}
    for i, section in enumerate(cfg.sections):
        if section.group and section.group != group:
            lines.append(rf"\section*{{{tex_escape(section.group)}}}")
            lines.append(rf"\addcontentsline{{toc}}{{section}}{{{tex_escape(section.group)}}}")
        group = section.group
        if i in intros:
            lines += _intro_page(section, intros[i], keys.get(i), i)
        if section.kind == "trees":
            lines += _tree_pages(section, figs, rel, headed=i not in intros)
        else:
            lines += _grid_pages(section, cfg.months(), figs, rel, headed=i not in intros)
    if cfg.report.end_page:  # at the foot of the last page, tiny, in the round's zone
        zone = zoneinfo.ZoneInfo(cfg.report.timezone) if cfg.report.timezone else dt.UTC
        when = built_at.astimezone(zone).strftime("%Y-%m-%d %H:%M %Z")
        lines.append(rf"\par\vspace*{{\fill}}\hspace*{{1.5em}}{{\tiny Report generated: {when}}}")
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


def check_output(tex: Path, slots: list[str], after_last_figure: int = 0) -> int:
    """Every figure typeset, and the PDF ends ``after_last_figure`` pages after the last figure's
    page (the closing page, when the report has one). Returns the page count."""
    pages = figure_pages(tex.with_suffix(".aux"))
    missing = [slot for slot in slots if _label(slot) not in pages]
    if missing:
        raise BuildError(f"{len(missing)} figure(s) not typeset: {missing[:5]}")
    n = pdf_pages(tex.with_suffix(".log"))
    last = max(pages.values())
    if n != last + after_last_figure:
        raise BuildError(
            f"{tex.stem}.pdf: {n} pages but the last figure is on page {last}"
            + (f" and {after_last_figure} page(s) should follow it" if after_last_figure else "")
        )
    return n


def af_source() -> dict[str, Any]:
    """Which af code built this: version, and the git commit of the source actually imported.

    A report must say what produced it. When af runs from a git checkout (an editable install)
    the commit and whether it had uncommitted changes are recorded. A frozen af release (a
    non-editable install made by tools/make-release.py) has no checkout, but records the commit it
    was built from: that is used, with ``dirty`` false, since a release is read-only. Otherwise
    ``commit`` is null rather than guessed.
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
        if Path(git("rev-parse", "--show-toplevel")).resolve() == source.parent:
            record["commit"] = git("rev-parse", "HEAD")
            record["dirty"] = bool(git("status", "--porcelain", "--untracked-files=no"))
            return record
    except (OSError, subprocess.CalledProcessError):
        pass  # not a git checkout of acmacs-f: try the release record next
    release = _release_info()
    if release and release.get("commit"):
        record.update(commit=release["commit"], dirty=False, release=release)
    return record


def _release_info() -> dict[str, Any] | None:
    """af's release record, when running from a release made by tools/make-release.py."""
    import importlib

    try:  # by name: af.run.runtime arrives with the releases work, and may not exist yet
        runtime = importlib.import_module("af.run.runtime")
    except ImportError:
        return None  # an af without release support: nothing to record
    info = runtime.release_info()
    return dict(info) if info else None


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
        # The map step's own warnings (e.g. points left unpainted), verbatim, per slot.
        "warnings": [
            {"slot": slot, "warning": w} for slot, r in figs.items()
            for w in figure_warnings(r.figure.doc)
        ],
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
    manifest_path: Path | None, *, deep: bool, export_records: Path | None = None,
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
        check_against_store(
            use, figs, Store.open(store_root), deep=deep, export_records=export_records
        )
    elif manifest_path is not None:
        raise BuildError(
            "no figure was drawn from the store, so there is no report manifest to write; "
            "this is a bring-up build (drop --manifest)"
        )
    return use


def build(
    config_path: Path, figures_root: Path, out_dir: Path, *,
    store_root: Path | None = None, manifest_path: Path | None = None, deep: bool = False,
    ignore_busy: bool = False, export_records: Path | None = None,
) -> Path:  # fmt: skip
    """Build the report; return the PDF path. Raises on any problem, leaving no final PDF.

    With a store, everything that reads it runs under ``Store.reading``: the build refuses to
    start while a batch is publishing, and fails if a dataset it read moved before it finished
    (its output would mix versions). Only then are the PDF, manifest and build record written,
    and the record carries what the guard saw. ``ignore_busy`` reads anyway (diagnosis only);
    the record says so.
    """
    cfg = load(config_path)
    built_at = dt.datetime.now(dt.UTC)  # shown on the cover and recorded; decides nothing
    figs = resolve_all(cfg, figures_root)
    guarded = (
        Store.open(store_root).reading("report-build", override=ignore_busy)
        if store_root is not None
        else nullcontext(None)
    )
    with guarded as guard:
        use = check_provenance(
            cfg, figs, store_root, manifest_path, deep=deep, export_records=export_records
        )
        keys, key_record = clade_keys(cfg, figs, figures_root)
        intros = intro_texts(cfg, figs, Store.open(store_root) if store_root else None, key_record)
        gaps = [g for _, section_gaps in intros.values() for g in section_gaps]
        if gaps:  # bring-up too: a gap would print internal identifiers in place of a sentence
            raise BuildError(
                f"{len(gaps)} fact(s) in intro text no artefact records:\n  " + "\n  ".join(gaps)
            )
        build_dir = out_dir / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)  # never reuse a stale .aux/.toc or an old figure
        build_dir.mkdir(parents=True)
        tex = write_tex(cfg, figs, build_dir, built_at, len(use.not_from_store), intros, keys)
        if UNRECORDED in tex.read_text():  # from any source, config words included
            raise BuildError(f"{tex}: {UNRECORDED!r} would be printed in the report body")
        passes = run_latex(tex, LocalRunner())
        pages = check_output(tex, cfg.all_slots(), 1 if cfg.report.end_page else 0)
    final = out_dir / f"{cfg.report.id}.pdf"
    shutil.copyfile(tex.with_suffix(".pdf"), final)
    if manifest_path is not None:
        description = f"report {cfg.report.id}: {final.name} sha256 {sha256_path(final)}"
        write_report_manifest(manifest_path, use, description)
    record = build_record(
        cfg, config_path, figs, use, final, pages, passes, built_at, manifest_path
    )
    record["store_read"] = guard.to_json() if guard is not None else None
    for item in record["warnings"]:
        print(f"{item['slot']}: {item['warning']}", file=sys.stderr)
    record["text_gaps"] = gaps
    record["clade_key"] = key_record
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
    parser.add_argument(
        "--export-records", type=Path,
        help="af.tree's export records (<dir>/<dataset>/<alignment sha256[:16]>/<version>"
        ".export.json): needed when a tree was exported from another sequences version than "
        "the figures read",
    )  # fmt: skip
    parser.add_argument(
        "--ignore-busy", action="store_true",
        help="read the store even while a batch is publishing (diagnosis only; recorded)",
    )  # fmt: skip
    args = parser.parse_args(argv)
    try:
        pdf = build(
            args.config, args.figures, args.out,
            store_root=args.store, manifest_path=args.manifest, deep=args.deep,
            ignore_busy=args.ignore_busy, export_records=args.export_records,
        )  # fmt: skip
    except (BuildError, ConfigError, FigureError, ProvenanceError, StoreError) as err:
        print(f"report build FAILED: {err}", file=sys.stderr)
        return 1
    print(pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
