"""The map core must not know about influenza: a ratchet that keeps the boundary true.

Why this test exists (Sarah, 1 Oct 2026: "start the boundary work now"): af/chart and most of
af/map are antigenic CARTOGRAPHY: charts, titres, optimisation, orientation, frames and drawing.
They may one day be their own package, used for maps of anything, so they must not depend on
af's influenza knowledge (clades, sequences, serology, vaccines, strain names, passage meanings)
or on af's store and report machinery. notes/maps/BOUNDARY.md has the classification and why.

Knowledge travels in two ways, so this checks both: IMPORTS (any `import af.x` / `from af.x`,
including ones inside functions and TYPE_CHECKING blocks) and flu TERMS used in code (names and
string literals; docstrings and comments may tell history and are not checked).

Three kinds of file:
- CORE: no af-side import, no flu term. Ever.
- MIXED: does both jobs today, with a known seam to split along. Pinned to EXACTLY its current
  af-side imports and terms: adding one fails, and dropping one fails until you remove it from
  the pin, so the list can only shrink.
- AF_SIDE: flu, round configuration, or af's stores and formats. Not checked here.
A new file in af/chart or af/map must be put in one of the three, deliberately.

If this test fails you: do not just add your import or word to a pin. Ask whether the code
belongs on the af side (af.map.vaccines, af.map.colouring, af.map.matching, ...) and pass the
fact in from there. The scene carries marker shapes, not passages; the aligner takes the
caller's matching rule, not a strain-name parser.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: af packages the core must never reach: influenza knowledge, af's stores, tables and reports.
FORBIDDEN_PACKAGES = (
    "af.clades", "af.seq", "af.serology", "af.report", "af.tables", "af.tree", "af.store",
    "af.util.subtypes", "af.geo", "af.stat", "af.chain",
)  # fmt: skip
#: af.map modules on the af side of the line (flu, round configuration, af's figure format).
FORBIDDEN_MAP_MODULES = (
    "af.map.vaccines", "af.map.colouring", "af.map.matching", "af.map.build", "af.map.config",
    "af.map.roundconfig", "af.map.i7",
)  # fmt: skip
#: Influenza words that mean the code is interpreting viruses. ("reassortant" is not here: it is
#: a field of the acmacs chart format, read and written by af.chart as data.)
FLU_TERMS = re.compile(
    r"passage_class|vaccine|clade|lineage|strain|ferret|mdck|siat|h3n2|h1n1|b/vic|\begg\b"
    r"|\bcdc\b|\bniid\b|\bcrick\b|\bvidrl\b|\bcnic\b",
    re.IGNORECASE,
)

CORE = {
    "af/chart/__init__.py", "af/chart/ace.py", "af/chart/column_bases.py", "af/chart/control.py",
    "af/chart/identity.py", "af/chart/merge.py", "af/chart/model.py", "af/chart/procrustes.py",
    "af/chart/titre.py", "af/map/__init__.py", "af/map/align.py", "af/map/curate.py",
    "af/map/optimise.py", "af/map/orient.py", "af/map/viewport.py",
}  # fmt: skip

#: file -> (pinned af-side imports, pinned flu terms, the seam it splits along)
MIXED: dict[str, tuple[set[str], set[str], str]] = {
    "af/chart/sera.py": (
        set(), {"ferret"},
        "species/marker detection is core; FERRET_ONLY, Sarah's policy sentence, goes af-side",
    ),
    "af/map/render.py": (
        set(), {"egg", "vaccine"},
        "'egg'/'uglyegg' are marker SHAPE names (fine); ScenePoint.vaccine is to be renamed 'mark'",
    ),
    "af/map/finish.py": (
        {"af.map.i7"}, {"vaccine"},
        "frame/labels/PDF assembly is core; writing af's I7 description is af-side",
    ),
    "af/map/style.py": (
        {"af.map.vaccines"}, {"egg", "passage_class", "strain", "vaccine"},
        "Scene/ColourScheme types are core; 'a vaccine takes its cell preparation's colour' is flu",
    ),
    "af/map/labels.py": (
        {"af.map.vaccines"}, {"passage_class", "strain", "vaccine"},
        "label placement is core; label TEXT (strain abbreviations, passage suffixes) is flu",
    ),
    "af/map/figure.py": (
        {"af.map.vaccines", "af.map.colouring"}, {"passage_class", "vaccine"},
        "draw_axes/frame_around are core; chart_points/chart_scene build the scene from flu facts",
    ),
}  # fmt: skip

AF_SIDE = {
    "af/map/build.py", "af/map/colouring.py", "af/map/config.py", "af/map/i7.py",
    "af/map/matching.py", "af/map/roundconfig.py", "af/map/vaccines.py",
}  # fmt: skip


TIGHTEN = "remove it from the pin (the ratchet tightens)"


def af_imports(path: Path) -> set[str]:
    """Every af module the file imports, anywhere in it (functions and TYPE_CHECKING too)."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return {m for m in out if m == "af" or m.startswith("af.")}


def forbidden(modules: set[str]) -> set[str]:
    bad: set[str] = set()
    for m in modules:
        if any(m == p or m.startswith(p + ".") for p in FORBIDDEN_PACKAGES + FORBIDDEN_MAP_MODULES):
            bad.add(m)
    return bad


def flu_terms(path: Path) -> set[str]:
    """Flu words in the file's CODE: names and string literals, not docstrings or comments."""
    src = path.read_text()
    docstrings: set[int] = set()
    for node in ast.walk(ast.parse(src)):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings.update(range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1))
    found: set[str] = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.NAME, tokenize.STRING) and tok.start[0] not in docstrings:
            found.update(m.group(0).lower() for m in FLU_TERMS.finditer(tok.string))
    return found


def test_every_chart_and_map_file_is_classified() -> None:
    files = {
        str(p.relative_to(ROOT)) for d in ("af/chart", "af/map") for p in (ROOT / d).glob("*.py")
    }
    unclassified = files - CORE - MIXED.keys() - AF_SIDE
    assert not unclassified, (
        f"{sorted(unclassified)}: new file(s) in af/chart or af/map must be classified as CORE, "
        "MIXED or AF_SIDE in tests/test_map_core_boundary.py (see notes/maps/BOUNDARY.md). The "
        "map core must stay free of influenza knowledge; put flu code on the af side."
    )
    gone = (CORE | MIXED.keys() | AF_SIDE) - files
    assert not gone, f"{sorted(gone)} no longer exist: remove them from the boundary lists"


def test_core_files_know_nothing_of_influenza() -> None:
    problems = []
    for f in sorted(CORE):
        path = ROOT / f
        for m in sorted(forbidden(af_imports(path))):
            problems.append(f"{f} imports {m}")
        for t in sorted(flu_terms(path)):
            problems.append(f"{f} uses the influenza term {t!r} in code")
    assert not problems, (
        "The map core must not depend on influenza knowledge or on af's stores and reports, so it "
        "can serve maps of anything (Sarah, 1 Oct 2026; notes/maps/BOUNDARY.md):\n  "
        + "\n  ".join(problems)
        + "\nPass the fact in from the af side instead (af.map.vaccines, af.map.matching, ...)."
    )


def test_mixed_files_only_ever_lose_their_af_side_knowledge() -> None:
    problems = []
    for f, (imports, terms, seam) in sorted(MIXED.items()):
        path = ROOT / f
        got_imports, got_terms = forbidden(af_imports(path)), flu_terms(path)
        for m in sorted(got_imports - imports):
            problems.append(f"{f} gained the af-side import {m} (seam: {seam})")
        for t in sorted(got_terms - terms):
            problems.append(f"{f} gained the influenza term {t!r} (seam: {seam})")
        for m in sorted(imports - got_imports):
            problems.append(
                f"{f} no longer imports {m}: remove it from the pin (the ratchet tightens)"
            )
        for t in sorted(terms - got_terms):
            problems.append(
                f"{f} no longer uses {t!r}: remove it from the pin (the ratchet tightens)"
            )
    assert not problems, "\n  ".join(["Map core boundary ratchet:", *problems])
