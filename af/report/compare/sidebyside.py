"""Side-by-side documents of two builds of a report, written beside the newer report's PDF.

Sarah, 30 Oct 2026: a side by side of two reports, and a variant with Procrustes arrows, are
standard outputs, kept at their pages' own sizes with an automatic pixel measure, and "should go in
the same folder as the report pdf in the round folder". Two documents, from the two reports' build
records (``<round>/report/<id>.build.json``):

- ``<id>-sbs-vs-<ref>.pdf``: the two report PDFs page by page, portrait pages side by side and
  landscape pages one above the other, each pair with its pixel measure;
- ``<id>-procrustes-vs-<ref>.pdf``: per map slot in both reports, the two figures, the new one with
  Procrustes arrows (af.report.compare.arrows), and the numbers.

``<ref>`` names the reference: its round folder with the shared report id removed, so the
M1-maps round ``2026-1023-vidrl-interim-m1maps`` compared with ``2026-1023-vidrl-interim`` gives
``m1maps``. No date in the names: the documents are rebuilt by this command and overwrite
themselves; the build date is printed on their first pages, with the af that built them, the
rasteriser's version and what "no difference" measures (a page compared with itself).

    python -m af.report.compare.sidebyside <new round>/report/<id>.build.json \\
        <reference round>/report/<id>.build.json [--out DIR] [--ref-label TEXT]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from af.report.build import af_source, tex_escape
from af.report.compare import arrows, pixels
from af.report.compare.pages import GAP, Document, page_sizes, side_by_side, text_block
from af.util.artefacts import sha256_path

SBS = "sbs"
PROCRUSTES = "procrustes"
FAR_KEY = f"moved_gt_{arrows.FAR:g}"
# Beside every pixel measure of two maps: a reader sets it against the RMSD otherwise.
PIXELS_ARE_NOT_POSITIONS = (
    "Pixels count every drawn difference (colours, outlines, legend counts, labels), not only "
    "positions; the arrows and the RMSD measure positions"
)


class SideBySideError(ValueError):
    """The two builds cannot be put side by side."""


def _figure(entry: dict[str, Any]) -> tuple[Path, Path]:
    """A figure's I7 and PDF, checked against the hashes its build record holds (rule 3)."""
    i7 = Path(entry["i7"])
    pdf = i7.with_name("figure.pdf")
    for path, key in ((i7, "i7_sha256"), (pdf, "pdf_sha256")):
        if not path.is_file():
            raise SideBySideError(f"{entry['slot']}: {path} is missing")
        if entry.get(key) and sha256_path(path) != entry[key]:
            raise SideBySideError(f"{entry['slot']}: {path} is not the file its build recorded")
    return i7, pdf


def map_pairs(new: dict[str, Any], ref: dict[str, Any]) -> tuple[list[arrows.MapPair], list[str]]:
    """The map slots drawn in both builds, in the new report's order, and the map slots that one
    side has and the other lacks (listed, never dropped silently)."""

    def maps_of(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            f["slot"]: f for f in record["figures"]
            if f["slot"].startswith("map/") and not f.get("placeholder")
        }  # fmt: skip

    ours, theirs = maps_of(new), maps_of(ref)
    pairs = []
    for slot, entry in ours.items():
        if slot in theirs:
            new_i7, new_pdf = _figure(entry)
            ref_i7, ref_pdf = _figure(theirs[slot])
            pairs.append(arrows.MapPair(slot, new_i7, ref_i7, new_pdf, ref_pdf))
    one_sided = [f"{s} (new only)" for s in ours if s not in theirs]
    one_sided += [f"{s} (reference only)" for s in theirs if s not in ours]
    if not pairs:
        raise SideBySideError("no map slot is drawn in both builds")
    return pairs, one_sided


def reference_pairs(record: dict[str, Any], reference: Path) -> list[arrows.MapPair]:
    """The report's map slots that have a reference I7 (``<slot with dots>.i7.json``, as
    af.report.compare.run finds them). A reference of I7s has no pictures: no page 1."""
    pairs = []
    for entry in record["figures"]:
        ref_i7 = reference / (entry["slot"].replace("/", ".") + ".i7.json")
        if entry["slot"].startswith("map/") and not entry.get("placeholder") and ref_i7.is_file():
            new_i7, new_pdf = _figure(entry)
            pairs.append(arrows.MapPair(entry["slot"], new_i7, ref_i7, new_pdf, None))
    return pairs


def ref_name(new_record: Path, ref_record: Path, report_id: str) -> str:
    """The reference's short name: its round folder with the shared report id removed."""
    folder = ref_record.resolve().parent.parent.name
    short = folder.removeprefix(report_id).strip("-_")
    if not short:
        raise SideBySideError(
            f"{ref_record}: its round folder is named as the report id; pass --ref-label"
        )
    if new_record.resolve().parent.parent == ref_record.resolve().parent.parent:
        raise SideBySideError("the two build records are the same round")
    return short


def cover(title: str, sides: tuple[str, str], floor: str, lines: list[str]) -> str:
    """The first page: what is compared, how it was built, and how to read the numbers."""
    body = [
        rf"{{\Large\bfseries {tex_escape(title)}}}",
        rf"\textbf{{new}}: {tex_escape(sides[0])}",
        rf"\textbf{{reference}}: {tex_escape(sides[1])}",
        rf"built {dt.datetime.now(dt.UTC):%d %B %Y %H:%M} UTC by af "
        rf"{tex_escape(_af_text())}; rasterised by poppler {tex_escape(pixels.poppler_version())}",
        r"\medskip\textbf{Pixel measure}: each page at its own size at "
        rf"{pixels.DPI} dpi, RGB; pages of different sizes aligned top-left and padded white; "
        rf"a pixel differs when any channel differs by more than {pixels.THRESHOLD} of 255. "
        rf"No difference reads: {tex_escape(floor)} (the first compared page against itself).",
        *lines,
    ]
    return text_block(760.0, body, r"\normalsize")


def _af_text() -> str:
    src = af_source()
    commit = str(src.get("commit") or "unrecorded")[:12]
    return f"{src.get('version')} ({commit}{', with local changes' if src.get('dirty') else ''})"


def floor(pdf: Path) -> str:
    """What "no difference" reads with this rasteriser: page 1 of ``pdf`` against itself."""
    return pixels.pixel_diff(pdf, 1, pdf, 1).text()


def write_sbs(
    new_pdf: Path, ref_pdf: Path, sides: tuple[str, str], out_pdf: Path, banner: str = ""
) -> list[dict[str, Any]]:
    """The two report PDFs page by page, with the pixel measure of each page pair."""
    a, b = page_sizes(new_pdf), page_sizes(ref_pdf)
    doc = Document(banner=banner)
    rows: list[dict[str, Any]] = []
    for n in range(max(len(a), len(b))):
        if n >= len(a) or n >= len(b):
            only, pdf, label = ("reference", ref_pdf, sides[1]) if n >= len(a) else (
                "new", new_pdf, sides[0])  # fmt: skip
            doc.add(
                rf"\hbox{{\footnotesize {tex_escape(label)} p{n + 1}: "
                rf"no page {n + 1} in the other report}}\hbox{{{doc.picture(pdf, n + 1)}}}"
            )
            rows.append({"page": n + 1, "only": only})
            continue
        diff = pixels.pixel_diff(new_pdf, n + 1, ref_pdf, n + 1)
        w, h = a[n]
        labels = (f"{sides[0]} p{n + 1}", f"{sides[1]} p{n + 1}")
        pictures = (doc.picture(new_pdf, n + 1), doc.picture(ref_pdf, n + 1))
        pair = side_by_side(*pictures, labels, stacked=w > h)
        doc.add(rf"\hbox{{\footnotesize p{n + 1}: {tex_escape(diff.text())}}}\kern4pt" + pair)
        rows.append({"page": n + 1, "differing": diff.differing, "share": diff.share,
                     "mean_abs": diff.mean_abs, "sizes": [diff.size_a, diff.size_b]})  # fmt: skip
    pages = [
        rf"page {r['page']}: " + (f"{r['only']} report only" if "only" in r else
                                  f"{r['differing']:,} pixels differ ({100 * r['share']:.2f}\\%)")
        for r in rows
    ]  # fmt: skip
    lines = [r"\medskip new on the left (or above), reference on the right (or below).", *pages]
    doc.pages.insert(0, doc.page(cover("Two reports side by side", sides, floor(new_pdf), lines)))
    doc.write(out_pdf)
    return rows


def write_procrustes(
    pairs: list[arrows.MapPair], sides: tuple[str, str], one_sided: list[str], out_pdf: Path,
    how: str = "identity", banner: str = "",
) -> list[dict[str, Any]]:  # fmt: skip
    """Per map: the two figures of each window, then the new map with arrows, and the numbers.

    A map's windows share one layout, so their fits are usually the same. When every window's
    arrows have identical inputs (every matched point's two positions and distance, and the
    viewport), the map gets one arrows page and one numbers page, naming the windows they stand
    for (00 and 11, 30 Oct 2026); a map whose windows differ in any of these keeps one per window.
    """
    import tempfile

    doc = Document(banner=banner)
    rows: list[dict[str, Any]] = []
    groups: dict[str, list[tuple[arrows.MapPair, arrows.Arrows, dict[str, Any]]]] = {}
    for pair in pairs:
        new = json.loads(pair.new_i7.read_text())
        groups.setdefault(pair.slot.rsplit("/", 1)[0], []).append(
            (pair, arrows.read_pair(pair, how), new)
        )
    with tempfile.TemporaryDirectory() as tmp:
        for map_slot, members in groups.items():
            page_rows = []
            for pair, a, _ in members:
                row = arrows.summary(a)
                if pair.ref_pdf is not None:
                    diff = pixels.pixel_diff(pair.new_pdf, 1, pair.ref_pdf, 1)
                    row["pixels"] = {"differing": diff.differing, "share": diff.share,
                                     "mean_abs": diff.mean_abs}  # fmt: skip
                    doc.add(
                        rf"\hbox{{\large\bfseries {tex_escape(pair.slot)}}}"
                        rf"\hbox{{\footnotesize {tex_escape(diff.text())}. "
                        rf"{PIXELS_ARE_NOT_POSITIONS}}}\kern4pt"
                        + side_by_side(
                            doc.picture(pair.new_pdf),
                            doc.picture(pair.ref_pdf),
                            (f"new: {sides[0]}", f"reference: {sides[1]}"),
                        )  # fmt: skip
                    )
                page_rows.append(row)
            first, a0, new0 = members[0]
            same = all(
                a.moves == a0.moves and n["map"]["viewport"] == new0["map"]["viewport"]
                for _, a, n in members[1:]
            )
            drawn_for = [(members[0], page_rows)] if same else [
                (m, [r]) for m, r in zip(members, page_rows, strict=True)]  # fmt: skip
            for (pair, a, new), these in drawn_for:
                windows = [m[0].slot.rsplit("/", 1)[1] for m in members] if same else []
                label = (
                    f"{map_slot} ({', '.join(windows)})" if same and len(members) > 1 else pair.slot
                )
                layer = Path(tmp) / f"overlay-{len(rows)}.pdf"
                drawn = arrows.overlay(
                    a, new["map"]["viewport"], page_sizes(pair.new_pdf)[0], layer
                )
                doc.add(arrows_page(label, pair, a, drawn, layer, sides, doc, windows))
                doc.add(numbers_page(label, these, sides))
                for r in these:
                    r["arrows_drawn"] = drawn
            rows += page_rows
        heads = [
            rf"\textbf{{{tex_escape(r['slot'])}}}: RMSD {r['rmsd']:.3f} over {r['matched']:,} "
            rf"points; {sum(r[FAR_KEY].values()):,} moved over {arrows.FAR:g} unit"
            for r in rows
        ]
        extra = (
            [rf"Map slots in one report only: {tex_escape(', '.join(one_sided))}"]
            if one_sided
            else []
        )
        lines = [
            r"\medskip Points are matched by identity (isolate, year, passage class, date; sera by "
            r"serum id) and the reference is fitted onto the new map by rotation, reflection and "
            r"translation, without scaling: the comparison behind COMPARISON.json. A map whose "
            r"windows give the same arrows has one arrows page, naming its windows.",
            *extra, r"\medskip", *heads,
        ]  # fmt: skip
        title = "Maps of two reports: Procrustes arrows"
        doc.pages.insert(0, doc.page(cover(title, sides, floor(pairs[0].new_pdf), lines)))
        doc.write(out_pdf)
    return rows


def arrows_page(
    label: str, pair: arrows.MapPair, a: arrows.Arrows, drawn: int, layer: Path,
    sides: tuple[str, str], doc: Document, windows: list[str],
) -> str:  # fmt: skip
    """The new map as the report draws it, with the arrows laid over it, and the words."""
    words = [
        rf"{{\large\bfseries {tex_escape(label)}: Procrustes arrows}}",
        rf"The map is {tex_escape(sides[0])}, as the report draws it. Each arrow runs "
        rf"from where {tex_escape(sides[1])} puts a point, after it is fitted onto this "
        r"map, to where this map draws it.",
        "The reference is fitted onto this map by rotation, reflection and "
        "translation, without scaling, over every point drawn in both, matched as the "
        "report comparison matches them.",
        rf"Orientation, as COMPARISON reports it: {tex_escape(arrows.orientation(a))}.",
        rf"\textbf{{{tex_escape(arrows.headline(a))}}}.",
        rf"Arrows are drawn for moves over {arrows.THRESHOLD:g} units ({drawn:,}); "
        rf"{{\color[HTML]{{d62728}}red}} over {arrows.FAR:g} unit. Shorter moves draw no "
        "arrow, as in ae and AD.",
        rf"Drawn in one map only: {a.only_new:,} in this map, {a.only_ref:,} in the reference.",
    ]
    if len(windows) > 1:
        words.append(
            rf"One page for the windows {tex_escape(', '.join(windows))}: their arrows are "
            "identical (every matched point's positions and distance, and the frame). The map "
            rf"drawn is the {tex_escape(windows[0])} page."
        )
    # \vtop{\kern0pt...} hangs the map from the top, level with the text's first line
    return (
        rf"\hbox{{\vtop{{\kern0pt\hbox{{{doc.picture(pair.new_pdf, overlay=layer)}}}}}"
        rf"\kern{GAP}bp" + text_block(340.0, words, r"\large") + "}"
    )


def numbers_page(slot: str, rows: list[dict[str, Any]], sides: tuple[str, str]) -> str:
    """The numbers of a map (one row, or the identical windows' rows, whose pixels differ)."""
    row = rows[0]
    gt = row[FAR_KEY]
    lines = [
        rf"{{\large\bfseries {tex_escape(slot)}: numbers}}",
        rf"points matched: {row['matched']:,} ({row['antigens']:,} antigens, "
        rf"{row['sera']:,} sera); drawn in one map only: {row['only_new']:,} new, "
        rf"{row['only_ref']:,} reference",
        rf"distance after the fit: RMSD {row['rmsd']:.3f}, median {row['median']:.3f}, "
        rf"95th percentile {row['p95']:.3f}, largest {row['max']:.3f} units",
        rf"moved over {arrows.THRESHOLD:g} unit: {row[f'moved_gt_{arrows.THRESHOLD:g}']:,}; over "
        rf"{arrows.FAR:g}: {gt['antigens']:,} antigens, {gt['sera']:,} sera",
        rf"orientation, as COMPARISON reports it: {tex_escape(row['orientation'])}",
    ]
    for r in rows:
        if "pixels" in r:
            px = r["pixels"]
            lines.append(
                rf"pixels, {tex_escape(r['slot'])}: {px['differing']:,} differ "
                rf"({100 * px['share']:.2f}\%), mean |diff| {px['mean_abs']:.2f} of 255"
            )
    if any("pixels" in r for r in rows):
        lines.append(rf"{PIXELS_ARE_NOT_POSITIONS}.")
    vaccines = row["vaccines"]
    lines.append(r"\medskip\textbf{Vaccines} (units moved): " + (
        "; ".join(rf"{tex_escape(v['label'])} {v['distance']:.2f}" for v in vaccines)
        if vaccines else "none matched in both maps"))  # fmt: skip
    lines.append(rf"\medskip\textbf{{Largest {len(row['largest'])} moves}} (units):")
    lines += [
        rf"{tex_escape(m['label'])} ({m['kind']}) {m['distance']:.2f}" for m in row["largest"]
    ]
    return text_block(760.0, lines, r"\normalsize")


def report_pdf(record_path: Path, record: dict[str, Any]) -> Path:
    """The report PDF a build record names (relative to the record), checked against its hash."""
    pdf = record_path.resolve().parent / record["output"]["pdf"]
    if not pdf.is_file():
        raise SideBySideError(f"{record_path}: its report PDF {pdf} is missing")
    if record["output"].get("sha256") and sha256_path(pdf) != record["output"]["sha256"]:
        raise SideBySideError(f"{pdf} is not the PDF {record_path} recorded")
    return pdf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("new", type=Path, help="the newer report's build record")
    parser.add_argument("reference", type=Path, help="the reference report's build record")
    parser.add_argument(
        "--out", type=Path, help="directory to write to (default: beside the newer report's PDF)"
    )
    parser.add_argument("--ref-label", help="the reference's short name in the file names")
    parser.add_argument(
        "--allow-release-difference", action="store_true",
        help="compare builds made by different af commits; both are printed on every page",
    )  # fmt: skip
    args = parser.parse_args(argv)
    new, ref = json.loads(args.new.read_text()), json.loads(args.reference.read_text())
    commits = (str(new["af"].get("commit"))[:12], str(ref["af"].get("commit"))[:12])
    rounds = (args.new.resolve().parent.parent.name, args.reference.resolve().parent.parent.name)
    banner = ""
    if commits[0] != commits[1]:
        built = f"{rounds[0]} report on {commits[0]}, {rounds[1]} report on {commits[1]}"
        if not args.allow_release_difference:
            raise SideBySideError(
                f"{built}: build both on one release, or pass --allow-release-difference"
            )
        banner = f"Built by different af: {built}"
    report_id = new["report"]
    short = args.ref_label or ref_name(args.new, args.reference, report_id)
    new_pdf, ref_pdf = report_pdf(args.new, new), report_pdf(args.reference, ref)
    out = args.out or new_pdf.parent
    sides = (f"{rounds[0]} (af {commits[0]})", f"{rounds[1]} (af {commits[1]})")
    pairs, one_sided = map_pairs(new, ref)
    name = f"{report_id}-{PROCRUSTES}-vs-{short}.pdf"
    rows = write_procrustes(pairs, sides, one_sided, out / name, banner=banner)
    pages = write_sbs(new_pdf, ref_pdf, sides, out / f"{report_id}-{SBS}-vs-{short}.pdf", banner)
    (out / f"{report_id}-{PROCRUSTES}-vs-{short}.json").write_text(
        json.dumps({"sides": sides, "maps": rows, "pages": pages}, indent=1, default=str)
    )
    for r in rows:
        print(f"{r['slot']}: {r['matched']} matched, RMSD {r['rmsd']:.3f}, "
              f">{arrows.FAR:g}u {r[FAR_KEY]}", file=sys.stderr)  # fmt: skip
    print(f"wrote {out}/{report_id}-{{{SBS},{PROCRUSTES}}}-vs-{short}.pdf", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
