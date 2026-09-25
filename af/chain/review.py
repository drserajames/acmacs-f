"""Static review page for one chain: every step's map, stresses and diagnostics.

Writes `<chain>/review/index.html` and one PNG thumbnail per step. Thumbnails are oriented
step to step by Procrustes over common points so consecutive maps can be compared by
eye (the stored charts are not changed). A thumbnail's file name is a hash of its step's
map and record and of every earlier step (orientation depends on them), so a remade step
never shows an old picture and an unchanged chain redraws nothing.

The page shows strain names from the tables, so it lives in the store (private), never
in a git repo.
"""

from __future__ import annotations

import hashlib
import html
import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from af.chain.diagnostics import THRESHOLDS, Thresholds, flags, previous_in_current
from af.chart.ace import read_chart
from af.chart.model import Chart
from af.chart.procrustes import procrustes
from af.util.artefacts import sha256_path

# categorical slots 1 and 2 of the dataviz reference palette; FLAG is its red
OLD, NEW, SERUM, FLAG = "#b8b7b0", "#eb6834", "#2a78d6", "#e34948"

Step = tuple[dict[str, Any], dict[str, Any], Path]  # chain.json entry, step.json, thumbnail


def build_review(chain_root: Path, thresholds: Thresholds = THRESHOLDS) -> Path:
    """Flags are judged here from the recorded numbers, so thresholds can change freely."""
    chain_root = Path(chain_root)
    doc = json.loads((chain_root / "chain.json").read_text())
    out = chain_root / "review"
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    steps: list[Step] = []
    oriented_prev: np.ndarray | None = None
    prev_chart: Chart | None = None
    key = ""
    warnings = {t["table_id"]: t.get("warnings", []) for t in doc["config"]["tables"]}
    for s in doc["steps"]:
        d = chain_root / s["directory"]
        names = ("merge.ace", "incremental.ace", "scratch.ace", "chosen.ace", "step.json")
        s = {**s, "files": [n for n in names if (d / n).exists()]}
        record_text = (d / "step.json").read_text()
        record = json.loads(record_text)
        record["flags"] = flags(record["diagnostics"], thresholds)  # in memory only
        record["table_warnings"] = warnings.get(s["table_id"], [])  # current, not from the step
        if record["table_warnings"]:
            record["flags"].append(f"{len(record['table_warnings'])} table warnings")
        key = _sha(key, sha256_path(d / "chosen.ace"), _sha(record_text))
        thumb = out / "thumbs" / f"{s['index']:04d}-{key[:16]}.png"
        chart = read_chart(d / "chosen.ace")
        layout = chart.best().layout
        if prev_chart is not None and oriented_prev is not None:
            target = np.full_like(layout, np.nan)
            target[previous_in_current(prev_chart, chart)] = oriented_prev
            layout = procrustes(target, layout).apply(layout)
        if not thumb.exists():
            _thumbnail(chart, layout, record, thumb)
        steps.append((s, record, thumb.relative_to(out)))
        oriented_prev, prev_chart = layout, chart
    page = out / "index.html"
    page.write_text(_page(doc, steps))
    return page


def _sha(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def _thumbnail(chart: Chart, layout: np.ndarray, record: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    new = set(record.get("merge", {}).get("new_antigens", []))
    grid = record["diagnostics"].get("grid_test", [])
    trapped = sorted({g["point"] for g in grid if g["diagnosis"] == "trapped"})
    n_ag = chart.n_antigens
    ag = layout[:n_ag]
    old_idx = [i for i in range(n_ag) if i not in new]
    fig, ax = plt.subplots(figsize=(2.6, 2.6), dpi=100)
    ax.scatter(ag[old_idx, 0], ag[old_idx, 1], s=6, c=OLD, linewidths=0)
    if new:
        new_idx = sorted(new)
        ax.scatter(ag[new_idx, 0], ag[new_idx, 1], s=10, c=NEW, linewidths=0)
    sr = layout[n_ag:]
    ax.scatter(sr[:, 0], sr[:, 1], s=16, marker="s", facecolors="none", edgecolors=SERUM, lw=0.8)
    if trapped:
        t = layout[trapped]
        ax.scatter(t[:, 0], t[:, 1], s=60, facecolors="none", edgecolors=FLAG, linewidths=1.2)
    _square_unit_grid(ax, layout)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path)
    plt.close(fig)


def _square_unit_grid(ax: Any, layout: np.ndarray) -> None:
    """Equal aspect, 1-unit grid (one unit = a two-fold titre change), no tick labels."""
    ok = ~np.isnan(layout).any(axis=1)
    lo, hi = layout[ok].min(axis=0), layout[ok].max(axis=0)
    c, r = (lo + hi) / 2, max(hi - lo) / 2 + 0.5
    ax.set_aspect("equal")
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_xticks(np.arange(np.floor(c[0] - r), c[0] + r, 1.0), labels=[])
    ax.set_yticks(np.arange(np.floor(c[1] - r), c[1] + r, 1.0), labels=[])
    ax.grid(True, color="#e6e5e0", linewidth=0.4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ----------------------------------------------------------------------
# HTML


def _e(x: object) -> str:
    return html.escape(str(x))


def _fmt(v: float | None, spec: str = ".4f") -> str:
    return "–" if v is None else format(v, spec)


def _asset(name: str) -> str:
    """CSS and JS live beside this module so they can be read and edited as themselves."""
    return resources.files("af.chain").joinpath(name).read_text()


def _stress_points(steps: list[Step]) -> list[tuple[int, float, float, str, str]]:
    """(step, incremental, scratch, table id, chosen), stress per titre."""
    pts = []
    for s, r, _ in steps:
        terms = r["diagnostics"].get("stress_terms")
        if not terms:
            continue
        st = r["stress"]
        inc = st.get("incremental", st.get("scratch", st.get("seed")))
        scr = st.get("scratch", st.get("seed"))
        pts.append((s["index"], inc / terms, scr / terms, s["table_id"], r["chosen"]))
    return pts


def _stress_chart(steps: list[Step]) -> str:
    """Stress per titre by step, incremental and scratch, one axis; hover gives the numbers."""
    pts = _stress_points(steps)
    if len(pts) < 2:
        return ""
    w, h, left, right, top, bottom = 900, 220, 56, 16, 16, 28
    xs = [p[0] for p in pts]
    ys = [v for p in pts for v in p[1:3]]
    x0, x1 = min(xs), max(xs)
    pad = (max(ys) - min(ys)) * 0.08 or 0.01
    y0, y1 = min(ys) - pad, max(ys) + pad
    plot_w, plot_h = w - left - right, h - top - bottom

    def x(v: float) -> float:
        return left + (v - x0) / ((x1 - x0) or 1) * plot_w

    def y(v: float) -> float:
        return top + (1 - (v - y0) / (y1 - y0)) * plot_h

    parts = []
    for v in np.linspace(y0 + pad, y1 - pad, 4):
        parts.append(
            f'<line x1="{left}" x2="{w - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left - 6}" y="{y(v) + 4:.1f}" class="tick" text-anchor="end">{v:.3f}</text>'
        )
    for tick in sorted({int(round(t)) for t in np.linspace(x0, x1, min(8, len(xs)))}):
        parts.append(
            f'<text x="{x(tick):.1f}" y="{h - 8}" class="tick" text-anchor="middle">{tick}</text>'
        )
    for k, cls in ((1, "l1"), (2, "l2")):
        points = " ".join(f"{x(p[0]):.1f},{y(float(p[k])):.1f}" for p in pts)
        parts.append(f'<polyline class="{cls}" points="{points}"/>')
    band = plot_w / len(pts)
    for p in pts:
        tip = f"step {p[0]} · {_e(p[3])}&#10;"
        tip += f"incremental {p[1]:.4f} · scratch {p[2]:.4f} · chose {p[4]}"
        parts.append(
            f'<rect class="hit" x="{x(p[0]) - band / 2:.1f}" y="{top}" width="{band:.1f}" '
            f'height="{plot_h}" data-step="{p[0]}" data-tip="{tip}"></rect>'
        )
    key = (
        '<i style="background:var(--s1)"></i>incremental '
        '<i style="background:var(--s2)"></i>scratch'
    )
    return (
        f'<figure class="chart"><figcaption>Stress per titre by step <span class="key">{key}</span>'
        f'</figcaption><svg viewBox="0 0 {w} {h}" role="img" aria-label="Stress per titre by step">'
        + "".join(parts)
        + '</svg><div class="tip" hidden></div></figure>'
    )


def _details(title: str, rows: list[str]) -> str:
    if not rows:
        return ""
    items = "".join(f"<li>{r}</li>" for r in rows[:200])
    return f"<details><summary>{_e(title)} ({len(rows)})</summary><ul>{items}</ul></details>"


def _step_info(s: dict, r: dict) -> list[str]:
    d, m, st = r["diagnostics"], r.get("merge", {}), r["stress"]
    counts = f"antigens {d.get('antigens')} · sera {d.get('sera')}"
    if m:
        counts += (
            f" · new {len(m.get('new_antigens', []))} ag / {len(m.get('new_sera', []))} sr"
            f" · layers {m.get('layers')}"
        )
    stresses = " · ".join(f"{k} {_fmt(v)}" for k, v in st.items())
    rmsd_prev = _fmt(d.get("procrustes_to_previous_rmsd"), ".3f")
    rmsd_basin = _fmt(d.get("incremental_vs_scratch_rmsd"), ".3f")
    info = [
        f"<b>step {s['index']}</b> · {_e(s['table_id'])}",
        counts,
        f"stress {stresses} · <b>chose {r['chosen']}</b>",
        f"RMSD to previous {rmsd_prev} · incremental vs scratch RMSD {rmsd_basin}",
    ]
    if m.get("cheating_assay"):
        info.append(
            f"cheating assay: {m['skipped_reference_antigens']} reference antigens not merged"
        )
    return info


def _step_row(s: dict, r: dict, thumb: Path) -> str:
    d = r["diagnostics"]
    step_flags = r["flags"]
    details = "".join(
        [
            _details(
                "moved > 0.5 since the previous step",
                [f"{_e(x['point'])} — {x['distance']:.2f}" for x in d.get("moved_far", [])],
            ),
            _details(
                "grid test",
                [
                    f"{_e(g['name'])} — {g['diagnosis']}, {g['distance']:.2f} away, "
                    f"{g['contribution_diff']:+.2f}"
                    for g in d.get("grid_test", [])
                ],
            ),
            _details(
                "column-basis slack",
                [f"{_e(x['serum'])} — {x['slack']:.2f}" for x in d.get("column_basis_slack", [])],
            ),
            _details("disconnected", [_e(x) for x in d.get("disconnected", [])]),
            _details("table warnings", [_e(x) for x in r["table_warnings"]]),
        ]
    )
    flag_html = "".join(f'<span class="flag">{_e(f)}</span>' for f in step_flags)
    files = " · ".join(f'<a href="../{_e(s["directory"])}/{n}">{n}</a>' for n in s["files"])
    image = (
        f'<img src="{_e(thumb)}" width="200" height="200" '
        f'alt="map, step {s["index"]}" loading="lazy">'
    )
    return (
        f'<tr id="step-{s["index"]}" class="{"flagged" if step_flags else ""}"><td>{image}</td>'
        f"<td>{'<br>'.join(_step_info(s, r))}<div>{flag_html}</div>{details}"
        f'<div class="files">{files}</div></td></tr>'
    )


def _page(doc: dict, steps: list[Step]) -> str:
    name = doc["config"]["name"]
    flagged = [(s, r) for s, r, _ in steps if r["flags"]]
    summary = "".join(
        f'<li><a href="#step-{s["index"]}">step {s["index"]} · {_e(s["table_id"])}</a>: '
        f"{_e('; '.join(r['flags']))}</li>"
        for s, r in flagged
    )
    opts = doc["config"]["options"]
    reused = sum(1 for s in doc["steps"] if s["reused"])
    meta = (
        f"{len(steps)} steps · {reused} reused from an earlier run"
        f" · optimiser {_e(doc['optimiser'])}"
        f" · scratch starts {opts['scratch_starts']}"
        f", incremental starts {opts['incremental_starts']}"
        f" · minimum column basis {_e(opts['minimum_column_basis'])}"
        f" · column bases {_e(opts['column_bases'])}"
    )
    legend = (
        '<i class="dot old"></i>antigen <i class="dot new"></i>new this step '
        '<i class="sq"></i>serum '
        '<i class="ring"></i>trapped (grid test). Maps oriented to the previous step.'
    )
    rows = "".join(_step_row(s, r, t) for s, r, t in steps)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_e(name)} chain</title><style>{_asset('review.css')}</style></head><body>"
        f'<h1>{_e(name)} chain</h1><p class="meta">{meta}</p><p class="legend">{legend}</p>'
        f"<h2>Flagged steps ({len(flagged)})</h2>"
        f'<ul class="summary">{summary or "<li>No step flagged.</li>"}</ul>'
        f'{_stress_chart(steps)}<h2>Steps</h2><table class="steps">{rows}</table>'
        f"<script>{_asset('review.js')}</script></body></html>"
    )
