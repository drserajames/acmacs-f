"""Draw a round's report trees: each tree's figure and leaf book, in parallel processes.

Why: measured 18 Oct 2026, the three report trees with their books take 873 s on one core,
and no part waits on another. A tree's figure and its book each arrange the tree for
themselves, and the trees are independent. With every part in its own process, the stage
takes as long as its largest part: H1's book, about 260 s.

Each part runs in a fresh, spawned process that loads its tree from its source itself. A part
therefore draws the same bytes whichever process, order or number of workers runs it, and
nothing large is copied between processes. With ``workers`` 1 the parts run one after another
in this process, as before.

Parts already drawn are reused through :mod:`.cache` when a cache directory is given. The
cache is looked up for every part before any drawing starts, and entries are stored only
after the drawn files pass their checks.
"""

from __future__ import annotations

import json
import multiprocessing
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from af.store import Store, StoreRef

from .book import make_book
from .cache import CacheResult, canonical, code_identity, keep, reuse
from .figure import FigureConfig, make_figure
from .store import StoreTree, load_version


class TreeStageError(RuntimeError):
    """One or more parts failed to draw; every failure is listed."""


class TreeSource(Protocol):
    """Where a job's tree comes from. Must pickle: it is sent to the drawing process."""

    def load(self) -> StoreTree: ...

    def identity(self) -> dict[str, Any]:
        """What makes this tree this tree, for the cache key (a content hash, not a path)."""
        ...

    def refs(self) -> list[StoreRef]:
        """Store versions recorded in the drawn files' provenance."""
        ...


@dataclass(frozen=True)
class StoreVersion:
    """A tree store version, resolved and fully verified in the process that draws it."""

    store: Path
    ref: StoreRef

    def load(self) -> StoreTree:
        return load_version(Store.open(self.store).resolve(self.ref, verify=True))

    def identity(self) -> dict[str, Any]:
        result: dict[str, Any] = canonical(self.ref)  # the manifest hash names the content
        return result

    def refs(self) -> list[StoreRef]:
        return [self.ref]


@dataclass(frozen=True)
class TreeJob:
    name: str  # e.g. the subtype; names the job in messages and results
    source: TreeSource
    config: FigureConfig
    figure: Path  # figure.pdf; its I7 and draw report are written beside it
    book: Path | None = None  # book.pdf (its report beside it), or no book


@dataclass
class TreeResult:
    report: dict[str, Any]  # the figure's draw report, with the book's summary under "book"
    cache: dict[str, CacheResult] = field(default_factory=dict)  # per part


def outputs(part: str, pdf: Path) -> dict[str, Path]:
    """Entry file name -> where it is written, for one part."""
    if part == "figure":
        return {
            "figure.pdf": pdf,
            "figure.i7.json": pdf.with_suffix(".i7.json"),
            "figure.draw.json": pdf.with_suffix(".draw.json"),
        }
    return {"book.pdf": pdf, "book.book.json": pdf.with_suffix(".book.json")}


def draw_trees(
    jobs: Sequence[TreeJob],
    *,
    cache: Path | None = None,
    workers: int = 1,
    code: Mapping[str, Any] | None = None,
) -> dict[str, TreeResult]:
    """Draw (or reuse) every job's figure and book; return each job's draw report.

    ``code`` defaults to :func:`.cache.code_identity` (tests pass their own).
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, not {workers}")
    names = [job.name for job in jobs]
    if len(set(names)) != len(names):
        raise ValueError(f"job names repeat: {names}")
    code = code_identity() if code is None else code
    todo: list[tuple[TreeJob, str, Path]] = []
    results = {job.name: TreeResult({}) for job in jobs}
    for job in jobs:
        for part, pdf in _parts(job):
            hit = reuse(cache, _key(job, part), code, outputs(part, pdf), _step(job, part))
            results[job.name].cache[part] = hit
            if not hit.hit:
                todo.append((job, part, pdf))
    todo.sort(key=lambda t: t[1] != "book")  # books first: the longest parts should not start last
    failures = _run(todo, workers)
    if failures:
        lines = "\n".join(f"  - {name}: {error}" for name, error in failures)
        raise TreeStageError(f"{len(failures)} tree part(s) failed:\n{lines}")
    for job, part, pdf in todo:
        keep(cache, _key(job, part), code, outputs(part, pdf), _step(job, part))
    for job in jobs:
        results[job.name].report = _report(job)
    return results


def draw_part(source: TreeSource, config: FigureConfig, part: str, pdf: Path) -> None:
    """Draw one part from its source. Module level, so a spawned process can run it."""
    s = source.load()
    inputs = {**s.inputs, "clade_set": s.clade_set_version}
    pdf.parent.mkdir(parents=True, exist_ok=True)
    draw = make_figure if part == "figure" else make_book
    draw(
        s.tree,
        s.parents,
        config,
        pdf,
        inputs,
        s.flags,
        store_refs=source.refs(),
        depth_unit=s.branch_scale,
    )


def _parts(job: TreeJob) -> list[tuple[str, Path]]:
    return [("figure", job.figure)] + ([("book", job.book)] if job.book is not None else [])


def _key(job: TreeJob, part: str) -> dict[str, Any]:
    return {"tree": job.source.identity(), "figure": canonical(job.config), "part": part}


def _step(job: TreeJob, part: str) -> str:
    return f"tree {job.name} {part}"


def _run(todo: list[tuple[TreeJob, str, Path]], workers: int) -> list[tuple[str, str]]:
    """Draw every part; return (step, error) for each that failed, after all have finished."""
    failures: list[tuple[str, str]] = []
    if workers == 1 or len(todo) <= 1:
        for job, part, pdf in todo:
            try:
                draw_part(job.source, job.config, part, pdf)
            except Exception as error:
                failures.append((_step(job, part), f"{type(error).__name__}: {error}"))
        return failures
    context = multiprocessing.get_context("spawn")  # a fresh interpreter: no state carried over
    with ProcessPoolExecutor(max_workers=min(workers, len(todo)), mp_context=context) as pool:
        running = [
            (job, part, pool.submit(draw_part, job.source, job.config, part, pdf))
            for job, part, pdf in todo
        ]
        for job, part, future in running:
            problem = future.exception()
            if problem is not None:
                failures.append((_step(job, part), f"{type(problem).__name__}: {problem}"))
    return failures


def _report(job: TreeJob) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(job.figure.with_suffix(".draw.json").read_text())
    if job.book is not None:
        book = json.loads(job.book.with_suffix(".book.json").read_text())
        if book["rows"] != report["rows"]:
            raise TreeStageError(
                f"tree {job.name}: book has {book['rows']} rows, figure {report['rows']}"
            )
        report["book"] = {k: book[k] for k in ("pages", "rows_per_page", "seconds")}
    return report
