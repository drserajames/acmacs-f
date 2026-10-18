"""Adopt a layout from another version of the same problem: the next version of a merged map whose
chosen layout is another published chain's map of the same tables, placed point by point, minimised
once more under this dataset's own arithmetic, and passed through its own trapped-point loop.

    python -m af.chain.adopt <chain.toml> <run.toml> --version <id> --from <dataset>@<version>
        [--step NNNN]

Why (Q210 b, 16 Oct 2026; OWED 39): an incremental chain and a from-scratch merge of the same tables
are the same problem, but can sit in different minima (h3-hint: the incremental arrangement is 1.4%
lower). Without this a report that wants the incremental arrangement must point at another dataset;
with it the merged dataset carries that arrangement as its own next version, with its provenance
naming where the layout came from. `af.chain.finish` cannot do this: it refuses any layout whose
stress is not the one the version itself recorded, which is the guard against a problem read wrong.
Here the guard is problem identity instead.

It refuses, before anything is minimised, unless:
- the target version passes every check `af.chain.finish` makes of the version it finishes (CURRENT,
  one-step merge_all, today's inputs, parameters and merged table its build's, its own stress);
- the source is a published version of a chains dataset in the same store, its manifest intact;
- the two problems are the same: every point pairs, by its identity key (`af.chart.identity`) or,
  for a point with no key or a renamed one, by a titre row (antigen) or column (serum) identical to
  exactly one unpaired point's on the other side, each such pairing counted and recorded; and after
  pairing, the titre values and types, column bases, disconnected points and the dodgy flag (and any
  avidity adjustments and unmovable points) are identical;
- the source layout, placed, has the stress its map recorded (to the last bits an arithmetic moves).
The new version is published only when the run config says publish, with the target as its parent.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import sys
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from af.chain.__main__ import (
    RunSettings,
    _optimiser,
    check_tables_dataset,
    dataset_from_path,
    measure_references,
)
from af.chain.config import ChainConfig, ChainConfigError, load_chain_config
from af.chain.diagnostics import step_diagnostics
from af.chain.engine import (
    CHOSEN,
    STEP_RECORD,
    Mapper,
    StepResult,
    _ace,
    _action,
    _json,
    _map_chart,
    _merge_all_record,
    _projections,
    _write_chain_record,
    _write_step_record,
    chain_steps,
    merge_all_tables,
    step_dir,
)
from af.chain.finish import FinishError, Source, check_source
from af.chain.publish import KIND, publish_chain
from af.chain.review import build_review
from af.chart.ace import read_chart, write_chart
from af.chart.identity import AntigenKey, SerumKey, antigen_identity, serum_identity
from af.chart.model import Chart
from af.pipeline import Pipeline, Step
from af.run.job import Runner, run_main
from af.run.local import LocalRunner
from af.run.runtime import require_python
from af.store.ref import StoreRef
from af.store.store import Store
from af.store.work import Work
from af.util.artefacts import Artefact, sha256_path
from af.util.config import load_config

log = logging.getLogger(__name__)

# arrays indexed by point (antigens, then sera) that must agree once the points are paired
POINT_ARRAYS = ("disconnected", "avidity_adjust", "unmovable")


class AdoptError(RuntimeError):
    pass


@dataclass(frozen=True)
class Pairing:
    """Which source point each target point is: `antigens[i]` is the source antigen of target
    antigen i, `sera[j]` the source serum of target serum j. `by_titres` lists the points paired
    by their titres rather than their key (no key, or renamed), for the record."""

    antigens: np.ndarray
    sera: np.ndarray
    by_titres: tuple[dict[str, Any], ...]

    def points(self, n_source_antigens: int) -> np.ndarray:
        """Source point index of each target point, antigens then sera."""
        return np.concatenate([self.antigens, n_source_antigens + self.sera])


@dataclass(frozen=True)
class Adopted:
    """The source map whose layout is adopted."""

    dataset: str
    version: str
    step: str  # its steps/NNNN directory name
    directory: Path
    map_sha256: str
    stress: float

    @property
    def chosen(self) -> Path:
        return self.directory / CHOSEN

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "version": self.version,
            "step": self.step,
            "map_sha256": self.map_sha256,
            "stress": self.stress,
        }


def _keys(chart: Chart) -> tuple[list[AntigenKey | None], list[SerumKey | None]]:
    ag = [antigen_identity(a.name, a.reassortant, a.annotations, a.passage) for a in chart.antigens]
    sr = [serum_identity(s.name, s.reassortant, s.annotations, s.serum_id) for s in chart.sera]
    return ag, sr


def _by_key(target: Sequence[Hashable | None], source: Sequence[Hashable | None], what: str
            ) -> dict[int, int]:  # fmt: skip
    """target index -> source index for the points whose keys match. A key held twice on one side
    cannot say which point is which: refused."""
    for side, keys in (("target", target), ("source", source)):
        counts = Counter(k for k in keys if k is not None)
        doubled = [k for k, n in counts.items() if n > 1]
        if doubled:
            raise AdoptError(f"{len(doubled)} {what} identity keys held twice in the {side}")
    where = {k: i for i, k in enumerate(source) if k is not None}
    return {i: where[k] for i, k in enumerate(target) if k is not None and k in where}


def _signature(value: np.ndarray, kind: np.ndarray) -> bytes:
    """A titre row's (or column's) cells as bytes: missing cells compare equal to each other."""
    return np.nan_to_num(value, nan=np.inf).tobytes() + kind.tobytes()


def _by_titres(
    pairs: dict[int, int], n_target: int, n_source: int, target_cells, source_cells, what: str
) -> list[tuple[int, int]]:
    """Pair the points keys left unpaired by their cells against the points already paired on the
    other axis: a target point pairs with the one unpaired source point whose cells are the same,
    and only if no other point on either side has those cells too."""
    loose_t = [i for i in range(n_target) if i not in pairs]
    used = set(pairs.values())
    loose_s = [i for i in range(n_source) if i not in used]
    t_sig = {i: target_cells(i) for i in loose_t}
    s_sig = {i: source_cells(i) for i in loose_s}
    found = []
    for i, sig in t_sig.items():
        same_s = [j for j, s in s_sig.items() if s == sig]
        same_t = [k for k, t in t_sig.items() if t == sig]
        if len(same_s) == 1 and len(same_t) == 1:
            found.append((i, same_s[0]))
    left_t, left_s = len(loose_t) - len(found), len(loose_s) - len(found)
    if left_t or left_s:
        raise AdoptError(
            f"not the same problem: {left_t} target and {left_s} source {what} pair with nothing"
            " (no matching key, and no unique identical titres)"
        )
    return found


def pair_points(target: Chart, t_arrays: dict, source: Chart, s_arrays: dict) -> Pairing:
    """Every target point's source point, by key, then by titres for what keys leave (see the
    module docstring). Antigens are paired first so that sera can be compared over them."""
    t_ag, t_sr = _keys(target)
    s_ag, s_sr = _keys(source)
    ag = _by_key(t_ag, s_ag, "antigen")
    sr = _by_key(t_sr, s_sr, "serum")
    tv, tk = t_arrays["titre_value"], t_arrays["titre_type"]
    sv, sk = s_arrays["titre_value"], s_arrays["titre_type"]
    by_titres: list[dict[str, Any]] = []

    cols_t, cols_s = list(sr), [sr[j] for j in sr]
    found = _by_titres(
        ag, target.n_antigens, source.n_antigens,
        lambda i: _signature(tv[i, cols_t], tk[i, cols_t]),
        lambda i: _signature(sv[i, cols_s], sk[i, cols_s]),
        "antigens",
    )  # fmt: skip
    for i, j in found:
        ag[i] = j
        by_titres.append(_named("antigen", i, j, target.antigens[i].name, source.antigens[j].name))
    rows_t, rows_s = list(ag), [ag[i] for i in ag]
    found = _by_titres(
        sr, target.n_sera, source.n_sera,
        lambda j: _signature(tv[rows_t, j], tk[rows_t, j]),
        lambda j: _signature(sv[rows_s, j], sk[rows_s, j]),
        "sera",
    )  # fmt: skip
    for i, j in found:
        sr[i] = j
        by_titres.append(_named("serum", i, j, target.sera[i].name, source.sera[j].name))
    return Pairing(
        np.array([ag[i] for i in range(target.n_antigens)], dtype=int),
        np.array([sr[j] for j in range(target.n_sera)], dtype=int),
        tuple(by_titres),
    )


def _named(what: str, target: int, source: int, t_name: str, s_name: str) -> dict[str, Any]:
    return {"what": what, "target": target, "source": source, "target_name": t_name,
            "source_name": s_name}  # fmt: skip


def check_same_problem(t_arrays: dict, s_arrays: dict, pairing: Pairing, n_source_ag: int) -> None:
    """Refuse unless the source's arrays, reordered by `pairing`, are the target's: every cell,
    column basis and point flag. Says which differ and in how many places."""
    a, s, p = pairing.antigens, pairing.sera, pairing.points(n_source_ag)
    differ = []
    value = s_arrays["titre_value"][np.ix_(a, s)]
    same = (value == t_arrays["titre_value"]) | (
        np.isnan(value) & np.isnan(t_arrays["titre_value"])
    )
    if not same.all():
        differ.append(f"titre values ({int((~same).sum())} cells)")
    kind = s_arrays["titre_type"][np.ix_(a, s)] != t_arrays["titre_type"]
    if kind.any():
        differ.append(f"titre types ({int(kind.sum())} cells)")
    bases = np.asarray(s_arrays["column_bases"], dtype=float)[s]
    if not np.array_equal(bases, np.asarray(t_arrays["column_bases"], dtype=float)):
        n = int((bases != np.asarray(t_arrays["column_bases"], dtype=float)).sum())
        differ.append(f"column bases ({n} sera)")
    for name in POINT_ARRAYS:
        t, src = t_arrays.get(name), s_arrays.get(name)
        if t is None and src is None:
            continue
        if t is None or src is None or not np.array_equal(np.asarray(src)[p], np.asarray(t)):
            differ.append(name)
    if bool(t_arrays.get("dodgy_is_regular")) != bool(s_arrays.get("dodgy_is_regular")):
        differ.append("dodgy_is_regular")
    if differ:
        raise AdoptError(f"not the same problem: {', '.join(differ)} differ after pairing")


def find_source(store_root: Path, spec: str, step: str | None) -> Adopted:
    """`<dataset>@<version>` of a chains dataset in this store: a published version whose
    manifest is intact (a version id is its manifest's sha256, cut), and one of its steps."""
    dataset, sep, version = spec.rpartition("@")
    if not sep or not dataset:
        raise AdoptError(f"--from {spec!r}: expected <dataset>@<version>")
    store = Store.open(store_root)
    if version not in {h.get("version") for h in store.history(KIND, dataset)}:
        raise AdoptError(f"{dataset}@{version} is not a published version in {store_root}")
    manifest = store.dataset_dir(KIND, dataset) / "versions" / version / "MANIFEST.json"
    ref = StoreRef(KIND, dataset, version, sha256_path(manifest))
    vdir = store.resolve(ref)
    doc = json.loads((vdir / "chain.json").read_text())
    steps = [s["directory"] for s in doc["steps"]]
    name = f"steps/{step}" if step is not None else steps[-1]
    if name not in steps:
        raise AdoptError(f"{dataset}@{version} has no {name} (steps {steps[0]}..{steps[-1]})")
    sdir = vdir / name
    stress = read_chart(sdir / CHOSEN).best().stress_value
    return Adopted(dataset, version, Path(name).name, sdir, sha256_path(sdir / CHOSEN), stress)


def source_arrays(adopted: Adopted) -> tuple[Chart, dict]:
    """The source map's own problem, from its chosen chart and the options its step recorded."""
    sidecar = json.loads((adopted.directory / f"{CHOSEN}.provenance.json").read_text())
    o = sidecar["parameters"]["options"]
    chart = read_chart(adopted.chosen)
    arrays = chart.optimiser_arrays(
        o["minimum_column_basis"], chart.best(), disconnect_threshold=o["disconnect_threshold"]
    )
    return chart, arrays


def place(cfg: ChainConfig, target: Source, adopted: Adopted) -> tuple[np.ndarray, Pairing, dict]:
    """The source's best layout in the target's point order, once the problems are shown to be
    the same; with the pairing and the target's arrays."""
    m = merge_all_tables(cfg)
    t_chart = read_chart(target.chosen)
    s_chart, s_arrays = source_arrays(adopted)
    pairing = pair_points(t_chart, m.arrays, s_chart, s_arrays)
    check_same_problem(m.arrays, s_arrays, pairing, s_chart.n_antigens)
    layout = np.asarray(s_chart.best().layout, dtype=float)[pairing.points(s_chart.n_antigens)]
    return layout, pairing, m.arrays


def adopt_step(
    cfg: ChainConfig,
    root: Path,
    step: Step,
    target: Source,
    adopted: Adopted,
    pairing: Pairing,
    layout: np.ndarray,
    mapper: Mapper,
) -> Step:
    """The adoption as a pipeline step: the chain's own inputs and parameters plus the source
    map, so it is never taken for a search, and a rerun that changed nothing reuses it."""
    d0 = step_dir(root, 0)
    action = functools.partial(
        _adopt, cfg, target=target, adopted=adopted, pairing=pairing, layout=layout, mapper=mapper
    )
    return Step(
        name="step-0000",
        action=_action(action, d0),
        inputs={**dict(step.inputs), "adopted": adopted.chosen},  # type: ignore[arg-type]
        outputs=[Artefact(d0 / n, parse=_ace) for n in ("merge.ace", CHOSEN)]
        + [Artefact(d0 / STEP_RECORD, parse=_json)],
        parameters={
            **step.parameters,
            "adopted_from": adopted.to_json(),
            "adopted_onto": {"version": target.version, "map_sha256": target.map_sha256},
        },
    )


def _adopt(
    cfg: ChainConfig,
    directory: Path,
    runner: Runner,
    *,
    target: Source,
    adopted: Adopted,
    pairing: Pairing,
    layout: np.ndarray,
    mapper: Mapper,
) -> None:
    m = merge_all_tables(cfg)
    write_chart(m.merged, directory / "merge.ace")
    if sha256_path(directory / "merge.ace") != sha256_path(target.directory / "merge.ace"):
        raise AdoptError(f"{target.version}: the tables no longer merge to its merge.ace")
    o = cfg.options
    try:
        made = mapper.optimiser.adopt(
            m.arrays, layout, adopted.stress, move_groups=o.move_groups,
            move_lower=o.move_lower, move_lower_rule=o.move_lower_rule,
        )  # fmt: skip
    except ValueError as e:
        raise AdoptError(f"{adopted.dataset}@{adopted.version}: {e}") from e
    # The adopted map is projection 0 BY DECISION, not by stress: the target's own basins follow
    # it unchanged, and one of them may be lower (the step records both stresses).
    own = read_chart(target.chosen).projections
    chart = _map_chart(m.chart, [*_projections([made], m.arrays, o.minimum_column_basis), *own])
    write_chart(chart, directory / CHOSEN)
    base = _merge_all_record(cfg, m)
    record = {
        **{k: base[k] for k in ("index", "table_id", "merge_all")},
        "optimiser": mapper.optimiser.name,
        "chosen": "adopted",  # projection 0 by decision; not necessarily the lowest stress
        "stress": {"adopted": made["stress"], "target": target.stress, "source": adopted.stress},
        "adopted_from": {**adopted.to_json(), "paired_by_titres": list(pairing.by_titres)},
        "adopted_onto": target.to_json(),
        "adopt": made["adopt"],
        **{k: v for k, v in base.items() if k not in ("index", "table_id", "merge_all")},
    }
    diagnostics = step_diagnostics(
        chart, None, None, None, m.arrays, mapper.optimiser, cfg, m.report
    )
    diagnostics.pop("control_flags", None)  # as for any merge_all map
    record["diagnostics"] = diagnostics
    _write_step_record(directory, record, mapper)


def adopt_chain(
    cfg: ChainConfig,
    root: Path,
    target: Source,
    adopted: Adopted,
    pairing: Pairing,
    layout: np.ndarray,
    optimiser: Any,
    step: Step,
    site: str | None = None,
) -> StepResult:
    """Run the adoption into `root` (its own chain root) and write its chain.json."""
    mapper = Mapper(optimiser, None, site)
    one = adopt_step(cfg, root, step, target, adopted, pairing, layout, mapper)
    [outcome] = Pipeline([one], state_dir=root / "state", runner=LocalRunner(), root=root).run()
    d0 = step_dir(root, 0)
    result = StepResult(
        0, d0, outcome.status == "skipped", json.loads((d0 / STEP_RECORD).read_text())
    )
    _write_chain_record(root, cfg, [result], optimiser, n_steps=1)
    path = root / "chain.json"
    doc = json.loads(path.read_text())
    doc["adopted_from"] = adopted.to_json()
    doc["adopted_onto"] = target.to_json()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(path)
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m af.chain.adopt", description=__doc__.split("\n")[0]
    )
    parser.add_argument("chain", type=Path, help="chain config (TOML) of the dataset adopting")
    parser.add_argument("run", type=Path, help="run config (TOML)")
    parser.add_argument("--version", required=True, help="the dataset's CURRENT version")
    parser.add_argument(
        "--from", dest="source", required=True,
        help="<dataset>@<version>: the published chain whose map is adopted",
    )  # fmt: skip
    parser.add_argument("--step", help="its step (NNNN); default its last")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dataset = dataset_from_path(args.chain)
    check_tables_dataset(args.chain, dataset)
    run = load_config(args.run, RunSettings)
    if run.python is not None:
        require_python(run.python)
    if run.optimiser != "core":
        raise ChainConfigError(f"adopting needs the core optimiser, not {run.optimiser!r}")
    optimiser = _optimiser(run)
    work = Work.open(run.paths.work).dataset("chains", dataset)
    cfg = load_chain_config(
        args.chain, inputs_dir=work.root / "inputs", tables_store=run.paths.store
    )
    if not cfg.options.merge_all:
        raise ChainConfigError(f"{dataset}: only a merge_all dataset adopts a layout")
    [step] = chain_steps(cfg, work.root, Mapper(optimiser, None, run.site))
    try:
        target = check_source(cfg, Path(run.paths.store), dataset, args.version, step, work.root)
    except FinishError as e:
        raise AdoptError(str(e)) from e
    adopted = find_source(Path(run.paths.store), args.source, args.step)
    if adopted.dataset == dataset:
        raise AdoptError(f"{dataset}: a dataset adopts another's layout, not its own")
    layout, pairing, _ = place(cfg, target, adopted)
    log.info(
        "%s@%s and %s@%s are the same problem: %d points, %d paired by titres",
        dataset, args.version, adopted.dataset, adopted.version, len(layout),
        len(pairing.by_titres),
    )  # fmt: skip
    root = work.root / "adopt" / args.version / f"{adopted.version}-{adopted.step}"
    root.mkdir(parents=True, exist_ok=True)
    result = adopt_chain(cfg, root, target, adopted, pairing, layout, optimiser, step, run.site)
    log.info(
        "%s@%s adopted %s@%s step %s: stress %.6f (source %.6f, target %.6f; %s)",
        dataset, args.version, adopted.dataset, adopted.version, adopted.step,
        result.record["stress"]["adopted"], adopted.stress, target.stress,
        "reused" if result.reused else "made",
    )  # fmt: skip
    if cfg.references:
        measure_references(cfg, run, root)
    page = build_review(root)
    log.info("review page: %s", page)
    if run.publish:
        ref = publish_chain(Path(run.paths.store), dataset, root)
        log.info("published %s (parent %s)", ref, args.version)
    return 0


if __name__ == "__main__":
    run_main(lambda: main(sys.argv[1:]))
