"""Which optimiser the chain uses: workstream 6's core (`af.map.optimise`), or a stand-in.

Every backend returns maps as dicts {layout, stress, dimensions, n_iterations, start_seed}
sorted by stress, and can run a map's starts in chunks (`relax_chunk`) and put the chunks
back together (`combine`) so that the result is the same however the starts were split
(start i always draws from its own seed; COORDINATION.md I1).
"""

from __future__ import annotations

import inspect
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np

MapResult = dict[str, Any]

# How far a published layout's stress, evaluated again, may be from the one its map recorded
# before a finish refuses it: last-bit differences between the optimiser's own final value and a
# fresh evaluation, never a different problem (CoreOptimiser.finish).
FINISH_STRESS_TOLERANCE = 1e-9

# The move-lower rules a chain or a finish may name (MapOptions.move_lower_rule), by the name each
# rule records, with the af.map.optimise constant that holds it. Names, not the rules, so a
# config can be checked where the core is not built.
MOVE_LOWER_RULES = {"move-lower-1": "MOVE_LOWER_RULE", "ad-last-round-1": "AD_LAST_ROUND"}
DEFAULT_MOVE_LOWER_RULE = "move-lower-1"  # what move_lower meant before rules had names


class Optimiser(Protocol):
    name: str
    key: str  # the name `python -m af.chain.starts` knows it by

    @property
    def numerics(self) -> str | None:
        """The name of the arithmetic this optimiser's maps come from (`af.map.optimise.NUMERICS`),
        or None when it has no name: the stand-in, and the core's original arithmetic, whose
        results belong to one machine. Two values may give different maps from the same problem,
        so chunks, steps and checks made under one are never reused under another. Nothing is
        written for None, so maps made before arithmetics had names keep their digests and
        records."""
        ...

    @property
    def refine_best(self) -> int | None:
        """How many of a map's lowest-stress starts the optimiser minimises again at fine
        precision when the starts were relaxed rough (incremental maps, and scratch maps under
        MapOptions.scratch_precision = "rough"); None when it has no such stage. Recorded with
        each step and chain; maps made before it was recorded used 5."""
        ...

    def move_lower_rule(self, name: str = DEFAULT_MOVE_LOWER_RULE) -> dict[str, Any] | None:
        """The rule `name` the trapped-point loop applies with `move_lower` (`move_lower_rule()`),
        as a step records it; None for an optimiser that has no such rule (the stand-in)."""
        ...

    def optimise(
        self,
        arrays: dict,
        n_starts: int,
        dim: int,
        seed: int,
        start_layout: np.ndarray | None,
        move_groups: bool = False,
        resolve_best: int = 1,
        move_lower: bool = False,
        move_lower_rule: str = DEFAULT_MOVE_LOWER_RULE,
    ) -> list[MapResult]: ...

    def relax_chunk(
        self,
        arrays: dict,
        first_start: int,
        n_starts: int,
        dim: int,
        seed: int,
        start_layout: np.ndarray | None,
    ) -> list[MapResult]: ...

    def combine(
        self,
        arrays: dict,
        chunks: list[MapResult],
        incremental: bool,
        move_groups: bool = False,
        resolve_best: int = 1,
        move_lower: bool = False,
        move_lower_rule: str = DEFAULT_MOVE_LOWER_RULE,
    ) -> list[MapResult]: ...

    def grid_test(self, layout: np.ndarray, arrays: dict) -> list[dict]: ...

    def finish(
        self, arrays: dict, layout: np.ndarray, rule: FinishRule, recorded_stress: float
    ) -> MapResult: ...


def move_lower_rule(name: str = DEFAULT_MOVE_LOWER_RULE) -> dict[str, Any]:
    """The move-lower rule `name` (MOVE_LOWER_RULES) as a step records it (JSON), with the round
    cap it runs under: its own (AD_LAST_ROUND: 1), else resolve_trapped's max_rounds, which the
    chain never sets. A map made under another rule, or another cap, is another map."""
    from af.map.optimise import resolve_trapped

    rule = _rule(name)
    default = inspect.signature(resolve_trapped).parameters["max_rounds"].default
    return json.loads(json.dumps({**asdict(rule), "max_rounds": rule.max_rounds or default}))


def _rule(name: str) -> Any:
    """The af.map.optimise.MoveLowerRule called `name`; an unknown name is an error."""
    import af.map.optimise

    if name not in MOVE_LOWER_RULES:
        raise ValueError(f"unknown move-lower rule {name!r}: known {sorted(MOVE_LOWER_RULES)}")
    return getattr(af.map.optimise, MOVE_LOWER_RULES[name])


def _move_lower_record(fixed: Any) -> dict[str, Any]:
    """resolve_trapped's move-lower rounds, in step-record terms: the rule that ran (its own
    record of it, so a map says what ran on it, not only what its config asked for), why it
    stopped, and per round tried (the rejected last one included) which points it tried and kept,
    and the stresses."""
    return {
        "rule": fixed.move_lower_rule.name,
        "stop": fixed.move_lower_stop,
        "rounds": [
            {
                "round": int(r.round),
                "candidates": [int(i) for i in r.candidates],
                "joint_stress": float(r.joint_stress),
                "how": r.how,
                "moved": [int(i) for i in r.moved],
                "stress_before": float(r.stress_before),
                "stress_after": float(r.stress_after),
                "kept": bool(r.kept),
            }
            for r in fixed.move_lower
        ],
    }


def _group_record(groups: Any) -> list[dict[str, Any]]:
    return [
        {
            "members": [int(m) for m in g.members],
            "shift": [float(x) for x in g.shift],
            "stress_before": float(g.stress_before),
            "stress_after": float(g.stress_after),
            "kept": bool(g.kept),
            "n_antigens": int(g.n_antigens),
            "n_sera": int(g.n_sera),
            "mixed": bool(g.mixed),
        }
        for g in groups
    ]


@dataclass(frozen=True)
class FinishRule:
    """How `af.chain.finish` improves a published layout without new starts: the trapped-point
    loop on that layout alone, with the options named here. Recorded verbatim with the map it
    makes, so a reader knows which rule a version got."""

    move_groups: bool = False  # af.map.optimise.resolve_trapped's group pass (DECISIONS Q61, Q78)
    move_lower: bool = False  # its move-lower rule (af.map.optimise.MOVE_LOWER_RULE; Q195)
    # which move-lower rule, by name (MOVE_LOWER_RULES); AD's last round is "ad-last-round-1" (Q198)
    move_lower_rule: str = DEFAULT_MOVE_LOWER_RULE

    def __post_init__(self) -> None:
        if not (self.move_groups or self.move_lower):
            raise ValueError("a finish rule must turn something on, or it would change nothing")
        if self.move_lower_rule not in MOVE_LOWER_RULES:
            raise ValueError(
                f"unknown move-lower rule {self.move_lower_rule!r}:"
                f" known {sorted(MOVE_LOWER_RULES)}"
            )
        if self.move_lower_rule != DEFAULT_MOVE_LOWER_RULE and not self.move_lower:
            raise ValueError(f"move-lower rule {self.move_lower_rule!r} without move_lower")

    def to_json(self) -> dict[str, Any]:
        out = {"loop": "af.map.optimise.resolve_trapped", **asdict(self)}
        del out["move_lower_rule"]  # the name is in the record below, which says what it does
        if self.move_lower:  # which rule, so a later rule is never taken for this one
            out["move_lower_rule"] = move_lower_rule(self.move_lower_rule)
        return out


class CoreOptimiser:
    """ae's chain maps: N starts (incremental: from the previous layout, rough, then the best 5
    fine), then the trapped-point loop on the best (ae `chart-relax-grid` /
    `chart-relax-incremental --grid-test`)."""

    name = "af.map.optimise (alglib CG) + resolve_trapped"
    key = "core"

    def __init__(self, threads: int = 0, arithmetic: str | None = None):
        """`arithmetic` is the word that asks for one of the build's two compiled cores
        (`af.map.optimise.NUMERICS_OFFERED`): the arithmetic under contract, the default as in
        `MapProblem`, or "original". A run names it in its config; nothing here guesses."""
        from af.map.optimise import NUMERICS, NUMERICS_OFFERED

        self.threads = threads
        self.arithmetic = NUMERICS if arithmetic is None else arithmetic
        if self.arithmetic not in NUMERICS_OFFERED:
            raise ValueError(
                f"unknown optimiser arithmetic {arithmetic!r}: this build offers {NUMERICS_OFFERED}"
            )

    @property
    def numerics(self) -> str | None:
        from af.map.optimise import ORIGINAL

        return None if self.arithmetic == ORIGINAL else self.arithmetic

    @property
    def refine_best(self) -> int:
        from af.map.optimise import INCREMENTAL_REFINED  # the one copy: refine()'s own default

        return INCREMENTAL_REFINED

    def move_lower_rule(self, name: str = DEFAULT_MOVE_LOWER_RULE) -> dict[str, Any]:
        return move_lower_rule(name)

    def _problem(self, arrays: dict) -> Any:
        from af.map.optimise import MapProblem

        return MapProblem(
            titre_value=arrays["titre_value"],
            titre_type=arrays["titre_type"],
            column_bases=np.asarray(arrays["column_bases"], dtype=float),
            disconnected=arrays["disconnected"],
            dodgy_is_regular=bool(arrays.get("dodgy_is_regular", False)),
            weights=arrays.get("weights"),
            avidity_adjust=arrays.get("avidity_adjust"),
            unmovable=arrays.get("unmovable"),
            numerics=self.arithmetic,
        )

    @staticmethod
    def _as_dict(p: Any, threads: int | None = None) -> MapResult:
        """`threads` is what the relax that made `p` ran on; with it go the CPUs this process
        may use, read here because a split chunk runs in its own SLURM task. None for maps
        that are re-read or refined, which keep what their relax recorded."""
        out: MapResult = {
            "layout": p.layout,
            "stress": p.stress,
            "dimensions": p.dimensions,
            "n_iterations": p.n_iterations,
            "start_seed": p.start_index,
            "rng_seed": p.start_seed,
            "termination": p.termination,
        }
        if threads is not None:
            from af.map.optimise import available_cpus

            out["threads"] = int(threads)
            out["cpus"] = available_cpus()
        return out

    @staticmethod
    def _as_projection(r: MapResult) -> Any:
        from af.map.optimise import Projection

        return Projection(
            layout=r["layout"],
            stress=r["stress"],
            dimensions=r["dimensions"],
            n_iterations=r["n_iterations"],
            start_seed=r["rng_seed"],
            start_index=r["start_seed"],
            termination=r["termination"],
        )

    def optimise(
        self,
        arrays,
        n_starts,
        dim,
        seed,
        start_layout,
        move_groups=False,
        resolve_best=1,
        move_lower=False,
        move_lower_rule=DEFAULT_MOVE_LOWER_RULE,
    ):
        from af.map.optimise import refine, relax

        problem = self._problem(arrays)
        rough = start_layout is None and _scratch_rough(arrays)
        result = relax(
            problem,
            n_starts=n_starts,
            seed=seed & (2**64 - 1),
            dimensions=dim,
            start_layout=start_layout,
            precision="rough" if rough else "fine",
            keep=None,
            threads=self.threads,
        )
        projections = result.projections
        if rough:  # as AD chart-relax-grid: every start rough, then the best few fine
            projections = refine(problem, projections, threads=self.threads)
        maps = [self._as_dict(p, result.threads) for p in projections]
        # an incremental relax refines its best few itself (af.map.optimise.relax)
        refined = rough or start_layout is not None
        return self._resolve(
            problem, maps, move_groups, resolve_best=resolve_best, refined=refined,
            move_lower=move_lower, move_lower_rule=move_lower_rule,
        )  # fmt: skip

    def relax_chunk(self, arrays, first_start, n_starts, dim, seed, start_layout):
        """One job's share of a map's starts. Incremental chunks, and scratch chunks relaxed
        rough (MapOptions.scratch_precision), stay rough: the fine stage must see the best
        starts of all chunks (see `combine`)."""
        from af.map.optimise import relax

        result = relax(
            self._problem(arrays),
            n_starts=n_starts,
            first_start=first_start,
            seed=seed & (2**64 - 1),
            dimensions=dim,
            start_layout=start_layout,
            precision="rough" if start_layout is not None or _scratch_rough(arrays) else "fine",
            keep=None,
            threads=self.threads,
        )
        return [self._as_dict(p, result.threads) for p in result.projections]

    def combine(
        self,
        arrays,
        chunks,
        incremental,
        move_groups=False,
        resolve_best=1,
        move_lower=False,
        move_lower_rule=DEFAULT_MOVE_LOWER_RULE,
    ):
        from af.map.optimise import refine, sort_projections

        problem = self._problem(arrays)
        projections = sort_projections([self._as_projection(r) for r in chunks])
        refined = incremental or _scratch_rough(arrays)
        if refined:
            projections = refine(problem, projections, threads=self.threads)
        # the relax threads of the chunks, whatever order the maps are in now
        relax = [{"threads": r["threads"], "cpus": r["cpus"]} for r in chunks if "threads" in r]
        maps = [self._as_dict(p) for p in projections]
        return self._resolve(
            problem, maps, move_groups, relax, resolve_best, refined, move_lower, move_lower_rule
        )

    def _resolve(
        self,
        problem: Any,
        maps: list[MapResult],
        move_groups: bool,
        relax: list[dict] | None = None,
        resolve_best: int = 1,
        refined: bool | None = None,
        move_lower: bool = False,
        move_lower_rule: str = DEFAULT_MOVE_LOWER_RULE,
    ) -> list[MapResult]:
        """The trapped-point loop on the best `resolve_best` maps (MapOptions.resolve_best); the
        lowest is kept. `relax` is the threads/CPUs of each split chunk's starts; unsplit, the
        maps carry their own. With more than one resolved, each map records the rank it had
        before the loop (`refined_rank`), so a step says which one it kept.

        Every map the loop ran on records the stress it had before (`stress_before_resolve`):
        the loop lowers only those maps, so their final stress is not comparable with the other
        starts', and without the earlier figure a list of start stresses reads as one population
        when it is not. For the same reason the kept map records how many of the maps (the
        lowest) are at fine precision, the rest being rough (`fine_starts`), and, when a refine
        stage made that so (`refined`), how many it refined (`refine_best`). `refined` None: the
        caller did not say, and neither is recorded.

        With `move_lower`, the move-lower rule `move_lower_rule` runs too (by name,
        MOVE_LOWER_RULES: AD's last round runs after every map's loop, as AD's grid test made
        it on every map of every step; 06, Q198), and the kept map records its
        rounds (`resolved_lower`), also when it found nothing to move."""
        from af.map.optimise import available_cpus, resolve_trapped

        if relax is None:
            relax = [{"threads": m["threads"], "cpus": m["cpus"]} for m in maps if "threads" in m]
        changed = False
        groups_of: dict[int, list[dict]] = {}
        lower_of: dict[int, dict] = {}
        threads = 0
        for rank in range(min(resolve_best, len(maps))):
            maps[rank] = {**maps[rank], "stress_before_resolve": maps[rank]["stress"]}
            fixed = resolve_trapped(
                problem,
                maps[rank]["layout"],
                threads=self.threads,
                move_groups=move_groups,
                move_lower=move_lower,
                move_lower_rule=_rule(move_lower_rule),
            )
            threads = int(fixed.threads)
            groups_of[rank] = _group_record(fixed.groups)
            if move_lower:
                lower_of[rank] = _move_lower_record(fixed)
            lowered = move_lower and any(r["kept"] for r in lower_of[rank]["rounds"])
            if fixed.moved or lowered or any(g["kept"] for g in groups_of[rank]):
                p = fixed.projection
                maps[rank] = {
                    **maps[rank],
                    "layout": p.layout,
                    "stress": p.stress,
                    "resolved_rounds": fixed.rounds,
                    "resolved_moves": fixed.moved,
                }
                changed = True
            if resolve_best > 1:
                maps[rank] = {**maps[rank], "refined_rank": rank}
        run_threads: dict[str, Any] = {
            "relax": _count_runs(relax),
            "trapped": {"threads": threads, "cpus": available_cpus()},
        }
        if resolve_best > 1:
            run_threads["trapped"]["maps"] = len(groups_of)
        if changed:
            maps.sort(key=map_order)
        best = min(range(len(maps)), key=lambda i: map_order(maps[i]))
        maps[best] = {**maps[best], "run_threads": run_threads}
        if refined is not None:
            fine = min(self.refine_best, len(maps)) if refined else len(maps)
            maps[best] = {**maps[best], "fine_starts": fine}
            if refined:
                maps[best] = {**maps[best], "refine_best": self.refine_best}
        if move_groups:  # recorded on the best map, also when no group was found
            groups = groups_of[maps[best].get("refined_rank", 0)]
            maps[best] = {**maps[best], "resolved_groups": groups}
        if move_lower:  # likewise
            lower = lower_of[maps[best].get("refined_rank", 0)]
            maps[best] = {**maps[best], "resolved_lower": lower}
        return maps

    def finish(self, arrays, layout, rule, recorded_stress):
        """The trapped-point loop on `layout` alone, under `rule`: no starts, no refine. The
        loop only ever lowers the stress, so the result is never worse than the layout given.

        Refused unless `layout`'s stress in `arrays` is `recorded_stress`, the stress its map
        recorded: otherwise the arrays are not the problem the layout was made for (a chain
        whose column bases were adjusted, read without the adjustments, measured 18x higher),
        and the loop would move points for a problem the map never had."""
        from af.map.optimise import resolve_trapped, stress

        problem = self._problem(arrays)
        before = stress(problem, layout)
        if not math.isclose(before, recorded_stress, rel_tol=FINISH_STRESS_TOLERANCE, abs_tol=0.0):
            raise ValueError(
                f"the layout's stress is {before!r} in these arrays, but its map recorded"
                f" {recorded_stress!r}: these are not the arrays it was made from"
            )
        fixed = resolve_trapped(
            problem,
            layout,
            threads=self.threads,
            move_groups=rule.move_groups,
            move_lower=rule.move_lower,
            move_lower_rule=_rule(rule.move_lower_rule),
        )
        p = fixed.projection
        return {
            "layout": p.layout,
            "stress": p.stress,
            "dimensions": p.dimensions,
            "n_iterations": p.n_iterations,
            "start_seed": 0,
            "finish": {
                "rule": rule.to_json(),
                "stress_before": before,
                "stress_after": p.stress,
                "rounds": fixed.rounds,
                "moves": fixed.moved,
                "groups": _group_record(fixed.groups),
                **({"move_lower": _move_lower_record(fixed)} if rule.move_lower else {}),
                "threads": int(fixed.threads),
            },
        }

    def grid_test(self, layout, arrays):
        from af.map.optimise import grid_test

        results = grid_test(self._problem(arrays), layout, threads=self.threads)
        return [
            {
                "point": int(g.point),
                "diagnosis": g.diagnosis,
                "distance": float(g.distance),
                "contribution_diff": float(g.stress_diff),
            }
            for g in results
            if g.diagnosis in ("trapped", "hemisphering")
        ]


def map_order(r: MapResult) -> tuple[bool, float, int]:
    """The core's order (`af.map.optimise.sort_projections`): stress ascending, NaN last, ties by
    start index. Used wherever maps are ranked after the trapped-point loop, so the map kept does
    not depend on the order the maps happened to arrive in."""
    return (math.isnan(r["stress"]), r["stress"], r["start_seed"])


def _scratch_rough(arrays: dict) -> bool:
    """A map from scratch relaxed rough, then its best 5 fine (MapOptions.scratch_precision).
    Absent means af's original method, every start fine."""
    return arrays.get("scratch_precision", "fine") == "rough"


def _count_runs(runs: list[dict]) -> list[dict]:
    """Distinct (threads, CPUs) among a map's starts, with how many starts ran on each: one
    entry unsplit, one per kind of SLURM node when split."""
    counts = Counter((r["threads"], r["cpus"]) for r in runs)
    return [{"threads": t, "cpus": c, "starts": n} for (t, c), n in sorted(counts.items())]


class StubOptimiser:
    """Numpy stand-in (af.chain.stub_optimiser): for tests and machines without the core."""

    name = "stub-numpy-gradient-descent"
    key = "stub"
    numerics: str | None = None  # numpy on whatever machine: it names no arithmetic
    refine_best: int | None = None  # every start is relaxed once

    def move_lower_rule(self, name: str = DEFAULT_MOVE_LOWER_RULE) -> dict[str, Any] | None:
        return None  # it has no trapped-point loop

    def optimise(
        self,
        arrays,
        n_starts,
        dim,
        seed,
        start_layout,
        move_groups=False,
        resolve_best=1,
        move_lower=False,
        move_lower_rule=DEFAULT_MOVE_LOWER_RULE,
    ):
        if move_lower:
            raise ValueError("the stand-in optimiser has no move-lower rule: that needs the core")
        return self.relax_chunk(arrays, 0, n_starts, dim, seed, start_layout)

    def relax_chunk(self, arrays, first_start, n_starts, dim, seed, start_layout):
        from af.chain import stub_optimiser

        return stub_optimiser.optimise(
            arrays,
            n_starts=n_starts,
            dim=dim,
            seed=seed,
            start_layout=start_layout,
            keep=n_starts,
            first_start=first_start,
        )

    def combine(
        self,
        arrays,
        chunks,
        incremental,
        move_groups=False,
        resolve_best=1,
        move_lower=False,
        move_lower_rule=DEFAULT_MOVE_LOWER_RULE,
    ):
        if move_lower:
            raise ValueError("the stand-in optimiser has no move-lower rule: that needs the core")
        return sorted(chunks, key=map_order)

    def finish(self, arrays, layout, rule, recorded_stress):
        raise ValueError("the stand-in optimiser cannot finish a map: finishing needs the core")

    def grid_test(self, layout, arrays):
        from af.chain import stub_optimiser

        return stub_optimiser.grid_test(layout, arrays)


BACKENDS = {"core": CoreOptimiser, "stub": StubOptimiser}


def optimiser_by_key(key: str, threads: int = 0, arithmetic: str | None = None) -> Optimiser:
    """`arithmetic` is the run config's word for the core's arithmetic (CoreOptimiser); the
    stand-in has none."""
    if key == "core":
        return CoreOptimiser(threads=threads, arithmetic=arithmetic)
    if key == "stub":
        return StubOptimiser()
    raise ValueError(f"unknown optimiser {key!r}; known: {sorted(BACKENDS)}")


def optimiser_for_problem(key: str, numerics: str | None, threads: int = 0) -> Optimiser:
    """The optimiser a chunk problem was written for: `numerics` is the arithmetic's name in the
    problem file, absent (None) for the core's original arithmetic and for the stand-in. Raises
    when this build cannot make it (the driver and the task run different builds)."""
    if key == "core":
        from af.map.optimise import ORIGINAL

        return CoreOptimiser(threads=threads, arithmetic=ORIGINAL if numerics is None else numerics)
    opt = optimiser_by_key(key, threads=threads)
    if opt.numerics != numerics:
        raise ValueError(
            f"the {key!r} optimiser has numerics {opt.numerics!r}, not the problem's {numerics!r}"
        )
    return opt


def default_optimiser() -> Optimiser:
    try:
        import af.map.optimise  # noqa: F401
    except ImportError:
        return StubOptimiser()
    return CoreOptimiser()
