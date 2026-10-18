"""A monthly chain that keeps its published order: `[tables] order = "append-to-published"`.

Sarah, Q164 (13 Oct 2026), "cut-off only": a table that arrives after a month's cut-off joins the
chain at its END, whatever its test date, and is recorded as late; earlier months' maps never
change. So the chain's published CURRENT version fixes the order of every table it holds (the
prefix), and the tables selected today that it does not hold follow, in date order, as this
month's arrivals. One that is dated before the prefix's latest table is LATE. "Arrival order" is
thereby the run a table first appeared in; no arrival time is needed. Every prefix step is reused
untouched (a step is keyed by the one before it), so a month costs one step per new table.

"Earlier months' maps never change" is checked exactly, on what keys a step (03, 18 Oct 2026):
every prefix step's table chart (the `table_sha256` its step.json records) and its step
parameters must be today's. Anything else would silently remake the published chain from the
first step that differs. So the run STOPS (Sarah, Q224 a: "if the table is recent then likely will
want to re-run") when:

- a prefix table was REVISED: its chart changed. The refusal says what changed, field by field
  in the terms of the tables store's map content (`Table.map_content`, what its map hash covers),
  its position from the chain's end and how many steps a re-run from it makes. A metadata-only
  tables version (03's EPI_ISL ids, Q221) leaves the chart alone and stops nothing;
- a prefix table is no longer selected: retired by the tables store (with or without a successor
  under the same source key), or left out by the chain file (exclude, date_from, date_to);
- the step parameters changed (`check_settings`, when the steps are built): options, rules, or a
  `[[select.remove_by_sequence]]` decision moved by a metadata version, a pin or a new table.

Each is acknowledged per case in the chain file, never automatically, and an acknowledgement that
matches nothing pending is an error (design rule 1):

    [[tables.rerun_from]]               # a revised table: re-run from it, where it stands
    table = "<table id>"
    table_sha256 = "<its new chart hash, as the refusal printed it>"
    decided = "Sarah, <date>: <why>"

    [[tables.rerun_from]]               # a table no longer selected: re-run from where it stood
    table = "<table id>"
    dropped = "retired"                 # or "excluded", as the refusal classed it
    decided = "..."

    [[tables.rerun_settings]]           # changed step parameters: re-run the chain from step 0
    keys = ["select_remove_by_sequence"]
    decided = "..."
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ORDERS = ("date", "append-to-published")
DROPPED = ("retired", "excluded")


class OrderError(ValueError):
    pass


@dataclass(frozen=True)
class Rerun:
    """`[[tables.rerun_from]]`: a revised table re-run where it stands (`table_sha256`), or a
    table no longer selected left out and the chain re-run from where it stood (`dropped`)."""

    table: str
    decided: str
    table_sha256: str = ""
    dropped: str = ""

    def __post_init__(self) -> None:
        if not (self.table.strip() and self.decided.strip()):
            raise OrderError("tables.rerun_from needs table and decided")
        if bool(self.table_sha256) == bool(self.dropped):
            raise OrderError(
                f"tables.rerun_from {self.table}: exactly one of table_sha256, dropped"
            )
        if self.dropped and self.dropped not in DROPPED:
            raise OrderError(f"tables.rerun_from {self.table}: dropped must be one of {DROPPED}")


@dataclass(frozen=True)
class SettingsRerun:
    """`[[tables.rerun_settings]]`: changed step parameters accepted; the chain re-runs."""

    keys: list[str]
    decided: str

    def __post_init__(self) -> None:
        if not self.keys or not self.decided.strip():
            raise OrderError("tables.rerun_settings needs keys and decided")


@dataclass(frozen=True)
class Published:
    """The chain's published CURRENT version as the order needs it."""

    version: str
    tables: Sequence[tuple[str, str]]  # (table id, table_sha256 its step recorded), in order
    directory: Path | None = None  # the version's directory, for its step parameters


@dataclass(frozen=True)
class Revision:
    table_id: str
    position: int  # in the published order, from 0
    from_end: int  # 0 = the published chain's last table
    rerun_steps: int  # steps a re-run from it makes: it, the rest of the prefix, and the new tables
    new_sha256: str
    changes: Mapping[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        what = ", ".join(f"{k} {v}" for k, v in self.changes.items()) or "the chart"
        return (
            f"  {self.table_id}: position {self.position} ({self.from_end} from the end); a"
            f" re-run from it makes {self.rerun_steps} steps; changed: {what}"
        )

    def acknowledgement(self) -> str:
        return (
            f'  {{ table = "{self.table_id}", table_sha256 = "{self.new_sha256}",'
            ' decided = "<who, date: why>" },'
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "position": self.position,
            "from_end": self.from_end,
            "rerun_steps": self.rerun_steps,
            "new_table_sha256": self.new_sha256,
            "changes": dict(self.changes),
        }


class RevisedTablesError(OrderError):
    """A prefix table changed: the monthly run stops and shows it (Q224 a)."""

    def __init__(self, dataset: str, version: str, revisions: Sequence[Revision]) -> None:
        self.revisions = tuple(revisions)
        first = min(revisions, key=lambda r: r.position)
        super().__init__(
            f"{dataset}: {len(revisions)} table(s) of the published chain {version} were revised"
            " since it was built; nothing was run. Each re-run is a decision (Q224). Re-running"
            f" makes {first.rerun_steps} steps, from the earliest, {first.table_id}:\n"
            + "\n".join(r.describe() for r in revisions)
            + "\nTo re-run, every revised table is acknowledged under [tables]:\nrerun_from = [\n"
            + "\n".join(r.acknowledgement() for r in revisions)
            + "\n]"
        )


class DroppedTablesError(OrderError):
    """A prefix table is no longer selected: the published order cannot be kept as it is."""

    def __init__(self, dataset: str, version: str, dropped: Sequence[dict[str, Any]]) -> None:
        self.dropped = tuple(dropped)
        lines = [
            f"  {d['table_id']}: position {d['position']} ({d['from_end']} from the end);"
            f" {d['why']}"
            for d in dropped
        ]
        acks = [
            f'  {{ table = "{d["table_id"]}", dropped = "{d["kind"]}",'
            ' decided = "<who, date: why>" },'
            for d in dropped
        ]
        super().__init__(
            f"{dataset}: {len(dropped)} table(s) of the published chain {version} are no longer"
            " selected; nothing was run:\n"
            + "\n".join(lines)
            + "\nTo leave them out and re-run from where they stood, acknowledge under [tables]:"
            "\nrerun_from = [\n" + "\n".join(acks) + "\n]"
        )


class SettingsChangedError(OrderError):
    """The steps' parameters are not the published chain's: every step would be remade."""


@dataclass(frozen=True)
class Order:
    """The chain's table order today, and what it says about each table."""

    tables: list[Any]  # TableRefs
    late: dict[str, dict[str, Any]]  # table id -> its lateness record
    reruns: tuple[dict[str, Any], ...] = ()  # acknowledged revisions and drops, as recorded
    prefix: int = 0  # tables kept from the published order

    def to_json(self, published: Published | None) -> dict[str, Any]:
        return {
            "order": "append-to-published",
            "published_version": None if published is None else published.version,
            "prefix": self.prefix,
            "late_tables": self.late,
            "reruns": list(self.reruns),
        }


def append_order(
    dataset: str,
    published: Published | None,
    today: Sequence[Any],
    today_sha256: Mapping[str, str],
    reruns: Sequence[Rerun],
    changes_of: Callable[[str], Mapping[str, Any]],
    why_dropped: Callable[[str], tuple[str, str]],
) -> Order:
    """The published order, then today's new tables in date order (see the module docstring).
    `today` are the tables selected today (TableRefs in date order), `today_sha256` their charts'
    hashes, `changes_of(table_id)` what changed in a revised table, `why_dropped(table_id)` the
    kind ("retired" or "excluded") and the reason a published table is not selected."""
    if published is None:  # the first version: nothing to keep, the order is the dates'
        if reruns:
            raise OrderError(f"{dataset}: tables.rerun_from with no published version")
        return Order(list(today), {})
    by_id = {t.table_id: t for t in today}
    acked = {r.table: r for r in reruns}
    n = len(published.tables)
    held = {tid for tid, _ in published.tables}
    tail = [t for t in today if t.table_id not in held]
    dropped, revisions, done, used = [], [], [], set()
    for i, (tid, sha) in enumerate(published.tables):
        if tid not in by_id:
            kind, why = why_dropped(tid)
            r = acked.get(tid)
            if r is not None and r.dropped == kind:
                used.add(tid)
                done.append({"table_id": tid, "position": i, "dropped": kind, "why": why,
                             "decided": r.decided})  # fmt: skip
            else:
                dropped.append({"table_id": tid, "position": i, "from_end": n - 1 - i,
                                "kind": kind, "why": why})  # fmt: skip
        elif today_sha256[tid] != sha:
            rev = Revision(tid, i, n - 1 - i, n - i + len(tail), today_sha256[tid], changes_of(tid))
            r = acked.get(tid)
            if r is not None and r.table_sha256 == rev.new_sha256:
                used.add(tid)
                done.append({**rev.to_json(), "decided": r.decided})
            else:
                revisions.append(rev)
    unknown = sorted(set(acked) - used)
    if unknown:
        raise OrderError(
            f"{dataset}: tables.rerun_from names {', '.join(unknown)}, but nothing such is pending"
            " (a revision to that chart hash, or a drop of that kind): an acknowledgement that"
            " matches nothing is an error"
        )
    if dropped:
        raise DroppedTablesError(dataset, published.version, dropped)
    if revisions:
        raise RevisedTablesError(dataset, published.version, revisions)
    prefix = [by_id[tid] for tid, _ in published.tables if tid in by_id]
    latest = max((t.date, t.suffix) for t in prefix) if prefix else None
    late = {
        t.table_id: {
            "date": t.date.isoformat(),
            "published_until": datetime.date.isoformat(latest[0]),
            "published_version": published.version,
        }
        for t in tail
        if latest is not None and (t.date, t.suffix) < latest
    }
    return Order([*prefix, *tail], late, tuple(done), len(prefix))


def check_settings(
    dataset: str,
    published: Published,
    table_ids: Sequence[str],
    parameters: Sequence[Mapping[str, Any]],
    accepted: Sequence[SettingsRerun],
) -> None:
    """Every prefix step's parameters (canonical, as its build recorded them) must be today's,
    unless the keys that changed are accepted (`[[tables.rerun_settings]]`). Compared while the
    tables stand where they stood: after a dropped table the steps re-run by decision anyway."""
    import json

    assert published.directory is not None
    keys_ok = {k for a in accepted for k in a.keys}
    changed: dict[str, tuple[Any, Any]] = {}
    for k, (tid, _) in enumerate(published.tables):
        if k >= len(table_ids) or table_ids[k] != tid:
            break
        built = json.loads(
            (published.directory / f"steps/{k:04d}/chosen.ace.provenance.json").read_text()
        )["parameters"]
        now = parameters[k]
        for key in sorted(set(built) | set(now)):
            if built.get(key) != now.get(key) and key not in changed:
                changed[key] = (built.get(key), now.get(key))
    unused = sorted(keys_ok - set(changed))
    if unused:
        raise OrderError(
            f"{dataset}: tables.rerun_settings accepts {', '.join(unused)}, which did not change"
        )
    pending = {k: v for k, v in changed.items() if k not in keys_ok}
    if pending:
        detail = [_setting_detail(k, *v) for k, v in pending.items()]
        keys = ", ".join(f'"{k}"' for k in sorted(pending))
        raise SettingsChangedError(
            f"{dataset}: the step parameters are not those of the published chain"
            f" {published.version}, so every step from the first would be remade; nothing was"
            " run:\n" + "\n".join(detail) + "\nTo re-run the chain, accept them under [tables]:"
            f'\nrerun_settings = [ {{ keys = [{keys}], decided = "<who, date: why>" }} ]'
        )


def _setting_detail(key: str, was: Any, now: Any) -> str:
    """One changed parameter, in words; for the sequence rule, the decisions that moved."""
    if key == "select_remove_by_sequence" and isinstance(was, dict) and isinstance(now, dict):
        a = {tuple(r) for r in was.get("removed", [])}
        b = {tuple(r) for r in now.get("removed", [])}
        moved = [f"now removed: {' | '.join(r)}" for r in sorted(b - a)]
        moved += [f"now kept: {' | '.join(r)}" for r in sorted(a - b)]
        rules = "" if was.get("rules") == now.get("rules") else " (and the rules changed)"
        return f"  {key}: {len(moved)} decision(s) moved{rules}" + "".join(
            f"\n    {m}" for m in moved
        )
    return f"  {key}: {_short(was)} -> {_short(now)}"


def _short(v: Any) -> str:
    text = repr(v)
    return text if len(text) <= 120 else text[:117] + "..."


def table_changes(old: Any, new: Any) -> dict[str, int]:
    """What changed between two versions of one table (af.tables.model.Table), counted for the
    refusal, in the terms of `Table.map_content`: per field, how many points both hold changed
    it (a serum's species, which the ferret-only policy reads, is one); points added or removed;
    cells among the points both hold; other table fields.

    Points pair by identity and occurrence (the n-th point with one key pairs with the n-th:
    repeats in a test, as af.chain.tables marks them DISTINCT), then any left over by row order,
    so a corrected passage or name reads as that field, not as one point removed and one added."""
    a, b = old.map_content(), new.map_content()
    out: dict[str, int] = {}
    matched: dict[str, list[tuple[int, int]]] = {}
    for what, key in (("antigens", _antigen_key), ("sera", _serum_key)):
        pairs = _pair(a[what], b[what], key)
        matched[what] = pairs
        if added := len(b[what]) - len(pairs):
            out[f"{what} added"] = added
        if removed := len(a[what]) - len(pairs):
            out[f"{what} removed"] = removed
        fields: dict[str, int] = {}
        for i, j in pairs:
            x, y = a[what][i], b[what][j]
            for f in sorted(set(x) | set(y)):
                if x.get(f) != y.get(f):
                    fields[f] = fields.get(f, 0) + 1
        for f, n in fields.items():
            out[f"{what} {f}"] = n
    cells = sum(
        a["titres"][i][j] != b["titres"][k][m]
        for i, k in matched["antigens"]
        for j, m in matched["sera"]
    )  # fmt: skip
    if cells:
        out["cells changed"] = cells
    for f in sorted((set(a) | set(b)) - {"antigens", "sera", "titres"}):
        if a.get(f) != b.get(f):
            out[f"table {f}"] = 1
    return out


def _pair(
    a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]], key: Callable[[Any], tuple]
) -> list[tuple[int, int]]:
    def occurrences(points: Sequence[Mapping[str, Any]]) -> dict[tuple, int]:
        seen: dict[tuple, int] = {}
        out = {}
        for i, p in enumerate(points):
            k = key(p)
            out[(k, seen.get(k, 0))] = i
            seen[k] = seen.get(k, 0) + 1
        return out

    ka, kb = occurrences(a), occurrences(b)
    pairs = [(ka[k], kb[k]) for k in ka.keys() & kb.keys()]
    left_a = sorted(set(range(len(a))) - {i for i, _ in pairs})
    left_b = sorted(set(range(len(b))) - {j for _, j in pairs})
    pairs += list(zip(left_a, left_b, strict=False))  # leftovers by row order
    return sorted(pairs)


def _antigen_key(p: Mapping[str, Any]) -> tuple:
    return (p.get("name"), p.get("reassortant"), tuple(p.get("annotations") or ()),
            p.get("passage"), p.get("passage_date"))  # fmt: skip


def _serum_key(p: Mapping[str, Any]) -> tuple:
    return (p.get("name"), p.get("reassortant"), tuple(p.get("annotations") or ()),
            p.get("serum_id"), p.get("passage"), p.get("passage_date"))  # fmt: skip
