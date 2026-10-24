"""Convert a seed chart's point identities to the tables store's (a chain from a round's map).

A chain can start from a map made elsewhere (`first_map`, e.g. the map as shipped for a report)
and add the store's later tables one step at a time. Each step pairs points by the strict rule
(af.chart.identity), so the seed's names, passages and serum ids must be spelled as the store
spells them, or every later table brings its known points in again as new ones. The seed and
the store do not spell them alike: a lab code in front of serum ids, harvest dates in passages,
passage notation, place names with or without a hyphen (measured on the 18 report maps of the
Sep 2026 round, 24 Oct 2026; acmacs-f notes/chains).

The tables the seed already contains are the bridge. The seed keeps one titre layer per source
table, so each store table it shares is paired with its own layer, point by point:

1. sera by name and serum id, a configured lab code dropped as a first word on both sides,
   case and spaces ignored;
2. antigens by name, reassortant and passage without its date, each pairing confirmed by equal
   titres on the paired sera (at least one compared);
3. what is left by titres alone: the one candidate in the layer whose titres equal the store
   point's on at least two values; among identical rows, a name equal once case, spaces and
   punctuation are ignored decides. Antigens, then sera, then antigens again.

A paired seed point takes the store point's identity fields; nothing else in the chart changes
(layout, projections, titres, layers). Nothing is resolved silently: a seed point that is a
key candidate whose titres disagree, that pairs to two store identities, or whose new identity
would collide with another point's, keeps the seed's identity and is recorded, with every
unpaired seed point and its reason. The layer finds pairs; it is never merged again.
"""

from __future__ import annotations

import dataclasses
import datetime
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.chain.config import TableRef
from af.chart.ace import read_chart
from af.chart.identity import antigen_identity, serum_identity
from af.chart.model import Antigen, Chart, Layer, Serum

_PASSAGE_DATE = re.compile(r"\s*\(\d{4}-\d{2}-\d{2}\)\s*$")

# how a pairing was made
KEY = "key"
TITRES = "titres"
TITRES_NAME = "titres+name"


class BridgeError(ValueError):
    pass


# ----------------------------------------------------------------------
# which store tables the seed contains


@dataclass(frozen=True)
class SeedSplit:
    """The chain's tables against the seed's source list.

    `in_seed` pairs each table with the seed layers of its date (the bridge picks among them);
    `af_only` are dated on or before the seed's last source but absent from it (the store has
    them, the seed lacks them); `post_seed` come after the seed's last source.
    """

    in_seed: list[tuple[TableRef, list[int]]]
    af_only: list[TableRef]
    post_seed: list[TableRef]

    def counts(self) -> dict[str, int]:
        return {
            "in_seed": len(self.in_seed),
            "af_only": len(self.af_only),
            "post_seed": len(self.post_seed),
        }


def seed_source_dates(chart: Chart) -> list[datetime.date]:
    """One date per source table (`i.S`), in layer order; a seed without them cannot be bridged."""
    sources = chart.info.get("S") or []
    if not sources:
        raise BridgeError("seed chart has no source list (info S): cannot tell its tables")
    if len(sources) != len(chart.titres.layers):
        raise BridgeError(
            f"seed chart has {len(sources)} sources but {len(chart.titres.layers)} titre layers"
        )
    dates = []
    for n, s in enumerate(sources):
        digits = re.sub(r"\D", "", str(s.get("D", "")))[:8]
        try:
            dates.append(datetime.datetime.strptime(digits, "%Y%m%d").date())
        except ValueError:
            raise BridgeError(f"seed source {n} has no readable date: {s.get('D')!r}") from None
    return dates


def seed_split(seed_dates: Sequence[datetime.date], tables: Sequence[TableRef]) -> SeedSplit:
    """A table is in the seed if its date occurs there at least as often as it has occurred so
    far among the chain's tables (two tables of one date need two sources of that date)."""
    if not seed_dates:
        raise BridgeError("seed has no source dates")
    last = max(seed_dates)
    available = Counter(seed_dates)
    layers_of: dict[datetime.date, list[int]] = defaultdict(list)
    for i, d in enumerate(seed_dates):
        layers_of[d].append(i)
    seen: Counter[datetime.date] = Counter()
    in_seed, af_only, post_seed = [], [], []
    for t in sorted(tables, key=lambda x: (x.date, x.suffix)):
        if t.date > last:
            post_seed.append(t)
            continue
        seen[t.date] += 1
        if seen[t.date] <= available[t.date]:
            in_seed.append((t, list(layers_of[t.date])))
        else:
            af_only.append(t)
    return SeedSplit(in_seed, af_only, post_seed)


# ----------------------------------------------------------------------
# pairing one store table with one seed layer


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.upper())


def _loose_name(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def _passage(p: str) -> str:
    return _norm(_PASSAGE_DATE.sub("", p))


def _serum_id(serum_id: str, prefix: str | None) -> str:
    parts = serum_id.split(" ", 1)
    if prefix and len(parts) == 2 and parts[0].upper() == prefix.upper():
        serum_id = parts[1]
    return _norm(serum_id)


def _cells(layer: Layer) -> dict[tuple[int, int], str]:
    return {k: str(v) for k, v in layer.items() if not v.is_missing}


def _table_cells(chart: Chart) -> dict[tuple[int, int], str]:
    return {
        (i, j): str(t)
        for i, row in enumerate(chart.titres.table)
        for j, t in enumerate(row)
        if not t.is_missing
    }


@dataclass
class _TablePairs:
    antigens: dict[int, tuple[int, str]] = field(default_factory=dict)  # table index -> (seed, how)
    sera: dict[int, tuple[int, str]] = field(default_factory=dict)
    rejected: list[tuple[int, list[int]]] = field(default_factory=list)  # antigen, seed candidates
    ambiguous: list[tuple[str, int, list[int]]] = field(default_factory=list)  # kind, index, seed

    @property
    def size(self) -> int:
        return len(self.antigens) + len(self.sera)


def _pair_table(table: Chart, seed: Chart, layer: Layer, prefix: str | None) -> _TablePairs:
    t_cells = _table_cells(table)
    s_cells = _cells(layer)
    layer_ag = sorted({a for a, _ in s_cells})
    layer_sr = sorted({s for _, s in s_cells})
    out = _TablePairs()

    def compare_rows(i: int, k: int) -> tuple[int, int]:
        n = agree = 0
        for j, (kk, _) in out.sera.items():
            a, b = t_cells.get((i, j)), s_cells.get((k, kk))
            if a and b:
                n += 1
                agree += a == b
        return n, agree

    def compare_columns(j: int, kk: int) -> tuple[int, int]:
        n = agree = 0
        for i, (k, _) in out.antigens.items():
            a, b = t_cells.get((i, j)), s_cells.get((k, kk))
            if a and b:
                n += 1
                agree += a == b
        return n, agree

    # 1. sera by name and serum id
    for j, s in enumerate(table.sera):
        sid = _serum_id(s.serum_id, prefix)
        if not sid:
            continue
        c = [
            k
            for k in layer_sr
            if seed.sera[k].name == s.name and _serum_id(seed.sera[k].serum_id, prefix) == sid
        ]
        if len(c) == 1:
            out.sera[j] = (c[0], KEY)
        elif len(c) > 1:
            out.ambiguous.append(("serum", j, c))
    # 2. antigens by key, confirmed by titres
    for i, a in enumerate(table.antigens):
        c = [
            k
            for k in layer_ag
            if seed.antigens[k].name == a.name
            and seed.antigens[k].reassortant == a.reassortant
            and _passage(seed.antigens[k].passage) == _passage(a.passage)
        ]
        if not c:
            continue
        compared = [(k, *compare_rows(i, k)) for k in c]
        confirmed = [k for k, n, agree in compared if n >= 1 and agree == n]
        if len(confirmed) == 1:
            out.antigens[i] = (confirmed[0], KEY)
        elif len(confirmed) > 1:
            out.ambiguous.append(("antigen", i, confirmed))
        else:
            disagree = [k for k, n, agree in compared if n >= 1 and agree < n]
            if disagree:  # compared and found different: never paired by the key alone
                out.rejected.append((i, disagree))

    # 3. titres alone
    def by_titres_antigens() -> None:
        used = {k for k, _ in out.antigens.values()}
        for i, a in enumerate(table.antigens):
            if i in out.antigens:
                continue
            hits = [k for k in layer_ag if k not in used and _equal(compare_rows(i, k))]
            how = TITRES
            if len(hits) > 1:  # identical rows: a name spelled alike decides
                hits = [
                    k for k in hits if _loose_name(seed.antigens[k].name) == _loose_name(a.name)
                ]
                how = TITRES_NAME
            if len(hits) == 1:
                out.antigens[i] = (hits[0], how)
                used.add(hits[0])

    def by_titres_sera() -> None:
        used = {k for k, _ in out.sera.values()}
        for j, s in enumerate(table.sera):
            if j in out.sera:
                continue
            hits = [kk for kk in layer_sr if kk not in used and _equal(compare_columns(j, kk))]
            how = TITRES
            if len(hits) > 1:
                hits = [kk for kk in hits if _loose_name(seed.sera[kk].name) == _loose_name(s.name)]
                how = TITRES_NAME
            if len(hits) == 1:
                out.sera[j] = (hits[0], how)
                used.add(hits[0])

    by_titres_antigens()
    by_titres_sera()
    by_titres_antigens()
    return out


def _equal(compared: tuple[int, int]) -> bool:
    n, agree = compared
    return n >= 2 and agree == n


# ----------------------------------------------------------------------
# the bridge


def _ag_fields(a: Antigen) -> dict[str, Any]:
    return {
        "name": a.name,
        "reassortant": a.reassortant,
        "annotations": list(a.annotations),
        "passage": a.passage,
    }


def _sr_fields(s: Serum) -> dict[str, Any]:
    return {
        "name": s.name,
        "reassortant": s.reassortant,
        "annotations": list(s.annotations),
        "serum_id": s.serum_id,
    }


def _key(kind: str, f: dict[str, Any]) -> tuple[Any, ...] | None:
    if kind == "antigen":
        return antigen_identity(f["name"], f["reassortant"], f["annotations"], f["passage"])
    return serum_identity(f["name"], f["reassortant"], f["annotations"], f["serum_id"])


@dataclass
class Bridge:
    """The converted seed and every decision that made it (`to_json` is bridge.json)."""

    chart: Chart
    tables: list[dict[str, Any]]
    converted: list[dict[str, Any]]
    unpaired: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    split: dict[str, Any]
    drop_serum_id_prefix: str | None

    def counts(self) -> dict[str, Any]:
        def per(kind: str, items: list[dict[str, Any]], what: str) -> dict[str, int]:
            return dict(sorted(Counter(x[what] for x in items if x["kind"] == kind).items()))

        out: dict[str, Any] = {"tables": self.split["counts"]}
        out["layer_swaps"] = sum(1 for t in self.tables if t["swapped"])
        for kind, n in (("antigen", self.chart.n_antigens), ("serum", self.chart.n_sera)):
            conv = [x for x in self.converted if x["kind"] == kind]
            out[kind] = {
                "seed_points": n,
                "converted": len(conv),
                "spelling_changed": sum(1 for x in conv if x["seed"] != x["store"]),
                "how": dict(sorted(Counter(x["how"] for x in conv).items())),
                "unpaired": per(kind, self.unpaired, "reason"),
            }
        out["rejected_events"] = len(self.rejected)
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "drop_serum_id_prefix": self.drop_serum_id_prefix,
            "split": self.split,
            "tables": self.tables,
            "counts": self.counts(),
            "converted": self.converted,
            "unpaired": self.unpaired,
            "rejected": self.rejected,
        }


def bridge_seed(
    seed: Chart,
    split: SeedSplit,
    drop_serum_id_prefix: str | None = None,
    read: Callable[[Path], Chart] | None = None,
) -> Bridge:
    """Pair the seed with every store table it contains and convert what pairs (module doc)."""
    reader = read_chart if read is None else read
    layers = seed.titres.layers
    used_layers: set[int] = set()
    tables: list[dict[str, Any]] = []
    # seed point -> store identities it paired with, and how
    found: dict[tuple[str, int], dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
    in_layers: set[tuple[str, int]] = set()
    rejected_at: dict[tuple[str, int], list[str]] = defaultdict(list)
    ambiguous_at: set[tuple[str, int]] = set()
    rejected: list[dict[str, Any]] = []
    for ref, candidates in split.in_seed:
        free = [j for j in candidates if j not in used_layers]
        if not free:
            raise BridgeError(f"{ref.table_id}: no seed layer of its date left")
        table = reader(ref.path)
        best: tuple[int, _TablePairs] | None = None
        for j in free:
            pairs = _pair_table(table, seed, layers[j], drop_serum_id_prefix)
            if best is None or pairs.size > best[1].size:
                best = (j, pairs)
        assert best is not None
        j, pairs = best
        used_layers.add(j)
        tables.append({"table_id": ref.table_id, "layer": j, "swapped": j != free[0]})
        cells = _cells(layers[j])
        in_layers.update(("antigen", a) for a, _ in cells)
        in_layers.update(("serum", s) for _, s in cells)
        for i, (k, how) in pairs.antigens.items():
            f = _ag_fields(table.antigens[i])
            entry = found[("antigen", k)].setdefault(_freeze(f), {"fields": f, "how": Counter()})
            entry["how"][how] += 1
            entry.setdefault("tables", []).append(ref.table_id)
        for jj, (k, how) in pairs.sera.items():
            f = _sr_fields(table.sera[jj])
            entry = found[("serum", k)].setdefault(_freeze(f), {"fields": f, "how": Counter()})
            entry["how"][how] += 1
            entry.setdefault("tables", []).append(ref.table_id)
        for i, ks in pairs.rejected:
            rejected.append(
                {
                    "table_id": ref.table_id,
                    "store": _ag_fields(table.antigens[i]),
                    "seed_candidates": ks,
                }
            )
            for k in ks:
                rejected_at[("antigen", k)].append(ref.table_id)
        for kind, _, ks in pairs.ambiguous:
            ambiguous_at.update((kind, k) for k in ks)

    antigens = list(seed.antigens)
    sera = list(seed.sera)
    converted: dict[tuple[str, int], dict[str, Any]] = {}
    unpaired: list[dict[str, Any]] = []
    for kind, points, fields_of in (
        ("antigen", seed.antigens, _ag_fields),
        ("serum", seed.sera, _sr_fields),
    ):
        for k, p in enumerate(points):
            where = (kind, k)
            seed_f = fields_of(p)  # type: ignore[arg-type]
            got = found.get(where, {})
            if where in rejected_at:
                unpaired.append(
                    _unpaired(kind, k, seed_f, "titre-rejected", rejected_at[where], got)
                )
            elif len(got) > 1:
                unpaired.append(_unpaired(kind, k, seed_f, "conflicting", [], got))
            elif len(got) == 1:
                (e,) = got.values()
                converted[where] = {
                    "kind": kind,
                    "index": k,
                    "seed": seed_f,
                    "store": e["fields"],
                    "how": _how(e["how"]),
                    "tables": len(e["tables"]),
                }
            elif where in ambiguous_at:
                unpaired.append(_unpaired(kind, k, seed_f, "ambiguous", [], got))
            elif where in in_layers:
                unpaired.append(_unpaired(kind, k, seed_f, "no key or titre match", [], got))
            else:
                unpaired.append(_unpaired(kind, k, seed_f, "not in any shared layer", [], got))

    # a new identity must not collide with another point's: revert both, never pick one
    while True:
        keys: dict[tuple[str, tuple[Any, ...]], list[tuple[str, int]]] = defaultdict(list)
        for kind, points, fields_of in (
            ("antigen", seed.antigens, _ag_fields),
            ("serum", seed.sera, _sr_fields),
        ):
            for k, p in enumerate(points):
                f = converted[(kind, k)]["store"] if (kind, k) in converted else fields_of(p)  # type: ignore[arg-type]
                key = _key(kind, f)
                if key is not None:
                    keys[(kind, key)].append((kind, k))
        clash = {w for ws in keys.values() if len(ws) > 1 for w in ws if w in converted}
        if not clash:
            break
        for where in sorted(clash):
            c = converted.pop(where)
            unpaired.append(
                {
                    "kind": c["kind"],
                    "index": c["index"],
                    "seed": c["seed"],
                    "reason": "conflicting",
                    "store_candidates": [c["store"]],
                    "note": "the store identity would collide with another point's",
                }
            )

    for (kind, k), c in converted.items():
        f = c["store"]
        if kind == "antigen":
            antigens[k] = dataclasses.replace(
                antigens[k],
                name=f["name"],
                reassortant=f["reassortant"],
                annotations=tuple(f["annotations"]),
                passage=f["passage"],
            )
        else:
            sera[k] = dataclasses.replace(
                sera[k],
                name=f["name"],
                reassortant=f["reassortant"],
                annotations=tuple(f["annotations"]),
                serum_id=f["serum_id"],
            )
    chart = dataclasses.replace(seed, antigens=antigens, sera=sera)
    split_json = {
        "counts": split.counts(),
        "in_seed": [t.table_id for t, _ in split.in_seed],
        "af_only": [t.table_id for t in split.af_only],
        "post_seed": [t.table_id for t in split.post_seed],
    }
    return Bridge(
        chart,
        tables,
        [converted[w] for w in sorted(converted)],
        sorted(unpaired, key=lambda x: (x["kind"], x["index"])),
        rejected,
        split_json,
        drop_serum_id_prefix,
    )


def _freeze(f: dict[str, Any]) -> tuple[Any, ...]:
    return tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in sorted(f.items()))


def _how(c: Counter[str]) -> str:
    """The strongest way the point was paired in any table: a key beats titres alone."""
    for how in (KEY, TITRES, TITRES_NAME):
        if c.get(how):
            return how
    raise BridgeError("pairing without a method")


def _unpaired(
    kind: str,
    k: int,
    seed_f: dict[str, Any],
    reason: str,
    tables: list[str],
    got: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": kind, "index": k, "seed": seed_f, "reason": reason}
    if tables:
        out["tables"] = sorted(set(tables))
    if got:
        out["store_candidates"] = [e["fields"] for e in got.values()]
    return out


# ----------------------------------------------------------------------
# the declaration


def declaration_fragment(
    dataset: str, bridge: dict[str, Any], *, ruling: str, version: str, file: str
) -> str:
    """A `[[declaration]]` for the round's declarations file (af.report.howmade): counts only."""
    c = bridge["counts"]

    def side(kind: str, label: str) -> str:
        k = c[kind]
        how = ", ".join(f"{n} by {h}" for h, n in k["how"].items()) or "none"
        un = ", ".join(f"{n} {r}" for r, n in k["unpaired"].items()) or "none"
        return (
            f"{label}: {k['converted']} of {k['seed_points']} converted to the store's identity"
            f" ({how}; {k['spelling_changed']} spelled differently), unpaired: {un}"
        )

    t = c["tables"]
    text = (
        f"The chain starts from a seed map whose point identities were converted to the tables"
        f" store's through the {t['in_seed']} tables they share. {side('antigen', 'Antigens')}."
        f" {side('serum', 'Sera')}. Titre-rejected pairings: {c['rejected_events']}. Tables the"
        f" store has but the seed lacks, dated within it: {t['af_only']}."
    )
    return (
        "[[declaration]]\n"
        f"dataset = {_toml_str(dataset)}\n"
        f"ruling = {_toml_str(ruling)}\n"
        f"version = {_toml_str(version)}\n"
        f"file = {_toml_str(file)}\n"
        f"text = {_toml_str(text)}\n"
    )


def _toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
