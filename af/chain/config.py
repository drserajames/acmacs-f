"""Chain configuration: which tables, in what order, and how each step is mapped.

A chain is data, not code (today's chains are Python files that glob a directory). The
config is TOML read through `af.util.config`; the table list is resolved once into
(table id, path, date, suffix) and the pipeline hashes each table when it runs, which is
what makes restart-from-change work.

Example::

    name = "labx-h9-hi"
    seed = 1

    [tables]
    directory = "../tables/labx-h9-hi"
    group = "h9-hi-turkey-labx"
    date_from = "2020-04-21"
    exclude = []

    [options]
    scratch_starts = 1000
    incremental_starts = 1000
"""

from __future__ import annotations

import datetime
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from af.chart.merge import ColumnBasisConvention
from af.util.config import load_config


class ChainConfigError(ValueError):
    pass


@dataclass(frozen=True)
class MapOptions:
    """How every step's maps are made. A pipeline parameter: changing it remaps every step."""

    dimensions: int = 2
    minimum_column_basis: str = "none"
    scratch_starts: int = 1000
    incremental_starts: int = 1000
    projections_to_keep: int = 10  # distinct basins kept per map (Procrustes RMSD > 0.5)
    disconnect_threshold: int | None = 3  # fewer REGULAR titres than this -> disconnected
    combine_cheating_assays: bool = True
    column_bases: str = ColumnBasisConvention.ADJUST_TO_NEXT.value
    sd_limit: float = 1.0
    grid_test: bool = True
    code_version: int = 1  # bump to force every step to be remade after a code change

    def __post_init__(self) -> None:
        ColumnBasisConvention(self.column_bases)  # raises on an unknown convention


@dataclass(frozen=True)
class TableSelection:
    directory: Path
    group: str
    date_from: str | None = None  # ISO date, inclusive
    date_to: str | None = None
    exclude: list[str] = field(default_factory=list)  # table ids; each must match a table


@dataclass(frozen=True)
class ChainSettings:
    """The TOML schema."""

    name: str
    tables: TableSelection
    seed: int
    first_map: Path | None = None  # a seed chart with a projection (today's `first_source`)
    options: MapOptions = field(default_factory=MapOptions)


@dataclass(frozen=True)
class TableRef:
    table_id: str  # e.g. <group>-20260904 or <group>-20240617.2
    path: Path
    date: datetime.date
    suffix: int  # 1 for the first table of a date


@dataclass
class ChainConfig:
    """A resolved chain: settings plus the ordered table list."""

    name: str
    tables: list[TableRef]
    options: MapOptions = field(default_factory=MapOptions)
    seed: int = 0
    first_map: Path | None = None

    def __post_init__(self) -> None:
        if not self.tables:
            raise ChainConfigError(f"{self.name}: no tables")
        ids = [t.table_id for t in self.tables]
        if len(set(ids)) != len(ids):
            raise ChainConfigError(f"{self.name}: duplicate table ids")
        order = [(t.date, t.suffix) for t in self.tables]
        if order != sorted(order):
            raise ChainConfigError(f"{self.name}: tables are not in (date, suffix) order")


def load_chain_config(path: Path) -> ChainConfig:
    s = load_config(path, ChainSettings)
    t = s.tables
    tables = tables_from_directory(
        t.directory, t.group, _date(t.date_from), _date(t.date_to), set(t.exclude)
    )
    return ChainConfig(s.name, tables, s.options, s.seed, s.first_map)


def _date(text: str | None) -> datetime.date | None:
    return None if text is None else datetime.date.fromisoformat(text)


_TABLE_NAME = re.compile(r"^(?P<group>.+)-(?P<date>\d{8})(?:\.(?P<suffix>\d+))?\.ace$")


def tables_from_directory(
    directory: Path,
    group: str,
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
    exclude: set[str] = frozenset(),  # type: ignore[assignment]
) -> list[TableRef]:
    """A group's `.ace` tables, ordered by parsed (date, suffix); never by string or index.

    Every id in `exclude` must name a table of the group: an exclusion that matches
    nothing is a mistake in the config (design rule 1).
    """
    everything: list[TableRef] = []
    for path in Path(directory).glob(f"{group}-*.ace"):
        m = _TABLE_NAME.match(path.name)
        if not m or m["group"] != group:
            continue
        date = datetime.datetime.strptime(m["date"], "%Y%m%d").date()
        everything.append(TableRef(path.name[: -len(".ace")], path, date, int(m["suffix"] or 1)))
    unmatched = set(exclude) - {t.table_id for t in everything}
    if unmatched:
        raise ChainConfigError(f"exclusions match no table: {sorted(unmatched)}")
    chosen = [
        t
        for t in everything
        if t.table_id not in exclude
        and (date_from is None or t.date >= date_from)
        and (date_to is None or t.date <= date_to)
    ]
    if not chosen:
        raise ChainConfigError(f"no tables for {group} in {directory} ({date_from}..{date_to})")
    return sorted(chosen, key=lambda t: (t.date, t.suffix))


def config_to_json(cfg: ChainConfig) -> dict[str, Any]:
    return {
        "name": cfg.name,
        "seed": cfg.seed,
        "first_map": None if cfg.first_map is None else str(cfg.first_map),
        "options": asdict(cfg.options),
        "tables": [
            {
                "table_id": t.table_id,
                "path": str(t.path),
                "date": t.date.isoformat(),
                "suffix": t.suffix,
            }
            for t in cfg.tables
        ],
    }
