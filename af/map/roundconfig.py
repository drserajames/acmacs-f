"""Read a round's ``maps.toml`` into the schema, and the curated vaccine list it names.

Kept apart from :mod:`af.map.build` so the loading rules (relative paths, dates, which key means
what) can be read and tested on their own, and so a caller that already has a
:class:`~af.map.config.MapsConfig` need not touch TOML at all.

Relative paths in the file resolve against the file's own directory, never the working directory,
so a config means the same thing wherever it is run from.
"""

from __future__ import annotations

import datetime as dt
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from af.map.config import (
    BlockOffsetConfig,
    ChainUntilConfig,
    ColouringConfig,
    ColumnBaseConfig,
    Defaults,
    FrameConfig,
    HideConfig,
    MapConfig,
    MapsConfig,
    MoveConfig,
    RotationConfig,
    VaccineChooseConfig,
    VaccineDisableConfig,
    WindowConfig,
)


class ConfigError(ValueError):
    """The round's maps config does not make sense. The message says which key."""


def _optional_date(row: Mapping[str, Any], where: str) -> dt.date | None:
    """A row's ``decided`` date if it has one. Vaccine rules predate dated decisions, so the
    date is optional there; the figure records null and the report says it is missing."""
    return _date(row["decided"], where) if "decided" in row else None


def _date(value: Any, where: str) -> dt.date:
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value)
        except ValueError as exc:
            raise ConfigError(f"{where}: {exc}") from exc
    raise ConfigError(f"{where}: expected a date, got {value!r}")


def _known(table: dict[str, Any], allowed: set[str], where: str) -> None:
    """A key nobody declares is a typo that would otherwise leave a setting at its default."""
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {', '.join(unknown)}")


def _path(value: str, base: Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def load_maps_config(path: Path) -> tuple[MapsConfig, Path]:
    """Read ``maps.toml``; return the config and the path of the curated vaccine list."""
    path = Path(path)
    base = path.resolve().parent
    with path.open("rb") as f:
        data = tomllib.load(f)
    _known(
        data,
        {"defaults", "frames", "maps", "vaccine_list", "vaccine_defaults", "colouring", "af_data"},
        str(path),
    )

    d = data.get("defaults")
    if not d:
        raise ConfigError(f"{path}: no [defaults]")
    _known(
        d,
        {
            "must_show_since",
            "windows",
            "min_common_points",
            "previous_round",
            "orientation_reference",
        },
        "defaults",
    )
    windows = tuple(
        WindowConfig(
            w["name"], _date(w["since"], f"window {w.get('name')}") if "since" in w else None
        )
        for w in d.get("windows", ())
    )
    if not windows:
        raise ConfigError("defaults.windows: at least one window is needed")
    defaults = Defaults(
        must_show_since=_date(d["must_show_since"], "defaults.must_show_since"),
        windows=windows,
        min_common_points=int(d.get("min_common_points", 50)),
        previous_round=_path(d["previous_round"], base) if "previous_round" in d else None,
        orientation_reference=d.get("orientation_reference", "previous-round"),
    )

    frames = []
    for i, f in enumerate(data.get("frames", ())):
        _known(f, {"subtype", "assay", "size"}, f"frames[{i}]")
        frames.append(FrameConfig(f["subtype"], float(f["size"]), f.get("assay")))
    if not frames:
        raise ConfigError("frames: at least one frame size is needed")

    maps = []
    for m in data.get("maps", ()):
        maps.append(_map(m, base))
    if not maps:
        raise ConfigError("maps: nothing to build")

    if "af_data" not in data:
        raise ConfigError(
            f"{path}: af_data (the acmacs-f-data checkout) is required: the non-ferret sera "
            "markers and the store colouring's matching rules are read from it"
        )
    if "vaccine_list" not in data:
        raise ConfigError(f"{path}: vaccine_list (the curated list) is required")
    try:
        config = MapsConfig(
            defaults=defaults,
            frames=tuple(frames),
            maps=tuple(maps),
            vaccine_defaults=_path(data["vaccine_defaults"], base)
            if "vaccine_defaults" in data
            else None,
            colouring=_colouring(data.get("colouring"), base),
            af_data=_path(data["af_data"], base) if "af_data" in data else None,
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return config, _path(data["vaccine_list"], base)


_COLOURING_PATHS = ("acmacs_data", "nomenclature")


def _colouring(c: dict[str, Any] | None, base: Path) -> ColouringConfig:
    """``[colouring]``. Absent means the bring-up stand-in, which each figure's provenance names;
    present, ``source`` must be said outright."""
    if c is None:
        return ColouringConfig("stand-in")
    if "af_data" in c:
        raise ConfigError(
            "colouring.af_data has moved to the top-level af_data: one root for acmacs-f-data"
        )
    _known(c, {"source", *_COLOURING_PATHS}, "colouring")
    if "source" not in c:
        raise ConfigError("colouring: source is required (stand-in or store)")
    paths = {k: _path(c[k], base) for k in _COLOURING_PATHS if k in c}
    try:
        return ColouringConfig(c["source"], **paths)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _map(m: dict[str, Any], base: Path) -> MapConfig:
    where = f"map {m.get('folder', '?')}"
    _known(
        m,
        {
            "folder",
            "clade_scheme",
            "chain",
            "chain_until",
            "scheme_stand_in",
            "layout_stand_in",
            "colouring",
            "title",
            "column_bases",
            "moves",
            "blocks",
            "hides",
            "rotations",
            "vaccine_disable",
            "vaccine_choose",
        },
        where,
    )
    moves = []
    for mv in m.get("moves", ()):
        _known(
            mv,
            {
                "name",
                "reason",
                "decided",
                "movers",
                "movers_file",
                "target_legend",
                "max_stress_rise",
                "max_from_target",
                "min_target_points",
            },
            f"{where} move",
        )
        moves.append(
            MoveConfig(
                mv["name"],
                mv["reason"],
                _date(mv["decided"], f"{where} move {mv['name']}"),
                tuple(mv.get("movers", ())),
                mv["target_legend"],
                float(mv["max_stress_rise"]),
                float(mv["max_from_target"]),
                int(mv.get("min_target_points", 5)),
                _path(mv["movers_file"], base) if "movers_file" in mv else None,
            )
        )
    blocks = []
    for b in m.get("blocks", ()):
        _known(
            b,
            {
                "name",
                "reason",
                "decided",
                "movers",
                "movers_file",
                "shift",
                "derived_from",
                "to",
                "target_legend",
                "max_stress_rise",
                "settled_within",
                "min_settled",
                "max_other_move",
            },
            f"{where} block",
        )
        shift = b.get("shift")
        if "to" in b:
            from af.map.curate import TO_TARGET_MEDIAN

            if b["to"] != TO_TARGET_MEDIAN:
                raise ConfigError(
                    f"{where} block {b['name']}: to must be {TO_TARGET_MEDIAN!r}, not {b['to']!r}"
                )
            if "shift" in b or "derived_from" in b:
                raise ConfigError(
                    f"{where} block {b['name']}: to = {b['to']!r} computes the shift; "
                    "give no shift or derived_from"
                )
        elif not (isinstance(shift, list) and len(shift) == 2):
            raise ConfigError(f"{where} block {b['name']}: shift must be [dx, dy]")
        elif "derived_from" not in b:
            raise ConfigError(f"{where} block {b['name']}: a fixed shift needs derived_from")
        blocks.append(
            BlockOffsetConfig(
                b["name"],
                b["reason"],
                _date(b["decided"], f"{where} block {b['name']}"),
                tuple(b.get("movers", ())),
                (float(shift[0]), float(shift[1])) if shift is not None else None,
                b.get("derived_from"),
                b["target_legend"],
                float(b["max_stress_rise"]),
                float(b["settled_within"]),
                int(b["min_settled"]),
                float(b["max_other_move"]),
                _path(b["movers_file"], base) if "movers_file" in b else None,
                b.get("to"),
            )
        )
    column_bases = []
    for c in m.get("column_bases", ()):
        _known(c, {"name", "reason", "decided", "sera", "value"}, f"{where} column_bases")
        column_bases.append(
            ColumnBaseConfig(
                c["name"],
                c["reason"],
                _date(c["decided"], f"{where} column_bases {c['name']}"),
                tuple(c["sera"]),
                float(c["value"]),
            )
        )
    hides = []
    for h in m.get("hides", ()):
        _known(
            h, {"name", "reason", "decided", "designations", "designations_file"}, f"{where} hide"
        )
        hides.append(
            HideConfig(
                h["name"],
                h["reason"],
                _date(h["decided"], f"{where} hide {h['name']}"),
                tuple(h.get("designations", ())),
                _path(h["designations_file"], base) if "designations_file" in h else None,
            )
        )
    rotations = []
    for r in m.get("rotations", ()):
        _known(r, {"name", "reason", "decided", "degrees", "reflect"}, f"{where} rotation")
        rotations.append(
            RotationConfig(
                r["name"],
                r["reason"],
                _date(r["decided"], f"{where} rotation {r['name']}"),
                float(r["degrees"]),
                bool(r.get("reflect", False)),
            )
        )
    disable = []
    for v in m.get("vaccine_disable", ()):
        _known(v, {"name", "reason", "passage", "optional", "decided"}, f"{where} vaccine_disable")
        disable.append(
            VaccineDisableConfig(
                v["name"],
                v["reason"],
                v.get("passage", "any"),
                bool(v.get("optional", False)),
                _optional_date(v, f"{where} vaccine_disable {v['name']}"),
            )
        )
    choose = []
    for v in m.get("vaccine_choose", ()):
        _known(
            v,
            {"name", "reason", "passage_class", "passage", "optional", "decided"},
            f"{where} vaccine_choose",
        )
        choose.append(
            VaccineChooseConfig(
                v["name"],
                v["reason"],
                v["passage_class"],
                v["passage"],
                bool(v.get("optional", False)),
                _optional_date(v, f"{where} vaccine_choose {v['name']}"),
            )
        )
    until = None
    if "chain_until" in m:
        u = m["chain_until"]
        _known(u, {"table", "reason", "decided"}, f"{where} chain_until")
        until = ChainUntilConfig(
            u["table"], u["reason"], _date(u["decided"], f"{where} chain_until")
        )
    return MapConfig(
        folder=m["folder"],
        clade_scheme=m["clade_scheme"],
        chain=m.get("chain"),
        chain_until=until,
        layout_stand_in=_path(m["layout_stand_in"], base) if "layout_stand_in" in m else None,
        scheme_stand_in=_path(m["scheme_stand_in"], base) if "scheme_stand_in" in m else None,
        title=m.get("title"),
        column_bases=tuple(column_bases),
        moves=tuple(moves),
        blocks=tuple(blocks),
        hides=tuple(hides),
        rotations=tuple(rotations),
        vaccine_disable=tuple(disable),
        vaccine_choose=tuple(choose),
        colouring=m.get("colouring"),
    )


def load_vaccine_list(path: Path) -> dict[str, list[Any]]:
    """The curated vaccine list, as its tables.

    Kept as a mapping rather than flattened: the transition-period file holds a table per subtype
    plus "-disabled", "-seasonal" and historical tables, and flattening them marks superseded and
    other-subtype strains as vaccines. The caller picks the table for the chart's subtype.
    """
    from af.map.vaccines import read_org_vaccine_tables

    path = Path(path)
    if path.suffix == ".py":
        return {k: list(v) for k, v in read_org_vaccine_tables(path.read_text()).items()}
    raise ConfigError(f"{path}: unsupported vaccine list format")
