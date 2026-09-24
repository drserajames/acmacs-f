"""Explicit configuration: TOML files validated against dataclass schemas.

Why TOML rather than YAML: ``tomllib`` is in the standard library (no dependency), and
TOML has no implicit typing. YAML reads ``NO`` as false, ``3.10`` as 3.1 and bare dates
as date objects, which silently changes lab codes, versions and table dates (design
rule 7). TOML strings stay strings unless quoted otherwise.

Why a schema: design rule 4. Every key a step reads is declared as a dataclass field;
a key in the file that no field declares is an error (usually a typo that would
otherwise leave a setting at its default), and a field without a default that the file
does not set is an error. All problems are reported together, each with its dotted key
path, so one run shows everything wrong with a file.

Relative paths in a config file are resolved against the directory of that file, never
the current working directory, so a config means the same thing wherever it is run from.
Nothing here reads environment variables.

Example::

    @dataclass(frozen=True)
    class Store:
        root: Path

    @dataclass(frozen=True)
    class Settings:
        store: Store
        threads: int = 1

    settings = load_config(Path("af.toml"), Settings)
"""

from __future__ import annotations

import dataclasses
import tomllib
import types
import typing
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

T = TypeVar("T")


class ConfigError(ValueError):
    """A config file does not match its schema. ``problems`` lists every mismatch."""

    def __init__(self, source: Path | str, problems: list[str]) -> None:
        self.source = source
        self.problems = problems
        lines = "\n".join(f"  - {problem}" for problem in problems)
        super().__init__(f"invalid config {source}:\n{lines}")


def load_config(path: Path, schema: type[T]) -> T:
    """Read the TOML file at ``path`` and return it as an instance of ``schema``.

    A missing file is fatal (``FileNotFoundError``), as is a TOML syntax error.
    """
    path = Path(path)
    with path.open("rb") as file:
        try:
            data = tomllib.load(file)
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(path, [f"TOML syntax: {error}"]) from error
    return parse_config(data, schema, base_dir=path.resolve().parent, source=path)


def parse_config(
    data: dict[str, Any], schema: type[T], *, base_dir: Path, source: Path | str = "<config>"
) -> T:
    """Validate an already-parsed mapping against ``schema``.

    ``base_dir`` is required: it is what relative paths are resolved against, and there
    is deliberately no default (such as the working directory).
    """
    if not (dataclasses.is_dataclass(schema) and isinstance(schema, type)):
        raise TypeError(f"schema must be a dataclass type, not {schema!r}")
    problems: list[str] = []
    result = _convert_dataclass(data, schema, "", base_dir, problems)
    if problems:
        raise ConfigError(source, problems)
    return typing.cast(T, result)


def _convert_dataclass(
    data: Any, schema: type, key: str, base_dir: Path, problems: list[str]
) -> Any:
    if not isinstance(data, dict):
        problems.append(f"{key or '<top level>'}: expected a table, got {_describe(data)}")
        return None
    hints = typing.get_type_hints(schema)
    fields = {field.name: field for field in dataclasses.fields(schema) if field.init}
    for unknown in sorted(set(data) - set(fields)):
        problems.append(f"{_join(key, unknown)}: unknown key")
    values: dict[str, Any] = {}
    for name, field in fields.items():
        child = _join(key, name)
        if name in data:
            values[name] = _convert(data[name], hints[name], child, base_dir, problems)
        elif _has_default(field):
            continue
        else:
            problems.append(f"{child}: required key missing")
    if problems:
        return None
    return schema(**values)


def _convert(value: Any, hint: Any, key: str, base_dir: Path, problems: list[str]) -> Any:
    """Check ``value`` against the annotation ``hint``; convert paths and nested tables."""
    origin = get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        return _convert_union(value, hint, key, base_dir, problems)
    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        return _convert_dataclass(value, hint, key, base_dir, problems)
    if hint is Path:
        if not isinstance(value, str):
            problems.append(f"{key}: expected a path string, got {_describe(value)}")
            return None
        path = Path(value).expanduser()
        return path if path.is_absolute() else (base_dir / path).resolve()
    if origin in (list, tuple):
        return _convert_list(value, hint, key, base_dir, problems)
    if origin is dict:
        return _convert_dict(value, hint, key, base_dir, problems)
    if hint is Any:
        return value
    return _check_scalar(value, hint, key, problems)


def _convert_union(value: Any, hint: Any, key: str, base_dir: Path, problems: list[str]) -> Any:
    options = get_args(hint)
    if value is None and type(None) in options:
        return None
    for option in options:
        if option is type(None):
            continue
        trial: list[str] = []
        converted = _convert(value, option, key, base_dir, trial)
        if not trial:
            return converted
    problems.append(f"{key}: expected {_name(hint)}, got {_describe(value)}")
    return None


def _convert_list(value: Any, hint: Any, key: str, base_dir: Path, problems: list[str]) -> Any:
    if not isinstance(value, list):
        problems.append(f"{key}: expected a list, got {_describe(value)}")
        return None
    args = get_args(hint)
    item_hint = args[0] if args else Any
    items = [
        _convert(item, item_hint, f"{key}[{index}]", base_dir, problems)
        for index, item in enumerate(value)
    ]
    return tuple(items) if get_origin(hint) is tuple else items


def _convert_dict(value: Any, hint: Any, key: str, base_dir: Path, problems: list[str]) -> Any:
    if not isinstance(value, dict):
        problems.append(f"{key}: expected a table, got {_describe(value)}")
        return None
    args = get_args(hint)
    value_hint = args[1] if len(args) == 2 else Any
    return {
        name: _convert(item, value_hint, _join(key, name), base_dir, problems)
        for name, item in value.items()
    }


def _check_scalar(value: Any, hint: Any, key: str, problems: list[str]) -> Any:
    # bool is a subclass of int in Python; `threads = true` must not pass as 1.
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(hint, type) and isinstance(value, hint):
        if hint is int and isinstance(value, bool):
            problems.append(f"{key}: expected int, got bool")
            return None
        return value
    problems.append(f"{key}: expected {_name(hint)}, got {_describe(value)}")
    return None


def _has_default(field: dataclasses.Field[Any]) -> bool:
    return (
        field.default is not dataclasses.MISSING or field.default_factory is not dataclasses.MISSING
    )


def _join(key: str, name: str) -> str:
    return f"{key}.{name}" if key else name


def _describe(value: Any) -> str:
    return f"{type(value).__name__} {value!r}"


def _name(hint: Any) -> str:
    return getattr(hint, "__name__", None) or str(hint)
