"""Reuse a drawn tree figure when nothing it is drawn from has changed.

Why: drawing the three report trees with their leaf books takes about 13 minutes on one core
(measured 18 Oct 2026: the book 60%, the searchable name layer 27%), and a round draws the
same trees again for every rehearsal and build. A figure is a function of the tree store
version, the figure config, whether a book is drawn, and the code and libraries that draw it.
An entry keyed by all of those holds the same bytes a fresh draw would write, so reusing it
changes nothing in the output.

Entries are content-addressed and write-once, in a directory the caller names. They are never
kept in the store, because the drawing stage only reads the store. A hit copies the entry's
files into place and checks each copy against the hash recorded when it was drawn. A damaged
entry is an error, not a quiet redraw (design rule 3).

Code that cannot be identified is never trusted to match: a dirty checkout, or an af that is
neither a release nor a git checkout, draws every time and says why. Within a release the
commit decides everything, since the release pins every package. In a clean checkout the
commit is joined by the versions of the libraries that decide a PDF's bytes.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from af.util.artefacts import Artefact, CheckedArtefact, check_artefacts, sha256_path
from af.util.running import running_af

ENTRY_FORMAT = 1
"""Bumped when the key or the entry layout changes; entries in another format are not read."""
ENTRY_FILE = "ENTRY.json"
PDF_LIBRARIES = ("matplotlib", "numpy", "fonttools")  # each decides bytes in a drawn PDF


class DrawCacheError(RuntimeError):
    """A cache entry is damaged or does not match what it is asked for."""


@dataclass(frozen=True)
class CacheResult:
    hit: bool
    reason: str
    key: str | None  # None when no entry could be looked up


def code_identity() -> dict[str, Any] | None:
    """What decides a drawn PDF's bytes on the code side, or None when it cannot be known."""
    af = running_af()
    if af["commit"] is None or af["dirty"] is not False:
        return None
    from matplotlib import ft2font  # the bundled FreeType lays out every glyph

    return {
        "commit": af["commit"],
        "release": af["release"],
        "python": af["python"],
        "libraries": {name: _installed(name) for name in PDF_LIBRARIES}
        | {"freetype": ft2font.__freetype_version__},
    }


def canonical(value: Any) -> Any:
    """``value`` as plain JSON data, the same for equal values: dataclasses become dicts, sets
    sorted lists, paths strings. Anything else JSON cannot hold is a TypeError, so a config
    field that would make the key ambiguous is caught rather than dropped."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    return json.loads(json.dumps(value, sort_keys=True, default=_plain))


def entry_key(parts: Mapping[str, Any], code: Mapping[str, Any]) -> str:
    text = json.dumps(
        {"format": ENTRY_FORMAT, "parts": canonical(dict(parts)), "code": canonical(dict(code))},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode()).hexdigest()


def reuse_or_draw(
    cache: Path | None,
    parts: Mapping[str, Any],
    code: Mapping[str, Any] | None,
    outputs: Mapping[str, Path],
    draw: Callable[[], None],
    step: str,
) -> CacheResult:
    """Put ``outputs`` (entry file name -> where it goes) in place: copied from the entry for
    ``parts`` and ``code`` if there is one, else written by ``draw()`` and then stored.

    ``draw`` must write every output; they are checked (exists, non-empty) before an entry is
    made, so a failed draw never becomes a reusable entry.
    """
    for name in outputs:
        if Path(name).name != name or name == ENTRY_FILE or name.startswith("."):
            raise ValueError(f"{step}: output name {name!r} is not a plain file name")
    if cache is None:
        draw()
        return CacheResult(False, "no cache directory given", None)
    if code is None:
        draw()
        return CacheResult(False, "code not identified (a dirty or unknown af): not cached", None)
    key = entry_key(parts, code)
    entry = cache / key
    if entry.is_dir():
        _copy_out(entry, key, outputs, step)
        return CacheResult(True, "reused: tree, config and code unchanged", key)
    draw()
    checked = check_artefacts(step, [Artefact(path) for path in outputs.values()])
    _store(cache, key, parts, code, {name: c for name, c in zip(outputs, checked, strict=True)})
    return CacheResult(False, "no entry for this tree, config and code: drawn", key)


def _copy_out(entry: Path, key: str, outputs: Mapping[str, Path], step: str) -> None:
    record = json.loads((entry / ENTRY_FILE).read_text())
    if record.get("format") != ENTRY_FORMAT or record.get("key") != key:
        raise DrawCacheError(f"{step}: cache entry {entry} is not the entry for key {key}")
    files: dict[str, str] = record["files"]
    if sorted(files) != sorted(outputs):
        raise DrawCacheError(
            f"{step}: cache entry {entry} holds {sorted(files)}, asked for {sorted(outputs)}"
        )
    for name, target in outputs.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.from-cache")
        shutil.copyfile(entry / name, partial)
        if sha256_path(partial) != files[name]:
            partial.unlink()
            raise DrawCacheError(f"{step}: cache entry {entry}: {name} does not match its hash")
        partial.replace(target)


def _store(
    cache: Path,
    key: str,
    parts: Mapping[str, Any],
    code: Mapping[str, Any],
    checked: Mapping[str, CheckedArtefact],
) -> None:
    """Write the entry beside its final place and rename it in: never half-written, and when
    another process stored the same key first, theirs is kept (the bytes are the same)."""
    cache.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".tmp-", dir=cache))
    try:
        for name, artefact in checked.items():
            shutil.copyfile(artefact.path, staging / name)
        record = {
            "format": ENTRY_FORMAT,
            "key": key,
            "parts": canonical(dict(parts)),
            "code": canonical(dict(code)),
            "files": {name: artefact.sha256 for name, artefact in checked.items()},
        }
        (staging / ENTRY_FILE).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        try:
            staging.rename(cache / key)
        except OSError:
            if not (cache / key).is_dir():
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _plain(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} has no canonical form for a cache key")


def _installed(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
