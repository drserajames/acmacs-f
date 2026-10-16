"""The figure fonts installed with af: TeX Gyre Heros (Helvetica metrics) and TeX Gyre Cursor
(Courier metrics), under the GUST Font License (an instance of the LPPL 1.3c).

Why they come with af: a figure must not depend on what a machine happens to have installed,
and matplotlib falls back to DejaVu silently when a family is missing. The fonts are not in
git: the build fetches them at a pinned version, checks the archive's SHA-256 and installs them
here (fonts/CMakeLists.txt), as it does alglib. A figure names a font by its file name;
``path`` gives the installed file, or fails, so a missing font is an error and never a fallback.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path
from typing import Any

SOURCE_FILE = "SOURCE.toml"


class FontsMissing(RuntimeError):
    """af was installed without its fonts, or without the one asked for."""


def directory() -> Path:
    """Where the build installed the fonts (``_files`` is installed with them, never in git)."""
    try:
        files = importlib.import_module("af.fonts._files")
    except ImportError as err:
        raise FontsMissing(
            "this af was installed without its fonts (no af.fonts._files): reinstall it "
            "(pip install -e . in a checkout, or a new release), which fetches them"
        ) from err
    found: Path = files.DIRECTORY
    return found


def source() -> dict[str, Any]:
    """Where the fonts came from: the archive's URL, SHA-256 and version, and every installed
    file with its SHA-256 (``files``), as the build wrote them."""
    record = directory() / SOURCE_FILE
    if not record.is_file():
        raise FontsMissing(f"{record} is missing: the fonts' install is incomplete")
    return tomllib.loads(record.read_text())


def names() -> list[str]:
    """The installed font files (.otf), by file name."""
    return sorted(name for name in source()["files"] if name.endswith(".otf"))


def path(name: str) -> Path:
    """The installed font file called ``name`` (e.g. ``texgyreheros-regular.otf``)."""
    available = names()
    if name not in available:
        raise FontsMissing(f"no installed font {name!r}; af has: {', '.join(available)}")
    found = directory() / name
    if not found.is_file():
        raise FontsMissing(f"{found} is listed in {SOURCE_FILE} but not installed")
    return found


def register(files: list[str] | None = None) -> dict[str, str]:
    """Add the fonts (``files``, or all) to matplotlib's font manager; their family names.

    Registering makes a family resolvable by name; a figure that wants a particular file should
    still check, with ``findfont(..., fallback_to_default=False)``, that the name gives it.
    """
    from matplotlib import font_manager

    families = {}
    for name in names() if files is None else files:
        file = path(name)
        font_manager.fontManager.addfont(str(file))
        families[name] = font_manager.FontProperties(fname=file).get_name()
    return families
