"""The figure fonts: one sans (regular and bold) and one mono, shared by maps, geo and trees.

Why font FILES, named in a round's ``figures.toml``, rather than family names: matplotlib finds a
family among whatever fonts the machine has, and falls back to DejaVu Sans with only a warning
when it finds none, so the same config could draw in different fonts on two machines. A file is
explicit (design rule 4), and its hash goes into each figure's provenance (rule 5), so a font
change is a recorded input change.

``figures.toml``::

    [fonts]
    sans = "texgyreheros-regular.otf"
    sans_bold = "texgyreheros-bold.otf"
    mono = "texgyrecursor-regular.otf"

The names are bare file names of fonts the af build installs (:mod:`af.fonts`, pinned by hash at
build time), so a round config never names a release path. :func:`load_fonts` reads the file,
:meth:`FontNames.locate` finds the installed files, :meth:`FontFiles.resolve` registers them with
matplotlib and checks that each family and weight resolves back to exactly that file, and
:meth:`FigureFonts.context` is the rc context a figure is drawn in. Text widths measured for
placement (TextPath, FontProperties without a family) take the rc family too, so a caller wraps
the WHOLE drawing in the context, not only the save.
"""

from __future__ import annotations

import hashlib
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from af.util.config import ConfigError, parse_config


class FontError(ValueError):
    """A figure font file is missing, unreadable, or does not resolve to itself."""


@dataclass(frozen=True)
class FontNames:
    """The three installed font files a round names, by bare file name."""

    sans: str
    sans_bold: str
    mono: str

    def locate(self, path_of: Callable[[str], Path] | None = None) -> FontFiles:
        """The installed files. ``path_of`` maps a name to its file; by default the fonts the
        af build installed (:func:`af.fonts.path`, which raises listing what is installed)."""
        for role, name in vars(self).items():
            if Path(name).name != name:
                raise FontError(f"figure font {role}: {name!r} is a path; name an installed file")
        if path_of is None:
            try:
                import af.fonts as fonts  # type: ignore[import-not-found,unused-ignore]
            except ImportError as exc:
                raise FontError("af.fonts is missing: this af build installed no fonts") from exc
            path_of = fonts.path
        return FontFiles(path_of(self.sans), path_of(self.sans_bold), path_of(self.mono))


@dataclass(frozen=True)
class FontFiles:
    """The three font files, located."""

    sans: Path
    sans_bold: Path
    mono: Path

    def resolve(self) -> FigureFonts:
        """Register the files with matplotlib and return the families they define.

        Refused (rule 1): a missing file; a regular and bold that are not one family; or a
        family and weight that matplotlib resolves to a DIFFERENT font than the one named
        (another installed font of the same name would otherwise win silently); a byte-identical
        copy is the same font."""
        from matplotlib import font_manager

        roles = {"sans": self.sans, "sans_bold": self.sans_bold, "mono": self.mono}
        for role, path in roles.items():
            if not path.is_file():
                raise FontError(f"figure font {role}: no file {path}")
            font_manager.fontManager.addfont(str(path))
        sans = font_manager.FontProperties(fname=str(self.sans)).get_name()
        bold = font_manager.FontProperties(fname=str(self.sans_bold)).get_name()
        mono = font_manager.FontProperties(fname=str(self.mono)).get_name()
        if sans != bold:
            raise FontError(f"figure fonts: sans is {sans!r} but sans_bold is {bold!r}")
        for family, weight, path in (
            (sans, "normal", self.sans),
            (sans, "bold", self.sans_bold),
            (mono, "normal", self.mono),
        ):
            found = font_manager.findfont(
                font_manager.FontProperties(family=family, weight=weight),
                fallback_to_default=False,
            )
            same = Path(found).resolve() == path.resolve() or _sha256(Path(found)) == _sha256(path)
            if not same:  # an identical copy installed elsewhere draws the same glyphs
                raise FontError(
                    f"figure font {family!r} ({weight}) resolves to {found}, not {path}"
                )
        return FigureFonts(sans, mono, self, _advance(self.mono))


@dataclass(frozen=True)
class FigureFonts:
    """Resolved figure fonts: the families to draw with and the files they came from."""

    sans: str
    mono: str
    files: FontFiles
    mono_advance: float  # the mono font's advance width per point of font size

    def context(self) -> Any:
        """The matplotlib rc context to draw a whole figure in (placement included)."""
        import matplotlib.pyplot as plt

        return plt.rc_context(
            {
                "font.family": "sans-serif",
                "font.sans-serif": [self.sans],
                "font.monospace": [self.mono],
                "pdf.fonttype": pdf_fonttype(
                    (self.files.sans, self.files.sans_bold, self.files.mono)
                ),
            }
        )

    def inputs(self) -> dict[str, dict[str, str]]:
        """Provenance inputs, one per file (name -> path, family, sha256), as rule 5 wants."""
        rows = (
            ("font_sans", self.files.sans, self.sans),
            ("font_sans_bold", self.files.sans_bold, self.sans),
            ("font_mono", self.files.mono, self.mono),
        )
        return {
            name: {"path": str(path), "family": family, "sha256": _sha256(path)}
            for name, path, family in rows
        }


def load_fonts(path: Path, path_of: Callable[[str], Path] | None = None) -> FigureFonts:
    """Read the ``[fonts]`` table of a round's ``figures.toml`` and resolve its fonts
    (``path_of``: see :meth:`FontNames.locate`).

    The file is the round's one figure-style file: other tables (such as geo's per-subtype
    look) belong to, and are validated by, their own readers; ``[fonts]`` is read strictly."""
    path = Path(path)
    with path.open("rb") as file:
        try:
            data = tomllib.load(file)
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(path, [f"TOML syntax: {error}"]) from error
    if "fonts" not in data:
        raise ConfigError(path, ["fonts: required table missing"])
    names = parse_config(data["fonts"], FontNames, base_dir=path.resolve().parent, source=path)
    return names.locate(path_of).resolve()


def pdf_fonttype(paths: tuple[Path, ...]) -> int:
    """42 (embedded TrueType, as the shipped maps have) when every file has TrueType outlines;
    3 otherwise. matplotlib writes a CFF-outline OpenType font ("OTTO", e.g. TeX Gyre) under
    Type 42 as a TrueType font program, which PDF readers flag as a type mismatch; as Type 3 it
    is drawn from the same outlines and stays searchable."""
    truetype = (b"\x00\x01\x00\x00", b"true")
    return 42 if all(p.read_bytes()[:4] in truetype for p in paths) else 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _advance(path: Path) -> float:
    """Advance width of "M" per point of font size; a mono font's every glyph has it."""
    from matplotlib.ft2font import FT2Font

    font = FT2Font(str(path))
    font.set_size(100, 72)
    return font.load_char(ord("M")).linearHoriAdvance / 65536 / 100
