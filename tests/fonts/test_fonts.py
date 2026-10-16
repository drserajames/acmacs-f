"""af.fonts: the figure fonts the build fetches and installs (fonts/CMakeLists.txt).

These run against the fonts this environment's af was built with (check-committed copies them
into its export): what a figure would get, not stand-ins.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest

import af.fonts as fonts
from tests.helpers import REPO_ROOT

FAMILIES = {"heros": "TeX Gyre Heros", "cursor": "TeX Gyre Cursor"}
STYLES = ("regular", "bold", "italic", "bolditalic")


def test_the_fonts_come_from_the_archive_the_build_pins() -> None:
    recipe = (REPO_ROOT / "fonts" / "CMakeLists.txt").read_text()
    pinned = re.search(r'set\(AF_FONTS_SHA256 "([0-9a-f]{64})"\)', recipe)
    version = re.search(r'set\(AF_FONTS_VERSION "([^"]+)"\)', recipe)
    assert pinned and version
    source = fonts.source()
    assert source["archive_sha256"] == pinned[1]
    assert source["version"] == version[1]
    assert source["archive_url"].endswith(".zip")


def test_every_installed_file_is_the_one_the_build_recorded() -> None:
    files = fonts.source()["files"]
    assert {
        "GUST-FONT-LICENSE.txt",
        "README-TeX-Gyre-Heros.txt",
        "MANIFEST-TeX-Gyre-Cursor.txt",
    } <= set(files)
    for name, sha256 in files.items():
        installed = fonts.directory() / name
        assert hashlib.sha256(installed.read_bytes()).hexdigest() == sha256, name


def test_the_licence_text_is_the_committed_one() -> None:
    committed = (REPO_ROOT / "fonts" / "GUST-FONT-LICENSE.txt").read_bytes()
    assert (fonts.directory() / "GUST-FONT-LICENSE.txt").read_bytes() == committed


def test_fonts_are_named_by_file_and_an_unknown_name_is_an_error() -> None:
    expected = sorted(f"texgyre{family}-{style}.otf" for family in FAMILIES for style in STYLES)
    assert fonts.names() == expected
    assert all(fonts.path(name).is_file() for name in expected)
    with pytest.raises(fonts.FontsMissing, match=r"no installed font 'helvetica\.otf'; af has: "):
        fonts.path("helvetica.otf")


def test_an_af_built_without_its_fonts_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "af.fonts._files", None)  # as if never installed
    with pytest.raises(fonts.FontsMissing, match="installed without its fonts"):
        fonts.path("texgyreheros-regular.otf")


def test_registered_families_resolve_to_these_files_without_a_fallback() -> None:
    from matplotlib import font_manager

    families = fonts.register()
    assert set(families.values()) == set(FAMILIES.values())
    for family, name in FAMILIES.items():
        for weight, style in (("normal", "regular"), ("bold", "bold")):
            found = font_manager.findfont(
                font_manager.FontProperties(family=name, weight=weight),
                fallback_to_default=False,
            )
            wanted = fonts.path(f"texgyre{family}-{style}.otf")
            assert Path(found).read_bytes() == wanted.read_bytes(), (name, weight)


@pytest.mark.parametrize("fonttype", [3, 42])
def test_matplotlib_embeds_both_families_in_a_pdf(tmp_path: Path, fonttype: int) -> None:
    """Embedded and subset (a "XXXXXX+" prefix), not merely referenced by name."""
    import matplotlib

    matplotlib.use("pdf", force=True)
    import matplotlib.pyplot as plt

    fonts.register()
    with matplotlib.rc_context({"pdf.fonttype": fonttype}):
        fig = plt.figure(figsize=(2, 1))
        fig.text(0.1, 0.6, "Sans text 0123", family="TeX Gyre Heros")
        fig.text(0.1, 0.2, "Mono text 0123", family="TeX Gyre Cursor")
        fig.savefig(tmp_path / "fonts.pdf")
        plt.close(fig)
    pdf = (tmp_path / "fonts.pdf").read_bytes()
    for postscript_name in (b"TeXGyreHeros-Regular", b"TeXGyreCursor-Regular"):
        assert re.search(rb"/BaseFont /[A-Z]{6}\+" + postscript_name, pdf), postscript_name
    assert b"DejaVu" not in pdf, "a fallback font was used"
