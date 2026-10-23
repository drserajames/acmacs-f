"""The figure fonts: named installed files, resolved strictly, drawn in and recorded."""

from pathlib import Path

import matplotlib
import pytest

from af.util.config import ConfigError
from af.util.fonts import FontError, FontFiles, FontNames, load_fonts, pdf_fonttype

matplotlib.use("pdf")

# matplotlib's own bundled fonts stand in for the ones the af build installs: every env has them,
# and no binary goes into the repository.
TTF = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def bundled(name: str) -> Path:
    path = TTF / name
    if not path.is_file():
        raise FontError(f"not installed: {name}")
    return path


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "figures.toml"
    p.write_text(text)
    return p


def fonts_toml(sans: str = "STIXGeneral.ttf", bold: str = "STIXGeneralBol.ttf") -> str:
    return f'[fonts]\nsans = "{sans}"\nsans_bold = "{bold}"\nmono = "DejaVuSansMono.ttf"\n'


def test_fonts_resolve_to_their_families_and_record_their_files(tmp_path: Path) -> None:
    fonts = load_fonts(write(tmp_path, fonts_toml()), bundled)
    assert (fonts.sans, fonts.mono) == ("STIXGeneral", "DejaVu Sans Mono")
    assert fonts.mono_advance == pytest.approx(0.602, abs=0.001)
    inputs = fonts.inputs()
    assert set(inputs) == {"font_sans", "font_sans_bold", "font_mono"}
    assert all(len(row["sha256"]) == 64 for row in inputs.values())
    assert inputs["font_sans_bold"]["path"] == str(TTF / "STIXGeneralBol.ttf")


def test_other_tables_are_left_to_their_readers(tmp_path: Path) -> None:
    text = fonts_toml() + "\n[geo.example]\noutline = '#808080'\n"
    assert load_fonts(write(tmp_path, text), bundled).sans == "STIXGeneral"


def test_drawing_in_the_context_uses_the_named_font(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    fonts = load_fonts(write(tmp_path, fonts_toml()), bundled)
    with fonts.context():
        fig = plt.figure(figsize=(2, 1))
        fig.text(0.1, 0.5, "an invented title", weight="bold")
        fig.savefig(tmp_path / "f.pdf")
        plt.close(fig)
    pdf = (tmp_path / "f.pdf").read_bytes()
    assert b"STIXGeneral-Bold" in pdf and b"DejaVu" not in pdf


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("[other]\nx = 1\n", "fonts: required table missing"),
        (fonts_toml().replace("mono =", "mono_typo ="), "mono"),
    ],
)
def test_a_bad_file_is_refused(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        load_fonts(write(tmp_path, text), bundled)


def test_a_path_is_refused_where_a_name_belongs(tmp_path: Path) -> None:
    """A round config never names a release (or any) path: only an installed file's name."""
    with pytest.raises(FontError, match="is a path"):
        load_fonts(write(tmp_path, fonts_toml(sans=str(TTF / "STIXGeneral.ttf"))), bundled)


def test_a_font_that_is_not_installed_is_refused() -> None:
    with pytest.raises(FontError, match="not installed"):
        FontNames("none.ttf", "STIXGeneralBol.ttf", "DejaVuSansMono.ttf").locate(bundled)


def test_a_missing_font_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FontError, match="no file"):
        FontFiles(
            tmp_path / "none.ttf", TTF / "STIXGeneralBol.ttf", TTF / "DejaVuSansMono.ttf"
        ).resolve()


def test_regular_and_bold_must_be_one_family(tmp_path: Path) -> None:
    with pytest.raises(FontError, match="sans_bold is"):
        load_fonts(write(tmp_path, fonts_toml(bold="DejaVuSans-Bold.ttf")), bundled)


def test_cff_outline_fonts_embed_as_type_3(tmp_path: Path) -> None:
    """matplotlib's Type 42 mislabels a CFF-outline OpenType font; such a set draws as Type 3."""
    truetype, cff = tmp_path / "a.ttf", tmp_path / "b.otf"
    truetype.write_bytes(b"\x00\x01\x00\x00 rest of an invented font")
    cff.write_bytes(b"OTTO rest of an invented font")
    assert pdf_fonttype((truetype, truetype)) == 42
    assert pdf_fonttype((truetype, cff)) == 3


def test_a_mono_font_af_does_not_install_is_named_by_path_and_hash(tmp_path: Path) -> None:
    import hashlib
    import shutil

    mono = tmp_path / "Mono.ttf"  # a file af does not install, as a system font would be
    shutil.copyfile(TTF / "DejaVuSansMono.ttf", mono)
    sha = hashlib.sha256(mono.read_bytes()).hexdigest()
    head = (
        '[fonts]\nsans = "STIXGeneral.ttf"\nsans_bold = "STIXGeneralBol.ttf"\nmono = "Mono.ttf"\n'
    )
    fonts = load_fonts(
        write(tmp_path, head + f'mono_path = "{mono}"\nmono_sha256 = "{sha}"\n'), bundled
    )
    assert fonts.inputs()["font_mono"]["path"] == str(mono)
    with pytest.raises(FontError, match="has sha256"):
        load_fonts(
            write(tmp_path, head + f'mono_path = "{mono}"\nmono_sha256 = "{"0" * 64}"\n'), bundled
        )
    with pytest.raises(FontError, match="both mono_path and mono_sha256"):
        load_fonts(write(tmp_path, head + f'mono_path = "{mono}"\n'), bundled)
    other = (
        '[fonts]\nsans = "STIXGeneral.ttf"\nsans_bold = "STIXGeneralBol.ttf"\nmono = "Other.ttf"\n'
    )
    with pytest.raises(FontError, match="file name"):
        load_fonts(
            write(tmp_path, other + f'mono_path = "{mono}"\nmono_sha256 = "{sha}"\n'), bundled
        )
