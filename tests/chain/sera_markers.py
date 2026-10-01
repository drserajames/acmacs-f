"""A minimal non-ferret markers file for chain configs in tests (synthetic patterns only)."""

from pathlib import Path

MARKERS = (
    "field\tpattern\tkind\treason\tevidence\n"
    "name\tTESTPOOL\tpooled test serum\tsynthetic\ttests only\n"
)


def write_markers(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MARKERS)
    return path


def select_block(tmp_path: Path) -> str:
    """A `[select]` table naming a markers file, to append at the end of a chain file."""
    return f'\n[select]\nnon_ferret_markers = "{write_markers(tmp_path / "markers.tsv")}"\n'
