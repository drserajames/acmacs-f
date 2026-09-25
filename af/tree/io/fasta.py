"""Read and write aligned FASTA.

Deliberately small: af needs sequences keyed by name, all the same length, and a writer that does
not reflow. An alignment whose rows differ in length is an error here rather than something the
next tool discovers — ae's `export` only *printed* a wrong length and carried on (INVENTORY §2.2).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path


class FastaError(ValueError):
    """The FASTA is malformed, or its rows are not all the same length."""


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (name, sequence) without holding the whole file in memory."""
    name: str | None = None
    chunks: list[str] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line:
                continue
            if line[0] == ">":
                if name is not None:
                    yield name, "".join(chunks)
                name = line[1:].strip()
                chunks = []
            else:
                if name is None:
                    raise FastaError(f"{path}: sequence data before the first '>' header")
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


def read_alignment(path: Path) -> dict[str, str]:
    """Read an alignment, checking that every row has the same length and no name repeats."""
    out: dict[str, str] = {}
    length: int | None = None
    for name, sequence in iter_fasta(path):
        if name in out:
            raise FastaError(f"{path}: duplicate sequence name {name!r}")
        if length is None:
            length = len(sequence)
        elif len(sequence) != length:
            raise FastaError(
                f"{path}: {name!r} is {len(sequence)} long but the alignment is {length}; "
                "an unaligned or truncated row cannot be used for reconstruction"
            )
        out[name] = sequence
    if not out:
        raise FastaError(f"{path}: no sequences")
    return out


def write_alignment(path: Path, sequences: Mapping[str, str], wrap: int = 0) -> None:
    """Write FASTA. ``wrap=0`` keeps one line per sequence, which every tool here accepts."""
    with Path(path).open("w") as handle:
        for name, sequence in sequences.items():
            handle.write(f">{name}\n")
            if wrap:
                for start in range(0, len(sequence), wrap):
                    handle.write(sequence[start : start + wrap] + "\n")
            else:
                handle.write(sequence + "\n")
