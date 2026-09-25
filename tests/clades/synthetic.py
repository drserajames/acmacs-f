"""A synthetic nomenclature, built here rather than committed.

Everything is invented: the subtype uses af's real ``A(H3N2)`` coordinates, because that
is what the conversions are keyed on, but no clade name, sequence or virus name here
belongs to a real virus (public repo rule).

The shape mirrors what upstream publishes, because that is what the loader has to cope
with: per-branch mutations, an aliased clade level with older display names, a revoked
name, a deletion-defined clade and a locus outside the mature HA.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from af.clades.nomenclature import CladeSet, load_clade_set


def commit_command(repository: Path, message: str) -> list[str]:
    """Commit without depending on the machine's git identity."""
    return [
        "git",
        "-C",
        str(repository),
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "user.name=Test",
        "commit",
        "-qm",
        message,
    ]


def write_clade(directory: Path, name: str, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.yml").write_text(body.strip() + "\n")


def build_clone(tmp_path: Path) -> Path:
    """A synthetic single-subtype clone, committed so the pin check has something to read.

    Hierarchy: ``P`` → ``P.1`` → ``P.1.1``, with ``P.2`` a sister defined by a deletion
    and ``Q`` an older name aliasing ``P.1``.
    """
    root = tmp_path / "synthetic_HA"
    subclades = root / "subclades"
    write_clade(
        subclades,
        "P",
        """
name: P
parent: none
defining_mutations:
- locus: HA1
  position: 5
  state: K
clade: none
""",
    )
    write_clade(
        subclades,
        "P.1",
        """
name: P.1
parent: P
unaliased_name: P.1
defining_mutations:
- locus: HA1
  position: 9
  state: T
- locus: HA2
  position: 2
  state: W
clade: legacy-1
""",
    )
    write_clade(
        subclades,
        "P.1.1",
        """
name: P.1.1
parent: P.1
defining_mutations:
- locus: HA1
  position: 12
  state: N
- locus: nuc
  position: 71
  state: A
representatives: []
""",
    )
    write_clade(
        subclades,
        "P.2",
        """
name: P.2
parent: P
defining_mutations:
- locus: HA1
  position: 7
  state: '-'
- locus: SigPep
  position: 3
  state: M
""",
    )
    write_clade(
        subclades,
        "P.3",
        """
name: P.3
parent: P
revoked: true
comment: "renamed"
defining_mutations:
- locus: HA1
  position: 5
  state: K
""",
    )
    write_clade(
        root / "clades",
        "Q",
        """
name: Q
short_name: Q
alias_of: P.1
parent: none
representatives: []
""",
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(commit_command(root, "synthetic"), check=True)
    return root


def clone_commit(clone: Path) -> str:
    finished = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return finished.stdout.strip()


def load_synthetic(clones: Path) -> CladeSet:
    """The synthetic nomenclature, loaded from the directory holding the clone."""
    return load_clade_set("A(H3N2)", clones, repository="synthetic_HA")
