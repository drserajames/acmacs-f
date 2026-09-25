"""acmacs-f: phylogenetic trees and antigenic maps for influenza vaccine strain selection.

The distribution is called ``acmacs-f``; the import name is ``af`` (AD -> ae -> af).
"""

import sys
from pathlib import Path

__version__ = "0.1.0.dev0"


def _check_release(prefix: Path, module_file: Path) -> None:
    """Refuse an af from outside a frozen release when the python *is* a release.

    ``python -c`` / ``-m`` put the current directory first on sys.path, so a release's
    python started inside an acmacs-f checkout would silently import that checkout's af:
    the very drift releases exist to prevent (tools/make-release.py).
    """
    release = prefix.parent
    if not (release / "RELEASE.toml").is_file():
        return
    if not module_file.resolve().is_relative_to(prefix.resolve()):
        raise ImportError(
            f"this python belongs to the frozen af release {release}, but af was imported "
            f"from {module_file.parent}: run it from outside that checkout, or with `python -P`"
        )


_check_release(Path(sys.prefix), Path(__file__))
