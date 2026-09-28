"""tools/make-release.py's linkage gate, on real otool/ldd output (Linux can't run here)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tests.helpers import REPO_ROOT

RELEASE = Path("/syn/acmacs-f/releases/abc")


@pytest.fixture(scope="module")
def mr() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "make_release", REPO_ROOT / "tools" / "make-release.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_release"] = module
    spec.loader.exec_module(module)
    return module


# o's release b6da3f4e06df (28 Sep): _core built with system g++, no RUNPATH.
O_LDD = """\
\tlinux-vdso.so.1 (0x00007ffd1d7f8000)
\tlibgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1 (0x0000748df217d000)
\tlibstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x0000748df1e00000)
\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x0000748df1a00000)
\t/lib64/ld-linux-x86-64.so.2 (0x0000748df21e0000)
"""


def test_parse_ldd(mr: ModuleType) -> None:
    assert mr.parse_ldd(O_LDD) == [
        ("libgomp.so.1", "/lib/x86_64-linux-gnu/libgomp.so.1"),
        ("libstdc++.so.6", "/lib/x86_64-linux-gnu/libstdc++.so.6"),
        ("libc.so.6", "/lib/x86_64-linux-gnu/libc.so.6"),
    ]


def test_linux_system_copy_of_a_shipped_library_is_foreign(mr: ModuleType) -> None:
    """o's case: the env ships libgomp and libstdc++, ldd resolved the system's."""
    shipped = {"libgomp.so.1", "libgomp.so.1.0.0", "libstdc++.so.6", "libpython3.12.so"}
    foreign = mr.foreign_libraries(mr.parse_ldd(O_LDD), RELEASE, shipped, darwin=False)
    assert foreign == ["/lib/x86_64-linux-gnu/libgomp.so.1", "/lib/x86_64-linux-gnu/libstdc++.so.6"]


def test_linux_runpath_build_passes(mr: ModuleType) -> None:
    fixed = O_LDD.replace("/lib/x86_64-linux-gnu/libgomp.so.1", f"{RELEASE}/env/lib/libgomp.so.1")
    fixed = fixed.replace(
        "/lib/x86_64-linux-gnu/libstdc++.so.6", f"{RELEASE}/env/lib/libstdc++.so.6"
    )
    shipped = {"libgomp.so.1", "libstdc++.so.6"}
    assert mr.foreign_libraries(mr.parse_ldd(fixed), RELEASE, shipped, darwin=False) == []


def test_linux_non_system_path_is_foreign(mr: ModuleType) -> None:
    libs = [("libfoo.so.1", "/opt/other/lib/libfoo.so.1")]
    assert mr.foreign_libraries(libs, RELEASE, set(), darwin=False) == [
        "/opt/other/lib/libfoo.so.1"
    ]


def test_macos_rules(mr: ModuleType) -> None:
    otool = (
        "_core.so:\n"
        "\t@rpath/libomp.dylib (compatibility version 5.0.0)\n"
        "\t/usr/lib/libc++.1.dylib (compatibility version 1.0.0)\n"
    )
    shipped = {"libomp.dylib", "libc++.1.dylib"}  # conda ships libc++ too: fine on macOS
    assert mr.foreign_libraries(mr.parse_otool(otool), RELEASE, shipped, darwin=True) == []
    homebrew = otool.replace("@rpath/libomp.dylib", "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    assert mr.foreign_libraries(mr.parse_otool(homebrew), RELEASE, shipped, darwin=True) == [
        "/opt/homebrew/opt/libomp/lib/libomp.dylib"
    ]
