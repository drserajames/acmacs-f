"""How much two rendered pages differ, in pixels: the one measure the side-by-side documents print.

Sarah, 30 Oct 2026: "Keep the original page sizes in side by side and add automatic pixel diff
metric". Agreed with 00 the same day as the single definition for every side-by-side page:

- each page is rasterised at its own size at :data:`DPI`, RGB, by poppler's ``pdftoppm`` with its
  default antialiasing; nothing is cropped or rescaled;
- pages of different sizes are aligned top-left and the smaller is padded white to the union
  canvas, and both sizes are reported so a padded comparison is never silent;
- a pixel differs when any channel differs by more than :data:`THRESHOLD` of 255. Reported as a
  count and as a share of the canvas (a share of a padded canvas does not compare between pages
  of different sizes; the count does), with the mean absolute difference over every channel.

The rasteriser's version is part of the measure (two poppler builds can rasterise one PDF
differently), so it is reported with every result (:func:`poppler_version`).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DPI = 100
THRESHOLD = 32  # of 255, on any channel
PDFTOPPM = "pdftoppm"


class PixelError(RuntimeError):
    """A page could not be rasterised."""


@dataclass(frozen=True)
class PixelDiff:
    """Two pages compared at :data:`DPI`. Sizes are (width, height) in pixels."""

    differing: int
    canvas: int
    mean_abs: float
    size_a: tuple[int, int]
    size_b: tuple[int, int]

    @property
    def share(self) -> float:
        return self.differing / self.canvas if self.canvas else 0.0

    def text(self) -> str:
        sizes = (
            f"{self.size_a[0]} x {self.size_a[1]} px"
            if self.size_a == self.size_b
            else f"{self.size_a[0]} x {self.size_a[1]} and {self.size_b[0]} x {self.size_b[1]} px,"
            " aligned top-left and padded white"
        )
        return (
            f"{self.differing:,} pixels differ ({100 * self.share:.2f}% of {sizes}), "
            f"mean |diff| {self.mean_abs:.2f} of 255"
        )


def tool() -> str:
    path = shutil.which(PDFTOPPM)
    if path is None:
        raise PixelError(f"{PDFTOPPM} (poppler) is not installed: the pixel measure needs it")
    return path


def poppler_version() -> str:
    """The rasteriser's version, as ``pdftoppm -v`` prints it ("26.04.0")."""
    out = subprocess.run([tool(), "-v"], capture_output=True, text=True)
    match = re.search(r"version\s+(\S+)", out.stderr + out.stdout)
    if not match:
        raise PixelError(f"{PDFTOPPM} -v printed no version")
    return match.group(1)


def read_ppm(path: Path) -> np.ndarray:
    """A binary RGB PPM (P6, 8 bit) as a (height, width, 3) array."""
    data = path.read_bytes()
    fields: list[bytes] = []
    pos = 0
    while len(fields) < 4:
        while data[pos : pos + 1].isspace():
            pos += 1
        if data[pos : pos + 1] == b"#":  # a comment runs to the end of its line
            pos = data.index(b"\n", pos) + 1
            continue
        end = pos
        while not data[end : end + 1].isspace():
            end += 1
        fields.append(data[pos:end])
        pos = end
    if fields[0] != b"P6" or int(fields[3]) != 255:
        raise PixelError(f"{path}: not an 8-bit binary RGB PPM")
    width, height = int(fields[1]), int(fields[2])
    pixels = np.frombuffer(data, dtype=np.uint8, count=width * height * 3, offset=pos + 1)
    return pixels.reshape(height, width, 3)


def rasterise(pdf: Path, page: int, dpi: int = DPI) -> np.ndarray:
    """One page of ``pdf`` as RGB at ``dpi``. Written to a file, never to stdout: ``pdftoppm -``
    can write nothing and exit 0, and an empty raster would compare as a match."""
    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "page"
        result = subprocess.run(
            [tool(), "-f", str(page), "-l", str(page), "-r", str(dpi), "-singlefile",
             str(pdf), str(stem)],
            capture_output=True, text=True,
        )  # fmt: skip
        out = stem.with_suffix(".ppm")
        if result.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
            raise PixelError(f"{pdf} page {page}: {PDFTOPPM} failed: {result.stderr.strip()}")
        return read_ppm(out).copy()


def compare_arrays(a: np.ndarray, b: np.ndarray, threshold: int = THRESHOLD) -> PixelDiff:
    """Two rasters, aligned top-left on the union canvas with white padding."""
    height, width = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])

    def padded(x: np.ndarray) -> np.ndarray:
        canvas = np.full((height, width, 3), 255, dtype=np.int16)
        canvas[: x.shape[0], : x.shape[1]] = x
        return canvas

    diff = np.abs(padded(a) - padded(b))
    return PixelDiff(
        differing=int((diff.max(axis=2) > threshold).sum()),
        canvas=height * width,
        mean_abs=float(diff.mean()),
        size_a=(a.shape[1], a.shape[0]),
        size_b=(b.shape[1], b.shape[0]),
    )


def pixel_diff(pdf_a: Path, page_a: int, pdf_b: Path, page_b: int) -> PixelDiff:
    """Page ``page_a`` of ``pdf_a`` against page ``page_b`` of ``pdf_b`` (pages from 1)."""
    return compare_arrays(rasterise(pdf_a, page_a), rasterise(pdf_b, page_b))
