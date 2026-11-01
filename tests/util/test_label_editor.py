"""The shared label editor page (af.util.label_editor): the figures' adapters are tested with them
(tests/tree/draw/test_label_editor.py)."""

import base64
import json
import re
from pathlib import Path

import matplotlib
import pytest

pytest.importorskip("numpy", reason="numpy not installed")

from af.util.label_editor import embed_font, write_editor  # noqa: E402

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

TTF = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
ADAPTER = "const ADAPTER = {}; // invented"


def test_the_page_carries_its_data_picture_scripts_and_starts_the_adapter(tmp_path):
    fig = plt.figure(figsize=(1, 1))
    ax = fig.add_axes((0, 0, 1, 1))
    (line,) = ax.plot([0, 1], [0, 1], linewidth=0.01)  # far under a pixel
    data = {"title": "TEST <map>: labels", "save_as": "labels-t.toml", "note": "</script>"}
    path = tmp_path / "sub" / "f.labels.html"
    write_editor(path, fig, data, ADAPTER, dpi=72)
    html = path.read_text()
    assert "<title>TEST &lt;map&gt;: labels</title>" in html
    found = re.search(r"<script type='application/json' id='data'>(.*?)</script>", html, re.S)
    assert found and json.loads(found.group(1).replace("<\\/", "</")) == data
    png = re.search(r"data:image/png;base64,([^']*)'", html)
    assert png and base64.b64decode(png.group(1)).startswith(b"\x89PNG")
    # the common page, then the adapter, then the start: the adapter may use the page's helpers
    common, adapter, start = (
        html.index(s) for s in ("function startLabelEditor", ADAPTER, "startLabelEditor(ADAPTER);")
    )
    assert common < adapter < start
    assert line.get_linewidth() == 1.0  # drawn a pixel wide in the picture
    plt.close(fig)


def test_a_single_font_file_is_embedded_with_its_family():
    font = embed_font(TTF / "DejaVuSans.ttf")
    assert font["family"] == "DejaVu Sans" and font["format"] == "truetype"
    assert base64.b64decode(font["data"] or "") == (TTF / "DejaVuSans.ttf").read_bytes()
