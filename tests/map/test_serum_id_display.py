"""Serum ids as printed on maps and report pages (Q123)."""

import pytest

from af.map.vaccines import display_serum_id
from af.tables.labs import Lab


def test_serum_ids_print_as_the_lab_wrote_them() -> None:
    """Q123: stored "<LAB> <id>", printed bare for a "bare" lab, as stored for "with-lab"."""

    bare = Lab("LABA", "Lab A", "none yet", "bare")
    with_lab = Lab("LABB", "Lab B", "none yet", "with-lab")
    assert display_serum_id("LABA F0227", bare) == "F0227"
    assert display_serum_id("LABA LABB1", bare) == "LABB1"  # names another lab: as written
    assert display_serum_id("LABB 2023-004", with_lab) == "LABB 2023-004"
    assert display_serum_id("F0227", bare) == "F0227"  # no prefix: unchanged
    assert display_serum_id("LABA ", bare) == "LABA "  # nothing after the token: unchanged
    with pytest.raises(ValueError, match="unknown serum_id_print"):
        display_serum_id("LABA F1", Lab("LABA", "Lab A", "none yet", "short"))
