"""A round's place abbreviations for vaccine labels: a named, hashed table (style 08-27)."""

from pathlib import Path

import pytest

from af.map.labels import LabelAbbreviations, label_text
from af.map.roundconfig import load_maps_config

from .test_roundconfig import GOOD, write

TABLE = "# invented places, for the test\nlocation\tabbreviation\nNEWPLACE\tNP\nOTHER PLACE\tOP\n"


def name(place: str) -> str:
    return "/".join(("A(H3N2)", place, "1", "2021"))  # invented; joined for the gate


def test_the_table_abbreviates_its_places_and_the_rule_the_rest(tmp_path: Path) -> None:
    (tmp_path / "abbr.tsv").write_text(TABLE)
    table = LabelAbbreviations.read(tmp_path / "abbr.tsv")
    assert table.table == {"NEWPLACE": "NP", "OTHER PLACE": "OP"} and len(table.sha256) == 64
    assert label_text(name("NEWPLACE"), "cell", "", table.table) == "NP/21-cell"
    assert label_text(name("NEWTOWN"), "cell", "", table.table) == "Ne/21-cell"
    assert table.input() == {"path": str(tmp_path / "abbr.tsv"), "sha256": table.sha256}


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("place\tabbr\nX\tY\n", "header"),
        ("location\tabbreviation\nX\n", "two non-empty cells"),
        ("location\tabbreviation\nX\tY\nX\tZ\n", "two rows"),
    ],
)
def test_a_bad_table_is_refused(tmp_path: Path, text: str, match: str) -> None:
    (tmp_path / "abbr.tsv").write_text(text)
    with pytest.raises(ValueError, match=match):
        LabelAbbreviations.read(tmp_path / "abbr.tsv")


def test_the_round_names_its_table(tmp_path: Path) -> None:
    text = GOOD.replace("[defaults]", 'label_abbreviations = "abbr.tsv"\n\n[defaults]', 1)
    config, _ = load_maps_config(write(tmp_path, text))
    assert config.label_abbreviations == (tmp_path / "abbr.tsv").resolve()
    assert load_maps_config(write(tmp_path, GOOD))[0].label_abbreviations is None
