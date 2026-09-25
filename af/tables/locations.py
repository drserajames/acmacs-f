"""Chinese location names in lab tables, romanised the way ae romanised them.

CNIC writes locations in Chinese (上海奉贤). ae romanised them through two sources, and af
reads the same two, in this order, never transliterating:

1. a lab's alias file (whocc-tables ``cnic-location-aliases.tsv``: Chinese, romanised,
   evidence), for names whose romanisation was established by matching isolates;
2. acmacs-data locationdb ``replacements`` whose key contains Chinese characters:
   the value is the spelling to write (上海奉贤 -> SHANGHAI FENGXIAN);
3. locationdb ``names`` with Chinese keys point at a location *key*, not a spelling
   (江苏梁溪 -> WUXI), so ae kept the Chinese; af does the same and warns
   (Q29 asks whether to write the key instead).

Anything else is an error: a Chinese location nobody has mapped must be looked at, not
guessed (江苏梁溪 is *not* JIANGSU LIYANG, though three isolates agree on number and date).
Both files are shared facts that stay where they are (DECISIONS: acmacs-data is read through
an importer during the transition); their paths come from config.
"""

from __future__ import annotations

import json
import lzma
import re
from dataclasses import dataclass
from pathlib import Path

CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


@dataclass(frozen=True)
class Romanised:
    text: str  # what to write in the name
    source: str  # "aliases", "locdb replacement", "locdb name (kept)"


class ChineseLocations:
    def __init__(self, aliases: dict[str, str], replacements: dict[str, str], names: set[str]):
        self.aliases = aliases
        self.replacements = replacements
        self.names = names

    @classmethod
    def load(cls, locdb: Path, aliases_tsv: Path) -> ChineseLocations:
        db = json.loads(lzma.decompress(locdb.read_bytes()))
        replacements = {k: v for k, v in db["replacements"].items() if CJK.search(k)}
        names = {k for k in db["names"] if CJK.search(k)}
        aliases = {}
        for no, line in enumerate(aliases_tsv.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 2 or not fields[0] or not fields[1]:
                raise ValueError(f"{aliases_tsv}:{no}: expected Chinese<TAB>romanised")
            aliases[fields[0].strip()] = fields[1].strip()
        return cls(aliases, replacements, names)

    def romanise(self, location: str) -> Romanised | None:
        """None when the location has no Chinese characters (nothing to do)."""
        key = location.strip()
        if not CJK.search(key):
            return None
        if key in self.aliases:
            return Romanised(self.aliases[key], "aliases")
        if key in self.replacements:
            return Romanised(self.replacements[key], "locdb replacement")
        if key in self.names:
            return Romanised(key, "locdb name (kept)")
        raise LookupError(f"Chinese location {key!r} is in no alias file and not in locationdb")
