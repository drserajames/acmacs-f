"""Passage histories: parse a lab's passage string into steps and write it canonically.

A passage is segments separated by "/", each a run of steps "<name><count>", e.g. CDC's
"S1C4/C2" (SIAT 1, MDCK 4, then MDCK 2). Step names come from the ``passage_tokens`` rule
table; a count of "X" (unknown) is written "?".

The canonical form is ae's, so that af's strings read like the ones people know:
``C1S1/S1`` -> ``MDCK1SIAT1/SIAT1``; a "/" is written between two consecutive steps of the
same name (``E3E9`` -> ``E3/E9``). A string that does not parse completely is kept upper-cased
and reported: a passage is part of an antigen's identity, so it is never guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .rules import RuleTable

CLASSES = ("egg", "cell", "original")  # in precedence order; anything else is "unknown"


def class_of(classes: list[str]) -> str:
    """One class for a whole passage history (eu-e0's rule, used to choose among a name's
    sequences): "egg" if any step is egg, else "cell" if any is cell, else "original" if
    any is an original specimen, else "unknown" (no passage, or one that could not be read)."""
    return next((c for c in CLASSES if c in classes), "unknown")


@dataclass
class Passage:
    text: str  # canonical
    classes: list[str] = field(default_factory=list)  # class of each step, in order
    problems: list[str] = field(default_factory=list)

    @property
    def is_egg(self) -> bool:
        return "egg" in self.classes


class PassageParser:
    def __init__(self, tokens: RuleTable, lab: str):
        self.tokens = tokens
        self.lab = lab
        in_scope = [r for r in tokens.rules if tokens.in_scope(r, lab=lab)]
        self.canonical_classes = {r["canonical"].upper(): r["class"] for r in in_scope}
        self.canonical_step = re.compile(
            "("
            + "|".join(map(re.escape, sorted(self.canonical_classes, key=len, reverse=True)))
            + r")(\d+|\?)?(\([^()]*\))?",
            re.IGNORECASE,
        )
        names = sorted(
            {r["pattern"] for r in tokens.rules if tokens.in_scope(r, lab=lab)},
            key=len,
            reverse=True,
        )
        self.step = re.compile("(" + "|".join(map(re.escape, names)) + r")(\d+|X)?", re.IGNORECASE)

    def passage_class(self, canonical: str) -> str:
        """The class (see ``class_of``) of a passage already written canonically. Readers
        call this on the text they store, so every lab's tables get the same answer from the
        same rules, whichever way the text was assembled."""
        classes = []
        for segment in canonical.split("/"):
            pos = 0
            while pos < len(segment):
                m = self.canonical_step.match(segment, pos)
                if m is None:
                    return "unknown"  # kept as written because it did not parse
                classes.append(self.canonical_classes[m[1].upper()])
                pos = m.end()
        return class_of(classes)

    def parse(self, raw: str) -> Passage:
        text = raw.strip()
        if not text:
            return Passage("")
        segments, classes = [], []
        for segment in text.split("/"):
            steps, pos = [], 0
            while pos < len(segment):
                m = self.step.match(segment, pos)
                if m is None:
                    return Passage(
                        text.upper(), problems=[f"passage {raw!r}: cannot read {segment[pos:]!r}"]
                    )
                rule = self.tokens.find(m[1], lab=self.lab)
                assert rule is not None  # the alternation is built from the same rules
                count = m[2] or ""
                steps.append((rule["canonical"], "?" if count.upper() == "X" else count))
                classes.append(rule["class"])
                pos = m.end()
            if not steps:
                return Passage(text.upper(), problems=[f"passage {raw!r}: empty segment"])
            out = steps[0][0] + steps[0][1]
            for (prev, _), (name, count) in zip(steps, steps[1:], strict=False):
                out += ("/" if name == prev else "") + name + count
            segments.append(out)
        return Passage("/".join(segments), classes=classes)
