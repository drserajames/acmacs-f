"""Titres: parsing, logs, and the lispmds layer-merge rules.

A titre is kept as the string that appears in the `.ace` file ("40", "<10", ">1280",
"~80", "*"). Everything derived from it (type, value, log) is computed here so there is
exactly one place that knows the grammar.

The merge rules reproduce ae `cc/chart/v3/titers.cc:584-676` (lispmds 2014-12-06 plus
acmacs' handling of `>`), because the chains must replay today's science. Where ae's
behaviour is a parameter (SD limit, SD denominator) it is a parameter here too.
"""

from __future__ import annotations

import enum
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


class TitreType(enum.IntEnum):
    """Numeric codes are interface I1's `titre_type`."""

    MISSING = 0
    REGULAR = 1
    LESS_THAN = 2
    MORE_THAN = 3
    DODGY = 4


_TITRE_RE = re.compile(r"^(?P<prefix>[<>~]?)(?P<value>[0-9]+)$")

DONT_CARE = "*"


class TitreError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Titre:
    """One parsed titre. `value` is the number after the prefix (0 for missing)."""

    type: TitreType
    value: int

    @classmethod
    def parse(cls, text: str) -> Titre:
        if text == DONT_CARE or text == "":
            return MISSING_TITRE
        m = _TITRE_RE.match(text)
        if not m or int(m["value"]) <= 0:
            raise TitreError(f"invalid titre {text!r}")
        prefix = m["prefix"]
        kind = {
            "": TitreType.REGULAR,
            "<": TitreType.LESS_THAN,
            ">": TitreType.MORE_THAN,
            "~": TitreType.DODGY,
        }[prefix]
        return cls(kind, int(m["value"]))

    def __str__(self) -> str:
        if self.type == TitreType.MISSING:
            return DONT_CARE
        prefix = {
            TitreType.REGULAR: "",
            TitreType.LESS_THAN: "<",
            TitreType.MORE_THAN: ">",
            TitreType.DODGY: "~",
        }[self.type]
        return f"{prefix}{self.value}"

    @property
    def is_missing(self) -> bool:
        return self.type == TitreType.MISSING

    def logged(self) -> float:
        """log2(value/10); the threshold prefix is ignored (ae `Titer::logged`)."""
        if self.is_missing:
            return math.nan
        return math.log2(self.value / 10.0)

    def logged_with_thresholded(self) -> float:
        """`<X` one dilution down, `>X` one up: used for the merge SD and mean."""
        if self.type == TitreType.LESS_THAN:
            return self.logged() - 1.0
        if self.type == TitreType.MORE_THAN:
            return self.logged() + 1.0
        return self.logged()

    def logged_for_column_bases(self) -> float | None:
        """Contribution to a serum's column basis; None when the titre does not count.

        Regular and `<` count at their log, `>` one dilution higher; dodgy and missing
        do not count (ae `titers.cc:117-133`).
        """
        if self.type in (TitreType.REGULAR, TitreType.LESS_THAN):
            return self.logged()
        if self.type == TitreType.MORE_THAN:
            return self.logged() + 1.0
        return None


MISSING_TITRE = Titre(TitreType.MISSING, 0)


def from_logged(logged: float, prefix: str = "") -> Titre:
    """ae `Titer::from_logged`: lround(2^logged * 10), so 40 and 80 merge to 57."""
    value = _lround(2.0**logged * 10.0)
    return Titre.parse(f"{prefix}{value}")


def _lround(x: float) -> int:
    """C `lround`: halves round away from zero (Python's round() is banker's)."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


# ----------------------------------------------------------------------
# layer merge


class MoreThanOnly(enum.Enum):
    """What a cell holding only `>` titres becomes (ae `more_than_thresholded`).

    `TO_DONT_CARE` is the table the map is made from; `ADJUST_TO_NEXT` keeps `>X` and is
    used only to compute column bases (see `merged_column_bases`).
    """

    TO_DONT_CARE = "to-dont-care"
    ADJUST_TO_NEXT = "adjust-to-next"


class MergeOutcome(enum.Enum):
    """Why a merged cell got its value (ae `titer_merge`); counted in the merge report."""

    ALL_DONT_CARE = "all-dont-care"
    LESS_AND_MORE_THAN = "less-and-more-than"
    LESS_THAN_ONLY = "less-than-only"
    MORE_THAN_ONLY_ADJUST_TO_NEXT = "more-than-only-adjust-to-next"
    MORE_THAN_ONLY_TO_DONT_CARE = "more-than-only-to-dont-care"
    SD_TOO_BIG = "sd-too-big"
    REGULAR_ONLY = "regular-only"
    MAX_LESS_THAN_BIGGER_THAN_MAX_REGULAR = "max-less-than-bigger-than-max-regular"
    LESS_THAN_AND_REGULAR = "less-than-and-regular"
    MIN_MORE_THAN_LESS_THAN_MIN_REGULAR = "min-more-than-less-than-min-regular"
    MORE_THAN_AND_REGULAR = "more-than-and-regular"


@dataclass(frozen=True)
class MergeSettings:
    sd_limit: float = 1.0  # NaN disables the SD rule
    population_sd: bool = True  # ae/AD default; False = sample SD (Racmacs)


def merge_titres(
    titres: Sequence[Titre],
    more_than: MoreThanOnly = MoreThanOnly.TO_DONT_CARE,
    settings: MergeSettings | None = None,
) -> tuple[Titre, MergeOutcome]:
    """Merge one cell's titres from several layers (missing titres already removed).

    The numbered rules are lispmds' (quoted in ae `titers.cc:584-596`):
    1. `<` and `>` both present -> `*`;  2. nothing -> `*`;
    3. only thresholded -> min `<` / max `>`;  4-5. SD of logs (thresholded moved one
    step) above the limit -> `*`;  6. only regular -> mean of logs;
    7-9. mixed regular and thresholded -> a thresholded value just beyond the regulars.
    """
    settings = settings or MergeSettings()
    titres = [t for t in titres if not t.is_missing]
    if any(t.type == TitreType.DODGY for t in titres):
        # ae throws here too; a dodgy titre in a layer means the input needs a decision.
        raise TitreError("cannot merge dodgy (~) titres")
    if not titres:
        return MISSING_TITRE, MergeOutcome.ALL_DONT_CARE

    regular = [t.value for t in titres if t.type == TitreType.REGULAR]
    less = [t.value for t in titres if t.type == TitreType.LESS_THAN]
    more = [t.value for t in titres if t.type == TitreType.MORE_THAN]

    if less and more:
        return MISSING_TITRE, MergeOutcome.LESS_AND_MORE_THAN
    if not regular:
        if less:
            return Titre(TitreType.LESS_THAN, min(less)), MergeOutcome.LESS_THAN_ONLY
        if more_than == MoreThanOnly.ADJUST_TO_NEXT:
            return Titre(TitreType.MORE_THAN, max(more)), MergeOutcome.MORE_THAN_ONLY_ADJUST_TO_NEXT
        return MISSING_TITRE, MergeOutcome.MORE_THAN_ONLY_TO_DONT_CARE

    logs = [t.logged_with_thresholded() for t in titres]
    mean, sd = _mean_sd(logs, settings.population_sd)
    if sd > settings.sd_limit:  # NaN limit: comparison is False, rule skipped (as in ae)
        return MISSING_TITRE, MergeOutcome.SD_TOO_BIG
    if not less and not more:
        return from_logged(mean), MergeOutcome.REGULAR_ONLY

    max_regular, min_regular = max(regular), min(regular)
    if less:
        if max(less) > max_regular:
            result = min(v for v in less if v > max_regular)
            return Titre(
                TitreType.LESS_THAN, result
            ), MergeOutcome.MAX_LESS_THAN_BIGGER_THAN_MAX_REGULAR
        return Titre(TitreType.LESS_THAN, max_regular * 2), MergeOutcome.LESS_THAN_AND_REGULAR
    if min(more) < min_regular:
        result = max(v for v in more if v < min_regular)
        return Titre(TitreType.MORE_THAN, result), MergeOutcome.MIN_MORE_THAN_LESS_THAN_MIN_REGULAR
    return Titre(TitreType.MORE_THAN, min_regular // 2), MergeOutcome.MORE_THAN_AND_REGULAR


def _mean_sd(values: Sequence[float], population: bool) -> tuple[float, float]:
    """Mean and SD computed as ae `utils/statistics.hh` does (sum of squared deviations)."""
    n = len(values)
    mean = sum(values) / n
    var_n = sum((v - mean) ** 2 for v in values)
    if population:
        return mean, math.sqrt(var_n / n)
    return mean, (math.sqrt(var_n / (n - 1)) if n > 1 else math.nan)


def column_basis(titres: Iterable[Titre]) -> float:
    """Raw column basis of one serum: the maximum `logged_for_column_bases`, floored at 0.

    ae starts the maximum at 0.0 (`titers.cc:681-690`), so a serum whose best titre is
    `<10` has column basis 0, not -inf.
    """
    best = 0.0
    for t in titres:
        v = t.logged_for_column_bases()
        if v is not None and v > best:
            best = v
    return best
