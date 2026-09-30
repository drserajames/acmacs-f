"""Read today's ``semantic_clades.py`` read-only, and report what does not survive.

While ae is still in production the colour and grouping tables are edited only in
``acmacs-data`` (one editable copy of each fact), so af reads them from there at run time
(:func:`read_user_clades`, used by maps and geo alike) rather than keeping a converted
copy that would silently go stale, and says which rows cannot be carried across and why.
They move into af's own tables once, at switch-over.

Two things are worth carrying over and two are not:

* **Carried:** the sub-groupings (``attributes``) and the colour schemes
  (``clades``, ``clades-vN``), which are the user's own choices.
* **Not carried:** rows that match nothing today. 24 attribute rows name a clade
  ``clades.json`` no longer defines, and four colour rows name neither a clade nor an
  attribute (INVENTORY E §2.3). They are inert: the labels never appear, and nobody is
  told. Rather than copy that silence forward, the importer lists every such row with its
  table and line so the user can decide to re-anchor or drop it.

The old file is a Python module wrapping Org-mode tables, and importing it needs ae on
the path. Rather than depend on ae, :func:`read_semantic_clades` runs the module with a
stand-in for ``ae.utils.org`` that parses the tables here, compiling the file itself so no
bytecode cache is read or written. The module's own Python structure is then whatever
Python says it is — no regular expression over the source, which is how three other
readers of this file each broke on a layout change (trap T20).
"""

from __future__ import annotations

import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from af.clades.colours import ColourScheme, scheme_from_rows
from af.clades.groups import SUBSTITUTION, Group, GroupSet
from af.clades.nomenclature import CladeSet
from af.store import ExternalInput

#: The old file's subtype keys, and af's names for them.
SUBTYPE_KEYS = {"A(H1N1)": "A(H1N1)", "A(H3N2)": "A(H3N2)", "BV": "B/Vic", "BY": "B/Yam"}


class ImportError_(ValueError):
    """The old file could not be read at all."""


@dataclass
class DeadRow:
    """A row that can never match anything, and the reason it cannot."""

    subtype: str
    table: str
    row: int
    name: str
    reason: str

    def __str__(self) -> str:
        return f"{self.subtype} {self.table}[{self.row}] {self.name!r}: {self.reason}"


@dataclass
class ImportReport:
    """What came across, what did not, and why.

    Three outcomes, deliberately distinguished. A row is **carried** when its clade exists
    upstream; it **needs a local definition** when the name is one the old system defined
    but the nomenclature does not (a legacy or locally invented clade, which af keeps in
    its local layer); and it is **dead** when nothing anywhere defines the name, which is
    the 24 rows that have been labelling nothing at all.
    """

    groups: dict[str, GroupSet] = field(default_factory=dict)
    colour_rows: dict[tuple[str, str], list[dict[str, str]]] = field(default_factory=dict)
    dead: list[DeadRow] = field(default_factory=list)
    needs_local: list[DeadRow] = field(default_factory=list)
    renamed: list[tuple[str, str, str]] = field(default_factory=list)
    legacy: list[tuple[str, str, int, str]] = field(default_factory=list)
    """Colour rows keyed by an older upstream clade (subtype, table, row, name): legacy entries."""
    repeated: list[DeadRow] = field(default_factory=list)
    """Rows that repeat an earlier row of the same name exactly: kept once, listed here."""

    def summary(self) -> str:
        lines = [
            f"groups: {sum(len(group_set) for group_set in self.groups.values())} "
            f"across {len(self.groups)} subtypes",
            f"colour schemes: {len(self.colour_rows)}",
            f"rows needing a local definition: {len(self.needs_local)}",
            f"dead rows: {len(self.dead)}",
            f"repeated rows (kept once): {len(self.repeated)}",
        ]
        lines.extend(f"  {dead}" for dead in self.dead)
        return "\n".join(lines)


def org_table_to_dict(data: str) -> list[dict[str, str]]:
    """Parse one Org-mode table into rows, as ae's own reader does.

    Only the first table is read, and only its named columns; ae's version also coerces
    some columns to int or bool, which af does not want (it is what made a year column an
    int and a "current vaccine" list never match, trap T12).
    """
    rows: list[dict[str, str]] = []
    names: list[str] = []
    for line in data.split("\n"):
        if line[:2] == "|-":
            continue
        if line[:1] == "|":
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if not names:
                names = cells
            else:
                rows.append(
                    {key: cells[index] for index, key in enumerate(names) if index < len(cells)}
                )
        elif names:
            break
    return rows


def read_semantic_clades(path: Path) -> Mapping[str, Mapping[str, list[dict[str, str]]]]:
    """Import the old module with a stand-in for ``ae.utils.org`` and return its data."""
    path = Path(path)
    if not path.is_file():
        raise ImportError_(f"semantic_clades.py not found: {path}")
    stand_in = types.ModuleType("ae.utils.org")
    stand_in.org_table_to_dict = org_table_to_dict  # type: ignore[attr-defined]
    injected = {
        "ae": types.ModuleType("ae"),
        "ae.utils": types.ModuleType("ae.utils"),
        "ae.utils.org": stand_in,
    }
    saved = {name: sys.modules.get(name) for name in injected}
    sys.modules.update(injected)
    try:
        # compiled from the file's bytes, not imported: the import system would run a cached
        # .pyc keyed on mtime and size, so a same-length edit within one second is silently
        # ignored while the content hash says the new file was read. It would also write
        # __pycache__ into a directory af must only read.
        module = types.ModuleType("_af_semantic_clades")
        module.__file__ = str(path)
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    data = getattr(module, "sData", None)
    if not isinstance(data, dict):
        raise ImportError_(f"{path} has no sData dictionary")
    return data


def import_semantic_clades(
    path: Path,
    clade_sets: Mapping[str, CladeSet],
    *,
    legacy_names: Mapping[str, Mapping[str, str]] | None = None,
    defined_locally: Mapping[str, set[str]] | None = None,
) -> ImportReport:
    """Convert the old tables, reporting every row that cannot be carried across.

    ``clade_sets`` are af's clade sets, keyed by af subtype name. Old clade names resolve
    through upstream's own aliases (:meth:`~af.clades.nomenclature.CladeSet.legacy_aliases`),
    each resolution listed in ``report.renamed``; ``legacy_names`` adds to them, per
    subtype. The old file writes a clade as its current name with the superseded one in
    brackets, and names some clades that were since renamed, so without a mapping every
    such row would be reported dead when it is really just spelled differently.

    ``defined_locally`` is the set of clade names the *old* system defines (the entry
    names of ``clades.json``), per subtype. Without it every legacy and locally invented
    clade looks dead, which overstates the problem by a factor of seven; with it, those
    rows are reported separately as needing a local definition.
    """
    report = ImportReport()
    data = read_semantic_clades(path)
    for old_key, subtype in SUBTYPE_KEYS.items():
        tables = data.get(old_key)
        if tables is None:
            continue
        clade_set = clade_sets.get(subtype)
        # upstream's own aliases first (V1A.3 is A.3 under its old name), then any the
        # caller adds; without them a row under an old name looks dead or local when it
        # names a live subclade
        mapping = clade_set.legacy_aliases() if clade_set is not None else {}
        mapping.update((legacy_names or {}).get(subtype, {}))
        local_names = set((defined_locally or {}).get(subtype, ()))
        groups: list[Group] = []
        by_name: dict[str, Group] = {}
        for index, row in enumerate(tables.get("attributes", []), start=1):
            name = (row.get("name") or "").strip()
            if not name:
                continue
            anchor_raw = (row.get("clade") or "").strip()
            anchor = _resolve(
                anchor_raw, mapping, clade_set, report, subtype, index, name, local=local_names
            )
            if isinstance(anchor, _Unresolved):
                continue
            substitutions, bad = _substitutions(row.get("aa") or "")
            if bad:
                report.dead.append(
                    DeadRow(subtype, "attributes", index, name, f"unreadable substitutions: {bad}")
                )
                continue
            if not substitutions:
                report.dead.append(DeadRow(subtype, "attributes", index, name, "no substitutions"))
                continue
            if anchor_raw and anchor != anchor_raw:
                report.renamed.append((subtype, anchor_raw, str(anchor)))
            group = Group(subtype, name, anchor, substitutions, row.get("note", ""), index)
            first = by_name.get(name)
            if first is not None:
                # the old file repeats some rows verbatim; a repeat is harmless and kept once,
                # but two different definitions under one name is a real conflict
                if (first.anchor, first.substitutions) == (group.anchor, group.substitutions):
                    report.repeated.append(
                        DeadRow(
                            subtype, "attributes", index, name, f"repeats row {first.source_line}"
                        )
                    )
                else:
                    report.dead.append(
                        DeadRow(
                            subtype,
                            "attributes",
                            index,
                            name,
                            f"redefines the group of row {first.source_line} differently",
                        )
                    )
                continue
            by_name[name] = group
            groups.append(group)
        if groups:
            report.groups[subtype] = GroupSet(subtype, tuple(groups))
        imported_groups = {group.name for group in groups}
        # every attribute name, including rows that did not survive: a colour row naming
        # one of those is not dead, it is waiting on the same decision as its group
        all_group_names = {
            (row.get("name") or "").strip()
            for row in tables.get("attributes", [])
            if (row.get("name") or "").strip()
        }
        for table, rows in tables.items():
            if table == "attributes":
                continue
            kept: list[dict[str, str]] = []
            for index, row in enumerate(rows, start=1):
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                colour = (row.get("color") or row.get("colour") or "").strip().lower()
                legend = (row.get("legend") or name).strip()
                if name in imported_groups:
                    kept.append(
                        {"order": str(index), "key": name, "legend": legend, "colour": colour}
                    )
                    continue
                if name in all_group_names:
                    report.needs_local.append(
                        DeadRow(
                            subtype,
                            table,
                            index,
                            name,
                            "colours a group whose own row needs a local definition",
                        )
                    )
                    continue
                resolved = _resolve(
                    name,
                    mapping,
                    clade_set,
                    report,
                    subtype,
                    index,
                    name,
                    table,
                    local=local_names,
                    allow_legacy=True,
                )
                if isinstance(resolved, _Unresolved):
                    continue
                if resolved != name:
                    report.renamed.append((subtype, name, str(resolved)))
                kept.append(
                    {
                        "order": str(index),
                        "key": str(resolved),
                        "legend": legend,
                        "colour": colour,
                    }
                )
            if kept:
                report.colour_rows[(subtype, table)] = kept
    return report


class _Unresolved:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unresolved>"


_UNRESOLVED = _Unresolved()


def _resolve(
    raw: str,
    mapping: Mapping[str, str],
    clade_set: CladeSet | None,
    report: ImportReport,
    subtype: str,
    index: int,
    name: str,
    table: str = "attributes",
    local: set[str] | None = None,
    allow_legacy: bool = False,
) -> str | None | _Unresolved:
    """An old clade name as af's name, or ``_UNRESOLVED`` (recorded as a dead row).

    An empty name means "no anchor", which is a valid group defined by substitutions
    alone, so it returns None rather than failing.
    """
    if not raw:
        return None
    candidate = mapping.get(raw, raw)
    if clade_set is None:
        return candidate
    if candidate in clade_set:
        return candidate
    # the old file writes "<current name> (<superseded name>)"; af uses the part before
    # the bracket
    bare = candidate.split(" (")[0].strip()
    if bare in clade_set:
        return bare
    if allow_legacy:
        # colour rows may name an older upstream clade that defines itself (af.clades.legacy),
        # by its full name only: short names ("1", "5b") are ambiguous to a reader
        defining = set(clade_set.legacy_defining())
        if bare in defining:
            report.legacy.append((subtype, table, index, bare))
            return bare
        by_short = {clade_set.legacy_clades[n].short_name: n for n in defining}
        if bare in by_short and bare not in defining:
            report.dead.append(
                DeadRow(
                    subtype,
                    table,
                    index,
                    name,
                    f"names legacy clade {by_short[bare]!r} by its short name {bare!r}; "
                    "use the full name",
                )
            )
            return _UNRESOLVED
    if local and (candidate in local or bare in local):
        report.needs_local.append(
            DeadRow(
                subtype,
                table,
                index,
                name,
                f"names {raw!r}, which the old system defines but the nomenclature does not: "
                "needs a local definition or re-anchoring",
            )
        )
    else:
        report.dead.append(
            DeadRow(
                subtype,
                table,
                index,
                name,
                f"names clade {raw!r}, which nothing defines — it has been matching nothing",
            )
        )
    return _UNRESOLVED


def _substitutions(text: str) -> tuple[tuple, str]:
    """Parse the ``aa`` column; return the substitutions and any unreadable tokens."""
    from af.clades.groups import Substitution

    good: list[Substitution] = []
    bad: list[str] = []
    for token in text.split():
        matched = SUBSTITUTION.fullmatch(token)
        if matched is None:
            bad.append(token)
        else:
            good.append(Substitution(int(matched.group("position")), matched.group("state")))
    return tuple(good), " ".join(bad)


def write_groups(report: ImportReport, path: Path) -> int:
    """Write the imported groups as ``groups.tsv``; returns the number of rows."""
    path = Path(path)
    lines = ["\t".join(("subtype", "group", "anchor", "substitutions", "note"))]
    count = 0
    for subtype in sorted(report.groups):
        for group in report.groups[subtype]:
            lines.append(
                "\t".join(
                    (
                        subtype,
                        group.name,
                        group.anchor or "",
                        " ".join(str(substitution) for substitution in group.substitutions),
                        group.note or "",
                    )
                )
            )
            count += 1
    path.write_text("\n".join(lines) + "\n")
    return count


def write_colour_schemes(report: ImportReport, directory: Path) -> dict[str, int]:
    """Write each imported colour table as ``<subtype>/<scheme>.tsv``."""
    directory = Path(directory)
    written: dict[str, int] = {}
    for (subtype, table), rows in sorted(report.colour_rows.items()):
        target = directory / _directory_name(subtype)
        target.mkdir(parents=True, exist_ok=True)
        lines = ["\t".join(("order", "key", "legend", "colour"))]
        lines.extend(
            "\t".join((row["order"], row["key"], row["legend"], row["colour"])) for row in rows
        )
        file = target / f"{table}.tsv"
        file.write_text("\n".join(lines) + "\n")
        written[str(file.relative_to(directory))] = len(rows)
    return written


def _directory_name(subtype: str) -> str:
    return {"A(H1N1)": "h1", "A(H3N2)": "h3", "B/Vic": "bvic", "B/Yam": "byam"}.get(
        subtype, subtype
    )


def clades_json_names(path: Path) -> dict[str, set[str]]:
    """Entry names per subtype from the old ``clades.json``, for ``defined_locally``."""
    import json

    with Path(path).open() as stream:
        data = json.load(stream)
    return {
        subtype: {entry["N"] for entry in entries if isinstance(entry, dict) and entry.get("N")}
        for subtype, entries in data.items()
        if isinstance(entries, list)
    }


def clades_json_signatures(path: Path) -> dict[str, dict[str, tuple[str, ...]]]:
    """Each old ``clades.json`` entry's signature, per subtype, in ``local.tsv`` notation.

    Amino acids as ``145S``, nucleotides as ``nuc1230A``. This is how carried-over local
    clades get their mutations while ``acmacs-data`` is still the one editable copy
    (:func:`af.clades.local.from_signatures`).
    """
    import json

    with Path(path).open() as stream:
        data = json.load(stream)
    signatures: dict[str, dict[str, tuple[str, ...]]] = {}
    for subtype, entries in data.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("N"):
                continue
            tokens = (entry.get("aa") or "").split()
            tokens += [f"nuc{token}" for token in (entry.get("nuc") or "").split()]
            known = signatures.setdefault(subtype, {})
            if entry["N"] in known and known[entry["N"]] != tuple(tokens):
                raise ImportError_(
                    f"{path}: {subtype} defines {entry['N']!r} twice with different signatures"
                )
            known[entry["N"]] = tuple(tokens)
    return signatures


def dead_row_report(dead: Sequence[DeadRow]) -> str:
    """The dead rows grouped by subtype and table, for a human to act on."""
    if not dead:
        return "no dead rows"
    lines = [f"{len(dead)} row(s) match nothing and were not imported:"]
    for subtype in sorted({row.subtype for row in dead}):
        for table in sorted({row.table for row in dead if row.subtype == subtype}):
            rows = [row for row in dead if row.subtype == subtype and row.table == table]
            lines.append(f"  {subtype} {table}: {len(rows)}")
            lines.extend(f"    line {row.row}: {row.name!r} — {row.reason}" for row in rows)
    return "\n".join(lines)


# ---------------------------------------------------------------- run-time reader


class UserCladesError(ValueError):
    """The user's tables cannot be read, or a caller asked for something they do not have."""


#: Where the user's tables live inside the acmacs-data directory.
SEMANTIC_CLADES = "semantic_clades.py"
CLADES_JSON = "clades.json"


@dataclass(frozen=True)
class UserClades:
    """The user's colour schemes and groups, read from acmacs-data on this run.

    ``inputs`` are the two files' content hashes, for the provenance of whatever is
    coloured with them: a figure drawn after the user edits a scheme must not look
    up-to-date against one drawn before. ``report`` keeps the rows that did not come
    across (dead, needing a local definition, repeated), so a caller can count them.
    """

    schemes: Mapping[str, Mapping[str, ColourScheme]]
    groups: Mapping[str, GroupSet]
    report: ImportReport
    inputs: tuple[ExternalInput, ...]

    def scheme(self, subtype: str, name: str) -> ColourScheme:
        """One scheme; an unknown subtype or scheme name is an error, never an empty scheme."""
        if subtype not in self.schemes:
            raise UserCladesError(
                f"no colour schemes for {subtype!r}: it has no clade set here "
                f"(read for: {', '.join(sorted(self.schemes)) or 'none'})"
            )
        schemes = self.schemes[subtype]
        if name not in schemes:
            raise UserCladesError(
                f"no colour scheme {name!r} for {subtype} in {self.inputs[0].path} "
                f"(there are: {', '.join(sorted(schemes)) or 'none'})"
            )
        return schemes[name]

    def group_set(self, subtype: str) -> GroupSet | None:
        """The subtype's groups, or None when it has none (a scheme may use clades only)."""
        if subtype not in self.schemes:
            raise UserCladesError(f"no groups for {subtype!r}: it has no clade set here")
        return self.groups.get(subtype)


def read_user_clades(
    acmacs_data: Path,
    clade_sets: Mapping[str, CladeSet],
    *,
    legacy_names: Mapping[str, Mapping[str, str]] | None = None,
) -> UserClades:
    """Read the user's schemes and groups from ``acmacs_data`` now, for these clade sets.

    ``acmacs_data`` is the directory (from config; there is no default). Only the subtypes
    in ``clade_sets`` are read: a subtype without a clade set (B/Yamagata, by design) has
    no schemes, and asking for one fails. Every scheme is checked exactly as a scheme file
    would be (:func:`af.clades.colours.scheme_from_rows`), so a key naming no clade or
    group is an error here, not a colour nobody sees. Rows that name nothing are not
    errors: they are the user's to decide on, and are listed in ``report``.

    Both files are hashed before and after parsing; a file edited mid-read is refused
    rather than recorded under a hash that does not describe what was parsed.
    """
    directory = Path(acmacs_data)
    if not directory.is_dir():
        raise UserCladesError(f"acmacs-data directory not found: {directory}")
    paths = (directory / SEMANTIC_CLADES, directory / CLADES_JSON)
    for path in paths:
        if not path.is_file():
            raise UserCladesError(f"{path.name} not found in {directory}")
    before = tuple(ExternalInput.of(path) for path in paths)
    report = import_semantic_clades(
        paths[0],
        clade_sets,
        legacy_names=legacy_names,
        defined_locally=clades_json_names(paths[1]),
    )
    after = tuple(ExternalInput.of(path) for path in paths)
    if before != after:
        changed = [b.path.name for b, a in zip(before, after, strict=True) if b != a]
        raise UserCladesError(f"changed while being read: {', '.join(changed)}; run again")
    schemes: dict[str, dict[str, ColourScheme]] = {subtype: {} for subtype in clade_sets}
    for (subtype, table), rows in sorted(report.colour_rows.items()):
        if subtype not in clade_sets:
            continue
        schemes[subtype][table] = scheme_from_rows(
            [(f"{subtype} {table} row {row['order']}", row) for row in rows],
            subtype,
            table,
            clade_sets[subtype],
            report.groups.get(subtype),
            paths[0],
        )
    groups = {subtype: group for subtype, group in report.groups.items() if subtype in clade_sets}
    return UserClades(schemes, groups, report, before)
