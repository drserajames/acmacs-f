#!/usr/bin/env python3
"""WHO-data gate — mechanical scanner blocking pre-publication WHO CC influenza
data from entering the public repository (copied from ae; used in acmacs-f) (files, staged blobs, commit messages).

Stdlib-only. See tools/WHO-DATA-GATE.md for the full policy and rationale.

WHAT IT CATCHES (conservative — false positives acceptable, false negatives are not):
  * Virus strain names           A/Somewhere/123/2021, B/Place Name/7/17
  * Bare strain names (no A/B)    SOMEWHERE/123/2021 (Capitalised location/number/year)
  * seq_id forms                 SOMEWHERE/123/2021_MDCK1_A1B2C3D4
  * Serum ids                    lab-prefixed ids, and a strain name followed by one
  * Titer table rows             an org/markdown row of >=4 titer-shaped cells
  * Long AA / nucleotide runs    pasted sequence data
  * Amino-acid substitutions     K160T, N145S
  * Clade tokens                 3C.2a1b.2a.2, 5a.1
  * Any entry of an external, PRIVATE current-season strain list (never stored
    in this repo; sourced at runtime from $WHO_STRAIN_LIST or a gitignored file).

HOW MATCHES ARE SUPPRESSED (so the existing, already-public tree passes):
  * subprojects/** (vendored third-party) and binary/data files are not scanned.
  * Hex colour codes (#rrggbb) and 0x.. literals are masked before matching.
  * An allowlist file (plaintext) lists genuine, non-WHO false-positive tokens,
    ignore regexes, and extra skip-path globs.
  * A baseline file grandfathers REVIEWED FILES. One line per file, recording a
    hash of that file's whole matched-token set (no plaintext), a written
    justification and an expiry date. Suppression is scoped to that one path, so
    a token grandfathered in file A still fails in file B; and adding *or*
    removing a token changes the set hash, which re-surfaces the whole file for
    review. Expired or unjustified entries fail under --strict.

The private strain list ALWAYS fails, unconditionally — it is never grandfathered.

Every run prints how much the baseline suppressed. Use --redact when the output
goes somewhere public (CI logs): findings are then reported by location and rule
only, never by value.

WHAT IT EXAMINES, and why that is printed every time:
  * --all          the current tree (git-tracked files)
  * --staged       the INDEX — not the working tree; stage first
  * --range A..B   every commit in the range, one at a time: the blobs that commit
                   touched and its message. A name added in one commit and removed in a
                   later one still ships in the history; the range's net diff would net it
                   out to nothing, and every tip-only mode above cannot see it at all.
  * paths / --message / --message-file   exactly what they say
Every run reports how many files and commit messages it examined, and never prints
"clean" having examined nothing: an empty index, an empty range or a run with no input
is reported as such, and fails under --strict.

Exit codes: 0 = clean, 1 = potential WHO data found, 2 = usage/environment error.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import re
import subprocess
import sys
from fnmatch import fnmatch

# --------------------------------------------------------------------------- #
# Detection rules
# --------------------------------------------------------------------------- #
# An isolate id is usually numeric but labs also use alphanumerics (R10785, 16V-9267,
# NIC-NIH-1721) — note these may start with a LETTER. Requiring a leading digit here was a
# false-negative hole: it hid every letter-initial isolate (RV02127-26, SWL13396, BPH3048742,
# and NIC-NIH-1721 above) from the strain rules, which is why the private list had to carry
# ~600 names that the regexes should have caught by themselves.
# The trailing (?![0-9_A-Za-z-]) — rather than \b — is what lets the year
# be followed by a seq_id passage/hash suffix (…/2021_MDCK1_A1B2C3D4) and still match:
# \b failed there because "_" is a word character, which hid every seq_id from the gate.
_ISOLATE = r"[0-9A-Za-z][0-9A-Za-z_-]{0,19}"
_YEAR = r"(?:19|20)?[0-9]{2}"
_END = r"(?![0-9])"
# Strain names: A/Location/123/2021, B/Some Place/7/17  (2- or 4-digit year)
RE_STRAIN = re.compile(rf"\b[AB]/[A-Za-z][A-Za-z .'_-]*/{_ISOLATE}/{_YEAR}{_END}")
# Bare strain names WITHOUT an A/B prefix: Location/number/year, e.g. EXAMPLETOWN/2570/2019.
# Requires a Capitalised location word (>=3 alpha) so lowercase paths ("results/2/2024")
# don't match; "_" and "-" are allowed because seq_ids write spaces as underscores
# (SOUTH_AFRICA/…, NIIGATA-C/…).
RE_STRAIN_BARE = re.compile(rf"\b[A-Z][A-Za-z][A-Za-z_-]+/{_ISOLATE}/{_YEAR}{_END}")
# Serum ids: WHO CC lab-prefixed forms, e.g. "CDC 2016-045", "CRICK 2019-021".
RE_SERUM_ID = re.compile(
    r"\b(?:CDC|CRICK|NIID|VIDRL|NIMR|MELB|CNIC|NIB)[ _-]?[0-9]{4}-[0-9]{2,3}\b")
# A strain name immediately followed by a short alphanumeric id — the "strain + serum id"
# shape that leaked into TODO.md. Very low false-positive rate, so worth its own rule.
RE_STRAIN_SERUM = re.compile(
    rf"\b[A-Z][A-Za-z][A-Za-z_-]+/{_ISOLATE}/{_YEAR}\s+[A-Z]{{1,3}}[0-9]{{3,6}}[A-Z]?\b")
# Long amino-acid / nucleotide runs — pasted sequence data.
RE_SEQUENCE = re.compile(r"\b(?:[ACDEFGHIKLMNPQRSTVWXY]{60,}|[ACGTNU]{60,})\b")
# Amino-acid substitutions: <AA><pos 2-3 digits><AA>, e.g. K160T
RE_AASUB = re.compile(r"\b[A-Z][0-9]{2,3}[A-Z]\b")
# Clade tokens
RE_CLADE_3C = re.compile(r"\b3C\.[0-9A-Za-z.]+\b")
RE_CLADE_GEN = re.compile(r"\b[0-9][a-z]\.[0-9][0-9A-Za-z.]*\b")

RULES = [
    ("strain", RE_STRAIN),
    ("strain", RE_STRAIN_BARE),
    ("serum-id", RE_SERUM_ID),
    ("serum-id", RE_STRAIN_SERUM),
    ("sequence", RE_SEQUENCE),
    ("aa-sub", RE_AASUB),
    ("clade", RE_CLADE_3C),
    ("clade", RE_CLADE_GEN),
]

# Titer tables are matched per line, not per token: an org/markdown row whose cells are
# mostly titer-shaped ("160", "<20", ">2560", "*"). The matched value is never reported.
RE_TITER_CELL = re.compile(r"^\s*(?:[<>]?[0-9]{1,5}|\*)\s*$")


def titer_row(line: str) -> bool:
    cells = [c for c in line.split("|")[1:-1]] if line.lstrip().startswith("|") else []
    titers = [c for c in cells if RE_TITER_CELL.match(c)]
    return len(titers) >= 4

# Masked (turned to spaces) before rules run — removes the commonest false positives.
RE_HEX_COLOUR = re.compile(r"#[0-9A-Fa-f]{3,8}\b")
RE_HEX_LITERAL = re.compile(r"0[xX][0-9A-Fa-f]+")

# Never scanned (binary / compressed / image / built artefacts) — content is opaque.
SKIP_EXT = {
    ".xz", ".gz", ".bz2", ".br", ".zip", ".ace", ".png", ".jpg", ".jpeg",
    ".gif", ".ico", ".pdf", ".so", ".o", ".a", ".dylib", ".bin", ".woff",
    ".woff2", ".ttf", ".otf", ".wraplock", ".DS_Store",
}
# Built-in path skips (vendored third-party). Extendable via the allowlist file.
DEFAULT_SKIP_PATHS = ["subprojects/**"]

HASH_LEN = 16  # truncated sha256 hex chars stored in the baseline
BASELINE_MONTHS = 12  # how long a --update-baseline entry stays valid
UNJUSTIFIED = "UNJUSTIFIED — review required"


def token_hash(tok: str) -> str:
    norm = tok.strip().upper()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:HASH_LEN]


def token_set_hash(tokens) -> str:
    """Hash of a file's WHOLE matched-token set. Adding or removing any token changes
    it, which re-surfaces the file for review. Stores no plaintext."""
    joined = "\n".join(sorted({t.strip().upper() for t in tokens}))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:HASH_LEN]


def today() -> datetime.date:
    return datetime.date.today()


def default_expiry() -> str:
    d = today()
    year, month = divmod(d.month - 1 + BASELINE_MONTHS, 12)
    return d.replace(year=d.year + year, month=month + 1, day=1).isoformat()


class BaselineEntry:
    def __init__(self, path, set_hash, count, expires, justification):
        self.path = path
        self.set_hash = set_hash
        self.count = count
        self.expires = expires
        self.justification = justification

    @property
    def expired(self) -> bool:
        try:
            return datetime.date.fromisoformat(self.expires) < today()
        except ValueError:
            return True

    @property
    def unjustified(self) -> bool:
        return not self.justification or self.justification.startswith("UNJUSTIFIED")

    def line(self) -> str:
        return (f"{self.path}\t{self.set_hash}\t{self.count}\t{self.expires}"
                f"\t# {self.justification}\n")


# --------------------------------------------------------------------------- #
# Allowlist + baseline loading
# --------------------------------------------------------------------------- #
class Config:
    def __init__(self):
        self.allow_literal: set[str] = set()   # uppercased exact tokens
        self.allow_regex: list[re.Pattern] = []
        self.skip_paths: list[str] = list(DEFAULT_SKIP_PATHS)
        self.baseline: dict[str, BaselineEntry] = {}   # relpath -> entry
        self.private_list: list[str] = []      # lowercased private strain names
        self.suppressed: dict[str, int] = {}   # relpath -> matches suppressed this run

    def allowed(self, tok: str) -> bool:
        """Allowlist only. Baseline suppression is per-file and handled in scan_text."""
        up = tok.strip().upper()
        if up in self.allow_literal:
            return True
        for rx in self.allow_regex:
            if rx.fullmatch(tok):
                return True
        return False

    def baselined(self, source: str, tokens) -> bool:
        """True if `source` is a reviewed, unexpired baseline entry whose recorded
        token set still matches exactly what was found."""
        entry = self.baseline.get(source)
        return entry is not None and entry.set_hash == token_set_hash(tokens)

    def path_skipped(self, relpath: str) -> bool:
        for pat in self.skip_paths:
            if fnmatch(relpath, pat) or fnmatch(relpath, pat.rstrip("*/") + "/*"):
                return True
        return False


def load_allowlist(path: str, cfg: Config) -> None:
    if not path or not os.path.exists(path):
        return
    section = "allow-literal"
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip().lower()
                continue
            if section == "allow-literal":
                cfg.allow_literal.add(line.upper())
            elif section == "allow-regex":
                try:
                    cfg.allow_regex.append(re.compile(line))
                except re.error as exc:
                    print(f"who-data-gate: bad allow-regex {line!r}: {exc}", file=sys.stderr)
            elif section == "skip-path":
                cfg.skip_paths.append(line)


def load_baseline(path: str, cfg: Config) -> None:
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            fields, _, justification = raw.rstrip("\n").partition("#")
            parts = fields.split("\t")
            if len(parts) < 4:
                print(f"who-data-gate: {path}:{lineno}: malformed baseline entry "
                      "(expected path\\thash\\tcount\\texpiry\\t# justification)",
                      file=sys.stderr)
                continue
            rel, set_hash, count, expires = (p.strip() for p in parts[:4])
            cfg.baseline[rel] = BaselineEntry(
                rel, set_hash.lower(), count, expires, justification.strip())


# Private-list matching is normalised: seq_ids write spaces as underscores
# (GUANGXI JIANGZHOU/... vs GUANGXI_JIANGZHOU/...), so a raw substring test on the
# space form silently MISSES the underscore form — a false negative, which is the one
# failure class this gate exists to prevent. Collapse both sides to single spaces.
def norm_private(s: str) -> str:
    return re.sub(r"[\s_]+", " ", s.lower())


# Entries shorter than this are rejected: a very short name substring-matches
# ordinary words and would fail the gate everywhere.
PRIVATE_MIN_LEN = 8


def load_private_list(cfg: Config, root: str) -> str | None:
    """Return the source path used, or None if no private list was found."""
    candidates = []
    env = os.environ.get("WHO_STRAIN_LIST")
    if env:
        candidates.append(env)
    # The list lives in the private acmacs-data repo. Look for it there directly, so the
    # check still runs when the environment is not set up -- git hooks do NOT inherit a
    # shell that sourced ae-env.sh, and a silently regex-only gate is the failure this
    # whole file exists to prevent. Mirrors ae-env.sh's own $ACMACS_DATA / sibling logic.
    data_dir = os.environ.get("ACMACS_DATA") or os.path.join(root, os.pardir, "acmacs-data")
    candidates.append(os.path.join(data_dir, ".who-strain-list"))
    # gitignored local fallbacks (legacy; keeping the list inside this public repo is
    # discouraged -- see WHO-DATA-GATE.md)
    candidates.append(os.path.join(root, ".who-strain-list"))
    candidates.append(os.path.join(root, ".who-strain-list.txt"))
    for cand in candidates:
        cand = os.path.normpath(cand) if cand else cand
        if cand and os.path.exists(cand):
            too_short = []
            with open(cand, "r", encoding="utf-8") as fh:
                for raw in fh:
                    name = raw.strip()
                    if not name or name.startswith("#"):
                        continue
                    if len(name) < PRIVATE_MIN_LEN:
                        too_short.append(name)
                        continue
                    cfg.private_list.append(norm_private(name))
            cfg.private_list = sorted(set(cfg.private_list))
            if too_short:
                print(f"who-data-gate: WARNING — {len(too_short)} private-list "
                      f"entr{'y' if len(too_short) == 1 else 'ies'} shorter than "
                      f"{PRIVATE_MIN_LEN} chars ignored (too generic to match safely)",
                      file=sys.stderr)
            return cand
    return None


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #
REDACT = False  # set from --redact; when on, no matched value is ever printed


class Finding:
    def __init__(self, source, lineno, rule, token, snippet):
        self.source = source
        self.lineno = lineno
        self.rule = rule
        self.token = token
        self.snippet = snippet

    def __str__(self):
        loc = f"{self.source}:{self.lineno}" if self.lineno else self.source
        if REDACT:
            return f"  {loc}: [{self.rule}] <redacted sha:{token_hash(self.token)}>"
        return f"  {loc}: [{self.rule}] {self.token!r}  in: {self.snippet.strip()[:120]}"


def matched_tokens(text: str):
    """Yield (lineno, rule, token, line) for every rule match, before suppression."""
    for i, line in enumerate(text.splitlines(), 1):
        masked = RE_HEX_LITERAL.sub(" ", RE_HEX_COLOUR.sub(" ", line))
        for rule, rx in RULES:
            for m in rx.finditer(masked):
                yield i, rule, m.group(0), line
        if titer_row(line):
            # never surface the row itself — the location is enough to act on
            yield i, "titer-row", f"<titer-row:{token_hash(line)}>", "<redacted>"


def scan_text(text: str, source: str, cfg: Config, collect_tokens=None) -> list[Finding]:
    """Scan text. If collect_tokens is a set, add every matched token to it
    (baseline-generation mode) and do NOT apply allowlist/baseline suppression."""
    findings: list[Finding] = []
    candidates = []   # (lineno, rule, token, line) surviving the allowlist
    for i, rule, tok, line in matched_tokens(text):
        if collect_tokens is not None:
            collect_tokens.add(tok)
            continue
        if cfg.allowed(tok):
            continue
        candidates.append((i, rule, tok, line))

    if collect_tokens is None:
        if cfg.baselined(source, (t for _, _, t, _ in candidates)):
            # accumulate: --range can scan the same path once per commit
            cfg.suppressed[source] = cfg.suppressed.get(source, 0) + len(candidates)
        else:
            findings += [Finding(source, i, r, t, l) for i, r, t, l in candidates]

    # Private strain list — unconditional, case-insensitive substring, not maskable,
    # and NEVER suppressed by the allowlist or the baseline.
    if collect_tokens is None and cfg.private_list:
        lines = text.splitlines()
        norm_lines = [norm_private(l) for l in lines]
        whole = norm_private(text)
        for i, (line, nl) in enumerate(zip(lines, norm_lines), 1):
            for name in cfg.private_list:
                if name in nl:
                    findings.append(Finding(source, i, "private-list", name, line))
        # also catch names split across the whole blob but not a single line (rare)
        for name in cfg.private_list:
            if name in whole and not any(name in nl for nl in norm_lines):
                findings.append(Finding(source, 0, "private-list", name, "(multi-line)"))
    return findings


def is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def read_text(path: str) -> str | None:
    try:
        data = open(path, "rb").read()
    except OSError:
        return None
    if is_binary(data):
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("latin-1")
        except UnicodeDecodeError:
            return None


# --------------------------------------------------------------------------- #
# Git helpers
# --------------------------------------------------------------------------- #
def git(root, *args) -> str:
    return subprocess.check_output(["git", "-C", root, *args], text=True)


def git_root(start: str) -> str:
    try:
        return git(start, "rev-parse", "--show-toplevel").strip()
    except subprocess.CalledProcessError:
        return os.path.abspath(start)


def tracked_files(root: str) -> list[str]:
    return [f for f in git(root, "ls-files").splitlines() if f]


def staged_files(root: str) -> list[str]:
    out = git(root, "diff", "--cached", "--name-only", "--diff-filter=ACM")
    return [f for f in out.splitlines() if f]


def decode_blob(data: bytes) -> str | None:
    if is_binary(data):
        return None
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def staged_blob(root: str, relpath: str) -> str | None:
    try:
        data = subprocess.check_output(["git", "-C", root, "show", f":{relpath}"])
    except subprocess.CalledProcessError:
        return None
    return decode_blob(data)


class RangeCommit:
    def __init__(self, sha: str, parents: list[str], message: str):
        self.sha = sha
        self.parents = parents
        self.message = message
        self.tip = False   # no other commit in the range has this one as a parent

    @property
    def short(self) -> str:
        return self.sha[:12]


def range_commits(root: str, rev_range: str) -> list[RangeCommit]:
    """Every commit in `rev_range`, oldest first. Raises CalledProcessError on a bad range."""
    listing = git(root, "rev-list", "--topo-order", "--reverse", "--parents", rev_range, "--")
    commits = []
    for line in listing.splitlines():
        sha, *parents = line.split()
        message = git(root, "log", "-1", "--format=%B", sha)
        commits.append(RangeCommit(sha, parents, message))
    in_range_parents = {p for c in commits for p in c.parents}
    for c in commits:
        c.tip = c.sha not in in_range_parents
    return commits


def commit_files(root: str, commit: RangeCommit) -> list[str]:
    """The paths a commit touched, with the same A/C/M filter --staged uses. For a merge,
    the combined diff (-c): only paths whose merged content differs from EVERY parent,
    i.e. content the merge itself introduced. What it brought in from a parent is scanned
    at the commit that introduced it — which is in the range whenever it is not already
    reachable from the range's base."""
    if len(commit.parents) > 1:
        out = git(root, "diff-tree", "-r", "-c", "-z", "--no-commit-id", "--name-only",
                  commit.sha)
    else:
        out = git(root, "diff-tree", "-r", "--root", "-z", "--no-commit-id", "--name-only",
                  "--diff-filter=ACM", commit.sha)
    return [f for f in out.split("\0") if f]


def commit_blob(root: str, sha: str, relpath: str) -> str | None:
    """Mirror of staged_blob, reading from a commit instead of the index. None for a path
    the commit deleted (possible in a merge's combined diff) and for binary content."""
    try:
        data = subprocess.check_output(["git", "-C", root, "cat-file", "blob", f"{sha}:{relpath}"],
                                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    return decode_blob(data)


# --------------------------------------------------------------------------- #
# Baseline generation
# --------------------------------------------------------------------------- #
BASELINE_HEADER = """\
# who-data-gate baseline — REVIEWED grandfathered files.
#
# One line per file:   <path>\\t<token-set-hash>\\t<n>\\t<expires>\\t# <justification>
#
# The hash covers that file's WHOLE set of matched tokens (sha256 of the sorted,
# uppercased set, truncated) — no strain/serum/clade plaintext is ever stored here.
# Suppression is scoped to the one path, so a token grandfathered here still fails
# anywhere else; and adding OR removing a token changes the hash, which re-surfaces
# the whole file for review. Entries expire, and --strict fails on expired or
# unjustified ones, so nothing stays grandfathered by inertia.
#
# THE BASELINE IS THE DANGEROUS PART OF THIS TOOL. Every line is a standing decision
# that some real-looking data may live in the repo. Justify each one in words, and
# prefer fixing the data or using [allow-literal]/[allow-regex] for true false
# positives. Never add an entry to silence a hit you have not read.
#
# Regenerate: tools/who-data-gate.py --update-baseline --justification "..."
#   (existing justifications and expiry dates are carried over per path)
"""


def update_baseline(root: str, baseline_path: str, cfg: Config, justification: str) -> None:
    previous = dict(cfg.baseline)
    entries: list[BaselineEntry] = []
    for rel in tracked_files(root):
        if os.path.splitext(rel)[1] in SKIP_EXT or cfg.path_skipped(rel):
            continue
        text = read_text(os.path.join(root, rel))
        if text is None:
            continue
        tokens = [t for _, _, t, _ in matched_tokens(text) if not cfg.allowed(t)]
        if not tokens:
            continue
        old = previous.get(rel)
        entries.append(BaselineEntry(
            path=rel,
            set_hash=token_set_hash(tokens),
            count=str(len(tokens)),
            expires=(old.expires if old and not old.expired else default_expiry()),
            justification=(justification or (old.justification if old else "") or UNJUSTIFIED),
        ))
    entries.sort(key=lambda e: e.path)
    with open(baseline_path, "w", encoding="utf-8") as fh:
        fh.write(BASELINE_HEADER)
        fh.write(f"# {len(entries)} grandfathered file(s), "
                 f"{sum(int(e.count) for e in entries)} suppressed match(es).\n")
        for e in entries:
            fh.write(e.line())
    unjustified = sum(1 for e in entries if e.unjustified)
    print(f"who-data-gate: wrote {len(entries)} baseline entries to {baseline_path}"
          + (f" — {unjustified} still UNJUSTIFIED" if unjustified else ""))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WHO-data gate scanner")
    ap.add_argument("paths", nargs="*", help="explicit files/dirs to scan")
    ap.add_argument("--all", action="store_true", help="scan all git-tracked files (CI)")
    ap.add_argument("--staged", action="store_true",
                    help="scan git staged blobs (pre-commit) — the INDEX, so stage first")
    ap.add_argument("--range", metavar="REV-RANGE",
                    help="scan every commit in a revision range (A..B, A...B): the blobs "
                         "each commit touched and each commit's message (pre-push)")
    ap.add_argument("--message-file", help="also scan the commit message in this file")
    ap.add_argument("--message", help="also scan this literal message text")
    ap.add_argument("--root", help="repo root (default: git toplevel of cwd)")
    ap.add_argument("--allowlist", help="allowlist file")
    ap.add_argument("--baseline", help="baseline file")
    ap.add_argument("--update-baseline", action="store_true",
                    help="regenerate the baseline from the current tree and exit")
    ap.add_argument("--justification",
                    help="with --update-baseline: written reason for grandfathering")
    ap.add_argument("--strict", action="store_true",
                    help="fail if no private strain list is configured, or if any "
                         "baseline entry is expired or unjustified (use in CI/hooks)")
    ap.add_argument("--redact", action="store_true",
                    help="report findings by location only — never print a matched "
                         "value (use whenever output goes somewhere public, e.g. CI logs)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    global REDACT
    REDACT = args.redact

    root = git_root(args.root or ".")
    tools_dir = os.path.join(root, "tools")
    allowlist_path = args.allowlist or os.path.join(tools_dir, "who-data-gate-allowlist.txt")
    baseline_path = args.baseline or os.path.join(tools_dir, "who-data-gate-baseline.txt")

    cfg = Config()
    load_allowlist(allowlist_path, cfg)
    load_baseline(baseline_path, cfg)

    if args.update_baseline:
        update_baseline(root, baseline_path, cfg, args.justification)
        return 0

    hard_errors: list[str] = []

    priv_src = load_private_list(cfg, root)
    if priv_src:
        if not args.quiet:
            print(f"who-data-gate: private strain list loaded ({len(cfg.private_list)} names) "
                  f"from {priv_src}", file=sys.stderr)
    else:
        msg = ("no private strain list found ($WHO_STRAIN_LIST or .who-strain-list) — "
               "the high-signal check is OFF and only the regex rules are running")
        if args.strict:
            hard_errors.append(msg)
        else:
            print(f"who-data-gate: WARNING — {msg}.\n"
                  "  Point $WHO_STRAIN_LIST at the current-season name list "
                  "(see tools/WHO-DATA-GATE.md).", file=sys.stderr)

    expired = sorted(e.path for e in cfg.baseline.values() if e.expired)
    unjustified = sorted(e.path for e in cfg.baseline.values() if e.unjustified)
    for label, paths in (("expired", expired), ("unjustified", unjustified)):
        if not paths:
            continue
        msg = (f"{len(paths)} baseline entr{'y' if len(paths) == 1 else 'ies'} "
               f"{label}: {', '.join(paths[:5])}"
               + (" …" if len(paths) > 5 else ""))
        if args.strict:
            hard_errors.append(msg)
        elif not args.quiet:
            print(f"who-data-gate: WARNING — {msg}", file=sys.stderr)

    findings: list[Finding] = []
    # What was actually examined. "clean" is only ever printed over a non-zero count: the
    # gate has twice passed while reading nothing (an empty index; a tip-only scan of work
    # whose history carried the data), and the output could not tell that from a pass.
    examined = {"files": 0, "messages": 0, "skipped": 0}
    empty_inputs: list[str] = []   # requested inputs that yielded nothing to scan

    def scan_blob(text: str | None, rel: str) -> list[Finding]:
        if text is None:           # binary, undecodable or unreadable
            examined["skipped"] += 1
            return []
        examined["files"] += 1
        return scan_text(text, rel, cfg)

    def excluded(rel: str) -> bool:
        if os.path.splitext(rel)[1] in SKIP_EXT or cfg.path_skipped(rel):
            examined["skipped"] += 1
            return True
        return False

    if args.all:
        before = examined["files"]
        for rel in tracked_files(root):
            if not excluded(rel):
                findings += scan_blob(read_text(os.path.join(root, rel)), rel)
        if examined["files"] == before:
            empty_inputs.append("--all examined 0 files (no scannable tracked files)")
    if args.staged:
        before = examined["files"]
        staged = staged_files(root)
        for rel in staged:
            if not excluded(rel):
                findings += scan_blob(staged_blob(root, rel), rel)
        if not staged:
            empty_inputs.append("nothing staged — examined 0 files. --staged reads the INDEX, "
                                "not the working tree: `git add` first")
        elif examined["files"] == before:
            empty_inputs.append(f"--staged examined 0 files ({len(staged)} staged, all binary "
                                "or skip-listed)")

    commits: list[RangeCommit] = []
    hits_by_commit: dict[str, int] = {}
    if args.range is not None:
        if not args.range or args.range.startswith("-"):
            print(f"who-data-gate: --range {args.range!r} is not a revision range", file=sys.stderr)
            return 2
        try:
            commits = range_commits(root, args.range)
        except subprocess.CalledProcessError:
            print(f"who-data-gate: --range {args.range!r}: git cannot resolve it "
                  "(see git's message above)", file=sys.stderr)
            return 2
        if not commits:
            empty_inputs.append(f"--range {args.range} contains 0 commits — examined nothing")
        # Per commit, never the range's net diff: a name added in one commit and removed in
        # a later one nets out to nothing, yet still ships in the history.
        for c in commits:
            before = len(findings)
            for rel in commit_files(root, c):
                if excluded(rel):
                    continue
                for f in scan_blob(commit_blob(root, c.sha, rel), rel):
                    f.source = f"{c.short}:{rel}"   # rev:path — pasteable into `git show`
                    findings.append(f)
            examined["messages"] += 1
            for f in scan_text(c.message, "<commit-message>", cfg):
                f.source = f"{c.short}:<commit-message>"
                findings.append(f)
            if len(findings) > before:
                hits_by_commit[c.sha] = len(findings) - before

    if args.paths:
        before = examined["files"]
        for p in args.paths:
            if os.path.isdir(p):
                for dirpath, _dirs, names in os.walk(p):
                    for n in names:
                        fp = os.path.join(dirpath, n)
                        rel = os.path.relpath(fp, root)
                        if not excluded(rel):
                            findings += scan_blob(read_text(fp), rel)
            elif os.path.isfile(p):
                rel = os.path.relpath(p, root)
                if not excluded(rel):
                    findings += scan_blob(read_text(p), rel)
            else:
                print(f"who-data-gate: no such path: {p}", file=sys.stderr)
        if examined["files"] == before:
            empty_inputs.append("the given paths examined 0 files")

    if args.message_file:
        try:
            msg = open(args.message_file, "r", encoding="utf-8", errors="replace").read()
            # git commit-msg passes the whole file incl. comment lines starting with '#';
            # strip them so scissored/commented text isn't scanned.
            msg = "\n".join(l for l in msg.splitlines() if not l.lstrip().startswith("#"))
            findings += scan_text(msg, "<commit-message>", cfg)
            examined["messages"] += 1
        except OSError as exc:
            print(f"who-data-gate: cannot read message file: {exc}", file=sys.stderr)
            return 2
    if args.message:
        findings += scan_text(args.message, "<commit-message>", cfg)
        examined["messages"] += 1

    if not (args.all or args.staged or args.range is not None or args.paths
            or args.message_file or args.message):
        empty_inputs.append("no input given (--all, --staged, --range, paths or a message) "
                            "— examined nothing")

    summary = (f"examined {examined['files']} file(s) and {examined['messages']} "
               "commit message(s)")
    if args.range is not None:
        summary += f" across {len(commits)} commit(s) in {args.range}"
    if examined["skipped"]:
        summary += f"; {examined['skipped']} binary or skip-listed file(s) not scanned"

    # Loud whatever --quiet says: this is the line that distinguishes a pass from a run
    # that looked at nothing.
    for msg in empty_inputs:
        print(f"who-data-gate: WARNING — {msg}", file=sys.stderr)
        if args.strict:
            hard_errors.append(msg)

    # Always report what the baseline hid — a silent suppression is how real data
    # sat in TODO.md unnoticed. Counts only; never the values.
    if cfg.suppressed and not args.quiet:
        total = sum(cfg.suppressed.values())
        print(f"who-data-gate: baseline SUPPRESSED {total} match(es) in "
              f"{len(cfg.suppressed)} reviewed file(s):", file=sys.stderr)
        for rel in sorted(cfg.suppressed, key=lambda r: -cfg.suppressed[r]):
            entry = cfg.baseline[rel]
            print(f"    {cfg.suppressed[rel]:5d}  {rel}  (expires {entry.expires}) "
                  f"— {entry.justification}", file=sys.stderr)

    if findings:
        print("\nWHO-DATA GATE: potential WHO influenza data detected "
              f"({len(findings)} hit(s)) — commit/push BLOCKED:\n", file=sys.stderr)
        for f in findings:
            print(str(f), file=sys.stderr)
        print(f"\n  ({summary})", file=sys.stderr)
        if hits_by_commit:
            # Tip and history hits need different remedies, so name which each one is.
            print(f"\nHits by commit in {args.range}, oldest first:", file=sys.stderr)
            for c in commits:
                if c.sha in hits_by_commit:
                    print(f"  {c.short} [{'tip' if c.tip else 'history'}]  "
                          f"{hits_by_commit[c.sha]} hit(s)", file=sys.stderr)
            if any(not c.tip for c in commits if c.sha in hits_by_commit):
                print("  [history] — NOT the newest commit on its line. A later commit that "
                      "removes the data does\n"
                      "      not remove it from the history that ships: rewrite that commit "
                      "(interactive rebase)\n"
                      "      or redo the work on a fresh branch.", file=sys.stderr)
            if any(c.tip for c in commits if c.sha in hits_by_commit):
                print("  [tip] — the newest commit on its line: `git commit --amend` removes it, "
                      "provided no\n"
                      "      [history] commit above carries it too.", file=sys.stderr)
        print("\nIf this is a genuine false positive, extend the allowlist "
              "(tools/who-data-gate-allowlist.txt) — see tools/WHO-DATA-GATE.md.\n"
              "Do NOT bypass the gate to commit real WHO data.", file=sys.stderr)
        return 1

    if hard_errors:
        print("\nWHO-DATA GATE: --strict preconditions not met — BLOCKED:\n",
              file=sys.stderr)
        for msg in hard_errors:
            print(f"  {msg}", file=sys.stderr)
        print("\nSee tools/WHO-DATA-GATE.md.", file=sys.stderr)
        return 1

    if empty_inputs:
        # Not "clean": at least one requested input had nothing in it.
        if not args.quiet:
            print(f"who-data-gate: no findings, but NOT a pass — see WARNING above ({summary}).",
                  file=sys.stderr)
        return 0
    if not args.quiet:
        print(f"who-data-gate: clean — {summary}.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
