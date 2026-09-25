# acmacs-f: guide for agents

acmacs-f (import `af`) builds phylogenetic trees and antigenic maps for influenza vaccine strain
selection, from raw inputs, through long-lived stores, into reports. Licence GPL-3.0-or-later.
**This repository is public.**

## Layout

| Path | What |
|---|---|
| `af/util` | config loading, artefact checks, provenance |
| `af/run` | local and SLURM runners for external tools |
| `af/store` | store layout (`raw/`, `sequences/`, `clades/`, `tables/`, `trees/`, `chains/`, `serology/`, `snapshots/`), versions, cache, report manifests |
| `af/pipeline` | step driver with incremental skipping |
| `af/seq`, `af/tables`, `af/clades` | sequences, titre tables, clade assignment |
| `af/tree` (`build`, `asr`, `draw`) | trees |
| `af/chart`, `af/chain`, `af/map` | charts, incremental chains, map optimisation and finishing |
| `af/serology`, `af/geo`, `af/stat` | serology store, geographic maps, stat counts |
| `af/report` | reports built from the stores |
| `cpp/` | C++ optimiser (`cpp/optimiser/`), alglib fetched by `cpp/CMakeLists.txt` |
| `tests/` | mirrors `af/` |
| `tools/` | the WHO-data gate (MIT, copied from ae) |

## Build and test

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
```

On macOS use `/opt/homebrew/bin/python3`. `/usr/local/bin/python3` and `timeout` are x86_64
there, and running under them silently builds for the wrong architecture.

The build is scikit-build-core + CMake. With no `cpp/optimiser/` it is pure Python. With it,
`cpp/CMakeLists.txt` fetches alglib 3.19.0 (hash-checked). Offline, pass
`-Ccmake.define.AF_ALGLIB_URL=/path/to/alglib-3.19.0.cpp.gpl.tgz`.

## Design rules

1. A rule or selector that **matches nothing is an error** unless marked optional; report counts.
2. **Never select by index.** Use EPI_ISL ids, designations, clades.
3. **Every step checks its artefacts** (exists, non-empty, parses, content hash) and propagates
   failure. No exit 0 after a failure.
4. **Missing inputs are fatal.** No silent defaults, no environment-variable fallbacks. Paths
   come from explicit config. The one exception is `AF_DATA`, and only in tests.
5. **Provenance and content hashes** on everything derived; recompute only what changed.
6. **One editable copy of each fact.**
7. **Parse, don't sort strings**; no date logic that depends on today's date.
8. **Deterministic when seeded.**
9. **Automate curation first**; hand overrides are named data, counted and reported.
10. **General and config-driven**: no site-specific assumptions or fixed lab lists in code.

Plain Python with type hints, short functions, and docstrings that say *why*. C++ only in the
optimiser core.

## Public code, private data

- Never commit WHO CC or GISAID data: no real strain names, serum ids, titres, sequences or clade
  assignments of real viruses, in code, tests, docstrings or commit messages.
- Real test data lives in the private data repo, found through the `af_data` fixture
  (`$AF_DATA`, default `../acmacs-f-data`). Those tests **skip** when it is absent.
  Synthetic fixtures are generated in test code with invented names.
- `.gitignore` blocks binary data formats, which the gate cannot read. Keep it that way.
- Generated stores live outside git, at a path set in config.

## The WHO-data gate

Install once per clone: `git config core.hooksPath .githooks`. It scans staged contents, commit
messages and pushed ranges. **Never bypass it with `--no-verify`.** If it blocks you, the data
is in the wrong place. See [tools/WHO-DATA-GATE.md](tools/WHO-DATA-GATE.md).

## Git

Work on a branch, never directly on `master`. Merging and pushing are the maintainer's call.
