# acmacs-f

Phylogenetic trees and antigenic maps for influenza vaccine strain selection.

acmacs-f (Python package `af`) is the successor to the acmacs-d (AD) and acmacs-e (ae)
toolkits. It builds the trees and antigenic maps used in WHO influenza vaccine composition
reports, and runs the same way on a laptop and on a SLURM cluster.

- Distribution name: `acmacs-f`. Import name: `af`.
- Python ≥ 3.11. The map optimiser is a small C++ core bound into the package (not yet
  present; until then the package is pure Python and needs no compiler).

## Install

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

The build uses [scikit-build-core](https://scikit-build-core.readthedocs.io) and CMake.
pip fetches both into an isolated build environment, so nothing needs installing first.

External tools used by the tree steps (CMAPLE, RAxML-NG, UShER, Nextclade, gotree, TreeTime)
come from conda: `conda env create -f environment.yml`.

## Test, lint, type-check

Four commands, no `make`:

```sh
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
```

Tests that need real data use the private data repo, found at `$AF_DATA` (default:
`../acmacs-f-data` next to this checkout). When it is absent those tests are **skipped**,
never failed, so a public machine runs the rest.

Tests marked `slurm` need a real SLURM cluster and are skipped elsewhere.

## Layout

| Package | What it holds |
|---|---|
| `af.util` | Shared helpers: artefact checks, provenance records, config loading |
| `af.run` | Running external tools locally or on SLURM, from one call site |
| `af.seq` | Sequence store |
| `af.tree` | Trees: `build` (inference steps), `io` (formats), `draw` (report figures) |
| `af.clades` | Clade definitions from the upstream nomenclature, plus local additions |
| `af.chart` | Antigenic charts and their file formats |
| `af.serology` | Serology store and queries |
| `af.map` | Map optimisation (C++ core) |

`tests/` mirrors `af/`.

## Public code, private data

This repository is public. It contains **no** WHO Collaborating Centre data: no real strain
names, serum ids, titres, sequences or clade assignments. Real data and test fixtures live in
a separate private repository. A WHO-data gate runs as git hooks; install it once per clone:

```sh
git config core.hooksPath .githooks
```

See [tools/WHO-DATA-GATE.md](tools/WHO-DATA-GATE.md).
