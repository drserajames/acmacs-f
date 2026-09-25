# acmacs-f

Phylogenetic trees and antigenic maps for influenza vaccine strain selection.

acmacs-f (Python package `af`) is the successor to the acmacs-d (AD) and acmacs-e (ae)
toolkits. It builds phylogenetic trees and antigenic maps from raw sequence and titre data,
keeps them in long-lived stores that are updated incrementally, and produces reports from
them. It runs the same way on a laptop and on a SLURM cluster.

- Distribution name: `acmacs-f`. Import name: `af`.
- Python ≥ 3.11. The map optimiser is a small C++ core (on [alglib](https://www.alglib.net))
  bound into the package. Until its sources land the package is pure Python and needs no
  compiler.

## Install

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

The build uses [scikit-build-core](https://scikit-build-core.readthedocs.io) and CMake.
pip fetches both into an isolated build environment, so nothing needs installing first.
Once the C++ optimiser is present, the build also needs a C++20 compiler and downloads alglib
3.19.0 (checked against its SHA-256). On a machine without internet access, point it at a
local copy of the tarball:

```sh
pip install -e . -Ccmake.define.AF_ALGLIB_URL=/path/to/alglib-3.19.0.cpp.gpl.tgz
```

External tools used by the tree steps (CMAPLE, IQ-TREE, RAxML-NG, UShER, Nextclade, gotree,
TreeTime) come from conda: `conda env create -f environment.yml`.

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

## Running on SLURM

Every external tool and every heavy Python step goes through `af.run`, which has a local
runner and a SLURM runner behind the same call. Which one is used comes from config:

```toml
[runner]
kind = "slurm"

[runner.slurm]
work_dir = "/shared/scratch/af/slurm"   # batch scripts and status files; nodes must see it
partition = "..."                       # optional
account = "..."                         # optional
extra_args = ["--qos=..."]              # optional, passed to sbatch
max_parallel_tasks = 50                 # optional throttle for arrays (%N)
max_array_size = 1000                   # the cluster's MaxArraySize (scontrol show config)
output_wait_seconds = 30                # shared-filesystem lag allowance
```

Before relying on a new cluster, run the smoke test from a login node. It submits real jobs
(success, failure, missing output, a job array, a Python job on a node, and optionally a
time-limit kill) and checks af reports each one correctly:

```sh
python -m af.run.smoke --work-dir /shared/scratch/af-smoke [--partition P] [--account A] \
    [--extra-arg=--qos=Q] [--max-array-size N] [--with-timeout]
```

It prints a PASS/FAIL table and writes `smoke-report.json` into the work directory.

Independent pipeline steps run side by side with `[pipeline] max_parallel = N` (default 1,
one after another), so the three subtypes' trees can build at the same time.

What the cluster must provide:

- **A shared filesystem** for the conda env / venv (jobs run the submitting interpreter),
  the work directory, the store and the work area.
- **`sbatch --wait`** (SLURM ≥ 17). Jobs inherit the submitting environment (`--export=ALL`).
- **A driver that outlives the jobs.** `run()` blocks until the jobs finish. Ctrl-C,
  SIGTERM, SIGHUP (a dropped ssh session) or an error cancels every job in flight
  (`scancel --name`) before the driver exits; only SIGKILL cannot be handled. Still, run
  long pipelines in tmux, or submit the driver itself as a SLURM job.
- **The interpreter's `af` is the code that runs.** Python jobs (`Job.python_module`, the smoke
  test) run `python -m …` with the submitting interpreter, so install the checked-out code into
  it (`pip install -e .` in that env). An env whose editable install points at another
  checkout runs that checkout's `af` on the nodes.
- **Threads are passed twice.** A job's `Resources(threads=N)` sets `--cpus-per-task`; the
  program must also be told N in its arguments.

## Layout

| Package | What it holds |
|---|---|
| `af.util` | Config loading, artefact checks, provenance records |
| `af.run` | Running external tools locally or on SLURM, from one call site |
| `af.store` | The store layout and report manifests |
| `af.pipeline` | Step driver with incremental skipping |
| `af.seq` | Sequence store (GISAID sequences and metadata) |
| `af.tables` | Titre-table parsers with error checks |
| `af.clades` | Clade assignment from the upstream nomenclature |
| `af.tree` | Trees: `build`, `asr` (ancestral reconstruction), `draw` (figures) |
| `af.chart` | Antigenic charts and their file formats |
| `af.chain` | Incremental chains, restartable from the first changed table |
| `af.map` | Map optimisation (C++ core) and map finishing |
| `af.serology` | Serology store and queries |
| `af.geo`, `af.stat` | Geographic maps and stat counts |
| `af.report` | Reports built from the stores |

`tests/` mirrors `af/`.

## Public code, private data

This repository is public. It contains **no** WHO Collaborating Centre data: no real strain
names, serum ids, titres, sequences or clade assignments. Real data and test fixtures live in
a separate private repository. A WHO-data gate runs as git hooks; install it once per clone:

```sh
git config core.hooksPath .githooks
```

See [tools/WHO-DATA-GATE.md](tools/WHO-DATA-GATE.md).

## Licence

acmacs-f is free software, licensed under the GNU General Public License, version 3 or (at
your option) any later version: see [LICENSE](LICENSE). The optimiser links
[alglib](https://www.alglib.net), which is GPL-2.0-or-later.

`tools/who-data-gate.py` and its companion files are copied from
[ae](https://github.com/acorg/ae) and remain under ae's MIT licence:

```text
MIT License

Copyright (c) 2021 Eugene Skepner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
