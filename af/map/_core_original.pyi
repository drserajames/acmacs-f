"""Type stub for the optimiser core compiled with the ORIGINAL arithmetic (AF_ORIGINAL_ARITHMETIC in
cpp/optimiser): the same interface as af.map._core, except that it has no arithmetic name and no
portable_exp / portable_log2. Its Problem is a different class at run time: an object of one core is
refused by the other.
"""

from af.map._core import Problem as Problem
from af.map._core import column_bases as column_bases
from af.map._core import contracts_multiply_add as contracts_multiply_add
from af.map._core import fit_reactivity as fit_reactivity
from af.map._core import grid_test as grid_test
from af.map._core import optimise as optimise
from af.map._core import reactivity_column_bases as reactivity_column_bases
from af.map._core import reactivity_objective as reactivity_objective
from af.map._core import refine as refine
from af.map._core import relax as relax
from af.map._core import relax_incremental as relax_incremental
from af.map._core import resolve_threads as resolve_threads
from af.map._core import start_seed as start_seed

openmp: bool
library: str
NUMERICS: None
