"""The identity rules the serology store uses: the chain engine's, and nothing else.

``af.chart.identity`` is the one copy of "when is an antigen or serum in one table the
same as one in another" (design rule 6); this module only wraps it for the store.

The rules' version, which decides whether stored partitions are still valid, is a hash
of the two functions' source code. It changes by itself whenever the rules are edited, so
it can never be forgotten; an edit that changes only a comment or docstring costs one
unnecessary rebuild, never a stale store.
"""

from __future__ import annotations

import hashlib
import inspect

from af.chart import identity
from af.serology.rows import IdentityRules


def chain_rules() -> IdentityRules:
    source = inspect.getsource(identity.antigen_identity) + inspect.getsource(
        identity.serum_identity
    )
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    return IdentityRules(
        antigen=identity.antigen_identity,
        serum=identity.serum_identity,
        version=f"af.chart.identity:{digest}",
    )
