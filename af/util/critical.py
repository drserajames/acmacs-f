"""Critical sections: short sequences that a prompt exit must not cut through.

A store publish renames a version into place, appends to HISTORY and rewrites CURRENT.
Stopping between those steps leaves HISTORY naming a version CURRENT doesn't point to.
Drivers stop promptly on Ctrl-C, SIGTERM and SIGHUP (af.run.job), so such sequences are
wrapped in :func:`critical`:

- in a worker thread, the driver's exit (``af.run.job.run_main``) waits for every open
  critical section to finish (with a time limit) before leaving;
- in the main thread, where the signal handler would otherwise raise in the middle of
  the sequence, the signal is deferred and re-raised as the section ends.

Keep critical sections short: file renames and small writes, never computation.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager

_condition = threading.Condition()
_open = 0  # critical sections open in any thread
_main_depth = 0  # of which in the main thread (nesting)
_deferred: int | None = None  # a signal that arrived during a main-thread section


@contextmanager
def critical() -> Iterator[None]:
    global _open, _main_depth, _deferred
    in_main = threading.current_thread() is threading.main_thread()
    with _condition:
        _open += 1
        if in_main:
            _main_depth += 1
    try:
        yield
    finally:
        pending = None
        with _condition:
            _open -= 1
            if in_main:
                _main_depth -= 1
                if _main_depth == 0 and _deferred is not None:
                    pending, _deferred = _deferred, None
            _condition.notify_all()
        if pending is not None:
            signal.raise_signal(pending)  # now outside the section: the handler acts on it


def defer_in_main(signum: int) -> bool:
    """Called by a signal handler: True if it must return now and let the section re-raise."""
    global _deferred
    if threading.current_thread() is not threading.main_thread():
        return False
    with _condition:
        if _main_depth == 0:
            return False
        _deferred = signum
        return True


def wait_for_critical_sections(timeout: float) -> bool:
    """Wait until no critical section is open; False if ``timeout`` seconds pass first."""
    with _condition:
        return _condition.wait_for(lambda: _open == 0, timeout)
