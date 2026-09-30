"""af.util.critical: a stop signal never lands in the middle of a store publish."""

import os
import signal
import threading
import time

import pytest

from af.run.job import Terminated, signals_as_exceptions
from af.util.critical import critical, wait_for_critical_sections


def test_wait_returns_when_the_last_section_closes() -> None:
    entered, release = threading.Event(), threading.Event()

    def worker() -> None:
        with critical():
            entered.set()
            release.wait(10)

    thread = threading.Thread(target=worker)
    thread.start()
    entered.wait(10)
    assert not wait_for_critical_sections(0.2), "a section is open"
    threading.Timer(0.3, release.set).start()
    assert wait_for_critical_sections(10)
    thread.join()


@pytest.mark.parametrize(
    ("signum", "expected"),
    [(signal.SIGTERM, Terminated), (signal.SIGHUP, Terminated), (signal.SIGINT, KeyboardInterrupt)],
    ids=["SIGTERM", "SIGHUP", "SIGINT"],
)
def test_main_thread_signal_waits_for_the_section_to_end(
    signum: int, expected: type[BaseException]
) -> None:
    steps: list[str] = []
    with signals_as_exceptions(), pytest.raises(expected):
        with critical():
            os.kill(os.getpid(), signum)
            time.sleep(0.2)  # the handler has run by now; it must not have raised
            steps.append("section finished")
        steps.append("after the section")  # never reached: the signal takes effect first
    assert steps == ["section finished"]


def test_signal_outside_a_section_acts_at_once() -> None:
    with signals_as_exceptions(), pytest.raises(Terminated):
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)
