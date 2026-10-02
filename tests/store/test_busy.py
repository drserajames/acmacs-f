"""af.store.busy: batches of publishes, and reads that refuse to see half of one."""

import datetime
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from af.store import Provenance, Store, StoreBusy, StoreError, busy
from af.store.__main__ import main as store_main

T0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


def publish(store: Store, kind: str, dataset: str, text: str) -> str:
    with store.build(kind, dataset) as build:
        (build.path / "data.txt").write_text(text)
        provenance = Provenance(step="t", inputs=(), parameters={}, started=T0, finished=T0)
        return build.publish(provenance).version


@pytest.fixture
def store(tmp_path: Path) -> Store:
    store = Store.create(tmp_path / "store")
    publish(store, "sequences", "h3", "one")
    publish(store, "clades", "h3", "one")
    return store


def write_marker(store: Store, **changes: object) -> Path:
    marker = {
        "name": "sweep",
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "process_started": busy.process_start(os.getpid()),
        "started": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "datasets": ["sequences/h3"],
        "max_age_hours": 12.0,
        **changes,
    }
    path = busy.busy_dir(store.root) / f"{marker['name']}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(marker))
    return path


def dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


# ---- writers ------------------------------------------------------------------------------


def test_batch_holds_a_marker_and_removes_it(store: Store) -> None:
    path = busy.busy_dir(store.root) / "sweep.json"
    with store.batch("sweep", ["sequences/h3", "sequences/h1"]) as marker:
        held = json.loads(path.read_text())
        assert held["pid"] == os.getpid() and held["datasets"] == ["sequences/h3", "sequences/h1"]
        assert marker.name == "sweep"
    assert not path.exists()


def test_batch_marker_removed_when_the_batch_fails(store: Store) -> None:
    with pytest.raises(RuntimeError), store.batch("sweep", []):
        raise RuntimeError("publish failed")
    assert not (busy.busy_dir(store.root) / "sweep.json").exists()


def test_one_name_one_holder_but_names_are_independent(store: Store) -> None:
    with store.batch("sweep", []):
        with pytest.raises(StoreBusy, match="'sweep'.*already running"), store.batch("sweep", []):
            pass
        with store.batch("rebuild", []):
            assert len(busy.read_markers(store.root)) == 2


def test_batch_name_is_checked(store: Store) -> None:
    with pytest.raises(StoreError, match="batch name"), store.batch("../escape", []):
        pass


# ---- readers ------------------------------------------------------------------------------


def test_reading_refuses_while_a_batch_holds_the_store(store: Store) -> None:
    holder = r"'sweep' \(pid .*datasets: sequences/h3"
    with (
        store.batch("sweep", ["sequences/h3"]),
        pytest.raises(StoreBusy, match=holder),
        store.reading("map-build"),
    ):
        pass


def test_override_reads_anyway_and_says_so_in_provenance(store: Store) -> None:
    with store.batch("sweep", ["sequences/h3"]), store.reading("map", override=True) as guard:
        store.current("sequences", "h3")
    record = guard.to_json()
    assert [m["name"] for m in record["overrode_batches"]] == ["sweep"]
    assert list(record["currents_read"]) == ["sequences/h3"]


def test_a_read_current_that_moves_fails_the_read(store: Store) -> None:
    before = store.current("sequences", "h3").version
    moved = f"sequences/h3 moved from {before} to "
    with pytest.raises(StoreBusy, match=moved), store.reading("map-build"):
        store.current("sequences", "h3")
        publish(store, "sequences", "h3", "two")


def test_a_publish_to_a_dataset_never_read_does_not_fail_the_read(store: Store) -> None:
    with store.reading("map-build") as guard:
        store.current("sequences", "h3")
        publish(store, "clades", "h3", "two")
    assert list(guard.reads) == [("sequences", "h3")]


def test_reading_one_dataset_at_two_versions_fails(store: Store) -> None:
    with pytest.raises(StoreBusy, match="sequences/h3 read as .* and "), store.reading("r"):
        store.current("sequences", "h3")
        publish(store, "sequences", "h3", "two")
        store.current("sequences", "h3")


def test_override_turns_a_moved_current_into_a_warning(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, "af.store"), store.reading("r", override=True):
        store.current("sequences", "h3")
        publish(store, "sequences", "h3", "two")
    assert "moved from" in caplog.text and "(override)" in caplog.text


def test_an_error_inside_the_read_is_not_masked(store: Store) -> None:
    with pytest.raises(ValueError, match="the build's own error"), store.reading("r"):
        store.current("sequences", "h3")
        publish(store, "sequences", "h3", "two")
        raise ValueError("the build's own error")


def test_reads_through_another_store_object_and_thread_are_recorded(store: Store) -> None:
    other = Store.open(store.root)
    with store.reading("r") as guard:
        worker = threading.Thread(target=other.current, args=("clades", "h3"))
        worker.start()
        worker.join()
    assert list(guard.reads) == [("clades", "h3")]


def test_no_reads_recorded_outside_a_guard(store: Store) -> None:
    with store.reading("r") as guard:
        pass
    store.current("sequences", "h3")
    assert guard.reads == {}


# ---- stale markers: a crashed writer never holds the store for ever ---------------------


def test_marker_of_a_dead_process_is_stale_reported_and_replaced(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    write_marker(store, pid=dead_pid())
    with caplog.at_level(logging.WARNING, "af.store"):
        with store.reading("r"):
            pass
        assert "stale batch marker" in caplog.text and "is gone" in caplog.text
        with store.batch("sweep", []) as marker:
            assert marker.pid == os.getpid()
        assert "replacing stale batch marker" in caplog.text


def test_a_reused_pid_is_not_the_holder(store: Store) -> None:
    write_marker(store, process_started="some other process")
    with store.reading("r"):
        pass  # our own pid, but not the process that took the marker


def test_marker_from_another_host_lives_until_its_age_limit(store: Store) -> None:
    write_marker(store, host="another-host")
    with pytest.raises(StoreBusy), store.reading("r"):
        pass
    old = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=13)).isoformat()
    write_marker(store, host="another-host", started=old)
    with store.reading("r"):
        pass


def test_sigterm_mid_batch_removes_the_marker_and_kill_9_leaves_a_stale_one(
    store: Store,
) -> None:
    script = (
        "import sys, time\n"
        "from af.store import Store\n"
        "from af.run.job import run_main, signals_as_exceptions\n"
        "def main():\n"
        "    with signals_as_exceptions(), Store.open(sys.argv[1]).batch('sweep', []):\n"
        "        print('held', flush=True)\n"
        "        time.sleep(60)\n"
        "    return 0\n"
        "run_main(main)\n"
    )
    marker = busy.busy_dir(store.root) / "sweep.json"
    for sig, removed in ((signal.SIGTERM, True), (signal.SIGKILL, False)):
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(store.root)], stdout=subprocess.PIPE, text=True
        )
        assert process.stdout is not None and process.stdout.readline().strip() == "held"
        process.send_signal(sig)
        process.wait(timeout=20)
        assert marker.exists() is not removed
    with store.reading("r"):  # the killed holder's marker is stale
        pass
    marker.unlink()


# ---- the CLI ------------------------------------------------------------------------------


def test_cli_lists_and_clears_markers(store: Store, capsys: pytest.CaptureFixture[str]) -> None:
    write_marker(store)
    write_marker(store, name="old", pid=dead_pid())
    assert store_main(["busy", str(store.root)]) == 0
    listing = capsys.readouterr().out
    assert "LIVE  'sweep'" in listing and "STALE 'old'" in listing
    assert store_main(["busy", str(store.root), "--clear", "sweep"]) == 0
    assert "removed 'sweep'" in capsys.readouterr().out
    assert [m.name for m in busy.read_markers(store.root)] == ["old"]
    with pytest.raises(StoreError, match="no batch marker named 'nothing'"):
        store_main(["busy", str(store.root), "--clear", "nothing"])
