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


def test_a_batch_that_fails_part_way_keeps_its_marker_until_resumed(store: Store) -> None:
    """7 Oct 2026: a sweep published its first pull, failed on the second, and released its
    marker, leaving the store half swept and unguarded for 3.6 min. Now the marker stays."""
    path = busy.busy_dir(store.root) / "sweep.json"
    with pytest.raises(RuntimeError), store.batch("sweep", ["sequences/h3", "sequences/h1"]):
        publish(store, "sequences", "h3", "pull 1")
        raise RuntimeError("unknown country")
    (marker,) = busy.read_markers(store.root)
    assert marker.failed and "RuntimeError: unknown country" in marker.failed["error"]
    assert marker.published == ("sequences/h3",) and busy.is_partial(marker)
    with pytest.raises(StoreBusy, match="PARTIAL|FAILED.*Resume it"), store.reading("map-build"):
        pass
    # nobody writes its datasets meanwhile, this process included, outside the batch
    with pytest.raises(StoreBusy, match="publish to sequences/h1 refused: a PARTIAL batch"):
        publish(store, "sequences", "h1", "stray")
    publish(store, "tables", "cdc-h3", "unrelated")  # a dataset it does not name: allowed
    with pytest.raises(StoreBusy), store.batch("other", ["sequences/h1"]):
        publish(store, "sequences", "h1", "another batch")
    # the same batch resumes it, carries what was published, and completes
    with store.batch("sweep", ["sequences/h1"]) as resumed:
        assert resumed.resumed_from and resumed.resumed_from["failed"]
        assert resumed.published == ("sequences/h3",) and not resumed.failed
        assert set(resumed.datasets) == {"sequences/h3", "sequences/h1"}
        publish(store, "sequences", "h1", "pull 2")
    assert not path.exists()
    with store.reading("map-build"):
        pass


def test_a_resumed_batch_that_fails_again_stays_partial_without_new_publishes(
    store: Store,
) -> None:
    with pytest.raises(RuntimeError), store.batch("sweep", ["sequences/h3"]):
        publish(store, "sequences", "h3", "pull 1")
        raise RuntimeError("first failure")
    with pytest.raises(RuntimeError), store.batch("sweep", ["sequences/h3"]):
        raise RuntimeError("second failure, before publishing anything")
    (marker,) = busy.read_markers(store.root)
    assert marker.failed and "second failure" in marker.failed["error"]
    assert marker.published == ("sequences/h3",)


def test_the_marker_exists_and_names_the_dataset_before_any_publish(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Marker first, always: the marker is on disk when the first publish starts."""
    seen = []
    check = busy.check_write

    def watching(root: Path, kind: str, dataset: str) -> None:
        (marker,) = busy.read_markers(root)
        seen.append((marker.name, f"{kind}/{dataset}" in marker.datasets))
        check(root, kind, dataset)

    monkeypatch.setattr(busy, "check_write", watching)
    with store.batch("sweep", ["sequences/h3"]):
        publish(store, "sequences", "h3", "two")
    assert seen == [("sweep", True)]


def test_a_second_writer_to_a_dataset_another_batch_names_is_refused(store: Store) -> None:
    """Another live process's batch (the test's parent stands in for it)."""
    parent = os.getppid()
    write_marker(store, pid=parent, process_started=busy.process_start(parent) or busy.UNKNOWN)
    with pytest.raises(StoreBusy, match="publish to sequences/h3 refused: a running batch"):
        publish(store, "sequences", "h3", "second writer")
    publish(store, "sequences", "h1", "a dataset it does not name")
    publish(store, "clades", "h3", "nor this one")


def test_own_batch_writes_its_datasets(store: Store) -> None:
    with store.batch("sweep", ["sequences/h3"]):
        publish(store, "sequences", "h3", "two")
        (marker,) = busy.read_markers(store.root)
        assert marker.published == ("sequences/h3",)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_a_holder_stopped_after_publishing_leaves_a_partial_marker(
    store: Store, sig: signal.Signals, capsys: pytest.CaptureFixture[str]
) -> None:
    """SIGTERM fails the batch (marked failed); kill -9 runs no Python at all, but the marker
    already lists the publish, so the dead holder's batch is partial, not stale."""
    script = (
        "import sys, time, datetime\n"
        "from af.store import Store, Provenance\n"
        "from af.run.job import run_main, signals_as_exceptions\n"
        "T = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)\n"
        "def main():\n"
        "    store = Store.open(sys.argv[1])\n"
        "    names = ['sequences/h3', 'sequences/h1']\n"
        "    with signals_as_exceptions(), store.batch('sweep', names):\n"
        "        with store.build('sequences', 'h3') as b:\n"
        "            (b.path / 'data.txt').write_text('pull 1')\n"
        "            p = Provenance(step='t', inputs=(), parameters={}, started=T, finished=T)\n"
        "            b.publish(p)\n"
        "        print('published', flush=True)\n"
        "        time.sleep(60)\n"
        "    return 0\n"
        "run_main(main)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(store.root)], stdout=subprocess.PIPE, text=True
    )
    assert process.stdout is not None and process.stdout.readline().strip() == "published"
    process.send_signal(sig)
    process.wait(timeout=20)
    (marker,) = busy.read_markers(store.root)
    assert busy.is_partial(marker) and marker.published == ("sequences/h3",)
    assert bool(marker.failed) is (sig == signal.SIGTERM)
    with pytest.raises(StoreBusy), store.reading("r"):
        pass
    assert store_main(["busy", str(store.root)]) == 0
    assert "PARTIAL 'sweep'" in capsys.readouterr().out


def test_clearing_a_partial_batch_needs_a_reason_and_is_recorded(
    store: Store, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(RuntimeError), store.batch("sweep", ["sequences/h3"]):
        publish(store, "sequences", "h3", "pull 1")
        raise RuntimeError("failed")
    with pytest.raises(StoreError, match="partial batch.*--reason"):
        store_main(["busy", str(store.root), "--clear", "sweep"])
    with pytest.raises(StoreError, match="--reason"):
        store_main(["busy", str(store.root), "--clear", "sweep", "--reason", "  "])
    reason = "pull 1 is complete on its own; the rest is a separate sweep"
    assert store_main(["busy", str(store.root), "--clear", "sweep", "--reason", reason]) == 0
    assert "recorded in" in capsys.readouterr().out
    (record,) = [
        json.loads(line)
        for line in (busy.busy_dir(store.root) / busy.CLEARED).read_text().splitlines()
    ]
    assert record["reason"] == reason and record["partial"] is True
    assert record["marker"]["name"] == "sweep" and record["marker"]["failed"]
    assert busy.read_markers(store.root) == []
    with store.reading("r"):
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


# ---- readers that declare their kinds -----------------------------------------------------


def test_a_batch_of_kinds_a_read_does_not_use_does_not_hold_it_off(store: Store) -> None:
    """A sequences sweep ran for hours while serology's update, which reads only tables and
    serology, could get in only by overriding: that trains people to override (7 Oct 2026)."""
    publish(store, "tables", "cdc-h3", "one")
    with store.batch("sweep", ["sequences/h3", "sequences/h1"]):
        with store.reading("serology-update", kinds={"tables", "serology"}) as guard:
            store.current("tables", "cdc-h3")
        assert guard.to_json()["kinds"] == ["serology", "tables"]
        assert [m["name"] for m in guard.to_json()["batches_of_other_kinds"]] == ["sweep"]
        assert guard.overridden == []
        with pytest.raises(StoreBusy, match="'sweep'"), store.reading("seq", kinds={"sequences"}):
            pass
        with pytest.raises(StoreBusy, match="'sweep'"), store.reading("any"):
            pass  # no kinds declared: every batch holds it off, as before


def test_a_read_of_an_undeclared_kind_fails_at_the_read(store: Store) -> None:
    reached = []
    with (
        pytest.raises(StoreError, match=r"read sequences/h3, but this read declared only"),
        store.reading("serology-update", kinds={"tables", "serology"}),
    ):
        store.current("sequences", "h3")
        reached.append("after the read")
    assert reached == []


def test_an_enclosing_guard_declares_the_kinds_of_the_guards_inside_it(store: Store) -> None:
    with (
        pytest.raises(StoreError, match=r"map-build: read sequences/h3"),
        store.reading("map-build", kinds={"clades"}),
        store.reading("colouring", kinds={"clades", "sequences"}),
    ):
        store.current("sequences", "h3")
    with (
        store.reading("map-build", kinds={"clades", "sequences"}),
        store.reading("colouring", kinds={"clades", "sequences"}),
    ):
        store.current("sequences", "h3")


def test_declared_kinds_are_checked(store: Store) -> None:
    with (
        pytest.raises(StoreError, match="unknown store kind"),
        store.reading("r", kinds={"tabels"}),
    ):
        pass
    with pytest.raises(StoreError, match="kinds is empty"), store.reading("r", kinds=set()):
        pass


def test_a_marker_entry_with_no_kind_holds_off_every_read(store: Store) -> None:
    write_marker(store, datasets=["h3"])
    with pytest.raises(StoreBusy), store.reading("r", kinds={"tables"}):
        pass


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


@pytest.mark.skipif(
    busy.process_start(os.getpid()) == busy.UNKNOWN,
    reason="processes cannot be inspected here (e.g. a sandbox that forbids ps)",
)
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


def test_where_processes_cannot_be_inspected_the_age_decides(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sandbox that forbids ps: a marker from this host falls back to the age limit."""
    monkeypatch.setattr(busy, "process_start", lambda pid: busy.UNKNOWN)
    write_marker(store, process_started=busy.UNKNOWN)
    with pytest.raises(StoreBusy), store.reading("r"):
        pass
    old = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=13)).isoformat()
    write_marker(store, process_started=busy.UNKNOWN, started=old)
    with store.reading("r"):
        pass
