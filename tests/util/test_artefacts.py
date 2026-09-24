"""af.util.artefacts: missing, empty and unparsable outputs are fatal; provenance records."""

import datetime
import hashlib
import json
from pathlib import Path

import pytest

import af
from af.util.artefacts import (
    Artefact,
    ArtefactError,
    check_artefact,
    check_artefacts,
    provenance_path,
    read_provenance,
    sha256_path,
    write_provenance,
)

STEP = "example-step"


def json_check(path: Path) -> object:
    return json.loads(path.read_text())


def test_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    path.write_text('{"a": 1}')
    checked = check_artefact(STEP, Artefact(path, parse=json_check))
    assert checked.sha256 == hashlib.sha256(b'{"a": 1}').hexdigest()
    assert checked.size == 8


def test_missing_artefact_is_fatal(tmp_path: Path) -> None:
    with pytest.raises(ArtefactError) as caught:
        check_artefact(STEP, Artefact(tmp_path / "absent.txt"))
    assert caught.value.step == STEP
    assert caught.value.failures == [(tmp_path / "absent.txt", "missing")]
    assert STEP in str(caught.value) and "absent.txt" in str(caught.value)


def test_empty_file_is_fatal_unless_allowed(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.touch()
    with pytest.raises(ArtefactError, match="empty file"):
        check_artefact(STEP, Artefact(path))
    assert check_artefact(STEP, Artefact(path, allow_empty=True)).size == 0


def test_unparsable_file_is_fatal(tmp_path: Path) -> None:
    path = tmp_path / "half.json"
    path.write_text('{"a": ')
    with pytest.raises(ArtefactError, match="does not parse: JSONDecodeError"):
        check_artefact(STEP, Artefact(path, parse=json_check))


def test_all_failures_reported_together(tmp_path: Path) -> None:
    good = tmp_path / "good.txt"
    good.write_text("x")
    empty = tmp_path / "empty.txt"
    empty.touch()
    with pytest.raises(ArtefactError) as caught:
        check_artefacts(STEP, [Artefact(good), Artefact(empty), Artefact(tmp_path / "gone")])
    assert [reason for _, reason in caught.value.failures] == ["empty file", "missing"]


def test_no_artefacts_is_an_error() -> None:
    with pytest.raises(ArtefactError, match="declared no artefacts"):
        check_artefacts(STEP, [])


def test_directory_artefact(tmp_path: Path) -> None:
    directory = tmp_path / "tree"
    directory.mkdir()
    with pytest.raises(ArtefactError, match="empty directory"):
        check_artefact(STEP, Artefact(directory))
    (directory / "sub").mkdir()
    (directory / "sub" / "a.txt").write_text("a")
    (directory / "b.txt").write_text("b")
    before = check_artefact(STEP, Artefact(directory))
    assert before.size == 2
    (directory / "b.txt").rename(directory / "c.txt")
    assert sha256_path(directory) != before.sha256, "a rename must change a directory hash"


def test_provenance_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("input")
    output = tmp_path / "out.txt"
    output.write_text("output")
    checked = check_artefact(STEP, Artefact(output))
    started = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.UTC)
    finished = started + datetime.timedelta(seconds=90)

    path = write_provenance(
        STEP,
        checked,
        inputs=[source],
        parameters={"threads": 2, "method": "example"},
        started=started,
        finished=finished,
    )

    assert path == provenance_path(output) == tmp_path / "out.txt.provenance.json"
    record = read_provenance(output)
    assert record == {
        "step": STEP,
        "af_version": af.__version__,
        "output": {"path": str(output), "sha256": checked.sha256, "size": 6},
        "inputs": [{"path": str(source), "sha256": hashlib.sha256(b"input").hexdigest()}],
        "parameters": {"method": "example", "threads": 2},
        "started": "2026-01-02T03:04:05+00:00",
        "finished": "2026-01-02T03:05:35+00:00",
    }
    assert not list(tmp_path.glob(".*.tmp")), "temporary file left behind"


def test_provenance_missing_input_is_fatal(tmp_path: Path) -> None:
    output = tmp_path / "out.txt"
    output.write_text("output")
    now = datetime.datetime.now(datetime.UTC)
    with pytest.raises(FileNotFoundError):
        write_provenance(
            STEP,
            check_artefact(STEP, Artefact(output)),
            inputs=[tmp_path / "absent"],
            parameters={},
            started=now,
            finished=now,
        )


def test_provenance_rejects_naive_times(tmp_path: Path) -> None:
    output = tmp_path / "out.txt"
    output.write_text("output")
    naive = datetime.datetime(2026, 1, 1)
    with pytest.raises(ValueError, match="timezone-aware"):
        write_provenance(
            STEP,
            check_artefact(STEP, Artefact(output)),
            inputs=[],
            parameters={},
            started=naive,
            finished=naive,
        )
