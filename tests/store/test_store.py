"""af.store: immutable versions, hard-linked updates, identical rebuilds, refs, snapshots."""

import datetime
import json
import os
from pathlib import Path

import pytest

from af.store import (
    ExternalInput,
    Provenance,
    Store,
    StoreError,
    StoreRef,
    check_refs,
    read_manifest,
    read_snapshot,
    write_manifest,
    write_snapshot,
)

T0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


def provenance(*inputs: StoreRef | ExternalInput, step: str = "example") -> Provenance:
    return Provenance(step=step, inputs=inputs, parameters={"n": 1}, started=T0, finished=T0)


def publish(store: Store, kind: str, dataset: str, files: dict[str, str], **kw: object) -> StoreRef:
    with store.build(kind, dataset) as build:
        for name, text in files.items():
            path = build.path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return build.publish(provenance(), summary=kw or None)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store.create(tmp_path / "store")


def test_create_and_open(tmp_path: Path) -> None:
    root = tmp_path / "store"
    Store.create(root)
    assert Store.open(root).root == root
    with pytest.raises(StoreError, match="non-empty"):
        Store.create(root)
    with pytest.raises(StoreError, match="not a store"):
        Store.open(tmp_path / "elsewhere")
    (root / "STORE.toml").write_text("layout = 99\n")
    with pytest.raises(StoreError, match="layout 99 is not supported"):
        Store.open(root)


def test_publish_and_read_back(store: Store) -> None:
    ref = publish(store, "trees", "h3/report", {"tree.nwk": "(a,b);", "states/x.tsv": "1"})
    assert store.current("trees", "h3/report") == ref
    directory = store.resolve(ref, verify=True)
    assert (directory / "tree.nwk").read_text() == "(a,b);"
    assert (directory / "states" / "x.tsv").read_text() == "1"
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    assert [entry["path"] for entry in manifest["files"]] == ["states/x.tsv", "tree.nwk"]
    assert json.loads((directory / "PROVENANCE.json").read_text())["parameters"] == {"n": 1}
    assert [event["event"] for event in store.history("trees", "h3/report")] == ["published"]
    assert not os.access(directory / "tree.nwk", os.W_OK), "published files are read-only"


def test_same_content_same_id_in_any_store(tmp_path: Path) -> None:
    a = publish(Store.create(tmp_path / "a"), "tables", "labx/h3-hi", {"t.txt": "same"})
    b = publish(Store.create(tmp_path / "b"), "tables", "labx/h3-hi", {"t.txt": "same"})
    assert a == b


def test_identical_rebuild_is_reconfirmed_not_republished(store: Store) -> None:
    first = publish(store, "serology", "all", {"part.parquet": "rows"})
    second = publish(store, "serology", "all", {"part.parquet": "rows"})
    assert first == second
    assert len(list((store.dataset_dir("serology", "all") / "versions").iterdir())) == 1
    history = store.history("serology", "all")
    assert [event["event"] for event in history] == ["published", "reconfirmed"]
    assert history[1]["parent"] == first.version


def test_new_version_links_unchanged_files(store: Store) -> None:
    first = publish(store, "sequences", "h3", {"pull=1/part.parquet": "old", "index.tsv": "1"})
    with store.build("sequences", "h3") as build:
        linked = build.link_unchanged("pull=1/part.parquet")
        (build.path / "pull=2").mkdir()
        (build.path / "pull=2" / "part.parquet").write_text("new")
        (build.path / "index.tsv").write_text("2")
        second = build.publish(provenance(first))
        assert not linked.exists(), "staging is renamed into place"
    old = store.version_dir(first) / "pull=1" / "part.parquet"
    new = store.version_dir(second) / "pull=1" / "part.parquet"
    assert Path(old).stat().st_ino == Path(new).stat().st_ino, "unchanged file is a hard link"
    assert store.current("sequences", "h3") == second
    assert store.history("sequences", "h3")[1]["parent"] == first.version
    store.verify(first)  # the old version is untouched


def test_linked_file_cannot_be_edited_in_place(store: Store) -> None:
    publish(store, "trees", "h1/weekly", {"tree.nwk": "(a,b);"})
    with store.build("trees", "h1/weekly") as build:
        linked = build.link_unchanged("tree.nwk")
        with pytest.raises(PermissionError):
            linked.write_text("edited")


def test_restoring_an_earlier_version(store: Store) -> None:
    first = publish(store, "chains", "labx/h3-hi", {"chain.json": "1"})
    publish(store, "chains", "labx/h3-hi", {"chain.json": "2"})
    again = publish(store, "chains", "labx/h3-hi", {"chain.json": "1"})
    assert again == first == store.current("chains", "labx/h3-hi")
    assert store.history("chains", "labx/h3-hi")[-1]["event"] == "restored"


def test_failed_build_leaves_nothing(store: Store) -> None:
    with pytest.raises(RuntimeError), store.build("trees", "h3/report") as build:
        (build.path / "tree.nwk").write_text("partial")
        raise RuntimeError("tool crashed")
    dataset = store.dataset_dir("trees", "h3/report")
    assert list((dataset / "versions").iterdir()) == []
    assert not (dataset / "CURRENT").exists()
    with pytest.raises(StoreError, match="no such dataset"):
        store.current("trees", "h3/report")


def test_empty_version_refused(store: Store) -> None:
    with pytest.raises(StoreError, match="empty version"), store.build("trees", "h3/x") as build:
        build.publish(provenance())


def test_reserved_names_not_counted(store: Store) -> None:
    """A stray PROVENANCE.json in the staging dir must not change the version id."""
    plain = publish(store, "tables", "a/b", {"t.txt": "x"})
    with store.build("tables", "a/b") as build:
        (build.path / "t.txt").write_text("x")
        (build.path / "PROVENANCE.json").write_text("{}")
        assert build.publish(provenance()) == plain


def test_symlinks_refused(store: Store, tmp_path: Path) -> None:
    (tmp_path / "outside.txt").write_text("x")
    with pytest.raises(StoreError, match="symlinks"), store.build("tables", "a/b") as build:
        (build.path / "link.txt").symlink_to(tmp_path / "outside.txt")
        build.publish(provenance())


def test_bad_keys(store: Store) -> None:
    for kind, dataset in [
        ("tree", "h3"),
        ("trees", ""),
        ("trees", "../h3"),
        ("trees", "h3/versions"),
        ("trees", ".hidden"),
        ("trees", "h3//x"),
    ]:
        with pytest.raises(StoreError):
            store.dataset_dir(kind, dataset)


def test_nested_datasets_refused(store: Store) -> None:
    publish(store, "chains", "labx/h3-hi", {"c": "1"})
    with pytest.raises(StoreError, match="nested inside dataset 'labx/h3-hi'"):
        publish(store, "chains", "labx/h3-hi/from-2020", {"c": "1"})
    publish(store, "chains", "laby/h1/main", {"c": "1"})
    with pytest.raises(StoreError, match="would contain dataset 'laby/h1/main'"):
        publish(store, "chains", "laby/h1", {"c": "1"})


def test_list_datasets(store: Store) -> None:
    refs = [
        publish(store, "tables", "labx/h3-hi", {"t": "1"}),
        publish(store, "tables", "labx/h1-hi", {"t": "2"}),
        publish(store, "tables", "laby/bvic-hi", {"t": "3"}),
    ]
    with store.build_cache_entry("tables", "labx/h3-hi", "k1") as entry:
        (entry / "CURRENT").write_text("decoy inside a cache entry")
    listed = store.list_datasets("tables")
    assert [ref.dataset for ref in listed] == ["labx/h1-hi", "labx/h3-hi", "laby/bvic-hi"]
    assert sorted(listed, key=lambda r: r.dataset) == sorted(refs, key=lambda r: r.dataset)
    assert store.list_datasets("trees") == []


def test_resolve_detects_tampering(store: Store) -> None:
    ref = publish(store, "trees", "h3/report", {"tree.nwk": "(a,b);"})
    path = store.version_dir(ref) / "tree.nwk"
    path.chmod(0o644)
    path.write_text("(a,c);")
    store.resolve(ref)  # the cheap check reads only MANIFEST.json
    with pytest.raises(StoreError, match="content changed: tree.nwk"):
        store.resolve(ref, verify=True)
    forged = StoreRef(ref.kind, ref.dataset, ref.version, ref.version + "0" * 48)
    with pytest.raises(StoreError, match="does not match the reference"):
        store.resolve(forged)


def test_cache_entries_are_write_once(store: Store) -> None:
    with store.build_cache_entry("chains", "labx/h3-hi", "step-0001-abc") as entry:
        (entry / "map.ace.json").write_text("first")
    with store.build_cache_entry("chains", "labx/h3-hi", "step-0001-abc") as entry:
        (entry / "map.ace.json").write_text("second")
    found = store.cache_entry("chains", "labx/h3-hi", "step-0001-abc")
    assert found is not None and (found / "map.ace.json").read_text() == "first"
    assert store.cache_entry("chains", "labx/h3-hi", "missing") is None
    with (
        pytest.raises(StoreError, match="empty"),
        store.build_cache_entry("chains", "labx/h3-hi", "empty"),
    ):
        pass
    cache = store.dataset_dir("chains", "labx/h3-hi") / "cache"
    assert sorted(path.name for path in cache.iterdir()) == ["step-0001-abc"], "no tmp left"


def test_version_links_cache_entries(store: Store) -> None:
    with store.build_cache_entry("chains", "labx/h3-hi", "k1") as entry:
        (entry / "step.json").write_text("{}")
    cached = store.cache_entry("chains", "labx/h3-hi", "k1")
    assert cached is not None
    with store.build("chains", "labx/h3-hi") as build:
        build.link(cached, "steps/0001-k1")
        ref = build.publish(provenance())
    linked = store.version_dir(ref) / "steps" / "0001-k1" / "step.json"
    assert Path(linked).stat().st_ino == Path(cached / "step.json").stat().st_ino


def test_provenance_mixed_inputs(store: Store, tmp_path: Path) -> None:
    raw = publish(store, "raw", "gisaid/pull-1", {"data.xls": "raw"})
    reference = tmp_path / "reference.fasta"
    reference.write_text(">ref\nACGT\n")
    with store.build("sequences", "h3") as build:
        (build.path / "part.parquet").write_text("rows")
        ref = build.publish(
            provenance(raw, ExternalInput.of(reference, version="2026-01-01"), step="align")
        )
    record = json.loads((store.version_dir(ref) / "PROVENANCE.json").read_text())
    assert record["inputs"] == [
        {"store": raw.to_json()},
        {
            "external": {
                "path": str(reference),
                "sha256": ExternalInput.of(reference).sha256,
                "version": "2026-01-01",
            }
        },
    ]
    with pytest.raises(FileNotFoundError):
        ExternalInput.of(tmp_path / "absent")


def test_snapshots_and_report_manifests(store: Store, tmp_path: Path) -> None:
    tree = publish(store, "trees", "h3/report", {"tree.nwk": "(a,b);"})
    chain = publish(store, "chains", "labx/h3-hi", {"chain.json": "1"})
    snapshot = write_snapshot(store, [tree, chain], "example")
    assert write_snapshot(store, [chain, tree], "same refs, other order") == snapshot
    assert set(read_snapshot(store, snapshot)) == {tree, chain}

    manifest = write_manifest(tmp_path / "report" / "manifest.json", [tree, chain], "report")
    refs = read_manifest(manifest)
    assert check_refs(store, refs, deep=True) == []

    missing = StoreRef("trees", "h1/report", tree.version, tree.manifest_sha256)
    problems = check_refs(store, [*refs, missing], deep=False)
    assert len(problems) == 1 and "h1/report" in problems[0]
    with pytest.raises(StoreError, match="cannot snapshot"):
        write_snapshot(store, [missing], "broken")
    with pytest.raises(StoreError, match="only once"):
        write_manifest(tmp_path / "m.json", [tree, tree], "dup")


def test_store_ref_validation() -> None:
    good = {"kind": "trees", "dataset": "h3", "version": "a" * 16, "manifest_sha256": "a" * 64}
    assert StoreRef.from_json(good).to_json() == good
    with pytest.raises(StoreError, match="exactly the keys"):
        StoreRef.from_json({**good, "extra": 1})
    with pytest.raises(StoreError, match="prefix"):
        StoreRef.from_json({**good, "manifest_sha256": "b" * 64})
