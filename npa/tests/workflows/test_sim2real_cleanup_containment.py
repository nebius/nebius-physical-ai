"""Cleanup controls use only task-owned directories and sentinel bytes."""

from types import SimpleNamespace

import pytest

from npa.clients.storage import StorageError
import npa.workflows.sim2real_rerun_regen as regen


def test_public_sync_rejects_symlinked_cleanup_ancestor(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside"
    (outside / "gold-heldout").mkdir(parents=True)
    sentinel = outside / "gold-heldout" / "sentinel.txt"
    sentinel.write_bytes(b"unrelated fixture bytes")
    (work / "eval").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        regen,
        "_latest_completed_inner_evidence_rel",
        lambda *_args: "inner_loop/outer-01/evidence.json",
    )
    config = SimpleNamespace(
        run_id="fixture-run", s3_bucket="demo-bucket", s3_prefix="runs"
    )
    with pytest.raises(regen.Sim2RealRerunRegenError, match="symlinked ancestor"):
        regen.sync_regen_inputs(
            config, work, client=object(), publication_snapshot=object()
        )
    assert sentinel.read_bytes() == b"unrelated fixture bytes"


@pytest.mark.parametrize("failure", ["storage", "unexpected", "incomplete"])
def test_download_failure_cleanup_rechecks_ancestors(tmp_path, failure):
    work = tmp_path / "work"
    (work / "eval").mkdir(parents=True)
    outside = tmp_path / "outside"
    (outside / "gold-heldout").mkdir(parents=True)
    sentinel = outside / "gold-heldout" / "sentinel.txt"
    sentinel.write_bytes(b"must survive")

    class Storage:
        def download_directory(self, _uri, _destination):
            (work / "eval").rename(work / "previous-eval")
            (work / "eval").symlink_to(outside, target_is_directory=True)
            if failure == "storage":
                raise StorageError("injected failure")
            if failure == "unexpected":
                raise RuntimeError("injected failure")

    with pytest.raises(regen.Sim2RealRerunRegenError, match="symlinked ancestor"):
        regen._download_directory_fresh(
            Storage(),
            "s3://demo-bucket/fixture/",
            work / "eval" / "gold-heldout",
            containment_root=work,
        )
    assert sentinel.read_bytes() == b"must survive"


def test_cleanup_rejects_symlinked_root_and_outside_target(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_bytes(b"must survive")
    root = tmp_path / "root"
    root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(regen.Sim2RealRerunRegenError, match="symlinked ancestor"):
        regen._remove_tree(root / "sentinel.txt", containment_root=root)
    with pytest.raises(regen.Sim2RealRerunRegenError, match="outside"):
        regen._remove_tree(sentinel, containment_root=tmp_path / "work")
    assert sentinel.read_bytes() == b"must survive"


def test_cleanup_only_removes_contained_scopes_and_unlinks_owned_leaf(tmp_path):
    (tmp_path / "inner_loop").mkdir()
    (tmp_path / "eval" / "gold-heldout").mkdir(parents=True)
    sentinel = tmp_path / "unrelated.txt"
    sentinel.write_bytes(b"preserve")
    regen._clear_regen_pair_scopes(tmp_path)
    assert sentinel.read_bytes() == b"preserve"
    assert not (tmp_path / "inner_loop").exists()
    assert not (tmp_path / "eval" / "gold-heldout").exists()
    leaf = tmp_path / "owned-leaf"
    leaf.symlink_to(sentinel)
    regen._remove_tree(leaf, containment_root=tmp_path)
    assert not leaf.is_symlink()
    assert sentinel.read_bytes() == b"preserve"
