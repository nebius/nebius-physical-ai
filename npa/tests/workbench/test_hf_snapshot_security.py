"""Freeze Hub transfers to resolved commits and preserve their provenance."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from npa.fiftyone_lerobot import _download_huggingface_source
from npa.workbench.lerobot.policy_container import (
    PolicyContainerError,
    _download_eval_checkpoint,
)


@pytest.fixture(params=["dataset", "model"])
def consumer(request, tmp_path):
    if request.param == "dataset":
        return (
            lambda reference="owner/dataset@" + "a" * 40: _download_huggingface_source(
                reference, "review", tmp_path
            ),
            tmp_path / "review/lerobot_source/hf",
            ValueError,
            "dataset",
        )
    return (
        lambda reference="owner/policy@" + "a" * 40: _download_eval_checkpoint(
            reference, tmp_path
        ),
        tmp_path / "owner__policy",
        PolicyContainerError,
        "model",
    )


def test_download_and_receipt_use_the_resolved_commit(consumer, monkeypatch):
    download, root, _, repo_type = consumer
    revision = "a" * 40
    calls = []

    def metadata(**kwargs):
        assert kwargs["revision"] == revision
        return SimpleNamespace(sha=revision)

    def snapshot(**kwargs):
        calls.append(kwargs)
        assert kwargs["revision"] == revision
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "selected-weights").write_text(kwargs["revision"])
        return str(target)

    api = SimpleNamespace(dataset_info=metadata, model_info=metadata)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: api, snapshot_download=snapshot),
    )
    target = download()
    assert (target / "selected-weights").read_text() == revision
    receipt = json.loads((target / ".npa-hf-source.json").read_text())
    assert receipt["revision"] == revision
    assert receipt["repo_type"] == repo_type
    if repo_type == "model":
        assert target == root / revision
    assert len(calls) == 1


@pytest.mark.parametrize(
    "revision", [None, "", "main", "a" * 7, "A" * 40, "a" * 39, "a" * 41, "b" * 40, 3]
)
def test_invalid_resolved_revision_never_downloads_or_removes_cache(
    consumer, monkeypatch, revision
):
    download, root, error, _ = consumer
    root.mkdir(parents=True)
    retained = root / "retained-cache-file"
    retained.write_text("previous verified bytes")
    api = SimpleNamespace(
        dataset_info=lambda **_: SimpleNamespace(sha=revision),
        model_info=lambda **_: SimpleNamespace(sha=revision),
    )

    def unexpected_download(**kwargs):
        pytest.fail("invalid revision reached snapshot download")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: api, snapshot_download=unexpected_download),
    )
    with pytest.raises(error, match="immutable commit"):
        download()
    assert retained.read_text() == "previous verified bytes"


@pytest.mark.parametrize(
    "reference",
    ["owner/custom", "owner/custom@main", "owner/custom@v1", "owner/custom@" + "a" * 7],
)
def test_custom_sources_require_operator_selected_commit_before_network(
    consumer, monkeypatch, reference
):
    download, _, error, _ = consumer

    def unexpected_network():
        pytest.fail("unreviewed reference reached Hub metadata")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=unexpected_network, snapshot_download=unexpected_network),
    )
    with pytest.raises(error, match="require owner/repository@<full commit>"):
        download(reference)


def test_bundled_examples_use_reviewed_commits():
    from npa.fiftyone_lerobot import _dataset_snapshot_reference
    from npa.workbench.lerobot.policy_container import _checkpoint_snapshot_reference
    from npa.workflows.lerobot_dataset import DEFAULT_PUBLIC_LEROBOT_REVISION

    assert _dataset_snapshot_reference("lerobot/pusht") == (
        "lerobot/pusht",
        DEFAULT_PUBLIC_LEROBOT_REVISION,
    )
    assert _checkpoint_snapshot_reference("lerobot/diffusion_pusht") == (
        "lerobot/diffusion_pusht",
        "84a7c23178445c6bbf7e1a884ff497017910f653",
    )


def test_distinct_model_revisions_cannot_share_stale_files(tmp_path, monkeypatch):
    revisions = iter(["a" * 40, "b" * 40])

    def snapshot(**kwargs):
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True)
        (target / "weights.bin").write_text(kwargs["revision"])
        return str(target)

    api = SimpleNamespace(model_info=lambda **_: SimpleNamespace(sha=next(revisions)))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: api, snapshot_download=snapshot),
    )
    first = _download_eval_checkpoint("owner/policy@" + "a" * 40, tmp_path)
    (first / "obsolete-file.bin").write_bytes(b"only present in the first revision")
    second = _download_eval_checkpoint("owner/policy@" + "b" * 40, tmp_path)
    assert first != second
    assert (first / "obsolete-file.bin").is_file()
    assert not (second / "obsolete-file.bin").exists()
