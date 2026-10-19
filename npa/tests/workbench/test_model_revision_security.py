"""Keep model artifacts immutable without enabling repository-supplied code."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from npa.workbench.cosmos import reason
from npa.workbench.lancedb import bdd100k_udfs as udfs


@pytest.mark.parametrize("revision", ["", "main", "release-tag", "a" * 39, "g" * 40])
def test_custom_reason_repository_requires_immutable_revision(monkeypatch, revision):
    monkeypatch.setenv("NPA_COSMOS_REASON_MODEL_REVISION", revision)
    with pytest.raises(reason.CosmosReasonError, match="immutable"):
        reason._reason_model_revision("operator/custom-reason")


def test_reason_revision_override_and_local_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("NPA_COSMOS_REASON_MODEL_REVISION", "a" * 40)
    assert reason._reason_model_revision("operator/custom-reason") == "a" * 40
    assert reason._reason_model_revision(str(tmp_path)) is None


def test_clip_model_and_processor_share_the_same_immutable_snapshot(monkeypatch):
    import torch

    model = Mock()
    model.to.return_value = model
    model.float.return_value = model
    model_loader = Mock(return_value=model)
    processor_loader = Mock(return_value=object())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            CLIPModel=SimpleNamespace(from_pretrained=model_loader),
            CLIPProcessor=SimpleNamespace(from_pretrained=processor_loader),
        ),
    )
    monkeypatch.setattr(udfs, "_CLIP_CACHE", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    loaded, *_ = udfs._clip_components(device="cpu", precision="float32")

    assert loaded is model
    for loader in (model_loader, processor_loader):
        loader.assert_called_once_with(
            "openai/clip-vit-base-patch32",
            revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        )
