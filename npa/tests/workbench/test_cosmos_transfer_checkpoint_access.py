from __future__ import annotations

import pytest

from npa.clients import huggingface
from npa.clients.huggingface import HFAccessResult
from npa.workbench.cosmos.checkpoint_access import (
    CosmosCheckpointAccessError,
    preflight_control_checkpoint_access,
)
from npa.workbench.cosmos.control_contract import COSMOS_TRANSFER_CHECKPOINTS


def test_exact_modality_checkpoint_contract_is_pinned() -> None:
    assert set(COSMOS_TRANSFER_CHECKPOINTS) == {"edge", "vis", "depth", "seg"}
    assert COSMOS_TRANSFER_CHECKPOINTS["vis"].upstream_key == "blur"
    assert COSMOS_TRANSFER_CHECKPOINTS["depth"].repo == ("nvidia/Cosmos-Transfer2.5-2B")
    assert COSMOS_TRANSFER_CHECKPOINTS["depth"].filename.startswith("general/depth/")
    rendered = repr(COSMOS_TRANSFER_CHECKPOINTS).lower()
    assert "video-depth-anything" not in rendered
    assert "depth-anything/video" not in rendered


def test_missing_token_fails_before_probe() -> None:
    called = False

    def validator(*_args):
        nonlocal called
        called = True

    with pytest.raises(CosmosCheckpointAccessError, match="HF_TOKEN is required"):
        preflight_control_checkpoint_access(
            modality="depth", token="", validator=validator
        )
    assert called is False


def test_unknown_modality_is_a_clean_preflight_error() -> None:
    with pytest.raises(CosmosCheckpointAccessError, match="unsupported.*bogus"):
        preflight_control_checkpoint_access(modality="bogus", token="caller-owned")


@pytest.mark.parametrize("status", [401, 403])
def test_denied_access_is_actionable_and_redacted(status: int) -> None:
    secret = "hf_SENTINEL_NEVER_PRINT"

    def denied(token, repo, revision, filename):
        assert token == secret
        return HFAccessResult(
            repo=repo,
            revision=revision,
            filename=filename,
            ok=False,
            status_code=status,
            error=f"provider leaked {secret}",
        )

    with pytest.raises(CosmosCheckpointAccessError) as excinfo:
        preflight_control_checkpoint_access(
            modality="seg", token=secret, validator=denied
        )
    message = str(excinfo.value)
    assert str(status) in message
    assert "huggingface.co/nvidia/Cosmos-Transfer2.5-2B" in message
    assert secret not in message


@pytest.mark.parametrize("modality", ["edge", "vis", "depth", "seg"])
def test_selected_modality_probes_only_its_exact_checkpoint(modality: str) -> None:
    seen: list[tuple[str, str, str]] = []

    def accepted(_token, repo, revision, filename):
        seen.append((repo, revision, filename))
        return HFAccessResult(
            repo=repo,
            revision=revision,
            filename=filename,
            ok=True,
            status_code=302,
        )

    evidence = preflight_control_checkpoint_access(
        modality=modality, token="caller-owned", validator=accepted
    )
    checkpoint = COSMOS_TRANSFER_CHECKPOINTS[modality]
    assert seen == [(checkpoint.repo, checkpoint.revision, checkpoint.filename)]
    assert evidence["modality"] == modality
    assert evidence["status_code"] == 302


def test_transient_or_unknown_access_failure_fails_closed() -> None:
    def transient(_token, repo, revision, filename):
        return HFAccessResult(
            repo=repo,
            revision=revision,
            filename=filename,
            ok=False,
            status_code=503,
            error="network unavailable",
        )

    with pytest.raises(CosmosCheckpointAccessError, match="unverified"):
        preflight_control_checkpoint_access(
            modality="depth", token="caller-owned", validator=transient
        )


@pytest.mark.parametrize(
    "location",
    [
        "/api/resolve-cache/models/nvidia/checkpoint",
        "https://cdn-lfs-us-1.hf.co/signed-object?X-Amz-Signature=abc",
        "https://cas-bridge.xethub.hf.co/signed-object?X-Xet-Signed=abc",
        "https://us.aws.cdn.hf.co/xet-bridge-us/object?X-Xet-Signed=abc",
        "https://cdn.hf.co/xet-bridge/object?X-Xet-Signed=abc",
    ],
)
def test_exact_file_probe_accepts_only_known_hf_redirects(
    monkeypatch: pytest.MonkeyPatch, location: str
) -> None:
    class Response:
        status_code = 302
        headers = {"location": location}

    monkeypatch.setattr(huggingface.httpx, "head", lambda *_a, **_k: Response())
    result = huggingface.validate_hf_file_access(
        "caller-owned", "nvidia/model", "revision", "checkpoint.pt"
    )
    assert result.ok is True


@pytest.mark.parametrize(
    ("status", "location"),
    [
        (302, ""),
        (302, "https://attacker.invalid/object"),
        (302, "//attacker.invalid/object"),
        (302, "/login?next=/nvidia/model/resolve/revision/checkpoint.pt"),
        (302, "/nvidia/model"),
        (302, "https://huggingface.co/join"),
        (302, "https://cdn-lfs-us-1.hf.co/unsigned-object"),
        (302, "https://cdn.hf.co.attacker.invalid/signed?X-Xet-Signed=abc"),
        (302, "https://evilcdn.hf.co/signed?X-Xet-Signed=abc"),
        (302, "http://us.aws.cdn.hf.co/signed?X-Xet-Signed=abc"),
        (302, "https://user@us.aws.cdn.hf.co/signed?X-Xet-Signed=abc"),
        (304, ""),
    ],
)
def test_exact_file_probe_rejects_arbitrary_three_xx(
    monkeypatch: pytest.MonkeyPatch, status: int, location: str
) -> None:
    class Response:
        status_code = status
        headers = {"location": location}

    monkeypatch.setattr(huggingface.httpx, "head", lambda *_a, **_k: Response())
    result = huggingface.validate_hf_file_access(
        "caller-owned", "nvidia/model", "revision", "checkpoint.pt"
    )
    assert result.ok is False
    assert "caller-owned" not in result.error
