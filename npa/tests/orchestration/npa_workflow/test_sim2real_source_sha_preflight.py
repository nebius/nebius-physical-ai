"""Preflight coverage for the config.source_sha renderer-contract alignment."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from npa.orchestration.npa_workflow.sim2real_preflight import static_prerequisites


def _config(**overrides):
    digest = f"registry.example/registry/image@sha256:{'a' * 64}"
    config = {
        "controller_image": digest,
        "transfer_image": digest,
        "envgen_image": digest,
        "isaac_image": digest,
        "viewer_image": digest,
        "isaac_cache_pvc": "npa-isaac-cache",
        "cosmos3_model": "MiniMaxAI/MiniMax-M3",
        "env_count": "640",
        "train_fraction": "0.8",
        "rollout_count": "64",
        "validation_count": "64",
        "gold_count": "64",
    }
    config.update(overrides)
    return config


def _valid_secret_envs():
    return [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "NEBIUS_TOKEN_FACTORY_KEY",
    ]


def _run_preflight(config):
    return static_prerequisites(
        config,
        requested_secret_envs=_valid_secret_envs(),
        secret_values={"HF_TOKEN": "redacted", "NEBIUS_TOKEN_FACTORY_KEY": "redacted"},
        hf_validator=lambda _token, repo: SimpleNamespace(ok=True, repo=repo),
        token_factory_validator=lambda _key, model: SimpleNamespace(
            ok=True, model=model
        ),
    )


@pytest.mark.parametrize(
    "source_sha",
    [
        "short",
        "not-a-hex-01010101010101010101010101010101",
        "g" * 40,
        ("a" * 39) + "x",
        ("a" * 39) + "G",
        "a" * 41,
    ],
)
@pytest.mark.parametrize("baked", ["1", "0"])
def test_malformed_nonempty_source_sha_is_rejected_in_either_mode(
    source_sha: str, baked: str
) -> None:
    issues = _run_preflight(_config(require_baked_npa=baked, source_sha=source_sha))
    rendered = "\n".join(item for item, _ in issues)
    assert "config.source_sha is not an exact 40-character hexadecimal" in rendered


@pytest.mark.parametrize("baked", ["1", "true", "yes", "on", "TRUE", "On"])
def test_missing_source_sha_is_required_for_every_enabled_truthy_form(
    baked: str,
) -> None:
    issues = _run_preflight(_config(require_baked_npa=baked))
    rendered = "\n".join(item for item, _ in issues)
    assert "config.source_sha is missing" in rendered


@pytest.mark.parametrize("baked", ["0", "false", "no", "off", "", None])
def test_empty_source_sha_is_allowed_outside_baked_mode(baked: str | None) -> None:
    config = _config(require_baked_npa=baked)
    config.pop("source_sha", None)
    issues = _run_preflight(config)
    assert not any("source_sha" in item for item, _ in issues)


@pytest.mark.parametrize("baked", ["1", "0"])
def test_exact_source_sha_is_accepted_in_either_mode(baked: str) -> None:
    issues = _run_preflight(_config(require_baked_npa=baked, source_sha="a" * 40))
    assert not any("source_sha" in item for item, _ in issues)


@pytest.mark.parametrize("baked", ["1", "0"])
def test_uppercase_and_whitespace_are_normalized_to_valid_sha(baked: str) -> None:
    for value in ("  " + ("A" * 40) + "  ", "\t" + ("A" * 40) + "\n"):
        issues = _run_preflight(_config(require_baked_npa=baked, source_sha=value))
        assert not any("source_sha" in item for item, _ in issues)
