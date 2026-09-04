from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from npa.orchestration.npa_workflow.sim2real_preflight import (
    _ready_schedulable_cpu_nodes,
    kubernetes_prerequisites,
    static_prerequisites,
)


def _config(**overrides):
    digest = f"registry.example/registry/image@sha256:{'a' * 64}"
    config = {
        "controller_image": digest,
        "transfer_image": digest,
        "envgen_image": digest,
        "isaac_image": digest,
        "viewer_image": digest,
        "isaac_cache_pvc": "npa-isaac-cache",
        "cosmos3_model": "nvidia/Cosmos3-Super-Reasoner",
    }
    config.update(overrides)
    return config


def test_static_preflight_checks_hf_models_and_hosted_model_before_submission():
    checked = []
    hosted = []

    def validate(_token, repo):
        checked.append(repo)
        return SimpleNamespace(ok=repo != "nvidia/Cosmos-Transfer2.5-2B", error="403")

    issues = static_prerequisites(
        _config(),
        requested_secret_envs=["HF_TOKEN", "NEBIUS_TOKEN_FACTORY_KEY"],
        secret_values={"HF_TOKEN": "redacted", "NEBIUS_TOKEN_FACTORY_KEY": "redacted"},
        hf_validator=validate,
        token_factory_validator=lambda _key, model: hosted.append(model) or SimpleNamespace(ok=True),
    )

    assert checked == ["nvidia/Cosmos-Transfer2.5-2B"]
    assert "nvidia/Cosmos-Reason2-8B" not in checked
    assert hosted == ["nvidia/Cosmos3-Super-Reasoner"]
    rendered = "\n".join(item for item, _ in issues)
    assert "Cosmos-Transfer2.5-2B" in rendered
    assert "AWS_ACCESS_KEY_ID" in rendered
    assert "AWS_SECRET_ACCESS_KEY" in rendered


def test_archived_reason3_config_key_does_not_change_canonical_hosted_probe():
    checked = []
    config = _config()
    config.pop("cosmos3_model")
    config["reason3_model"] = "nvidia/Cosmos-Reason2-2B"

    static_prerequisites(
        config,
        requested_secret_envs=[
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "HF_TOKEN", "NEBIUS_TOKEN_FACTORY_KEY",
        ],
        secret_values={"HF_TOKEN": "redacted", "NEBIUS_TOKEN_FACTORY_KEY": "redacted"},
        hf_validator=lambda _token, repo: checked.append(repo)
        or SimpleNamespace(ok=True),
        token_factory_validator=lambda _key, model: checked.append(model) or SimpleNamespace(ok=True),
    )

    assert "nvidia/Cosmos3-Super-Reasoner" in checked
    assert "nvidia/Cosmos-Reason2-2B" not in checked


def test_static_preflight_rejects_mutable_images_without_manual_eula_inputs():
    issues = static_prerequisites(
        _config(controller_image="registry/image:latest"),
        requested_secret_envs=[
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "HF_TOKEN",
            "NEBIUS_TOKEN_FACTORY_KEY",
        ],
        secret_values={"HF_TOKEN": "redacted", "NEBIUS_TOKEN_FACTORY_KEY": "redacted"},
        hf_validator=lambda _token, repo: SimpleNamespace(ok=True, repo=repo),
        token_factory_validator=lambda _key, model: SimpleNamespace(ok=True, model=model),
    )

    rendered = "\n".join(item for item, _ in issues)
    assert "controller_image" in rendered
    assert "accept_eula" not in rendered.lower()


def test_static_preflight_rejects_unresolved_token_factory_key():
    issues = static_prerequisites(
        _config(),
        requested_secret_envs=[
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "HF_TOKEN",
            "NEBIUS_TOKEN_FACTORY_KEY",
        ],
        secret_values={"HF_TOKEN": "redacted"},
        hf_validator=lambda _token, repo: SimpleNamespace(ok=True, repo=repo),
        token_factory_validator=lambda *_args: pytest.fail("missing key must not be probed"),
    )
    assert "could not be resolved" in "\n".join(item for item, _ in issues)


def _nodes(*, cpu="10", memory="40Gi", gpu="0", taints=None):
    return json.dumps(
        {
            "items": [
                {
                    "metadata": {"name": "cpu-0"},
                    "spec": {"taints": taints or []},
                    "status": {
                        "allocatable": {
                            "cpu": cpu,
                            "memory": memory,
                            "nvidia.com/gpu": gpu,
                        },
                        "conditions": [{"type": "Ready", "status": "True"}],
                    },
                }
            ]
        }
    )


def test_cpu_node_parser_requires_the_real_schedulable_profile():
    assert _ready_schedulable_cpu_nodes(_nodes()) == ["cpu-0"]
    assert _ready_schedulable_cpu_nodes(_nodes(gpu="8")) == ["cpu-0"]
    assert _ready_schedulable_cpu_nodes(_nodes(cpu="9500m")) == []
    assert _ready_schedulable_cpu_nodes(_nodes(memory="39Gi")) == []
    assert (
        _ready_schedulable_cpu_nodes(
            _nodes(taints=[{"key": "dedicated", "effect": "NoSchedule"}])
        )
        == []
    )


def test_kubernetes_preflight_parses_nodes_and_pvc():
    calls = []

    def run(args):
        calls.append(args)
        if args[:2] == ["get", "nodes"]:
            return SimpleNamespace(returncode=0, stdout=_nodes(), stderr="")
        if args[:2] == ["get", "pvc"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "spec": {"accessModes": ["ReadWriteMany"]},
                        "status": {"phase": "Bound"},
                    }
                ),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    assert kubernetes_prerequisites(_config(), runner=run) == []
    assert [call[:2] for call in calls] == [
        ["get", "nodes"],
        ["get", "pvc"],
    ]


def test_kubernetes_preflight_reports_every_missing_cluster_object_together():
    def run(args):
        if args[:2] == ["get", "nodes"]:
            return SimpleNamespace(returncode=0, stdout=_nodes(cpu="4"), stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="NotFound")

    issues = kubernetes_prerequisites(_config(), runner=run)
    rendered = "\n".join(item for item, _ in issues)
    assert "no Ready" in rendered
    assert "Isaac cache PVC" in rendered
