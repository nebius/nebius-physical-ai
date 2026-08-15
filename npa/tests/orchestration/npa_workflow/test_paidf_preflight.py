from __future__ import annotations

import json
from types import SimpleNamespace

from npa.orchestration.npa_workflow.paidf_preflight import (
    _ready_schedulable_cpu_nodes,
    kubernetes_prerequisites,
    static_prerequisites,
)


def _nodes(*, cpu="6", memory="24Gi", gpu="0", taints=None):
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


def test_static_preflight_checks_all_runtime_secrets_and_transfer_access():
    checked = []

    def validate(_token, repo):
        checked.append(repo)
        return SimpleNamespace(ok=False, error="403 gated")

    issues = static_prerequisites(
        requested_secret_envs=["HF_TOKEN"],
        secret_values={"HF_TOKEN": "redacted"},
        hf_validator=validate,
    )

    assert checked == ["nvidia/Cosmos-Transfer2.5-2B"]
    rendered = "\n".join(item for item, _ in issues)
    assert "NEBIUS_TOKEN_FACTORY_KEY" in rendered
    assert "AWS_ACCESS_KEY_ID" in rendered
    assert "AWS_SECRET_ACCESS_KEY" in rendered
    assert "Cosmos-Transfer2.5-2B" in rendered
    assert "--capability paidf" in "\n".join(remedy for _, remedy in issues)


def test_static_preflight_passes_with_forwarded_secrets_and_gated_access():
    names = [
        "NEBIUS_TOKEN_FACTORY_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
    ]
    assert (
        static_prerequisites(
            requested_secret_envs=names,
            secret_values={"HF_TOKEN": "redacted"},
            hf_validator=lambda _token, _repo: SimpleNamespace(ok=True),
        )
        == []
    )


def test_cpu_node_parser_budgets_controller_and_one_paidf_cpu_stage():
    assert _ready_schedulable_cpu_nodes(_nodes()) == ["cpu-0"]
    assert _ready_schedulable_cpu_nodes(_nodes(gpu="4")) == ["cpu-0"]
    assert _ready_schedulable_cpu_nodes(_nodes(cpu="5900m")) == []
    assert _ready_schedulable_cpu_nodes(_nodes(memory="23Gi")) == []
    assert _ready_schedulable_cpu_nodes(
        _nodes(taints=[{"key": "dedicated", "effect": "NoSchedule"}])
    ) == []


def test_kubernetes_preflight_reports_cpu_placement_or_api_failure():
    def ready(_args):
        return SimpleNamespace(returncode=0, stdout=_nodes(), stderr="")

    assert kubernetes_prerequisites(runner=ready) == []

    def too_small(_args):
        return SimpleNamespace(returncode=0, stdout=_nodes(cpu="4"), stderr="")

    issue, remedy = kubernetes_prerequisites(runner=too_small)[0]
    assert "6 CPU / 24 GiB" in issue
    assert "8vcpu-32gb" in remedy

    def unavailable(_args):
        return SimpleNamespace(returncode=1, stdout="", stderr="Forbidden")

    assert "cannot be listed" in kubernetes_prerequisites(runner=unavailable)[0][0]
