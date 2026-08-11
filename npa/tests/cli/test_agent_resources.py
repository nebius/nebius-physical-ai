from __future__ import annotations

import json

from npa.cli.agent_resources import (
    build_resource_inventory,
    category_payload,
    configured_k8s_backends,
    discover_mk8s_accelerators,
    discover_nebius_categories,
    format_resource_inventory,
    inventory_summary,
    merge_configured_references,
)


def test_k8s_grounding_normalizes_legacy_config_and_live_node_groups(monkeypatch) -> None:
    configured = configured_k8s_backends(
        {
            "k8s_context": "customer-context",
            "container_registry": "registry.example/customer",
        },
        "customer",
    )
    assert configured[0]["context"] == "customer-context"
    assert configured[0]["raw"] == {"container_registry": "registry.example/customer"}

    class Result:
        returncode = 0
        stdout = json.dumps(
            {
                "items": [
                    {"spec": {"template": {"resources": {"platform": "gpu-rtx6000"}}}},
                    {"spec": {"template": {"resources": {"platform": "cpu-d3"}}}},
                ]
            }
        )

    monkeypatch.setattr("npa.cli.agent_resources.subprocess.run", lambda *_a, **_kw: Result())
    discovered = discover_mk8s_accelerators("cluster-id", ["nebius"], {})
    assert discovered == {
        "available_accelerators": ["RTXPRO6000"],
        "gpu_platforms": ["cpu-d3", "gpu-rtx6000"],
        "gpu_accelerator": "RTXPRO6000",
    }


def test_nested_k8s_grounding_keeps_secret_names_but_redacts_secret_values() -> None:
    configured = configured_k8s_backends(
        {
            "kubernetes": {
                "context": "customer-context",
                "image_pull_secrets": "registry-pull",
                "env_secret_names": "runtime-env",
                "api_token": "must-not-leak",
                "registry_password": "must-not-leak",
            }
        },
        "customer",
    )

    assert configured[0]["raw"]["image_pull_secrets"] == "registry-pull"
    assert configured[0]["raw"]["env_secret_names"] == "runtime-env"
    assert "api_token" not in configured[0]["raw"]
    assert "registry_password" not in configured[0]["raw"]


def test_build_inventory_prefers_metadata_profile_and_includes_local_resources() -> None:
    inventory = build_resource_inventory(
        config={
            "default_project": "demo",
            "projects": {
                "demo": {
                    "project_id": "project-test",
                    "tenant_id": "tenant-test",
                    "region": "us-central1",
                }
            },
        },
        env={"NEBIUS_PROFILE": "stale-profile", "NPA_AGENT_NAME": "paidf"},
        state={"latest_submit": {"run_id": "run-test"}},
        tool_refs=["workbench.cosmos_evaluator.evaluate", "workbench.fiftyone.curate"],
        runner=_runner_for({"compute instance": {"items": [{"metadata": {"name": "paidf"}}]}}),
        generated_at="2026-08-09T00:00:00Z",
        metadata_token_available=True,
        force_refresh=True,
    )

    assert inventory["context"] == {
        "project_alias": "demo",
        "project_id": "project-test",
        "tenant_id": "tenant-test",
        "region": "us-central1",
        "profile": "cursor-sa",
        "profile_status": "authenticated",
    }
    by_id = {item["id"]: item for item in inventory["categories"]}
    assert by_id["compute"]["configured"][0]["name"] == "paidf"
    assert by_id["workbench"]["discovered_count"] == 2
    assert by_id["workflows"]["discovered"][0]["name"] == "run-test"


def _runner_for(payloads):
    def run(command):
        service = command[3]
        operation = " ".join(command[3:5])
        value = payloads.get(operation, payloads.get(service, {"items": []}))
        if isinstance(value, tuple):
            return value
        return 0, json.dumps(value), ""

    return run


def test_discovers_non_empty_and_empty_categories_without_secrets() -> None:
    categories = discover_nebius_categories(
        project_id="project-test",
        tenant_id="tenant-test",
        profile="cursor-sa",
        runner=_runner_for(
            {
                "iam project": {"metadata": {"id": "project-test", "name": "demo-project"}},
                "iam tenant": {"metadata": {"id": "tenant-test", "name": "demo-tenant"}},
                "compute instance": {
                    "items": [
                        {
                            "metadata": {"id": "instance-test", "name": "agent-demo"},
                            "status": {"state": "RUNNING", "token": "must-not-leak"},
                            "credentials": {"password": "must-not-leak"},
                        }
                    ]
                },
            }
        ),
    )

    by_id = {item["id"]: item for item in categories}
    assert by_id["compute"]["status"] == "discovered"
    assert by_id["compute"]["discovered_count"] == 1
    assert by_id["compute"]["discovered"][0] == {
        "kind": "instance",
        "source": "nebius_cli",
        "id": "instance-test",
        "name": "agent-demo",
        "status": "RUNNING",
    }
    assert by_id["kubernetes"]["status"] == "empty"
    rendered = json.dumps(categories)
    assert "must-not-leak" not in rendered
    assert "password" not in rendered


def test_permission_error_is_honest_and_keeps_configured_reference() -> None:
    categories = discover_nebius_categories(
        project_id="project-test",
        tenant_id="tenant-test",
        profile="cursor-sa",
        runner=lambda _command: (1, "", "PermissionDenied opaque-debug-token-should-not-leak"),
    )
    categories = merge_configured_references(
        categories,
        {"storage": [{"kind": "bucket", "name": "configured-bucket", "source": "staged_credentials"}]},
    )
    storage = next(item for item in categories if item["id"] == "storage")
    assert storage["status"] == "error"
    assert storage["configured_count"] == 1
    assert storage["discovered_count"] == 0
    assert storage["error"] == {
        "kind": "permission_denied",
        "message": "Credentials are authenticated but cannot enumerate this resource category.",
    }
    assert "opaque-debug-token" not in json.dumps(storage)


def test_category_states_and_summary_are_explicit() -> None:
    categories = [
        category_payload("a", "A", discovered=[{"name": "one"}]),
        category_payload("b", "B", configured=[{"name": "two"}], discovery_attempted=False),
        category_payload("c", "C"),
        category_payload("d", "D", error={"kind": "authentication_error", "message": "no"}),
    ]
    assert [item["status"] for item in categories] == ["discovered", "configured", "empty", "error"]
    assert inventory_summary(categories) == {
        "categories": 4,
        "discovered_categories": 1,
        "configured_only_categories": 1,
        "empty_categories": 1,
        "error_categories": 1,
        "configured_resources": 1,
        "discovered_resources": 1,
    }


def test_grounded_inventory_reply_reports_counts_and_errors() -> None:
    inventory = {
        "context": {
            "project_alias": "demo",
            "project_id": "project-test",
            "tenant_id": "tenant-test",
            "region": "us-central1",
            "profile": "cursor-sa",
        },
        "categories": [
            category_payload("compute", "Compute", discovered=[{"name": "agent-demo"}]),
            category_payload(
                "storage",
                "Object storage",
                configured=[{"name": "configured-bucket"}],
                error={"kind": "permission_denied", "message": "Not enumerable."},
            ),
        ],
    }
    reply = format_resource_inventory(inventory)
    assert "**Tenant resources**" in reply
    assert "**discovered_resources**: `1`" in reply
    assert "status=`error`" in reply
    assert "discovery_error=`permission_denied`" in reply
