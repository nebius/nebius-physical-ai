from __future__ import annotations

import re
from pathlib import Path

import yaml

from npa.workbench.antioch.openpi_bridge import ACTION_SHAPE, render_stack


REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_ROOT = REPO_ROOT / "skills/tools/antioch"
SKILL = SKILL_ROOT / "SKILL.md"
CONTRACTS = SKILL_ROOT / "references/contracts.yaml"
INDEX = REPO_ROOT / "skills/index.yaml"


def _contracts() -> dict[str, object]:
    value = yaml.safe_load(CONTRACTS.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rendered_items() -> list[dict[str, object]]:
    digest = "sha256:" + "a" * 64
    stack = render_stack(
        run_id="skill-contract",
        namespace="example",
        policy_image=f"registry.example/policy@{digest}",
        bridge_image=f"registry.example/bridge@{digest}",
        policy_terms_secret="model-terms",
        isaac_acceptance_secret="isaac-terms",
        antioch_config_secret="antioch-session",
        policy_gpu_selector_key="gpu.example/product",
        policy_gpu_selector_value="policy-gpu",
        bridge_gpu_selector_key="gpu.example/product",
        bridge_gpu_selector_value="render-gpu",
        policy_cache_pvc="model-cache",
    )
    return stack["items"]  # type: ignore[return-value]


def test_antioch_skill_references_are_direct_discoverable_and_indexed() -> None:
    text = SKILL.read_text(encoding="utf-8")
    linked = set(re.findall(r"\]\((references/[^)]+)\)", text))
    actual = {
        str(path.relative_to(SKILL_ROOT))
        for path in (SKILL_ROOT / "references").iterdir()
        if path.is_file()
    }
    assert linked == actual
    assert all("/" not in path.removeprefix("references/") for path in linked)

    index = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    entry = next(item for item in index["skills"] if item["name"] == "antioch")
    indexed_resources = {
        smoke["path"] for smoke in entry["smoke"] if smoke["type"] == "file_exists"
    }
    assert indexed_resources == {
        f"skills/tools/antioch/{relative}" for relative in actual
    }
    assert all((REPO_ROOT / path).is_file() for path in indexed_resources)


def test_antioch_contract_matches_rendered_network_and_secret_boundaries() -> None:
    contracts = _contracts()
    boundary = contracts["policy_boundary"]
    assert tuple(boundary["action_shape"]) == ACTION_SHAPE
    assert boundary["finite"] is True
    assert boundary["fail_closed"] is True
    assert boundary["production_mode"] == "continuous_soft_real_time"
    assert boundary["smoke_mode"] == "finite_one_observation"
    assert boundary["hard_real_time"] is False
    assert boundary["maximum_in_flight_requests"] == 1
    assert boundary["observation_queue_capacity"] == 1
    assert boundary["response_queue_capacity"] == 1
    assert boundary["reconnect_behavior"] == "reset_control_epoch"

    items = _rendered_items()
    deployment, service, bridge, network_policy = items
    assert bridge["kind"] == "Deployment"
    assert service["spec"]["type"] == "ClusterIP"
    assert service["spec"]["ports"] == [
        {"name": "policy", "port": boundary["service"]["port"], "targetPort": 8000}
    ]
    assert network_policy["spec"]["policyTypes"] == ["Ingress"]

    policy_text = yaml.safe_dump(deployment)
    bridge_text = yaml.safe_dump(bridge)
    assert "model-terms" in policy_text
    assert "model-terms" not in bridge_text
    assert "antioch-session" not in policy_text
    assert "antioch-session" in bridge_text
    assert "isaac-terms" not in policy_text
    assert "isaac-terms" in bridge_text

    policy_spec = deployment["spec"]["template"]["spec"]
    assert (
        policy_spec["initContainers"][0]["volumeMounts"][0].get("readOnly") is not True
    )
    assert policy_spec["containers"][0]["volumeMounts"][0]["readOnly"] is True


def test_antioch_contract_rejects_process_only_readiness_and_broad_cleanup() -> None:
    contracts = _contracts()
    readiness = contracts["readiness"]
    required = set(readiness["required_evidence"])
    insufficient = set(readiness["insufficient_evidence"])
    assert required == {
        "supported_api_state",
        "application_health",
        "camera_frames",
        "advancing_camera_sequence",
        "repeated_finite_action_chunks",
        "repeated_safe_target_application",
        "sustained_interval",
    }
    assert {"process_exists", "pid_exists", "tunnel_connected"} <= insufficient
    assert required.isdisjoint(insufficient)

    cleanup = contracts["cleanup"]
    assert cleanup == {
        "scope": "exact_run_resources_only",
        "order": [
            "cancel_exact_run",
            "wait_terminal",
            "stop_exact_services",
            "release_exact_machine",
        ],
    }


def test_antioch_contract_routes_mutation_away_from_read_only_hosts() -> None:
    authority = _contracts()["host_authority"]
    assert authority["mutation_requires"] == "explicit_authorization"
    assert set(authority["read_only_host"].values()) == {"forbidden"}
    assert authority["build_fallback"] == [
        "trusted_registry",
        "authorized_kubernetes",
    ]


def test_antioch_public_contract_covers_all_restricted_payload_classes() -> None:
    prohibited = set(_contracts()["public_artifacts"]["prohibited_payloads"])
    assert {
        "credentials",
        "customer_data",
        "live_infrastructure_identifiers",
        "antioch_local_state",
        "model_weights",
        "populated_model_cache",
        "nvidia_isaac",
        "proprietary_isaac_lab",
        "omniverse_kit",
        "driver_userspace",
    } <= prohibited

    secret_scopes = _contracts()["secret_scopes"]
    assert secret_scopes["model_entitlement"] == "cache_warmer_only"
    assert secret_scopes["model_cache_reader"] == "policy_server_read_only"
    assert secret_scopes["antioch_session"] == "simulator_only"
    assert secret_scopes["simulator_model_entitlement"] == "forbidden"
    assert secret_scopes["policy_antioch_session"] == "forbidden"
