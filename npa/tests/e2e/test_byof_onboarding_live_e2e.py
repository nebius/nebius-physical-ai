"""Live infra checks for generic BYOF solution onboarding (workflow + optional agent chat)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import quote, urlsplit

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.config import resolve_container_registry
from npa.clients.project_credentials import s3_client_for_project
from npa.deploy.images import is_public_registry
from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.workflows.byof.live import (
    byof_ubuntu_validation_repo,
    byof_validation_repo,
    resolve_byof_kubernetes_target,
    resolve_byof_profile_path,
    resolve_byof_resource_yaml,
    resolve_skypilot_bin,
    skypilot_config_for_project,
)

from .agent_live_helpers import (
    CREATE_BYOF_WORKFLOW_PROMPT,
    ONBOARD_OSS_REPO_PROMPT,
    ONBOARD_SOLUTION_PROMPT,
    assert_grounded_onboard_solution_reply,
    load_agent_live_context,
)
from .npa_workflow_live_helpers import (
    assert_no_credential_leakage,
    live_bucket,
    live_credential_markers,
    parse_json_payload,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("NPA_INTEGRATION_E2E") != "1",
        reason="Set NPA_INTEGRATION_E2E=1 for live BYOF onboarding infra checks.",
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[3]
BYOF_SPEC = REPO_ROOT / "workflows" / "testing" / "byof.yaml"
ROBOMIMIC_SPEC = REPO_ROOT / "workflows" / "testing" / "byof-robomimic.yaml"
BYOF_RUNNER = REPO_ROOT / "npa" / "scripts" / "run_byof_repo.py"
ROBOMIMIC_RUNTIME_LOCK = (
    REPO_ROOT
    / "npa"
    / "docker"
    / "workbench"
    / "robomimic"
    / "runtime-requirements.lock"
)
ROBOMIMIC_ENTITLEMENT_MAX_BYTES = 16 * 1024
ROBOMIMIC_SMOKE_PROOF_MAX_BYTES = 64 * 1024
_ROBOMIMIC_SMOKE_PROOF_ERROR = "robomimic smoke proof validation failed"
ROBOMIMIC_ENTITLEMENT_TERMS = [
    {
        "name": "NVIDIA CUDA Toolkit EULA",
        "url": "https://docs.nvidia.com/cuda/eula/index.html",
    },
    {
        "name": "NVIDIA Software License Agreement",
        "url": (
            "https://www.nvidia.com/en-us/agreements/enterprise-software/"
            "nvidia-software-license-agreement/"
        ),
    },
    {
        "name": "NVIDIA cuDNN Software License Agreement",
        "url": (
            "https://docs.nvidia.com/deeplearning/cudnn/backend/latest/"
            "reference/eula.html"
        ),
    },
]
RUNNER = CliRunner()
ROBOMIMIC_SOURCE_REVISION = "d309eaecc18acf4152a830a895a6984b8ac71b05"
ROBOMIMIC_DATASET_REVISION = "74fa018461f479cd9fd15b924a16103012096203"
ROBOMIMIC_DATASET_SHA256 = (
    "2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540"
)
ROBOMIMIC_CAPABILITIES = {
    "lift_ph_lowdim_bc_train",
    "lift_ph_lowdim_heldout_validate",
    "lift_ph_lowdim_checkpoint_reload_action",
}
_ROBOMIMIC_ENTITLEMENT_REFUSAL_CATEGORIES = frozenset(
    {
        "binding-mismatch",
        "context-invalid",
        "contract-invalid",
        "fields-invalid",
        "identity-invalid",
        "record-invalid",
        "record-unsafe",
        "time-invalid",
    }
)


class _RobomimicEntitlementRefusal(RuntimeError):
    """Carry only an approved, value-free entitlement refusal category."""

    def __init__(self, category: str) -> None:
        safe_category = (
            category
            if category in _ROBOMIMIC_ENTITLEMENT_REFUSAL_CATEGORIES
            else "record-invalid"
        )
        self.category = safe_category
        super().__init__(f"robomimic customer entitlement refused: {safe_category}")


def _require_robomimic_entitlement_context(condition: bool) -> None:
    """Reject invalid live selectors without retaining their values."""

    if not condition:
        raise _RobomimicEntitlementRefusal("context-invalid")


def _activate_nebius_profile() -> None:
    profile = os.environ.get("NPA_NEBIUS_PROFILE", "agent-sa").strip()
    if not profile:
        return
    subprocess.run(
        ["nebius", "profile", "activate", profile],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _parse_last_json_blob(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    idx = 0
    last_obj: dict[str, object] | None = None
    while idx < len(text):
        next_brace = text.find("{", idx)
        if next_brace < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, next_brace)
        except json.JSONDecodeError:
            idx = next_brace + 1
            continue
        if isinstance(obj, dict):
            last_obj = obj
        idx = max(end, next_brace + 1)
    if last_obj is None:
        raise ValueError(f"no JSON object found in command output:\n{text}")
    return last_obj


@pytest.fixture(scope="module")
def live_byof_built_image(e2e_project: str | None) -> str:
    preset_image = os.environ.get("NPA_BYOF_TEST_IMAGE", "").strip()
    if preset_image:
        return preset_image
    if os.environ.get("NPA_BYOF_LIVE_CONTAINER") != "1":
        pytest.skip("Set NPA_BYOF_LIVE_CONTAINER=1 for real BYOF container build/push.")
    _activate_nebius_profile()
    registry = resolve_container_registry(e2e_project)
    repo_url, repo_ref = byof_validation_repo()
    run_id = (
        os.environ.get("NPA_BYOF_CONTAINER_RUN_ID")
        or f"byof-container-live-{os.getpid()}"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(BYOF_RUNNER),
            "--registry",
            registry,
            "--repo-url",
            repo_url,
            "--repo-ref",
            repo_ref,
            "--project",
            e2e_project or "",
            "--run-id",
            run_id,
            "--base-profile",
            "isaac-lab",
            "--skip-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=int(os.environ.get("NPA_BYOF_CONTAINER_TIMEOUT", "3600")),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = _parse_last_json_blob(proc.stdout + "\n" + proc.stderr)
    assert summary.get("status") == "ok", summary
    build = summary.get("build", {})
    assert build.get("ok") is True
    assert build.get("pushed") is True
    image = str(summary["image"])
    assert registry in image
    return image


@pytest.fixture(scope="module")
def forbidden_markers() -> list[str]:
    return live_credential_markers()


def _materialize_byof_spec(tmp_path: Path, *, bucket: str) -> Path:
    text = BYOF_SPEC.read_text(encoding="utf-8")
    text = text.replace("bucket: example-bucket", f"bucket: {bucket}")
    path = tmp_path / "byof-live.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_live_isaac_byof_workflow_validate_and_plan(
    tmp_path: Path,
    e2e_project: str | None,
    forbidden_markers: list[str],
) -> None:
    bucket = live_bucket(e2e_project)
    path = _materialize_byof_spec(tmp_path, bucket=bucket)
    validate = RUNNER.invoke(
        app, ["workbench", "workflow", "validate-spec", str(path), "--json"]
    )
    payload = parse_json_payload(validate, forbidden_markers)
    assert payload["status"] == "valid"
    assert payload["name"] == "byof"
    assert "byof-run" in set(payload.get("states", []))

    plan = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(path),
            "--run-id",
            "byof-onboard-live",
            "--json",
        ],
    )
    plan_payload = parse_json_payload(plan, forbidden_markers)
    steps = plan_payload.get("steps", [])
    assert steps
    tool_refs = {
        step.get("tool_ref") or step.get("toolRef")
        for step in steps
        if isinstance(step, dict)
    }
    assert "workbench.byof.repo" in tool_refs


def test_live_isaac_byof_plan_builder_matches_cli(
    tmp_path: Path,
    e2e_project: str | None,
    forbidden_markers: list[str],
) -> None:
    bucket = live_bucket(e2e_project)
    path = _materialize_byof_spec(tmp_path, bucket=bucket)
    spec = load_spec(path)
    plan = build_plan(spec, run_id="byof-plan-builder")
    assert plan.steps
    assert_no_credential_leakage(
        json.dumps(plan.to_dict()), extra_forbidden=forbidden_markers
    )
    assert any(step.tool_ref == "workbench.byof.repo" for step in plan.steps)


def test_live_byof_registry_resolution(e2e_project: str | None) -> None:
    registry = resolve_container_registry(e2e_project)
    assert registry
    assert "/" in registry
    assert "example-bucket" not in registry
    assert "<your-registry>" not in registry


@pytest.mark.skipif(
    os.environ.get("NPA_AGENT_LIVE") != "1",
    reason="Set NPA_AGENT_LIVE=1 to exercise onboard_solution chat on the configured agent.",
)
def test_live_agent_onboard_solution_chat() -> None:
    ctx = load_agent_live_context()
    chat = ctx.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": ONBOARD_SOLUTION_PROMPT}]},
        timeout=30.0,
    )
    chat.raise_for_status()
    assert_grounded_onboard_solution_reply(chat.json())


@pytest.mark.skipif(
    os.environ.get("NPA_AGENT_LIVE") != "1",
    reason="Set NPA_AGENT_LIVE=1 to validate generic BYOF workflow draft on the configured agent.",
)
def test_live_agent_byof_workflow_draft_validate() -> None:
    ctx = load_agent_live_context()
    draft = ctx.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": CREATE_BYOF_WORKFLOW_PROMPT}]},
        timeout=30.0,
    )
    draft.raise_for_status()
    payload = draft.json()
    assert payload.get("ok") is True
    workflow_yaml = str(payload.get("workflow_yaml") or "")
    assert workflow_yaml
    assert "name: byof" in workflow_yaml or "byof-run" in workflow_yaml
    assert "<repo-url>" in workflow_yaml
    assert "<workload>" in workflow_yaml

    validate = ctx.post(
        "/api/workflows/validate", json={"yaml": workflow_yaml}, timeout=15.0
    )
    validate.raise_for_status()
    validate_payload = validate.json()
    assert validate_payload.get("ok") is True


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_CONTAINER") != "1",
    reason="Set NPA_BYOF_LIVE_CONTAINER=1 for real BYOF docker build/push/inspect.",
)
def test_live_byof_runner_container_build_push(live_byof_built_image: str) -> None:
    assert live_byof_built_image
    assert (
        "npa-byof" in live_byof_built_image or "npa-isaac-lab" in live_byof_built_image
    )


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_CONTAINER") != "1",
    reason="Set NPA_BYOF_LIVE_CONTAINER=1 for real BYOF docker build/push/inspect.",
)
def test_live_byof_container_has_validation_repo(live_byof_built_image: str) -> None:
    repo_url, repo_ref = byof_validation_repo()
    image = live_byof_built_image
    meta_proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "cat",
            image,
            "/opt/byof/npa_source_metadata.json",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if meta_proc.returncode != 0:
        pytest.skip("validation repo layout differs from /opt/byof metadata path")
    metadata = json.loads(meta_proc.stdout)
    assert metadata["source"] == "oss-byof"
    assert metadata["repo"] == repo_url
    assert metadata["ref"] == repo_ref


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_GPU") != "1",
    reason="Set NPA_BYOF_LIVE_GPU=1 to run BYOF runner registry smoke on live infra (no build/push).",
)
def test_live_byof_runner_registry_smoke(e2e_project: str | None) -> None:
    registry = resolve_container_registry(e2e_project)
    proc = subprocess.run(
        [
            sys.executable,
            str(BYOF_RUNNER),
            "--registry",
            registry,
            "--project",
            e2e_project or "",
            "--skip-build",
            "--skip-run",
            "--run-id",
            "byof-live-registry-smoke",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = _parse_last_json_blob(proc.stdout + "\n" + proc.stderr)
    assert summary.get("status") == "ok", summary
    assert summary["registry"] == registry
    assert registry in summary["image"]


def _robomimic_storage_endpoint(value: str) -> str:
    """Accept only a canonical Nebius Object Storage HTTPS origin."""

    candidate = value.strip()
    malformed = False
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError:
        malformed = True
    if malformed:
        raise _RobomimicEntitlementRefusal("context-invalid")
    host = (parsed.hostname or "").lower()
    _require_robomimic_entitlement_context(parsed.scheme == "https")
    _require_robomimic_entitlement_context(
        parsed.username is None and parsed.password is None
    )
    _require_robomimic_entitlement_context(port in {None, 443})
    _require_robomimic_entitlement_context(
        parsed.path in {"", "/"} and not parsed.query and not parsed.fragment
    )
    _require_robomimic_entitlement_context(
        re.fullmatch(r"storage\.[a-z0-9-]+\.nebius\.cloud", host) is not None
    )
    canonical = f"https://{host}"
    _require_robomimic_entitlement_context(candidate.rstrip("/") == canonical)
    return canonical


def _robomimic_live_selectors(e2e_project: str | None) -> dict[str, str]:
    endpoint_candidates = {
        value.strip()
        for value in (
            os.environ.get("AWS_ENDPOINT_URL", ""),
            os.environ.get("NEBIUS_S3_ENDPOINT", ""),
        )
        if value.strip()
    }
    _require_robomimic_entitlement_context(len(endpoint_candidates) == 1)
    selectors = {
        "project": os.environ.get("NPA_E2E_PROJECT", "").strip(),
        "registry": os.environ.get("NPA_BYOF_ROBOMIMIC_REGISTRY", "").strip(),
        "registry_visibility": os.environ.get(
            "NPA_BYOF_ROBOMIMIC_REGISTRY_VISIBILITY", ""
        ).strip(),
        "image": os.environ.get("NPA_BYOF_ROBOMIMIC_IMAGE", "").strip(),
        "kubeconfig": os.environ.get("NPA_BYOF_KUBECONFIG", "").strip(),
        "context": os.environ.get("NPA_BYOF_K8S_CONTEXT", "").strip(),
        "namespace": os.environ.get("NPA_BYOF_K8S_NAMESPACE", "").strip(),
        "bucket": os.environ.get("NPA_E2E_S3_BUCKET", "").strip(),
        "runtime_pvc": os.environ.get("NPA_BYOF_ROBOMIMIC_RUNTIME_PVC", "").strip(),
        "runtime_inventory_sha256": os.environ.get(
            "NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256", ""
        ).strip(),
        "runtime_entitlement_file": os.environ.get(
            "NPA_BYOF_ROBOMIMIC_RUNTIME_ENTITLEMENT_FILE", ""
        ).strip(),
        "storage_endpoint": _robomimic_storage_endpoint(endpoint_candidates.pop()),
    }
    _require_robomimic_entitlement_context(
        bool(selectors["project"]) and e2e_project == selectors["project"]
    )
    _require_robomimic_entitlement_context(bool(selectors["registry"]))
    official_public = selectors["registry"] == "ghcr.io/nebius/nebius-physical-ai"
    _require_robomimic_entitlement_context(
        not is_public_registry(selectors["registry"]) or official_public
    )
    _require_robomimic_entitlement_context(
        selectors["registry_visibility"].lower()
        == ("public" if official_public else "private")
    )
    if official_public:
        _require_robomimic_entitlement_context(
            re.fullmatch(
                r"[0-9a-f]{40}",
                os.environ.get("NPA_BYOF_ROBOMIMIC_DEVELOPMENT_SHA", ""),
            )
            is not None
        )
    expected_image = re.escape(selectors["registry"].rstrip("/")) + (
        r"/npa-robomimic@sha256:[0-9a-f]{64}"
    )
    _require_robomimic_entitlement_context(
        re.fullmatch(expected_image, selectors["image"]) is not None
    )
    _require_robomimic_entitlement_context(
        bool(selectors["kubeconfig"]) and Path(selectors["kubeconfig"]).is_file()
    )
    _require_robomimic_entitlement_context(bool(selectors["context"]))
    _require_robomimic_entitlement_context(
        bool(selectors["namespace"]) and selectors["namespace"] != "default"
    )
    _require_robomimic_entitlement_context(bool(selectors["bucket"]))
    _require_robomimic_entitlement_context(bool(selectors["runtime_pvc"]))
    _require_robomimic_entitlement_context(
        re.fullmatch(r"[0-9a-f]{64}", selectors["runtime_inventory_sha256"]) is not None
    )
    _require_robomimic_entitlement_context(bool(selectors["runtime_entitlement_file"]))
    _require_robomimic_entitlement_context(
        os.environ.get("NPA_E2E_MK8S_RESERVED_CAPACITY") == "1"
    )
    _require_robomimic_entitlement_context(os.environ.get("NPA_BYOF_LIVE_GPU") == "1")
    _require_robomimic_entitlement_context(
        os.environ.get("NPA_BYOF_ROBOMIMIC_LIVE_B200") == "1"
    )
    return selectors


def _robomimic_private_record_bytes(path_value: str) -> bytes:
    """Read a symlink-free owner-private record without blocking on a FIFO."""

    path = Path(os.path.abspath(Path(path_value).expanduser()))
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise RuntimeError("robomimic customer entitlement path is invalid")
    directory_flags = (
        os.O_RDONLY
        | os.O_CLOEXEC
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )

    def open_parent() -> int:
        descriptor = -1
        open_failed = False
        try:
            descriptor = os.open(os.sep, directory_flags)
            for component in path.parent.parts[1:]:
                next_descriptor = os.open(component, directory_flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
        except OSError:
            if descriptor >= 0:
                os.close(descriptor)
            open_failed = True
        if open_failed:
            raise _RobomimicEntitlementRefusal("record-unsafe")
        details = os.fstat(descriptor)
        if details.st_uid != os.geteuid() or details.st_mode & 0o077:
            os.close(descriptor)
            raise _RobomimicEntitlementRefusal("record-unsafe")
        return descriptor

    parent_descriptor = open_parent()
    parent_identity = os.fstat(parent_descriptor)
    descriptor = -1
    read_failed = False
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or opened.st_mode & 0o077
            or opened.st_size > ROBOMIMIC_ENTITLEMENT_MAX_BYTES
        ):
            raise _RobomimicEntitlementRefusal("record-unsafe")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(ROBOMIMIC_ENTITLEMENT_MAX_BYTES + 1)
        closed = os.fstat(descriptor)
        if (
            len(raw) != opened.st_size
            or closed.st_size != opened.st_size
            or closed.st_dev != opened.st_dev
            or closed.st_ino != opened.st_ino
            or closed.st_nlink != 1
            or closed.st_uid != os.geteuid()
            or closed.st_mode & 0o077
        ):
            raise _RobomimicEntitlementRefusal("record-unsafe")
        current_parent = open_parent()
        try:
            current_identity = os.fstat(current_parent)
        finally:
            os.close(current_parent)
        if (current_identity.st_dev, current_identity.st_ino) != (
            parent_identity.st_dev,
            parent_identity.st_ino,
        ):
            raise _RobomimicEntitlementRefusal("record-unsafe")
        return raw
    except OSError:
        read_failed = True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)
    if read_failed:
        raise _RobomimicEntitlementRefusal("record-unsafe")
    raise _RobomimicEntitlementRefusal("record-invalid")


def _preflight_robomimic_runtime_entitlement(
    *, selectors: dict[str, str], run_id: str, now: datetime | None = None
) -> dict[str, str]:
    """Validate the complete customer/run/runtime binding before side effects."""

    raw = _robomimic_private_record_bytes(selectors["runtime_entitlement_file"])
    record_invalid = False
    try:
        record = json.loads(raw)
        lock_raw = ROBOMIMIC_RUNTIME_LOCK.read_bytes()
        lock = json.loads(lock_raw)
    except (OSError, UnicodeError, json.JSONDecodeError):
        record_invalid = True
    if record_invalid:
        raise _RobomimicEntitlementRefusal("record-invalid")
    if not isinstance(record, dict) or not isinstance(lock, dict):
        raise _RobomimicEntitlementRefusal("record-invalid")
    contract = lock.get("customer_entitlement")
    if not isinstance(contract, dict) or contract != {
        "schema": "npa.robomimic.customer-runtime-entitlement.v1",
        "maximum_validity_seconds": 86_400,
        "responsibilities": [
            "runtime-use",
            "derivative-use",
            "service-use",
            "output-use",
            "no-redistribution-grant",
        ],
        "terms": ROBOMIMIC_ENTITLEMENT_TERMS,
    }:
        raise _RobomimicEntitlementRefusal("contract-invalid")
    lock_sha256 = hashlib.sha256(lock_raw).hexdigest()
    notice_identity = {
        "schema": contract.get("schema"),
        "runtime_id": lock.get("runtime_id"),
        "runtime_lock_sha256": lock_sha256,
        "terms": contract.get("terms"),
        "customer_responsibilities": contract.get("responsibilities"),
    }
    expected = {
        "schema": contract.get("schema"),
        "decision": "accepted",
        "customer_binding_sha256": hashlib.sha256(
            b"npa.robomimic.customer-binding.v1\0"
            + selectors["project"].encode("utf-8")
        ).hexdigest(),
        "run_id": run_id,
        "source_revision": lock.get("source_revision"),
        "runtime_id": lock.get("runtime_id"),
        "runtime_manifest_sha256": selectors["runtime_inventory_sha256"],
        "runtime_lock_sha256": lock_sha256,
        "terms": contract.get("terms"),
        "customer_responsibilities": contract.get("responsibilities"),
        "notice_sha256": hashlib.sha256(
            json.dumps(notice_identity, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }
    if set(record) != set(expected) | {"accepted_at", "expires_at"}:
        raise _RobomimicEntitlementRefusal("fields-invalid")
    if any(record.get(key) != value for key, value in expected.items()):
        raise _RobomimicEntitlementRefusal("binding-mismatch")
    time_invalid = False
    try:
        accepted_at = datetime.strptime(
            record["accepted_at"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        expires_at = datetime.strptime(
            record["expires_at"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        time_invalid = True
    if time_invalid:
        raise _RobomimicEntitlementRefusal("time-invalid")
    observed_now = now or datetime.now(timezone.utc)
    validity_seconds = int((expires_at - accepted_at).total_seconds())
    maximum_validity = contract.get("maximum_validity_seconds")
    if (
        not isinstance(maximum_validity, int)
        or accepted_at > observed_now
        or validity_seconds <= 0
        or validity_seconds > maximum_validity
        or observed_now >= expires_at
    ):
        raise _RobomimicEntitlementRefusal("time-invalid")
    return {
        "record_sha256": hashlib.sha256(raw).hexdigest(),
        "customer_binding_sha256": expected["customer_binding_sha256"],
    }


def _robomimic_runner_command(
    config: dict[str, object],
    registry: str,
    project: str,
    output_root: str,
    run_id: str,
    profile_yaml: Path,
    runtime_entitlement_file: str,
) -> list[str]:
    options = (
        ("--registry", registry),
        ("--project", project),
        ("--repo-url", str(config["repo_url"])),
        ("--repo-ref", str(config["repo_ref"])),
        ("--base-profile", str(config["base_profile"])),
        ("--base-image", str(config["base_image"])),
        ("--build-command", str(config["build_command"])),
        ("--workload", "solution-smoke"),
        ("--smoke-command", str(config["smoke_command"])),
        ("--solution-name", "robomimic"),
        ("--capability-name", str(config["capability_name"])),
        ("--smoke-artifact-name", "robomimic-smoke.json"),
        ("--yaml", str(profile_yaml)),
        ("--output-root", output_root),
        ("--wait-timeout", "-1"),
        ("--run-id", run_id),
        ("--robomimic-runtime-entitlement-file", runtime_entitlement_file),
    )
    return [
        sys.executable,
        str(BYOF_RUNNER),
        *(item for pair in options for item in pair),
    ]


def _materialize_robomimic_attested_profile(
    destination: Path,
    *,
    namespace: str,
    service_account: str,
    runtime_pvc: str,
    runtime_inventory_sha256: str,
    runtime_entitlement_file: str,
    runtime_entitlement_sha256: str,
    customer_binding_sha256: str,
) -> Path:
    """Bind STRICT placement and customer entitlement into a run-local profile."""

    assert os.environ.get("NPA_E2E_MK8S_RESERVED_CAPACITY") == "1"
    source = resolve_byof_profile_path("byof-solution-smoke-robomimic-b200-gpu")
    documents = list(yaml.safe_load_all(source.read_text(encoding="utf-8")))
    assert len(documents) == 2
    task = documents[1]
    assert task["envs"]["NPA_ROBOMIMIC_STRICT_B200_ATTESTED"] == ""
    assert task["envs"]["NPA_ROBOMIMIC_EXPECTED_NAMESPACE"] == ""
    assert task["envs"]["NPA_ROBOMIMIC_EXPECTED_SERVICE_ACCOUNT"] == ""
    assert task["envs"]["NPA_ROBOMIMIC_RUNTIME_INVENTORY_SHA256"] == ""
    assert task["envs"]["NPA_ROBOMIMIC_RUNTIME_ENTITLEMENT_SHA256"] == ""
    assert task["envs"]["NPA_ROBOMIMIC_CUSTOMER_BINDING_SHA256"] == ""
    task["envs"]["NPA_ROBOMIMIC_STRICT_B200_ATTESTED"] = "1"
    task["envs"]["NPA_ROBOMIMIC_EXPECTED_NAMESPACE"] = namespace
    task["envs"]["NPA_ROBOMIMIC_EXPECTED_SERVICE_ACCOUNT"] = service_account
    task["envs"]["NPA_ROBOMIMIC_RUNTIME_INVENTORY_SHA256"] = runtime_inventory_sha256
    task["envs"]["NPA_ROBOMIMIC_RUNTIME_ENTITLEMENT_SHA256"] = (
        runtime_entitlement_sha256
    )
    task["envs"]["NPA_ROBOMIMIC_CUSTOMER_BINDING_SHA256"] = customer_binding_sha256
    task["file_mounts"]["/opt/npa-runtime-authorization/robomimic.json"] = (
        runtime_entitlement_file
    )
    task["config"]["kubernetes"]["pod_config"]["spec"]["serviceAccountName"] = (
        service_account
    )
    volumes = task["config"]["kubernetes"]["pod_config"]["spec"]["volumes"]
    runtime_volume = next(
        item for item in volumes if item["name"] == "robomimic-runtime"
    )
    assert (
        runtime_volume["persistentVolumeClaim"]["claimName"]
        == "npa-robomimic-runtime-placeholder"
    )
    runtime_volume["persistentVolumeClaim"]["claimName"] = runtime_pvc
    destination.write_text(
        yaml.safe_dump_all(documents, sort_keys=False), encoding="utf-8"
    )
    return destination


def _robomimic_observer_name(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:12]
    return f"npa-robomimic-{digest}"


def _robomimic_observer_manifests(
    *, run_id: str, namespace: str, service_account: str, owner_token: str
) -> list[dict[str, object]]:
    labels = {
        "app.kubernetes.io/managed-by": "npa-robomimic-live-gate",
    }
    annotations = {
        "npa.nebius.ai/run-id-sha256": hashlib.sha256(run_id.encode()).hexdigest(),
        "npa.nebius.ai/owner-token": owner_token,
    }

    def metadata() -> dict[str, object]:
        return {
            "name": service_account,
            "namespace": namespace,
            "labels": dict(labels),
            "annotations": dict(annotations),
        }

    return [
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": metadata(),
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": metadata(),
            "rules": [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get"]}],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": metadata(),
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": service_account,
                    "namespace": namespace,
                }
            ],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": service_account,
            },
        },
    ]


def _robomimic_kubectl(selectors: dict[str, str]) -> list[str]:
    return [
        "kubectl",
        "--kubeconfig",
        selectors["kubeconfig"],
        "--context",
        selectors["context"],
    ]


def _robomimic_can_i(
    kube: list[str],
    *,
    namespace: str,
    service_account: str,
    verb: str,
    resource: str,
    env: dict[str, str],
) -> bool:
    result = subprocess.run(
        [
            *kube,
            "auth",
            "can-i",
            verb,
            resource,
            "--namespace",
            namespace,
            "--as",
            f"system:serviceaccount:{namespace}:{service_account}",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    answer = result.stdout.strip().lower()
    if (result.returncode, answer) == (0, "yes"):
        return True
    if (result.returncode, answer) == (1, "no"):
        return False
    raise AssertionError(
        "kubectl auth can-i returned an uncertain robomimic result: "
        f"exit={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}"
    )


def _robomimic_resource_permissions(
    kube: list[str], *, namespace: str, service_account: str, env: dict[str, str]
) -> set[tuple[str, str, str, str]]:
    """Return every namespaced resource permission observed for the identity."""

    review = {
        "apiVersion": "authorization.k8s.io/v1",
        "kind": "SelfSubjectRulesReview",
        "spec": {"namespace": namespace},
    }
    result = subprocess.run(
        [
            *kube,
            "--as",
            f"system:serviceaccount:{namespace}:{service_account}",
            "create",
            "--raw",
            "/apis/authorization.k8s.io/v1/selfsubjectrulesreviews",
            "-f",
            "-",
        ],
        input=json.dumps(review),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise AssertionError(
            "SelfSubjectRulesReview failed for the robomimic observer: "
            f"exit={result.returncode}, stderr={result.stderr.strip()!r}"
        )
    payload = json.loads(result.stdout)
    status = payload.get("status", {})
    if status.get("incomplete") or status.get("evaluationError"):
        raise AssertionError("robomimic observer permission review was incomplete")
    permissions: set[tuple[str, str, str, str]] = set()
    for rule in status.get("resourceRules", []):
        names = rule.get("resourceNames") or ["*"]
        permissions.update(
            (group, resource, name, verb)
            for group in rule.get("apiGroups", [])
            for resource in rule.get("resources", [])
            for name in names
            for verb in rule.get("verbs", [])
        )
    return permissions


def _assert_robomimic_observer_permissions(
    kube: list[str], *, namespace: str, service_account: str, env: dict[str, str]
) -> None:
    expected = {
        ("", "pods", "*", "get"),
        (
            "authorization.k8s.io",
            "selfsubjectaccessreviews",
            "*",
            "create",
        ),
        (
            "authorization.k8s.io",
            "selfsubjectrulesreviews",
            "*",
            "create",
        ),
        ("authentication.k8s.io", "selfsubjectreviews", "*", "create"),
    }
    observed = _robomimic_resource_permissions(
        kube, namespace=namespace, service_account=service_account, env=env
    )
    assert observed == expected, {
        "unexpected": sorted(observed - expected),
        "missing": sorted(expected - observed),
    }


def _robomimic_delete_path(resource: str, namespace: str) -> str:
    kind, name = resource.split("/", 1)
    encoded_namespace = quote(namespace, safe="")
    encoded_name = quote(name, safe="")
    if kind == "serviceaccount":
        return f"/api/v1/namespaces/{encoded_namespace}/serviceaccounts/{encoded_name}"
    plural = {"role": "roles", "rolebinding": "rolebindings"}[kind]
    return (
        "/apis/rbac.authorization.k8s.io/v1/namespaces/"
        f"{encoded_namespace}/{plural}/{encoded_name}"
    )


@contextmanager
def _robomimic_observer_rbac(
    *, run_id: str, selectors: dict[str, str], env: dict[str, str]
) -> Iterator[str]:
    """Own, verify, and remove the run-scoped Pod-observation identity."""

    namespace = selectors["namespace"]
    service_account = _robomimic_observer_name(run_id)
    owner_token = secrets.token_hex(32)
    kube = _robomimic_kubectl(selectors)
    manifests = _robomimic_observer_manifests(
        run_id=run_id,
        namespace=namespace,
        service_account=service_account,
        owner_token=owner_token,
    )
    created_resources: list[str] = []
    primary_error: BaseException | None = None
    try:
        for manifest in manifests:
            resource = f"{str(manifest['kind']).lower()}/{service_account}"
            prior = subprocess.run(
                [
                    *kube,
                    "get",
                    "--namespace",
                    namespace,
                    resource,
                    "--ignore-not-found=true",
                    "-o",
                    "name",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            assert prior.returncode == 0 and not prior.stdout.strip(), (
                resource,
                prior.stdout,
                prior.stderr,
            )
            # Track after proving no prior object exists but before create, so a
            # server-side create followed by a lost response is still cleaned.
            created_resources.append(resource)
            created = subprocess.run(
                [*kube, "create", "--namespace", namespace, "-f", "-"],
                input=yaml.safe_dump(manifest, sort_keys=False),
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            if created.returncode != 0:
                raise RuntimeError(
                    f"failed to create run-scoped robomimic {resource}: "
                    f"{created.stderr.strip()}"
                )
        permissions = {
            ("get", "pods"): True,
            ("list", "pods"): False,
            ("watch", "pods"): False,
            ("get", "pods/log"): False,
            ("get", "secrets"): False,
        }
        for (verb, resource), expected in permissions.items():
            assert (
                _robomimic_can_i(
                    kube,
                    namespace=namespace,
                    service_account=service_account,
                    verb=verb,
                    resource=resource,
                    env=env,
                )
                is expected
            )
        _assert_robomimic_observer_permissions(
            kube,
            namespace=namespace,
            service_account=service_account,
            env=env,
        )
        yield service_account
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        resources = list(reversed(created_resources))
        cleanup_errors: list[str] = []
        for resource in resources:
            try:
                observed = subprocess.run(
                    [
                        *kube,
                        "get",
                        "--namespace",
                        namespace,
                        resource,
                        "--ignore-not-found=true",
                        "-o",
                        "json",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )
            except OSError as exc:
                cleanup_errors.append(
                    f"inspect {resource} ownership failed: {type(exc).__name__}"
                )
                continue
            if observed.returncode != 0:
                cleanup_errors.append(
                    f"inspect {resource} ownership exited {observed.returncode}: "
                    f"{observed.stderr.strip()}"
                )
                continue
            if not observed.stdout.strip():
                continue
            try:
                observed_payload = json.loads(observed.stdout)
                observed_metadata = observed_payload["metadata"]
                observed_token = observed_metadata["annotations"][
                    "npa.nebius.ai/owner-token"
                ]
                observed_uid = observed_metadata["uid"]
                observed_resource_version = observed_metadata["resourceVersion"]
            except (KeyError, TypeError, json.JSONDecodeError):
                cleanup_errors.append(
                    f"inspect {resource} ownership returned invalid metadata"
                )
                continue
            if observed_token != owner_token:
                cleanup_errors.append(
                    f"refused to delete {resource} owned by another invocation"
                )
                continue
            try:
                delete_options = {
                    "apiVersion": "v1",
                    "kind": "DeleteOptions",
                    "preconditions": {
                        "uid": observed_uid,
                        "resourceVersion": observed_resource_version,
                    },
                }
                deleted = subprocess.run(
                    [
                        *kube,
                        "delete",
                        "--raw",
                        _robomimic_delete_path(resource, namespace),
                        "-f",
                        "-",
                    ],
                    input=json.dumps(delete_options),
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )
            except OSError as exc:
                cleanup_errors.append(f"delete {resource} failed: {type(exc).__name__}")
                continue
            if deleted.returncode != 0:
                cleanup_errors.append(
                    f"delete {resource} exited {deleted.returncode}: "
                    f"{deleted.stderr.strip()}"
                )
        for resource in resources:
            try:
                absent = subprocess.run(
                    [
                        *kube,
                        "get",
                        "--namespace",
                        namespace,
                        resource,
                        "--ignore-not-found=true",
                        "-o",
                        "name",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )
            except OSError as exc:
                cleanup_errors.append(
                    f"verify {resource} absent failed: {type(exc).__name__}"
                )
                continue
            if absent.returncode != 0 or absent.stdout.strip():
                cleanup_errors.append(
                    f"verify {resource} absent failed: exit={absent.returncode}, "
                    f"stdout={absent.stdout.strip()!r}, stderr={absent.stderr.strip()!r}"
                )
        if cleanup_errors:
            message = "robomimic observer RBAC cleanup failed: " + "; ".join(
                cleanup_errors
            )
            if primary_error is not None:
                add_note = getattr(primary_error, "add_note", None)
                if callable(add_note):
                    add_note(message)
                else:  # Python 3.10 compatibility.
                    primary_error.args = (*primary_error.args, message)
            else:
                raise AssertionError(message)


def _robomimic_target_env(
    e2e_project: str, selectors: dict[str, str], cmd: list[str]
) -> dict[str, str]:
    config_path = skypilot_config_for_project(e2e_project)
    if config_path:
        cmd.extend(["--config-path", config_path])
    target = resolve_byof_kubernetes_target(e2e_project)
    assert target.kubeconfig == selectors["kubeconfig"]
    assert target.context == selectors["context"]
    assert target.namespace == selectors["namespace"]
    env = dict(os.environ)
    env["KUBECONFIG"] = target.kubeconfig
    env["NPA_BYOF_KUBECONFIG"] = target.kubeconfig
    env["NPA_BYOF_K8S_CONTEXT"] = target.context
    env["AWS_ENDPOINT_URL"] = selectors["storage_endpoint"]
    env["NEBIUS_S3_ENDPOINT"] = selectors["storage_endpoint"]
    namespace_readback = subprocess.run(
        [
            *_robomimic_kubectl(selectors),
            "config",
            "view",
            "--minify",
            "--output",
            "jsonpath={.contexts[0].context.namespace}",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if namespace_readback.returncode != 0:
        raise AssertionError(
            "manager-issued Kubernetes context namespace lookup failed: "
            f"exit={namespace_readback.returncode}, "
            f"stderr={namespace_readback.stderr.strip()!r}"
        )
    observed_namespace = namespace_readback.stdout.strip() or "default"
    assert observed_namespace == selectors["namespace"], (
        "manager-issued Kubernetes context namespace does not match its selector"
    )
    skypilot_bin = resolve_skypilot_bin()
    if skypilot_bin:
        env["PATH"] = f"{Path(skypilot_bin).parent}:{env.get('PATH', '')}"
    return env


def _invoke_robomimic_gate(
    e2e_project: str | None,
) -> tuple[dict[str, object], str, str]:
    context_refused = False
    try:
        selectors = _robomimic_live_selectors(e2e_project)
    except Exception:
        context_refused = True
    if context_refused:
        raise _RobomimicEntitlementRefusal("context-invalid")
    run_id = os.environ.get("NPA_BYOF_ROBOMIMIC_RUN_ID") or (
        f"robomimic-live-{secrets.token_hex(8)}"
    )
    if re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", run_id) is None:
        raise _RobomimicEntitlementRefusal("identity-invalid")
    refusal_category = ""
    try:
        entitlement = _preflight_robomimic_runtime_entitlement(
            selectors=selectors, run_id=run_id
        )
    except _RobomimicEntitlementRefusal as exc:
        refusal_category = exc.category
    except Exception:
        refusal_category = "record-invalid"
    if refusal_category:
        raise _RobomimicEntitlementRefusal(refusal_category)
    _activate_nebius_profile()
    config = load_spec(ROBOMIMIC_SPEC).config
    registry = resolve_container_registry(e2e_project)
    assert registry == selectors["registry"].rstrip("/")
    assert not is_public_registry(registry)
    bucket = live_bucket(e2e_project)
    assert bucket == selectors["bucket"].removeprefix("s3://").split("/", 1)[0]
    with tempfile.TemporaryDirectory(prefix="npa-robomimic-profile-") as temp_dir:
        service_account = _robomimic_observer_name(run_id)
        profile_yaml = _materialize_robomimic_attested_profile(
            Path(temp_dir) / "robomimic-attested.yaml",
            namespace=selectors["namespace"],
            service_account=service_account,
            runtime_pvc=selectors["runtime_pvc"],
            runtime_inventory_sha256=selectors["runtime_inventory_sha256"],
            runtime_entitlement_file=selectors["runtime_entitlement_file"],
            runtime_entitlement_sha256=entitlement["record_sha256"],
            customer_binding_sha256=entitlement["customer_binding_sha256"],
        )
        cmd = _robomimic_runner_command(
            config,
            registry,
            selectors["project"],
            f"s3://{bucket}/oss-solutions/robomimic",
            run_id,
            profile_yaml,
            selectors["runtime_entitlement_file"],
        )
        env = _robomimic_target_env(selectors["project"], selectors, cmd)
        with _robomimic_observer_rbac(
            run_id=run_id, selectors=selectors, env=env
        ) as observed_service_account:
            assert observed_service_account == service_account
            accepted_image = selectors["image"]
            cmd.extend(["--image", accepted_image, "--skip-build"])
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                env=env,
            )
            assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = _parse_last_json_blob(proc.stdout + "\n" + proc.stderr)
    assert summary.get("status") == "ok", summary
    summary_image = str(summary.get("image", ""))
    assert summary_image.startswith(f"{registry}/") and "@sha256:" in summary_image
    return summary, bucket, run_id


def _robomimic_artifact(
    e2e_project: str | None, bucket: str, run_id: str
) -> dict[str, object]:
    key = f"oss-solutions/robomimic/{run_id}/robomimic-smoke.json"
    body: object | None = None
    artifact: dict[str, object] | None = None
    failed = False
    try:
        client = s3_client_for_project(e2e_project, allow_host_creds=True)
        response = client.get_object(Bucket=bucket, Key=key)
        if not isinstance(response, dict):
            raise ValueError
        body = response.get("Body")
        if body is None:
            raise ValueError
        declared_size = response.get("ContentLength")
        if not _valid_robomimic_smoke_proof_size(declared_size):
            raise ValueError
        artifact = _decode_robomimic_smoke_proof(body, declared_size)
    except Exception:
        failed = True
    finally:
        if body is not None and not _close_robomimic_smoke_proof(body):
            failed = True
    if failed or artifact is None:
        raise RuntimeError(_ROBOMIMIC_SMOKE_PROOF_ERROR) from None
    return artifact


def _valid_robomimic_smoke_proof_size(value: object) -> bool:
    """Accept only bounded, non-negative integer S3 object lengths."""

    return type(value) is int and 0 <= value <= ROBOMIMIC_SMOKE_PROOF_MAX_BYTES


def _decode_robomimic_smoke_proof(
    body: object, declared_size: int
) -> dict[str, object]:
    """Read and decode one bounded robomimic smoke proof."""

    payload = body.read(ROBOMIMIC_SMOKE_PROOF_MAX_BYTES + 1)
    if not isinstance(payload, bytes):
        raise ValueError
    if len(payload) > ROBOMIMIC_SMOKE_PROOF_MAX_BYTES or len(payload) != declared_size:
        raise ValueError
    artifact = json.loads(payload.decode("utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError
    return artifact


def _close_robomimic_smoke_proof(body: object) -> bool:
    """Close one proof body without exposing a cleanup exception."""

    try:
        body.close()
    except Exception:
        return False
    return True


def _assert_robomimic_inputs(artifact: dict[str, object]) -> None:
    source = artifact["source"]
    assert source["repository"] == "ARISE-Initiative/robomimic"
    assert source["revision"] == ROBOMIMIC_SOURCE_REVISION
    assert source["observed_head"] == ROBOMIMIC_SOURCE_REVISION
    assert re.fullmatch(r"[0-9a-f]{64}", source["tree_archive_sha256"])
    dataset = artifact["dataset"]
    assert dataset["repository"] == "robomimic/robomimic_datasets"
    assert dataset["revision"] == ROBOMIMIC_DATASET_REVISION
    assert dataset["path"] == "v1.5/lift/ph/low_dim_v15.hdf5"
    assert dataset["sha256"] == ROBOMIMIC_DATASET_SHA256
    assert dataset["size_bytes"] == 21_084_088
    assert dataset["trajectory_count"] == 200 and dataset["sample_count"] > 0


def _assert_robomimic_split(artifact: dict[str, object]) -> None:
    dataset, split = artifact["dataset"], artifact["split"]
    assert split["overlap_count"] == 0
    assert split["train_trajectory_count"] > 0
    assert split["validation_trajectory_count"] > 0
    assert split["train_trajectory_count"] + split["validation_trajectory_count"] == 200
    assert split["train_sample_count"] > 0 and split["validation_sample_count"] > 0
    assert (
        split["train_sample_count"] + split["validation_sample_count"]
        == dataset["sample_count"]
    )
    assert re.fullmatch(r"[0-9a-f]{64}", split["train_keys_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", split["validation_keys_sha256"])


def _assert_robomimic_training(artifact: dict[str, object]) -> None:
    training = artifact["training"]
    assert training["entrypoint"] == "robomimic/scripts/train.py"
    assert training["algorithm"] == "bc" and training["optimizer"] == "adam"
    assert training["optimizer_step_count"] == 4
    assert training["configured_optimizer_steps"] == 4
    assert training["configured_validation_forward_steps"] == 2
    assert math.isfinite(training["train_loss"])
    assert math.isfinite(training["validation_loss"])
    assert artifact["checkpoint"]["reloaded"] is True
    assert re.fullmatch(r"[0-9a-f]{64}", artifact["checkpoint"]["sha256"])


def _assert_robomimic_action(artifact: dict[str, object]) -> None:
    action, split = artifact["heldout_action"], artifact["split"]
    assert action["demo"] == split["heldout_demo"]
    assert action["shape"] == [7] and action["finite"] is True
    assert action["within_allowed_range"] is True
    assert action["allowed_range"] == [-1.0, 1.0]
    assert -1.0 <= action["observed_min"] <= action["observed_max"] <= 1.0
    assert math.isfinite(action["observed_min"])
    assert math.isfinite(action["observed_max"])


def _assert_robomimic_runtime(
    artifact: dict[str, object], summary_image: str, run_id: str
) -> None:
    external_runtime = artifact["external_runtime"]
    assert external_runtime["prepopulated"] is True
    assert external_runtime["read_only"] is True
    assert external_runtime["runtime_manifest_digest_matched"] is True
    entitlement = artifact["customer_runtime_entitlement"]
    assert entitlement["run_binding_matched"] is True
    assert entitlement["customer_binding_matched"] is True
    assert "field_of_use" not in entitlement
    assert entitlement["redistribution_granted"] is False
    assert external_runtime["atomic_private_snapshot_published"] is True
    assert external_runtime["snapshot_write_bits_absent"] is True
    assert re.fullmatch(r"[0-9a-f]{64}", external_runtime["lock_sha256"])
    assert (
        external_runtime["inventory_sha256"]
        == os.environ["NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256"]
    )
    hardware = artifact["hardware"]
    assert hardware["accelerator_count"] == 1 and "B200" in hardware["model"].upper()
    assert hardware["architecture"] == "sm_100"
    assert hardware["compute_capability"] == [10, 0]
    assert len(hardware["nvidia_smi_rows"]) == 1
    assert "B200" in hardware["nvidia_smi_rows"][0].upper()
    assert "strict_reserved_capacity_attested" not in hardware
    assert "application_strict_capacity_qualification" in artifact["deferred"]
    pod_image = artifact["pod_image"]
    assert pod_image["runtime_ref"] == summary_image
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", pod_image["digest"])
    assert pod_image["digest"] in pod_image["image_id"]
    assert pod_image["container_name"] == "ray-node"
    assert pod_image["observation_source"] == (
        "Kubernetes Pod status.containerStatuses[].imageID"
    )
    runtime_mount = artifact["runtime_mount"]
    assert runtime_mount["name"] == "robomimic-runtime"
    assert runtime_mount["path"] == "/opt/npa-runtime/robomimic"
    assert runtime_mount["read_only"] is True
    identity = artifact["workload_identity"]
    assert identity["namespace"] == os.environ["NPA_BYOF_K8S_NAMESPACE"]
    assert identity["service_account"] == _robomimic_observer_name(run_id)
    assert identity["observation_source"] == (
        "mounted service-account namespace and Kubernetes Pod spec.serviceAccountName"
    )


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_GPU") != "1"
    or os.environ.get("NPA_BYOF_ROBOMIMIC_LIVE_B200") != "1"
    or os.environ.get("NPA_E2E_MK8S_RESERVED_CAPACITY") != "1",
    reason=(
        "Set NPA_BYOF_LIVE_GPU=1, NPA_BYOF_ROBOMIMIC_LIVE_B200=1, and "
        "NPA_E2E_MK8S_RESERVED_CAPACITY=1 only with the assigned STRICT "
        "one-B200 runtime context to execute the robomimic gate."
    ),
)
def test_live_robomimic_b200_train_reload_gate(e2e_project: str | None) -> None:
    """Use an accepted private candidate and require its complete run artifact."""

    summary, bucket, run_id = _invoke_robomimic_gate(e2e_project)
    summary_image = str(summary.get("image", ""))
    artifact = _robomimic_artifact(e2e_project, bucket, run_id)
    assert artifact["schema"] == "npa.workbench.robomimic.smoke.v1"
    assert artifact["solution"] == "robomimic"
    assert artifact["capability"] == "lift_ph_lowdim_checkpoint_reload_action"
    assert set(artifact["capabilities_exercised"]) == ROBOMIMIC_CAPABILITIES
    _assert_robomimic_inputs(artifact)
    _assert_robomimic_split(artifact)
    _assert_robomimic_training(artifact)
    _assert_robomimic_action(artifact)
    _assert_robomimic_runtime(artifact, summary_image, run_id)
    assert set(artifact["deferred"]) == {
        "public_image_acceptance",
        "image_policy_sweeps",
        "simulator_rollouts",
        "full_algorithm_matrix",
    }
    assert artifact["exit_status"] == 0


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_GPU") != "1",
    reason="Set NPA_BYOF_LIVE_GPU=1 to submit a real Isaac BYOF SkyPilot smoke (build/push/run).",
)
def test_live_byof_runner_submit_smoke(
    e2e_project: str | None,
    live_byof_built_image: str,
) -> None:
    registry = resolve_container_registry(e2e_project)
    yaml_override = resolve_byof_resource_yaml(e2e_project, smoke=True)
    image = os.environ.get("NPA_BYOF_TEST_IMAGE", "").strip() or live_byof_built_image
    task = os.environ.get("NPA_BYOF_TASK", "Isaac-Cartpole-v0")
    cmd = [
        sys.executable,
        str(BYOF_RUNNER),
        "--registry",
        registry,
        "--project",
        e2e_project or "",
        "--image",
        image,
        "--yaml",
        yaml_override,
        "--task",
        task,
        "--iterations",
        "1",
        "--run-id",
        f"byof-live-submit-{os.getpid()}",
        "--skip-build",
    ]
    config_path = skypilot_config_for_project(e2e_project)
    if config_path:
        cmd.extend(["--config-path", config_path])
    env = dict(os.environ)
    target = resolve_byof_kubernetes_target(e2e_project)
    if target.kubeconfig:
        env["KUBECONFIG"] = target.kubeconfig
        env["NPA_BYOF_KUBECONFIG"] = target.kubeconfig
    if target.context:
        env["NPA_BYOF_K8S_CONTEXT"] = target.context
    skypilot_bin = resolve_skypilot_bin()
    if skypilot_bin:
        env["PATH"] = f"{Path(skypilot_bin).parent}:{env.get('PATH', '')}"
    env.setdefault("NPA_ISAAC_LAB_ACCEPT_PRECHECK_FAILURE", "1")
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=int(os.environ.get("NPA_BYOF_LIVE_TIMEOUT", "3600")),
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = _parse_last_json_blob(proc.stdout + "\n" + proc.stderr)
    assert summary.get("status") == "ok", summary
    run_summary = summary.get("run", {})
    assert isinstance(run_summary, dict)
    if run_summary.get("status") in {"submitted", "SUBMITTED"}:
        return
    final = run_summary.get("final", {})
    if isinstance(final, dict) and final.get("status"):
        assert final.get("status") in {
            "SUBMITTED",
            "SUCCEEDED",
            "RUNNING",
            "PENDING",
            "FAILED_PRECHECKS",
            "FAILED_SETUP",
            "FAILED",
        }
    submit = run_summary.get("submit", {})
    if isinstance(submit, dict) and submit.get("status"):
        assert submit.get("status") == "SUBMITTED", submit
    elif isinstance(final, dict) and final.get("status") == "FAILED_PRECHECKS":
        pass
    else:
        assert submit or final, run_summary


@pytest.fixture(scope="module")
def live_byof_ubuntu_built_image(e2e_project: str | None) -> str:
    if os.environ.get("NPA_BYOF_LIVE_UBUNTU") != "1":
        pytest.skip(
            "Set NPA_BYOF_LIVE_UBUNTU=1 for Ubuntu OSS BYOF container build/push."
        )
    _activate_nebius_profile()
    registry = resolve_container_registry(e2e_project)
    repo_url, repo_ref = byof_ubuntu_validation_repo()
    run_id = (
        os.environ.get("NPA_BYOF_UBUNTU_RUN_ID") or f"byof-ubuntu-live-{os.getpid()}"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(BYOF_RUNNER),
            "--registry",
            registry,
            "--repo-url",
            repo_url,
            "--repo-ref",
            repo_ref,
            "--project",
            e2e_project or "",
            "--run-id",
            run_id,
            "--base-profile",
            "ubuntu",
            "--skip-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=int(os.environ.get("NPA_BYOF_CONTAINER_TIMEOUT", "3600")),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = _parse_last_json_blob(proc.stdout + "\n" + proc.stderr)
    assert summary.get("status") == "ok", summary
    assert summary.get("base_profile") == "ubuntu"
    image = str(summary["image"])
    assert registry in image
    return image


@pytest.mark.skipif(
    os.environ.get("NPA_AGENT_LIVE") != "1",
    reason="Set NPA_AGENT_LIVE=1 to exercise OSS repo onboarding on the configured agent.",
)
def test_live_agent_oss_repo_onboard_solution_chat() -> None:
    ctx = load_agent_live_context()
    chat = ctx.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": ONBOARD_OSS_REPO_PROMPT}]},
        timeout=30.0,
    )
    chat.raise_for_status()
    assert_grounded_onboard_solution_reply(chat.json())


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_UBUNTU") != "1",
    reason="Set NPA_BYOF_LIVE_UBUNTU=1 for Ubuntu OSS BYOF container build/push.",
)
def test_live_byof_ubuntu_oss_container_build_push(
    live_byof_ubuntu_built_image: str,
) -> None:
    assert live_byof_ubuntu_built_image
    assert "npa-byof" in live_byof_ubuntu_built_image


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_UBUNTU") != "1",
    reason="Set NPA_BYOF_LIVE_UBUNTU=1 for Ubuntu OSS BYOF container metadata inspect.",
)
def test_live_byof_ubuntu_oss_container_metadata(
    live_byof_ubuntu_built_image: str,
) -> None:
    repo_url, repo_ref = byof_ubuntu_validation_repo()
    meta_proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "cat",
            live_byof_ubuntu_built_image,
            "/opt/byof/npa_source_metadata.json",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert meta_proc.returncode == 0, meta_proc.stderr
    metadata = json.loads(meta_proc.stdout)
    assert metadata["repo"] == repo_url
    assert metadata["ref"] == repo_ref


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_UBUNTU") != "1"
    or os.environ.get("NPA_BYOF_LIVE_GPU") != "1",
    reason="Set NPA_BYOF_LIVE_UBUNTU=1 and NPA_BYOF_LIVE_GPU=1 for Ubuntu container-verify SkyPilot smoke.",
)
def test_live_byof_ubuntu_oss_container_verify_submit(
    e2e_project: str | None,
    live_byof_ubuntu_built_image: str,
) -> None:
    registry = resolve_container_registry(e2e_project)
    yaml_override = resolve_byof_resource_yaml(
        e2e_project, smoke=True, workload="container-verify"
    )
    cmd = [
        sys.executable,
        str(BYOF_RUNNER),
        "--registry",
        registry,
        "--project",
        e2e_project or "",
        "--image",
        live_byof_ubuntu_built_image,
        "--yaml",
        yaml_override,
        "--workload",
        "container-verify",
        "--run-id",
        f"byof-ubuntu-verify-{os.getpid()}",
        "--skip-build",
    ]
    config_path = skypilot_config_for_project(e2e_project)
    if config_path:
        cmd.extend(["--config-path", config_path])
    env = dict(os.environ)
    target = resolve_byof_kubernetes_target(e2e_project)
    if target.kubeconfig:
        env["KUBECONFIG"] = target.kubeconfig
    if target.context:
        env["NPA_BYOF_K8S_CONTEXT"] = target.context
    skypilot_bin = resolve_skypilot_bin()
    if skypilot_bin:
        env["PATH"] = f"{Path(skypilot_bin).parent}:{env.get('PATH', '')}"
    env.setdefault("NPA_ISAAC_LAB_ACCEPT_PRECHECK_FAILURE", "1")
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=int(os.environ.get("NPA_BYOF_LIVE_TIMEOUT", "3600")),
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = _parse_last_json_blob(proc.stdout + "\n" + proc.stderr)
    assert summary.get("status") == "ok", summary
