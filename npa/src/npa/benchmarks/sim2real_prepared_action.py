"""Private, fail-closed prepared workflow actions for model-agent benchmarks."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import stat
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from npa.orchestration.npa_workflow.runtime import is_terminal_fail


RECEIPT_SCHEMA = "npa.sim2real.prepared_workflow_action.v1"
STATE_SCHEMA = "npa.sim2real.prepared_workflow_action.state.v1"
OUTPUT_SCHEMA = "npa.sim2real.prepared_workflow_action.output.v1"
IMAGE_ROLES = (
    "controller",
    "transfer",
    "envgen",
    "isaac",
    "viewer",
)
CANONICAL_SECRET_ENV_NAMES = frozenset(
    {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN", "NEBIUS_TOKEN_FACTORY_KEY"}
)
REQUIRED_PREFLIGHTS = (
    "health_preflight",
    "model_access",
    "skypilot_verify",
    "workflow_gpu_scheduler",
    "workflow_validate",
    "workflow_plan",
    "image_pullability",
    "submit_plan_only",
)
_ACTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_IMAGE_DIGEST_RE = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")


class PreparedActionError(RuntimeError):
    """A typed prepared action failed before or during its one allowed execution."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


@dataclass(frozen=True)
class PreparedActionContext:
    workspace: Path
    evidence: Path
    control_dir: Path
    private_root: Path
    environment: Mapping[str, str]
    isolation: Mapping[str, Path]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def argv_sha256(argv: Sequence[str]) -> str:
    return canonical_sha256(list(argv))


def workspace_state(workspace: Path) -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=workspace, text=True
    ).strip()
    status_bytes = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-z"], cwd=workspace
    )
    changed_paths = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", "-z", "HEAD", "--"], cwd=workspace
        ).split(b"\0")
    )
    changed_paths.update(
        subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=workspace,
        ).split(b"\0")
    )
    content_digest = hashlib.sha256()
    for encoded_path in sorted(path for path in changed_paths if path):
        relative = encoded_path.decode("utf-8", errors="surrogateescape")
        candidate = workspace / relative
        content_digest.update(encoded_path + b"\0")
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            content_digest.update(b"deleted\0")
            continue
        content_digest.update(str(stat.S_IMODE(info.st_mode)).encode() + b"\0")
        if candidate.is_symlink():
            content_digest.update(os.readlink(candidate).encode() + b"\0")
        elif candidate.is_file():
            content_digest.update(file_sha256(candidate).encode() + b"\0")
        else:
            content_digest.update(b"non-file\0")
    return {
        "head": head,
        "detached": not branch,
        "status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "status_entries": status_bytes.count(b"\0"),
        "changed_content_sha256": content_digest.hexdigest(),
    }


def sandbox_private_path(path: Path, *, evidence: Path) -> str:
    relative = path.resolve().relative_to(evidence.resolve())
    return str(Path("/tmp/npa-private-evidence") / relative)


def resolve_sandbox_private_path(value: str, *, evidence: Path) -> Path:
    sandbox_root = Path("/tmp/npa-private-evidence")
    candidate = Path(value)
    try:
        relative = candidate.relative_to(sandbox_root)
    except ValueError as exc:
        raise PreparedActionError(
            "receipt_schema_invalid",
            "receipt paths must use the private benchmark mount",
        ) from exc
    resolved = (evidence / relative).resolve()
    try:
        resolved.relative_to(evidence.resolve())
    except ValueError as exc:
        raise PreparedActionError(
            "receipt_schema_invalid", "receipt path escapes private evidence"
        ) from exc
    return resolved


def _require_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise PreparedActionError(
            "receipt_schema_invalid", f"{label} does not match the closed schema"
        )
    return value


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PreparedActionError("receipt_schema_invalid", f"{label} is not SHA-256")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
        raise PreparedActionError(
            "receipt_schema_invalid", f"{label} is not an immutable commit"
        )
    return value


def create_receipt(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create one immutable owner-only receipt; an existing path is never replaced."""

    body = dict(payload)
    if "receipt_sha256" in body:
        raise ValueError("receipt payload must not supply receipt_sha256")
    body["receipt_sha256"] = canonical_sha256(body)
    encoded = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    os.chmod(path, 0o600)
    return body


def create_receipt_from_request(request_path: Path, output_path: Path) -> dict[str, Any]:
    """Materialize a receipt from one private, value-bearing operator request."""

    request = json.loads(request_path.read_text(encoding="utf-8"))
    keys = {
        "schema",
        "action_id",
        "workspace",
        "evidence_dir",
        "private_root",
        "canonical_spec_path",
        "source_commit",
        "benchmark_base",
        "run_id",
        "project_alias",
        "project_infra",
        "staged_manifest_path",
        "staged_input_identity",
        "images",
        "runtime_policy",
        "accepted_eulas",
        "required_secret_env_names",
        "preflight_evidence",
        "argv",
    }
    _require_keys(request, keys, "receipt request")
    if request["schema"] != "npa.sim2real.prepared_workflow_action.request.v1":
        raise PreparedActionError("receipt_schema_invalid", "unsupported request schema")
    workspace = Path(request["workspace"]).resolve()
    evidence = Path(request["evidence_dir"]).resolve()
    private_root = Path(request["private_root"]).resolve()
    spec_path = Path(request["canonical_spec_path"]).resolve()
    manifest_path = Path(request["staged_manifest_path"]).resolve()
    request_info = request_path.stat()
    if (
        not stat.S_ISREG(request_info.st_mode)
        or stat.S_IMODE(request_info.st_mode) != 0o600
        or request_info.st_uid != os.getuid()
    ):
        raise PreparedActionError(
            "receipt_permissions_invalid",
            "prepared action request must be an owner-owned mode-0600 regular file",
        )
    for candidate, label in (
        (request_path, "request"),
        (output_path, "receipt"),
        (evidence, "evidence"),
        (spec_path, "canonical spec"),
        (manifest_path, "staged manifest"),
    ):
        try:
            candidate.resolve().relative_to(private_root)
        except ValueError as exc:
            raise PreparedActionError(
                "receipt_location_invalid", f"{label} is outside private evidence"
            ) from exc
    observed_workspace = workspace_state(workspace)
    if (
        observed_workspace["head"] != request["benchmark_base"]
        or observed_workspace["detached"] is not True
    ):
        raise PreparedActionError(
            "source_mismatch", "workspace is not detached at the benchmark base"
        )
    try:
        output_path.resolve().relative_to(evidence)
    except ValueError:
        pass
    else:
        raise PreparedActionError(
            "receipt_location_invalid",
            "receipt must be outside the agent-visible evidence mount",
        )
    try:
        request_path.resolve().relative_to(evidence)
    except ValueError:
        pass
    else:
        raise PreparedActionError(
            "receipt_location_invalid",
            "prepared action request must be outside the agent-visible evidence mount",
        )
    preflight_mapping = request["preflight_evidence"]
    if not isinstance(preflight_mapping, dict) or set(preflight_mapping) != set(
        REQUIRED_PREFLIGHTS
    ):
        raise PreparedActionError(
            "receipt_schema_invalid", "preflight evidence set is incomplete"
        )
    preflights = []
    for name in REQUIRED_PREFLIGHTS:
        evidence_path = Path(preflight_mapping[name]).resolve()
        try:
            evidence_path.relative_to(private_root)
        except ValueError as exc:
            raise PreparedActionError(
                "receipt_location_invalid", "preflight evidence is outside private root"
            ) from exc
        try:
            result = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PreparedActionError(
                "preflight_evidence_invalid", f"{name} evidence is not valid JSON"
            ) from exc
        if not _preflight_result_passed(result):
            raise PreparedActionError(
                "preflight_not_passed", f"{name} preflight is not passed"
            )
        preflights.append(
            {
                "name": name,
                "evidence_sandbox_path": sandbox_private_path(
                    evidence_path, evidence=evidence
                ),
                "evidence_sha256": file_sha256(evidence_path),
            }
        )
    payload = {
        "schema": RECEIPT_SCHEMA,
        "action_id": request["action_id"],
        "created_at": utc_now(),
        "canonical_spec": {
            "sandbox_path": sandbox_private_path(spec_path, evidence=evidence),
            "sha256": file_sha256(spec_path),
        },
        "source": {
            "workspace_commit": observed_workspace["head"],
            "source_commit": request["source_commit"],
            "workspace_state_sha256": canonical_sha256(observed_workspace),
        },
        "benchmark": {"base_commit": request["benchmark_base"]},
        "run": {"run_id": request["run_id"]},
        "project": {
            "alias": request["project_alias"],
            "infra": request["project_infra"],
            "selection_sha256": canonical_sha256(
                {"alias": request["project_alias"], "infra": request["project_infra"]}
            ),
        },
        "staged_input": {
            "manifest_sandbox_path": sandbox_private_path(
                manifest_path, evidence=evidence
            ),
            "manifest_sha256": file_sha256(manifest_path),
            "identity": request["staged_input_identity"],
            "identity_sha256": canonical_sha256(request["staged_input_identity"]),
        },
        "images": request["images"],
        "runtime_policy": request["runtime_policy"],
        "accepted_eulas": request["accepted_eulas"],
        "required_secret_env_names": sorted(request["required_secret_env_names"]),
        "preflights": preflights,
        "argv": request["argv"],
        "argv_sha256": argv_sha256(request["argv"]),
    }
    _validate_argv_contract(payload)
    receipt = create_receipt(output_path, payload)
    context = PreparedActionContext(
        workspace=workspace,
        evidence=evidence,
        control_dir=output_path.parent.resolve(),
        private_root=private_root,
        environment={
            "NPA_PROJECT": str(request["project_alias"]),
            "NPA_SIM2REAL_INFRA": str(request["project_infra"]),
            "NPA_SIM2REAL_RUN_ID": str(request["run_id"]),
            "NPA_SIM2REAL_SOURCE_SHA": str(request["source_commit"]),
        },
        isolation={
            "evidence": evidence,
            "private_root": private_root,
            "controller_repo": private_root,
        },
    )
    validate_receipt(
        output_path,
        requested_action_id=str(request["action_id"]),
        context=context,
        require_secrets=False,
    )
    return receipt


def _load_owner_only_json(path: Path, *, private_root: Path) -> dict[str, Any]:
    try:
        path.resolve().relative_to(private_root.resolve())
    except ValueError as exc:
        raise PreparedActionError(
            "receipt_location_invalid", "receipt must remain under private evidence"
        ) from exc
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise PreparedActionError(
            "receipt_permissions_invalid", "receipt must be a regular mode-0600 file"
        )
    if info.st_uid != os.getuid():
        raise PreparedActionError(
            "receipt_permissions_invalid", "receipt must be owned by the controller user"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreparedActionError(
            "receipt_schema_invalid", "receipt is not readable canonical JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise PreparedActionError("receipt_schema_invalid", "receipt must be an object")
    return payload


def _option_values(argv: Sequence[str], option: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(argv):
        if token == option:
            if index + 1 >= len(argv):
                raise PreparedActionError(
                    "argv_contract_invalid", f"{option} has no value"
                )
            values.append(argv[index + 1])
        elif token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
    return values


def _validate_argv_contract(receipt: Mapping[str, Any]) -> None:
    argv = receipt["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(token, str) or not token or "\0" in token for token in argv)
    ):
        raise PreparedActionError("argv_contract_invalid", "argv must be strings")
    if len(argv) > 256 or sum(map(len, argv)) > 64_000:
        raise PreparedActionError("argv_contract_invalid", "argv exceeds safe bounds")
    expected_prefix = [
        "npa/.venv/bin/npa",
        "workbench",
        "workflow",
        "submit",
        receipt["canonical_spec"]["sandbox_path"],
    ]
    if argv[:5] != expected_prefix:
        raise PreparedActionError(
            "argv_contract_invalid", "argv is not the canonical NPA workflow submit"
        )
    boolean_options = {"--runtime", "--resume", "--accept-eula"}
    valued_options = {
        "--run-id",
        "--project",
        "--infra",
        "--preset",
        "--assume-decision",
        "--max-wait-seconds",
        "--var",
        "--secret-env",
        "--output-format",
    }
    index = 5
    while index < len(argv):
        token = argv[index]
        if token in boolean_options:
            index += 1
            continue
        if token not in valued_options or index + 1 >= len(argv):
            raise PreparedActionError(
                "argv_contract_invalid", f"argv contains unsupported token {token!r}"
            )
        if argv[index + 1].startswith("--"):
            raise PreparedActionError(
                "argv_contract_invalid", f"{token} has no literal value"
            )
        index += 2
    for flag in boolean_options:
        if argv.count(flag) != 1:
            raise PreparedActionError(
                "argv_contract_invalid", f"argv must contain exactly one {flag}"
            )
    exact_options = {
        "--run-id": receipt["run"]["run_id"],
        "--project": receipt["project"]["alias"],
        "--infra": receipt["project"]["infra"],
        "--preset": "public-franka-lift",
        "--assume-decision": "promote_checkpoint",
        "--max-wait-seconds": str(receipt["runtime_policy"]["max_wait_seconds"]),
        "--output-format": "json",
    }
    for option, expected in exact_options.items():
        if _option_values(argv, option) != [expected]:
            raise PreparedActionError(
                "argv_contract_invalid", f"argv {option} does not match the receipt"
            )
    if sorted(_option_values(argv, "--secret-env")) != sorted(
        receipt["required_secret_env_names"]
    ):
        raise PreparedActionError(
            "argv_contract_invalid", "argv secret names do not match the receipt"
        )
    if set(receipt["required_secret_env_names"]) != CANONICAL_SECRET_ENV_NAMES:
        raise PreparedActionError(
            "argv_contract_invalid",
            "prepared Sim2Real action must declare the canonical runtime secret names",
        )
    variables = _option_values(argv, "--var")
    variable_mapping: dict[str, str] = {}
    for variable in variables:
        key, separator, value = variable.partition("=")
        if not separator or not key or key in variable_mapping:
            raise PreparedActionError(
                "argv_contract_invalid", "argv variables must be unique KEY=VALUE pairs"
            )
        variable_mapping[key] = value
    expected_variable_names = {
        "source_sha",
        "require_baked_npa",
        "bucket",
        "isaac_cache_pvc",
        *(f"{role}_image" for role in IMAGE_ROLES),
    }
    if set(variable_mapping) != expected_variable_names:
        raise PreparedActionError(
            "argv_contract_invalid", "argv does not contain the closed variable set"
        )
    if variable_mapping["require_baked_npa"] != "1":
        raise PreparedActionError(
            "argv_contract_invalid", "argv must require the baked NPA runtime"
        )
    for role, image in receipt["images"].items():
        if variable_mapping[f"{role}_image"] != image:
            raise PreparedActionError(
                "argv_contract_invalid", f"argv does not bind the {role} image"
            )
    if variable_mapping["source_sha"] != receipt["source"]["source_commit"]:
        raise PreparedActionError(
            "argv_contract_invalid", "argv does not bind the source commit"
        )


def _configured_secret_names(
    context: PreparedActionContext, project_alias: str
) -> set[str]:
    home = str(context.environment.get("HOME") or "")
    sandbox_prefix = "/tmp/npa-private-evidence/"
    if home.startswith(sandbox_prefix):
        home_path = context.evidence / home.removeprefix(sandbox_prefix)
    else:
        home_path = Path(home) if home else Path("/__npa_missing_private_home__")
    credentials_path = home_path / ".npa/credentials.yaml"
    try:
        credentials = yaml.safe_load(credentials_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError):
        return set()
    if not isinstance(credentials, dict):
        return set()
    available: set[str] = set()
    tokens = credentials.get("tokens")
    if isinstance(tokens, dict):
        for name in ("HF_TOKEN", "NEBIUS_TOKEN_FACTORY_KEY"):
            if tokens.get(name):
                available.add(name)
    ngc = credentials.get("ngc")
    if isinstance(ngc, dict) and ngc.get("api_key"):
        available.add("NGC_API_KEY")
    storage_candidates: list[Mapping[str, Any]] = []
    top_storage = credentials.get("storage")
    registry = credentials.get("project_credentials")
    projects = registry.get("projects") if isinstance(registry, dict) else None
    if isinstance(projects, dict):
        for project in projects.values():
            if not isinstance(project, dict):
                continue
            aliases = project.get("aliases") or []
            if project_alias not in aliases:
                continue
            storage = project.get("storage")
            if isinstance(storage, dict):
                storage_candidates.append(storage)
            break
    elif isinstance(top_storage, dict):
        storage_candidates.append(top_storage)
    for storage in storage_candidates:
        if storage.get("aws_access_key_id"):
            available.add("AWS_ACCESS_KEY_ID")
        if storage.get("aws_secret_access_key"):
            available.add("AWS_SECRET_ACCESS_KEY")
    return available


def _preflight_result_passed(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if "passed" in value:
        return value["passed"] is True
    return value.get("exit_code") == 0


def validate_receipt(
    receipt_path: Path,
    *,
    requested_action_id: str,
    context: PreparedActionContext,
    require_secrets: bool = True,
) -> dict[str, Any]:
    try:
        receipt_path.resolve().relative_to(context.evidence.resolve())
    except ValueError:
        pass
    else:
        raise PreparedActionError(
            "receipt_location_invalid",
            "receipt must be isolated from the agent-visible evidence mount",
        )
    if receipt_path.parent.resolve() != context.control_dir.resolve():
        raise PreparedActionError(
            "receipt_location_invalid", "receipt is outside the controller control directory"
        )
    receipt = _load_owner_only_json(receipt_path, private_root=context.private_root)
    top_keys = {
        "schema",
        "action_id",
        "created_at",
        "canonical_spec",
        "source",
        "benchmark",
        "run",
        "project",
        "staged_input",
        "images",
        "runtime_policy",
        "accepted_eulas",
        "required_secret_env_names",
        "preflights",
        "argv",
        "argv_sha256",
        "receipt_sha256",
    }
    _require_keys(receipt, top_keys, "receipt")
    if receipt["schema"] != RECEIPT_SCHEMA:
        raise PreparedActionError("receipt_schema_invalid", "unsupported receipt schema")
    expected_receipt_sha = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    if receipt["receipt_sha256"] != expected_receipt_sha:
        raise PreparedActionError("receipt_tampered", "receipt digest does not match")
    action_id = receipt["action_id"]
    if (
        not isinstance(action_id, str)
        or not _ACTION_ID_RE.fullmatch(action_id)
        or action_id != requested_action_id
    ):
        raise PreparedActionError("action_id_mismatch", "action id does not match")

    spec = _require_keys(
        receipt["canonical_spec"], {"sandbox_path", "sha256"}, "canonical_spec"
    )
    source = _require_keys(
        receipt["source"],
        {"workspace_commit", "source_commit", "workspace_state_sha256"},
        "source",
    )
    benchmark = _require_keys(receipt["benchmark"], {"base_commit"}, "benchmark")
    run = _require_keys(receipt["run"], {"run_id"}, "run")
    project = _require_keys(
        receipt["project"], {"alias", "infra", "selection_sha256"}, "project"
    )
    staged = _require_keys(
        receipt["staged_input"],
        {"manifest_sandbox_path", "manifest_sha256", "identity", "identity_sha256"},
        "staged_input",
    )
    runtime = _require_keys(
        receipt["runtime_policy"],
        {"runtime", "resume", "max_wait_seconds"},
        "runtime_policy",
    )
    _require_sha(spec["sha256"], "canonical_spec.sha256")
    _require_commit(source["workspace_commit"], "source.workspace_commit")
    _require_commit(source["source_commit"], "source.source_commit")
    _require_sha(source["workspace_state_sha256"], "source.workspace_state_sha256")
    _require_commit(benchmark["base_commit"], "benchmark.base_commit")
    _require_sha(project["selection_sha256"], "project.selection_sha256")
    _require_sha(staged["manifest_sha256"], "staged_input.manifest_sha256")
    _require_sha(staged["identity_sha256"], "staged_input.identity_sha256")
    _require_sha(receipt["argv_sha256"], "argv_sha256")
    _require_sha(receipt["receipt_sha256"], "receipt_sha256")
    if not isinstance(run["run_id"], str) or not _RUN_ID_RE.fullmatch(run["run_id"]):
        raise PreparedActionError("receipt_schema_invalid", "run id is invalid")
    if not isinstance(project["alias"], str) or not project["alias"]:
        raise PreparedActionError("receipt_schema_invalid", "project alias is empty")
    if project["selection_sha256"] != canonical_sha256(
        {"alias": project["alias"], "infra": project["infra"]}
    ):
        raise PreparedActionError("receipt_tampered", "project selection digest differs")
    if staged["identity_sha256"] != canonical_sha256(staged["identity"]):
        raise PreparedActionError("receipt_tampered", "staged input identity differs")
    if runtime != {"runtime": True, "resume": True, "max_wait_seconds": 0}:
        raise PreparedActionError(
            "receipt_schema_invalid", "runtime policy must be runtime resume with no deadline"
        )
    if receipt["accepted_eulas"] != ["isaac"]:
        raise PreparedActionError(
            "receipt_schema_invalid", "receipt must bind the scoped Isaac acceptance"
        )
    secret_names = receipt["required_secret_env_names"]
    if (
        not isinstance(secret_names, list)
        or not secret_names
        or secret_names != sorted(set(secret_names))
        or any(not isinstance(name, str) or not name for name in secret_names)
    ):
        raise PreparedActionError(
            "receipt_schema_invalid", "required secret names must be unique and sorted"
        )
    images = receipt["images"]
    if not isinstance(images, dict) or set(images) != set(IMAGE_ROLES):
        raise PreparedActionError(
            "receipt_schema_invalid", "receipt must bind exactly five image roles"
        )
    if any(
        not isinstance(value, str) or not _IMAGE_DIGEST_RE.fullmatch(value)
        for value in images.values()
    ):
        raise PreparedActionError(
            "receipt_schema_invalid", "all images must be immutable digest references"
        )
    preflights = receipt["preflights"]
    if not isinstance(preflights, list) or [item.get("name") for item in preflights if isinstance(item, dict)] != list(REQUIRED_PREFLIGHTS):
        raise PreparedActionError(
            "receipt_schema_invalid", "receipt does not bind every required preflight"
        )
    for item in preflights:
        _require_keys(
            item,
            {"name", "evidence_sandbox_path", "evidence_sha256"},
            "preflight",
        )
        _require_sha(item["evidence_sha256"], "preflight.evidence_sha256")

    if argv_sha256(receipt["argv"]) != receipt["argv_sha256"]:
        raise PreparedActionError("argv_digest_mismatch", "argv digest does not match")
    _validate_argv_contract(receipt)
    spec_path = resolve_sandbox_private_path(spec["sandbox_path"], evidence=context.evidence)
    manifest_path = resolve_sandbox_private_path(
        staged["manifest_sandbox_path"], evidence=context.evidence
    )
    if file_sha256(spec_path) != spec["sha256"]:
        raise PreparedActionError("spec_mismatch", "canonical spec digest changed")
    if file_sha256(manifest_path) != staged["manifest_sha256"]:
        raise PreparedActionError("staged_input_mismatch", "staged input digest changed")
    for item in preflights:
        evidence_path = resolve_sandbox_private_path(
            item["evidence_sandbox_path"], evidence=context.evidence
        )
        if file_sha256(evidence_path) != item["evidence_sha256"]:
            raise PreparedActionError(
                "preflight_evidence_mismatch",
                f"{item['name']} preflight evidence changed",
            )
        try:
            evidence_result = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PreparedActionError(
                "preflight_evidence_invalid",
                f"{item['name']} preflight evidence is not valid JSON",
            ) from exc
        if not _preflight_result_passed(evidence_result):
            raise PreparedActionError(
                "preflight_not_passed", f"{item['name']} preflight is not passed"
            )
    observed_workspace = workspace_state(context.workspace)
    if (
        observed_workspace["head"] != source["workspace_commit"]
        or observed_workspace["detached"] is not True
        or canonical_sha256(observed_workspace) != source["workspace_state_sha256"]
        or benchmark["base_commit"] != source["workspace_commit"]
    ):
        raise PreparedActionError("source_mismatch", "workspace/source state changed")
    environment = context.environment
    if environment.get("NPA_PROJECT") != project["alias"]:
        raise PreparedActionError("project_mismatch", "selected project changed")
    if environment.get("NPA_SIM2REAL_INFRA") != project["infra"]:
        raise PreparedActionError("project_mismatch", "selected infrastructure changed")
    if environment.get("NPA_SIM2REAL_RUN_ID") != run["run_id"]:
        raise PreparedActionError("run_identity_mismatch", "run identity changed")
    if environment.get("NPA_SIM2REAL_SOURCE_SHA") != source["source_commit"]:
        raise PreparedActionError("source_mismatch", "staged source commit changed")
    if require_secrets:
        configured = _configured_secret_names(context, project["alias"])
        missing = [
            name
            for name in secret_names
            if not str(environment.get(name) or "") and name not in configured
        ]
        if missing:
            raise PreparedActionError(
                "missing_required_secrets",
                "required secret environment names are unavailable: " + ", ".join(missing),
            )
    return receipt


def _append_private_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def _state_events(path: Path, action_id: str | None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if item.get("schema") == STATE_SCHEMA and (
            action_id is None or item.get("action_id") == action_id
        ):
            events.append(item)
    return events


def recover_occurrence(
    state_path: Path, *, action_id: str, occurrence_id: str
) -> tuple[str, dict[str, Any] | None]:
    events = [
        event
        for event in _state_events(state_path, action_id)
        if event.get("occurrence_id") == occurrence_id
    ]
    finished = next(
        (event for event in reversed(events) if event.get("phase") == "execution_finished"),
        None,
    )
    if finished and isinstance(finished.get("result"), dict):
        return "finished", dict(finished["result"])
    if any(event.get("phase") == "execution_started" for event in events):
        return "indeterminate", None
    return "not_started", None


def _prior_execution_state(state_path: Path, action_id: str | None = None) -> str:
    events = _state_events(state_path, action_id)
    starts = {str(item.get("occurrence_id")) for item in events if item.get("phase") == "execution_started"}
    finishes = {str(item.get("occurrence_id")) for item in events if item.get("phase") == "execution_finished"}
    if starts - finishes:
        return "indeterminate"
    if finishes:
        return "finished"
    return "unused"


def _run_argv_in_isolation(
    argv: Sequence[str],
    *,
    workspace: Path,
    environment: Mapping[str, str],
    isolation: Mapping[str, Path],
) -> subprocess.CompletedProcess[str]:
    namespace_script = r"""
set -eu
mount --make-rprivate /
mount -t tmpfs tmpfs /tmp
mkdir -p /tmp/npa-private-evidence /tmp/npa-trial-workspace
mount --bind "$1" /tmp/npa-private-evidence
mount --bind "$2" /tmp/npa-trial-workspace
cd /tmp/npa-trial-workspace
mount -t tmpfs tmpfs "$3"
mount -t tmpfs tmpfs "$4"
shift 4
exec "$@"
"""
    command = [
        "unshare",
        "--user",
        "--map-root-user",
        "--mount",
        "--pid",
        "--fork",
        "bash",
        "-c",
        namespace_script,
        "--",
        str(isolation["evidence"]),
        str(workspace),
        str(isolation["private_root"]),
        str(isolation["controller_repo"]),
        *argv,
    ]
    return subprocess.run(
        command,
        cwd=workspace,
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )


def _safe_result(
    *,
    receipt: Mapping[str, Any],
    completed: subprocess.CompletedProcess[str],
    duration_seconds: float,
    output_sha256: str,
    output_characters: int,
) -> dict[str, Any]:
    accepted = False
    status = "INDETERMINATE"
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        returned_run_id = payload.get("run_id") or payload.get("workflow_run_id")
        candidate = str(payload.get("status") or "").strip().upper()
        authoritative = (
            returned_run_id == receipt["run"]["run_id"]
            and candidate
            and len(candidate) <= 64
            and re.fullmatch(r"[A-Z0-9_-]+", candidate)
        )
        if authoritative and (
            completed.returncode == 0 or is_terminal_fail(candidate)
        ):
            accepted = True
            status = candidate
    safe_view = {
        "submission_accepted": accepted,
        "action_consumed": True,
        "safe_run_reference": receipt["run"]["run_id"],
        "status": status,
        "exit_code": completed.returncode,
    }
    result: dict[str, Any] = {
        "schema": "npa.sim2real.prepared_workflow_action.result.v1",
        **safe_view,
        "duration_seconds": duration_seconds,
        "error": None,
        "evidence": {
            "full_output_sha256": output_sha256,
            "full_output_characters": output_characters,
            "bounded_view": safe_view,
            "bounded_view_sha256": canonical_sha256(safe_view),
        },
    }
    terminal_failure = accepted and is_terminal_fail(status)
    if terminal_failure:
        result["error"] = {
            "classification": "workflow_terminal_failure",
            "retryable": False,
            "action": "inspect the terminal workflow failure and private evidence; do not replay this prepared action",
        }
    elif not accepted:
        result["error"] = {
            "classification": (
                "workflow_submit_response_invalid"
                if completed.returncode == 0
                else "workflow_submit_indeterminate"
            ),
            "retryable": False,
            "action": "inspect durable workflow state and private evidence; do not replay this prepared action",
        }
    return result


def rejected_result(
    *, action_id: str, classification: str, message: str
) -> dict[str, Any]:
    return {
        "schema": "npa.sim2real.prepared_workflow_action.result.v1",
        "submission_accepted": False,
        "action_consumed": False,
        "safe_run_reference": None,
        "status": "REJECTED",
        "duration_seconds": 0.0,
        "error": {
            "classification": classification,
            "retryable": False,
            "action": message,
        },
        "action_id": action_id,
    }


def _execute_prepared_action_locked(
    receipt_path: Path,
    *,
    requested_action_id: str,
    occurrence_id: str,
    context: PreparedActionContext,
    runner: Callable[..., subprocess.CompletedProcess[str]] = _run_argv_in_isolation,
) -> dict[str, Any]:
    state_path = context.control_dir / "prepared-action-state.jsonl"
    prior = _prior_execution_state(state_path)
    if prior == "indeterminate":
        return rejected_result(
            action_id=requested_action_id,
            classification="indeterminate_prior_execution",
            message="reconcile durable workflow state; this action will not be replayed",
        )
    if prior == "finished":
        return rejected_result(
            action_id=requested_action_id,
            classification="duplicate_submission_prevented",
            message="the prepared action occurrence has already finished",
        )
    try:
        receipt = validate_receipt(
            receipt_path,
            requested_action_id=requested_action_id,
            context=context,
            require_secrets=True,
        )
    except PreparedActionError as exc:
        return rejected_result(
            action_id=requested_action_id,
            classification=exc.classification,
            message=str(exc),
        )
    started_at = utc_now()
    _append_private_jsonl(
        state_path,
        {
            "schema": STATE_SCHEMA,
            "phase": "execution_started",
            "action_id": requested_action_id,
            "occurrence_id": occurrence_id,
            "receipt_sha256": receipt["receipt_sha256"],
            "argv_sha256": receipt["argv_sha256"],
            "at": started_at,
        },
    )
    started = time.monotonic()
    try:
        completed = runner(
            receipt["argv"],
            workspace=context.workspace,
            environment=context.environment,
            isolation=context.isolation,
        )
    except Exception as exc:
        result = {
            "schema": "npa.sim2real.prepared_workflow_action.result.v1",
            "submission_accepted": False,
            "action_consumed": True,
            "safe_run_reference": receipt["run"]["run_id"],
            "status": "INDETERMINATE",
            "duration_seconds": time.monotonic() - started,
            "error": {
                "classification": "workflow_submit_execution_indeterminate",
                "retryable": False,
                "action": "reconcile durable workflow state; do not replay this action",
            },
        }
        _append_private_jsonl(
            state_path,
            {
                "schema": STATE_SCHEMA,
                "phase": "execution_indeterminate",
                "action_id": requested_action_id,
                "occurrence_id": occurrence_id,
                "exception_type": type(exc).__name__,
                "result": result,
                "at": utc_now(),
            },
        )
        return result
    duration = time.monotonic() - started
    raw_output = {
        "schema": OUTPUT_SCHEMA,
        "action_id": requested_action_id,
        "occurrence_id": occurrence_id,
        "receipt_sha256": receipt["receipt_sha256"],
        "argv_sha256": receipt["argv_sha256"],
        "started_at": started_at,
        "completed_at": utc_now(),
        "duration_seconds": duration,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    raw_serialized = json.dumps(raw_output, sort_keys=True, separators=(",", ":"))
    raw_sha = hashlib.sha256(raw_serialized.encode()).hexdigest()
    _append_private_jsonl(
        context.control_dir / "prepared-action-output.jsonl",
        {**raw_output, "record_sha256": raw_sha},
    )
    result = _safe_result(
        receipt=receipt,
        completed=completed,
        duration_seconds=duration,
        output_sha256=raw_sha,
        output_characters=len(raw_serialized),
    )
    _append_private_jsonl(
        state_path,
        {
            "schema": STATE_SCHEMA,
            "phase": "execution_finished",
            "action_id": requested_action_id,
            "occurrence_id": occurrence_id,
            "receipt_sha256": receipt["receipt_sha256"],
            "argv_sha256": receipt["argv_sha256"],
            "result": result,
            "at": utc_now(),
        },
    )
    return result


def execute_prepared_action(
    receipt_path: Path,
    *,
    requested_action_id: str,
    occurrence_id: str,
    context: PreparedActionContext,
    runner: Callable[..., subprocess.CompletedProcess[str]] = _run_argv_in_isolation,
) -> dict[str, Any]:
    """Serialize check-and-execute so parallel controllers cannot cross the boundary."""

    lock_path = context.control_dir / "prepared-action.lock"
    context.control_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "a+", encoding="utf-8") as lock_handle:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        return _execute_prepared_action_locked(
            receipt_path,
            requested_action_id=requested_action_id,
            occurrence_id=occurrence_id,
            context=context,
            runner=runner,
        )
