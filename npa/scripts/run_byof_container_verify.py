#!/usr/bin/env python3
"""Submit BYOF container-verify SkyPilot workloads (CPU smoke for /opt/byof clone)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from npa.workflows.byof.live import resolve_byof_profile_path
from npa.clients.project_credentials import (
    s3_client_for_project,
    storage_env_for_project,
)
from npa.orchestration.skypilot import submit_workflow, workflow_status
from npa.orchestration.skypilot._bin import (
    SkyPilotConfigError,
    SkyPilotNotInstalledError,
    SkyPilotVersionError,
    resolve_sky_bin,
)
from npa.orchestration.skypilot.cleanup import sky_environment
from npa.orchestration.skypilot.signal_teardown import (
    SignalTeardown,
    install_teardown_signal_handlers,
    restore_signal_handlers,
)

DEFAULT_YAML = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "npa"
    / "workflows"
    / "byof"
    / "profiles"
    / "byof-container-smoke-rtxpro.yaml"
)
DEFAULT_IMAGE_PULL_SECRETS = ("agent-sa",)


#: Credentials every BYOF resource profile needs, because each one uploads its summary
#: and artifacts to S3. Forwarded as SkyPilot task secrets (never written into the
#: rendered YAML). Without this a run provisions, pulls the image, executes the profile
#: and then dies at the upload with
#: ``botocore.exceptions.NoCredentialsError: Unable to locate credentials``.
OPERATOR_RUNTIME_ENVS = ("NPA_WAN_ACCEPT_NVIDIA_RUNTIME_TERMS",)
DEFAULT_SECRET_ENVS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
)


def resolve_secret_envs(explicit: list[str] | None) -> list[str]:
    """Return the secret env names to forward to SkyPilot.

    An explicit ``--secret-env`` list replaces the default storage names. An explicitly
    set operator-runtime gate is appended in either case so acceptance cannot fall back
    to rendered YAML. Names with no value are dropped, since SkyPilot rejects a secret
    it cannot resolve.
    """

    names = list(explicit if explicit is not None else DEFAULT_SECRET_ENVS)
    # Operator acceptance is runtime state, not workflow configuration. Always
    # carry an explicitly set gate through SkyPilot's redacted secret channel,
    # even when a caller supplies an otherwise explicit secret allowlist.
    names.extend(OPERATOR_RUNTIME_ENVS)
    return [name for name in dict.fromkeys(names) if os.environ.get(name)]


def _normalize_s3_bucket(value: str) -> str:
    """Return a bare bucket name from ``bucket`` or ``s3://bucket[/prefix]``."""

    text = (value or "").strip()
    if not text:
        return ""
    if text.startswith("s3://"):
        remainder = text[len("s3://") :]
        return remainder.split("/", 1)[0].strip()
    return text.split("/", 1)[0].strip()


def _normalize_output_root(value: str, *, default_prefix: str = "byof") -> str:
    """Normalize output roots that may already include ``s3://`` or a path prefix."""

    text = (value or "").strip()
    if not text:
        bucket = _normalize_s3_bucket(os.environ.get("NPA_S3_BUCKET", ""))
        if not bucket:
            bucket = "your-bucket-name"
        return f"s3://{bucket}/{default_prefix}"
    if text.startswith("s3://"):
        # Collapse accidental ``s3://s3://bucket/...`` forms.
        while text.startswith("s3://s3://"):
            text = "s3://" + text[len("s3://s3://") :]
        return text.rstrip("/")
    bucket = _normalize_s3_bucket(text)
    remainder = text.split("/", 1)[1].strip("/") if "/" in text else default_prefix
    return f"s3://{bucket}/{remainder or default_prefix}"


DEFAULT_BUCKET = (
    _normalize_s3_bucket(os.environ.get("NPA_S3_BUCKET", "")) or "your-bucket-name"
)
DEFAULT_OUTPUT_ROOT = _normalize_output_root(
    os.environ.get("NPA_BYOF_OUTPUT_ROOT", ""), default_prefix="byof"
)
TERMINAL_STATUSES = {
    "SUCCEEDED",
    "CANCELLED",
    "FAILED",
    "FAILED_SETUP",
    "FAILED_PRECHECKS",
    "FAILED_NO_RESOURCE",
    "FAILED_CONTROLLER",
}


def render_workflow(
    yaml_path: Path,
    *,
    run_id: str,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    image: str = "",
    repo_root: str = "/opt/byof",
    smoke_command: str = "",
    solution_name: str = "",
    capability_name: str = "",
    smoke_artifact_name: str = "",
) -> list[dict[str, Any]]:
    docs = _load_yaml_documents(yaml_path)
    for doc in docs[1:]:
        envs = doc.get("envs")
        if not isinstance(envs, dict):
            continue
        envs["NPA_BYOF_RUN_ID"] = run_id
        envs["BYOF_REPO_ROOT"] = repo_root
        envs["BYOF_SMOKE_COMMAND"] = smoke_command
        envs["BYOF_SOLUTION_NAME"] = solution_name
        envs["BYOF_CAPABILITY_NAME"] = capability_name
        envs["BYOF_SMOKE_ARTIFACT_NAME"] = smoke_artifact_name
        normalized_root = _normalize_output_root(output_root)
        envs["S3_OUTPUT_PREFIX"] = normalized_root.rstrip("/") + f"/{run_id}/"
        bucket = _normalize_s3_bucket(normalized_root) or _normalize_s3_bucket(
            os.environ.get("NPA_S3_BUCKET", "")
        )
        if bucket:
            envs["NPA_S3_BUCKET"] = bucket
        storage_env = _resolved_storage_env()
        for key in (
            "AWS_ENDPOINT_URL",
            "NEBIUS_S3_ENDPOINT",
            "NPA_S3_BUCKET",
        ):
            value = ""
            for candidate in (
                os.environ.get(key, "").strip(),
                storage_env.get(key, "").strip(),
            ):
                if candidate and not (
                    candidate.startswith("${") and candidate.endswith("}")
                ):
                    value = candidate
                    break
            if not value:
                continue
            # Prefer the bucket derived from --output-root when already set.
            # Compare via `key` only — avoid a second "NPA_S3_BUCKET" literal that
            # trips gitleaks generic-api-key on `key == "…"`.
            if key.endswith("_S3_BUCKET") and envs.get(key):
                continue
            envs[key] = value
        if image:
            image_ref = image.removeprefix("docker:")
            envs["BYOF_IMAGE"] = image_ref
            resources = doc.setdefault("resources", {})
            if isinstance(resources, dict):
                resources["image_id"] = f"docker:{image_ref}"
    return docs


def _load_yaml_documents(path: Path) -> list[dict[str, Any]]:
    docs = [
        doc
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if doc is not None
    ]
    if not docs:
        raise ValueError(f"empty SkyPilot YAML: {path}")
    return docs


def _task_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if (
        len(docs) > 1
        and isinstance(docs[0], dict)
        and "execution" in docs[0]
        and "run" not in docs[0]
    ):
        return docs[1:]
    return docs


def _write_yaml_documents(path: Path, docs: list[dict[str, Any]]) -> None:
    path.write_text(
        yaml.safe_dump_all(_task_docs(docs), sort_keys=False), encoding="utf-8"
    )


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("byof-container-%Y%m%dT%H%M%SZ")


def _resolved_storage_env() -> dict[str, str]:
    """Resolve project/host S3 env so rendered BYOF YAMLs are not left with ${...}."""

    project = (
        os.environ.get("NPA_E2E_PROJECT", "").strip()
        or os.environ.get("NPA_PROJECT", "").strip()
        or os.environ.get("NPA_BYOF_PROJECT", "").strip()
    )
    try:
        return dict(storage_env_for_project(project or None, allow_host_creds=True))
    except Exception as exc:  # noqa: BLE001 - best-effort for render/launch paths
        print(f"WARN: skipped BYOF storage env resolution: {exc}", file=sys.stderr)
        return {}


def preflight_output_storage(*, output_root: str, run_id: str) -> None:
    """Reserve a new S3 run prefix and prove it is writable before compute."""

    parsed = urlparse(_normalize_output_root(output_root))
    bucket = parsed.netloc.strip()
    if parsed.scheme != "s3" or not bucket:
        raise ValueError("BYOF output root must be a valid s3:// URI")
    prefix = parsed.path.strip("/")
    run_prefix = "/".join(part for part in (prefix, run_id) if part).rstrip("/") + "/"
    key = run_prefix + ".npa-write-preflight"
    project = (
        os.environ.get("NPA_E2E_PROJECT", "").strip()
        or os.environ.get("NPA_PROJECT", "").strip()
        or os.environ.get("NPA_BYOF_PROJECT", "").strip()
    )
    client = s3_client_for_project(project or None, allow_host_creds=True)
    created = False
    try:
        existing = client.list_objects_v2(Bucket=bucket, Prefix=run_prefix, MaxKeys=1)
        if existing.get("Contents"):
            raise RuntimeError(
                "refusing to reuse a non-empty BYOF run prefix; choose a new run ID"
            )
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=b"npa BYOF write preflight\n",
            ContentType="text/plain",
            IfNoneMatch="*",
        )
        created = True
        head = client.head_object(Bucket=bucket, Key=key)
        if int(head.get("ContentLength", -1)) <= 0:
            raise RuntimeError("S3 write preflight object is unexpectedly empty")
        client.delete_object(Bucket=bucket, Key=key)
        created = False
    except Exception as exc:  # noqa: BLE001 - preserve provider error as launch blocker
        if created:
            try:
                client.delete_object(Bucket=bucket, Key=key)
            except Exception:  # noqa: BLE001 - retain the original preflight failure
                pass
        raise RuntimeError(
            f"BYOF output storage preflight failed for s3://{bucket}/{prefix}: {exc}"
        ) from exc


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # `resolve_byof_profile_path` also accepts a bare packaged profile NAME, which is what
    # an npa.workflow spec must pass: a stage runs in a pod with no repo checkout.
    parser.add_argument(
        "--yaml", dest="yaml_path", type=resolve_byof_profile_path, default=DEFAULT_YAML
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--image", default="")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default="/opt/byof")
    parser.add_argument("--smoke-command", default="")
    parser.add_argument("--solution-name", default="")
    parser.add_argument("--capability-name", default="")
    parser.add_argument("--smoke-artifact-name", default="")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--infra", default=os.environ.get("NPA_BYOF_INFRA", ""))
    parser.add_argument("--sky-bin", default="")
    parser.add_argument(
        "--secret-env",
        action="append",
        default=None,
        help=(
            "Env var name to forward as a SkyPilot task secret (repeatable). "
            "Defaults to the S3 credentials the profile needs for its uploads."
        ),
    )
    parser.add_argument("--submit-timeout", type=int, default=600)
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=3600,
        help="0 checks once, positive values bound the wait, and -1 waits until terminal.",
    )
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--isolated-config-dir", default="")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument(
        "--direct-launch",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("NPA_BYOF_DIRECT_LAUNCH", "1") != "0",
    )
    parser.add_argument(
        "--cleanup", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return _submit_and_wait(args)
    except (
        SkyPilotNotInstalledError,
        SkyPilotConfigError,
        SkyPilotVersionError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def _wait_for_terminal(
    run_id: str,
    *,
    sky_bin: str,
    wait_timeout: int,
    poll_interval: int,
) -> tuple[Any, dict[str, Any]]:
    """Poll with explicit immediate, bounded, or indefinite semantics."""

    if wait_timeout < -1:
        raise ValueError("--wait-timeout must be -1, zero, or a positive number")
    mode = (
        "indefinite"
        if wait_timeout == -1
        else ("immediate" if wait_timeout == 0 else "bounded")
    )
    deadline = None if wait_timeout == -1 else time.time() + wait_timeout
    statuses: list[str] = []
    final = workflow_status(run_id, sky_bin=sky_bin)
    statuses.append(final.status)
    polls = 1
    while (
        final.status not in TERMINAL_STATUSES
        and wait_timeout != 0
        and (deadline is None or time.time() < deadline)
    ):
        time.sleep(max(poll_interval, 1))
        final = workflow_status(run_id, sky_bin=sky_bin)
        statuses.append(final.status)
        polls += 1
    diagnostics = {
        "mode": mode,
        "polls": polls,
        "statuses": statuses,
        "terminal": final.status in TERMINAL_STATUSES,
        "deadline_exhausted": bool(
            wait_timeout > 0
            and final.status not in TERMINAL_STATUSES
            and deadline is not None
            and time.time() >= deadline
        ),
    }
    if not diagnostics["terminal"]:
        diagnostics["stuck_state"] = final.status
        diagnostics["hint"] = (
            "workflow is not terminal; inspect SkyPilot controller/job and pod events"
        )
    return final, diagnostics


def _submit_and_wait(args: argparse.Namespace) -> int:
    run_id = args.run_id or _default_run_id()
    output_root = _normalize_output_root(args.output_root)
    docs = render_workflow(
        args.yaml_path,
        run_id=run_id,
        output_root=output_root,
        image=args.image,
        repo_root=args.repo_root,
        smoke_command=args.smoke_command,
        solution_name=args.solution_name,
        capability_name=args.capability_name,
        smoke_artifact_name=args.smoke_artifact_name,
    )
    outputs = {
        "root": output_root.rstrip("/") + f"/{run_id}/",
        "summary": output_root.rstrip("/") + f"/{run_id}/npa_byof_summary.json",
    }

    if args.render_only:
        render_dir = Path(tempfile.mkdtemp(prefix=f"npa-byof-container-{run_id}-"))
        rendered_yaml = render_dir / "byof-container.rendered.yaml"
        _write_yaml_documents(rendered_yaml, docs)
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "rendered_yaml": str(rendered_yaml),
                    "outputs": outputs,
                },
                indent=2,
            )
        )
        return 0

    preflight_output_storage(output_root=output_root, run_id=run_id)

    with tempfile.TemporaryDirectory(prefix=f"npa-byof-container-{run_id}-") as tmp:
        tmp_path = Path(tmp)
        previous_kubeconfig = os.environ.get("KUBECONFIG")
        sky_bin = str(
            resolve_sky_bin(args.sky_bin or os.environ.get("NPA_SKYPILOT_BIN"))
        )
        try:
            _normalize_kubeconfig_current_context(tmp_path)
            rendered_yaml = Path(tmp) / "byof-container.rendered.yaml"
            _write_yaml_documents(rendered_yaml, docs)
            infra = args.infra or _default_infra()
            config_path = args.config_path or _write_default_k8s_config(tmp_path, infra)
            _ensure_infra_enabled(sky_bin=sky_bin, infra=infra, config_path=config_path)
            if args.direct_launch:
                return _direct_launch(
                    rendered_yaml=rendered_yaml,
                    run_id=run_id,
                    outputs=outputs,
                    sky_bin=sky_bin,
                    infra=infra,
                    config_path=config_path,
                    cleanup=args.cleanup,
                    secret_envs=resolve_secret_envs(args.secret_env),
                )
            teardown_guard = SignalTeardown(
                run_id=run_id,
                isolated_config_dir=args.isolated_config_dir,
                sky_bin=sky_bin,
                poll_interval=max(float(args.poll_interval), 0.0),
            )
            previous_handlers = install_teardown_signal_handlers(
                teardown_guard.teardown
            )
            summary: dict[str, Any] | None = None
            return_code = 1
            try:
                teardown_guard.mark_launched()
                submit_config_path = Path(config_path) if config_path else None
                result = submit_workflow(
                    rendered_yaml,
                    run_id,
                    isolated_config_dir=args.isolated_config_dir,
                    config_path=submit_config_path,
                    sky_bin=sky_bin,
                    infra=infra,
                    secret_envs=resolve_secret_envs(args.secret_env),
                    timeout=args.submit_timeout,
                )
                submitted_config_path = (
                    Path(result.log_paths["config"])
                    if result.log_paths.get("config")
                    else None
                )
                teardown_guard.mark_launched(config_path=submitted_config_path)
                summary = {
                    "run_id": run_id,
                    "submit": result.__dict__,
                    "outputs": outputs,
                }
                final, wait_diagnostics = _wait_for_terminal(
                    run_id,
                    sky_bin=sky_bin,
                    wait_timeout=args.wait_timeout,
                    poll_interval=args.poll_interval,
                )
                summary["final"] = final.__dict__
                summary["wait"] = wait_diagnostics
                return_code = 0 if final.status == "SUCCEEDED" else 1
                if (
                    os.environ.get("NPA_ISAAC_LAB_ACCEPT_PRECHECK_FAILURE") == "1"
                    and final.status == "FAILED_PRECHECKS"
                ):
                    return_code = 0
            finally:
                restore_signal_handlers(previous_handlers)
                if args.cleanup:
                    teardown_guard.teardown()
            print(json.dumps(summary or {"run_id": run_id}, indent=2, sort_keys=True))
            return return_code
        finally:
            # Restore KUBECONFIG and stop the Sky API so a temp kubeconfig path
            # written under TemporaryDirectory cannot poison later sky launches.
            if previous_kubeconfig is None:
                os.environ.pop("KUBECONFIG", None)
            else:
                os.environ["KUBECONFIG"] = previous_kubeconfig
            if os.environ.get("NPA_BYOF_REFRESH_SKY_API", "1") != "0":
                subprocess.run(
                    [sky_bin, "api", "stop"],
                    env=sky_environment(None),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )


def _direct_launch(
    *,
    rendered_yaml: Path,
    run_id: str,
    outputs: dict[str, str],
    sky_bin: str,
    infra: str,
    config_path: str = "",
    cleanup: bool = True,
    secret_envs: list[str] | None = None,
) -> int:
    cmd = [
        sky_bin,
        "launch",
        "--yes",
        "--cluster",
        run_id,
        "--name",
        run_id,
    ]
    if cleanup:
        cmd.append("--down")
    if infra:
        # Prefer k8s/<context> form when we know the context; bare "kubernetes"
        # is fine once sky check has enabled it.
        cmd.extend(["--infra", infra])
    if config_path:
        cmd.extend(["--config", config_path])
    launch_env = sky_environment(None)
    for secret_name in secret_envs or ():
        if launch_env.get(secret_name):
            cmd.extend(["--secret", secret_name])
    cmd.append(str(rendered_yaml))
    # Ensure kubeconfig is visible to sky even when only KUBECONTEXT was set.
    if not launch_env.get("KUBECONFIG"):
        default_kube = Path.home() / ".kube" / "config"
        if default_kube.is_file():
            launch_env["KUBECONFIG"] = str(default_kube)
    result = subprocess.run(
        cmd,
        env=launch_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    summary = {
        "run_id": run_id,
        "mode": "direct-launch",
        "outputs": outputs,
        "command": cmd,
        "final": {
            "status": "SUCCEEDED" if result.returncode == 0 else "FAILED",
            "returncode": result.returncode,
        },
    }
    if result.stdout:
        summary["stdout_tail"] = result.stdout[-8000:]
    if result.stderr:
        summary["stderr_tail"] = result.stderr[-8000:]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.returncode == 0 else 1


def _default_infra() -> str:
    configured = (
        os.environ.get("NPA_BYOF_INFRA", "").strip()
        or os.environ.get("NPA_SKYPILOT_INFRA", "").strip()
    )
    if configured:
        return configured
    context = (
        os.environ.get("NPA_BYOF_K8S_CONTEXT", "")
        or os.environ.get("NPA_K8S_CONTEXT", "")
        or os.environ.get("KUBECONTEXT", "")
    ).strip()
    if context:
        return f"k8s/{context}"
    return "kubernetes"


def _normalize_kubeconfig_current_context(tmp_path: Path) -> None:
    kubeconfig = os.environ.get("KUBECONFIG", "").strip()
    context = (
        os.environ.get("KUBECONTEXT", "")
        or os.environ.get("NPA_BYOF_K8S_CONTEXT", "")
        or os.environ.get("NPA_K8S_CONTEXT", "")
    ).strip()
    if not kubeconfig or not context:
        return
    path = Path(kubeconfig)
    if not path.is_file():
        return
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return
    data["current-context"] = context
    target = tmp_path / "kubeconfig"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    os.environ["KUBECONFIG"] = str(target)


def _write_default_k8s_config(tmp_path: Path, infra: str) -> str:
    normalized = infra.strip().lower()
    if not (normalized.startswith("k8s") or normalized.startswith("kubernetes")):
        return ""
    path = tmp_path / "skypilot-byof-k8s-config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "kubernetes": {
                    "pod_config": {
                        "spec": {
                            "imagePullSecrets": [
                                {"name": name} for name in DEFAULT_IMAGE_PULL_SECRETS
                            ],
                        }
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return str(path)


def _ensure_infra_enabled(*, sky_bin: str, infra: str, config_path: str = "") -> None:
    if os.environ.get("NPA_BYOF_SKIP_SKY_CHECK") == "1":
        return
    normalized = infra.strip().lower()
    if not (normalized.startswith("kubernetes") or normalized.startswith("k8s")):
        return
    if os.environ.get("NPA_BYOF_REFRESH_SKY_API", "1") != "0":
        subprocess.run(
            [sky_bin, "api", "stop"],
            env=sky_environment(None),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    cmd = [sky_bin, "check", "kubernetes", "-o", "json"]
    if config_path:
        cmd.extend(["--config", config_path])
    result = subprocess.run(
        cmd,
        env=sky_environment(None),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise SkyPilotConfigError(
            f"SkyPilot Kubernetes check failed before BYOF smoke submission: {detail}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
