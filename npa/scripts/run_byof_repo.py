#!/usr/bin/env python3
"""Build/push a BYOF image from an OSS repo and launch a live workload on Nebius."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from npa.clients.config import resolve_container_registry
from npa.clients.project_credentials import storage_env_for_project
from npa.deploy.images import container_image_for_tool
from npa.workflows.byof.live import resolve_byof_kubernetes_target

SCRIPT_DIR = Path(__file__).resolve().parent
ISAAC_RUNNER = SCRIPT_DIR / "run_isaac_lab_rl.py"
DATAGEN_RUNNER = SCRIPT_DIR / "run_byof_datagen.py"
CONTAINER_VERIFY_RUNNER = SCRIPT_DIR / "run_byof_container_verify.py"
BYOF_REPO_MOUNT = "/opt/byof"

DEFAULT_REPO_URL = "https://github.com/LightwheelAI/leisaac.git"
DEFAULT_REPO_REF = "main"
DEFAULT_UBUNTU_BASE_IMAGE = "ubuntu:22.04"
BASE_PROFILES = frozenset({"ubuntu", "isaac-lab"})
PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "<base-image>",
        "<repo-url>",
        "<repo-ref>",
        "<workload>",
        "<resource-profile.yaml>",
        "<task>",
        "<base-profile>",
    }
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _normalize_optional(value: str) -> str:
    cleaned = str(value or "").strip()
    if cleaned in PLACEHOLDER_VALUES:
        return ""
    return cleaned


def _run(
    cmd: list[str],
    *,
    stdin: str | None = None,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    runtime_env = dict(os.environ)
    # Avoid stale operator tokens overriding profile-based auth on shared VMs.
    runtime_env.pop("NEBIUS_IAM_TOKEN", None)
    runtime_env.pop("NEBIUS_IAM_TOKEN_FILE", None)
    if env is not None:
        runtime_env.update(env)
    kwargs: dict[str, Any] = {"text": True, "check": False}
    if stdin is not None:
        kwargs["input"] = stdin
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    kwargs["env"] = runtime_env
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, **kwargs)
    if proc.returncode != 0:
        if capture:
            raise RuntimeError(
                f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def _registry_server(image_ref: str) -> str:
    ref = image_ref.removeprefix("docker:")
    return ref.split("/", 1)[0]


def _bare_s3_bucket(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if text.startswith("s3://"):
        text = text[len("s3://") :]
    return text.split("/", 1)[0].strip()


def _live_runner_env(project: str) -> dict[str, str]:
    env: dict[str, str] = {}
    target = resolve_byof_kubernetes_target(project or None)
    if target.kubeconfig:
        env["KUBECONFIG"] = target.kubeconfig
    if target.context:
        env["KUBECONTEXT"] = target.context
        env["NPA_BYOF_K8S_CONTEXT"] = target.context
    if target.namespace:
        env["NPA_BYOF_K8S_NAMESPACE"] = target.namespace
    try:
        env.update(storage_env_for_project(project or None, allow_host_creds=True))
    except Exception as exc:
        print(f"WARN: skipped BYOF storage env resolution: {exc}", file=sys.stderr)
    # Project configs often store checkpoint_bucket as s3://bucket/prefix. BYOF
    # SkyPilot templates expect a bare bucket name in NPA_S3_BUCKET.
    for key in ("NPA_S3_BUCKET", "S3_BUCKET"):
        bare = _bare_s3_bucket(os.environ.get(key, "") or env.get(key, ""))
        if bare:
            env["NPA_S3_BUCKET"] = bare
            break
    if "NPA_S3_BUCKET" not in env:
        try:
            from npa.clients.config import _load_yaml, _resolve_project_section

            yml = _load_yaml()
            section = _resolve_project_section(yml, project or None) if project else {}
            storage = section.get("storage") if isinstance(section, dict) else {}
            if not isinstance(storage, dict):
                storage = yml.get("storage") if isinstance(yml.get("storage"), dict) else {}
            bare = _bare_s3_bucket(
                str(
                    storage.get("checkpoint_bucket")
                    or storage.get("bucket")
                    or storage.get("s3_bucket")
                    or ""
                )
            )
            if bare:
                env["NPA_S3_BUCKET"] = bare
        except Exception as exc:
            print(f"WARN: skipped BYOF bucket resolution: {exc}", file=sys.stderr)
    return env


def _refresh_registry_pull_secrets(image: str, project: str) -> None:
    if os.environ.get("NPA_BYOF_SKIP_REGISTRY_REFRESH") == "1":
        return
    registry_server = _registry_server(image)
    if "nebius.cloud" not in registry_server:
        return
    try:
        from npa.workflows.sim2real.registry_auth import ensure_nebius_registry_pull_secret

        target = resolve_byof_kubernetes_target(project or None)
        namespaces = {target.namespace or "default", "default"}
        for namespace in sorted(namespaces):
            for secret_name in ("agent-sa", "npa-nebius-registry"):
                ensure_nebius_registry_pull_secret(
                    registry_server=registry_server,
                    secret_name=secret_name,
                    namespace=namespace,
                    kubeconfig=target.kubeconfig,
                    k8s_context=target.context,
                )
    except Exception as exc:
        print(f"WARN: skipped registry pull-secret refresh: {exc}", file=sys.stderr)


def _registry_path(image_ref: str) -> str:
    ref = image_ref.removeprefix("docker:")
    without_digest = ref.split("@", 1)[0]
    last_slash = without_digest.rfind("/")
    if last_slash <= 0:
        return ""
    return without_digest[:last_slash]


def _docker_login_nebius(server: str, *, env: dict[str, str] | None = None) -> None:
    # Prefer an explicit write-capable profile when operators set one (e.g. agent-sa).
    merged = {**os.environ, **(env or {})}
    profile = (
        merged.get("NPA_NEBIUS_PROFILE", "").strip()
        or merged.get("NEBIUS_PROFILE", "").strip()
    )
    token_cmd = ["nebius"]
    if profile:
        token_cmd.extend(["--profile", profile])
    token_cmd.extend(["iam", "get-access-token"])
    token_proc = _run(token_cmd, capture=True)
    token = token_proc.stdout.strip()
    if not token:
        raise RuntimeError("nebius iam get-access-token returned empty token")
    _run(["docker", "login", "-u", "iam", "--password-stdin", server], stdin=token, env=env)


def _dockerfile_text() -> str:
    return (
        "ARG BYOF_BASE_IMAGE\n"
        "FROM ${BYOF_BASE_IMAGE}\n"
        "ARG OSS_REPO_URL\n"
        "ARG OSS_REPO_REF\n"
        "ARG BYOF_BUILD_COMMAND\n"
        "USER root\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
        "      git ca-certificates python3 python3-pip sudo \\\n"
        "  && rm -rf /var/lib/apt/lists/*\n"
        "RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu\n"
        "RUN echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \\\n"
        "  && chmod 440 /etc/sudoers.d/ubuntu\n"
        "RUN mkdir -p /workspace && chown ubuntu:ubuntu /workspace\n"
        f"RUN git clone --depth 1 --branch \"${{OSS_REPO_REF}}\" \"${{OSS_REPO_URL}}\" {BYOF_REPO_MOUNT} \\\n"
        f"  || (rm -rf {BYOF_REPO_MOUNT} \\\n"
        f"    && git clone \"${{OSS_REPO_URL}}\" {BYOF_REPO_MOUNT} \\\n"
        f"    && cd {BYOF_REPO_MOUNT} \\\n"
        "    && git checkout \"${OSS_REPO_REF}\") \\\n"
        f"  && chown -R ubuntu:ubuntu {BYOF_REPO_MOUNT}\n"
        f"WORKDIR {BYOF_REPO_MOUNT}\n"
        "RUN if [ -n \"${BYOF_BUILD_COMMAND}\" ]; then /bin/sh -lc \"${BYOF_BUILD_COMMAND}\"; fi\n"
        f"RUN printf '{{\\n  \"source\": \"oss-byof\",\\n  \"repo\": \"%s\",\\n  \"ref\": \"%s\"\\n}}\\n' \\\n"
        f"  \"${{OSS_REPO_URL}}\" \"${{OSS_REPO_REF}}\" > {BYOF_REPO_MOUNT}/npa_source_metadata.json \\\n"
        f"  && chown ubuntu:ubuntu {BYOF_REPO_MOUNT}/npa_source_metadata.json\n"
        "LABEL npa.byof.repo=\"${OSS_REPO_URL}\" npa.byof.ref=\"${OSS_REPO_REF}\" "
        "npa.packaging.tier=\"interactive\"\n"
        "USER ubuntu\n"
        f"WORKDIR {BYOF_REPO_MOUNT}\n"
    )


def _parse_last_json(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _ubuntu_base_image_candidates() -> list[str]:
    configured = os.environ.get("NPA_BYOF_UBUNTU_BASE_IMAGE", DEFAULT_UBUNTU_BASE_IMAGE).strip()
    return [configured or DEFAULT_UBUNTU_BASE_IMAGE]


def _isaac_lab_base_image_candidates(*, image: str, registry: str) -> list[str]:
    candidates: list[str] = []
    derived_registry = _registry_path(image) or registry
    try:
        canonical = str(container_image_for_tool("isaac-lab")).strip()
        if canonical:
            candidates.append(canonical)
    except TypeError:
        pass
    for candidate_registry in (registry, derived_registry):
        candidate = str(container_image_for_tool("isaac-lab", registry=candidate_registry)).strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for public_candidate in ("nvcr.io/nvidia/isaac-lab:2.3.2", "nvcr.io/nvidia/isaac-sim:4.5.0"):
        if public_candidate not in candidates:
            candidates.append(public_candidate)
    return candidates


def _base_image_candidates(
    *,
    profile: str,
    image: str,
    registry: str,
    explicit_base: str,
) -> list[str]:
    if explicit_base:
        return [explicit_base]
    normalized_profile = profile if profile in BASE_PROFILES else "ubuntu"
    if normalized_profile == "isaac-lab":
        return _isaac_lab_base_image_candidates(image=image, registry=registry)
    return _ubuntu_base_image_candidates()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--repo-ref", default=DEFAULT_REPO_REF)
    parser.add_argument("--project", default="", help="Project alias used for container-registry resolution.")
    parser.add_argument("--registry", default="", help="Override registry host/path.")
    parser.add_argument("--image", default="", help="Fully-qualified image ref to build/push and run.")
    parser.add_argument(
        "--base-profile",
        choices=sorted(BASE_PROFILES),
        default=os.environ.get("NPA_BYOF_BASE_PROFILE", "ubuntu"),
        help="Base image family: ubuntu (generic) or isaac-lab (sim workloads).",
    )
    parser.add_argument(
        "--base-image",
        default=os.environ.get("NPA_BYOF_BASE_IMAGE", ""),
        help="Explicit base image (overrides --base-profile), e.g. ubuntu:24.04.",
    )
    parser.add_argument("--run-id", default=f"byof-{_utc_stamp()}")
    parser.add_argument(
        "--workload",
        choices=("rl-train", "datagen", "container-verify", "solution-smoke"),
        default="rl-train",
        help="Live workload: RL training, scripted datagen, container-verify, or solution smoke.",
    )
    parser.add_argument(
        "--build-command",
        default=os.environ.get("NPA_BYOF_BUILD_COMMAND", ""),
        help="Optional shell command run during image build from /opt/byof.",
    )
    parser.add_argument(
        "--smoke-command",
        default=os.environ.get("NPA_BYOF_SMOKE_COMMAND", ""),
        help="Optional documented shell command run during solution-smoke from /opt/byof.",
    )
    parser.add_argument("--solution-name", default=os.environ.get("NPA_BYOF_SOLUTION_NAME", ""))
    parser.add_argument("--capability-name", default=os.environ.get("NPA_BYOF_CAPABILITY_NAME", ""))
    parser.add_argument("--smoke-artifact-name", default=os.environ.get("NPA_BYOF_SMOKE_ARTIFACT_NAME", ""))
    parser.add_argument("--num-envs", type=int, default=4, help="Parallel sim envs (datagen workload).")
    parser.add_argument("--num-demos", type=int, default=4, help="Demonstrations to record (datagen workload).")
    parser.add_argument("--task", default="Isaac-Cartpole-v0")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--yaml", default="", help="Optional SkyPilot YAML override for the selected workload.")
    parser.add_argument("--output-root", default="", help="Override workload output root.")
    parser.add_argument("--wait-timeout", type=int, default=21600)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--sky-bin", default="")
    parser.add_argument("--config-path", default="", help="SkyPilot global config YAML for kubernetes pod_config.")
    parser.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    explicit_base = _normalize_optional(args.base_image)
    base_profile = _normalize_optional(args.base_profile) or "ubuntu"
    registry = args.registry.strip() or resolve_container_registry(args.project or None)
    image = args.image.strip() or f"{registry.rstrip('/')}/npa-byof:{args.run_id}"
    base_candidates = _base_image_candidates(
        profile=base_profile,
        image=image,
        registry=registry,
        explicit_base=explicit_base,
    )
    if not base_candidates:
        raise RuntimeError("unable to resolve a BYOF base image candidate")
    base_image = base_candidates[0]
    base_registry = _registry_path(base_image) or (_registry_path(image) or registry)

    summary: dict[str, Any] = {
        "repo_url": args.repo_url,
        "repo_ref": args.repo_ref,
        "registry": registry,
        "base_profile": base_profile,
        "base_registry": base_registry,
        "image": image,
        "base_image": base_image,
        "base_image_candidates": base_candidates,
        "run_id": args.run_id,
        "workload": args.workload,
        "build_command": args.build_command,
        "smoke_command": args.smoke_command,
        "solution_name": args.solution_name,
        "capability_name": args.capability_name,
        "smoke_artifact_name": args.smoke_artifact_name,
    }

    docker_config_dir: str | None = None
    docker_env: dict[str, str] = {}
    try:
        if not args.skip_build:
            if not args.skip_push:
                docker_config_dir = tempfile.mkdtemp(prefix="npa-docker-auth-")
                docker_env = {"DOCKER_CONFIG": docker_config_dir}
                _docker_login_nebius(_registry_server(image), env=docker_env)
            with tempfile.TemporaryDirectory(prefix="npa-byof-build-") as tmp:
                context = Path(tmp)
                (context / "Dockerfile").write_text(_dockerfile_text(), encoding="utf-8")
                last_build_error: Exception | None = None
                for idx, candidate_base in enumerate(base_candidates):
                    base_image = candidate_base
                    summary["base_image"] = base_image
                    summary["base_registry"] = _registry_path(base_image) or (_registry_path(image) or registry)
                    try:
                        _run(
                            [
                                "docker",
                                "build",
                                "--platform",
                                "linux/amd64",
                                "--build-arg",
                                f"BYOF_BASE_IMAGE={base_image}",
                                "--build-arg",
                                f"OSS_REPO_URL={args.repo_url}",
                                "--build-arg",
                                f"OSS_REPO_REF={args.repo_ref}",
                                "--build-arg",
                                f"BYOF_BUILD_COMMAND={args.build_command}",
                                "-t",
                                image,
                                str(context),
                            ],
                            env=docker_env or None,
                            capture=True,
                        )
                        break
                    except Exception as exc:
                        last_build_error = exc
                        message = str(exc)
                        forbidden_pull = "403 Forbidden" in message and (
                            "BYOF_BASE_IMAGE" in message
                            or "ISAAC_BASE_IMAGE" in message
                            or "failed to resolve source metadata" in message
                            or "pull access denied" in message
                        )
                        if forbidden_pull and idx + 1 < len(base_candidates):
                            continue
                        raise
                else:
                    assert last_build_error is not None
                    raise last_build_error
            if not args.skip_push:
                push_proc = _run(["docker", "push", image], env=docker_env or None, capture=True)
                if push_proc.stdout:
                    sys.stdout.write(push_proc.stdout)
                if push_proc.stderr:
                    sys.stderr.write(push_proc.stderr)
                try:
                    _run(["docker", "buildx", "imagetools", "inspect", image], env=docker_env or None)
                except Exception:
                    pass
            summary["build"] = {"ok": True, "pushed": not args.skip_push}
        else:
            summary["build"] = {"ok": True, "skipped": True}

        if not args.skip_run:
            if args.workload == "datagen":
                cmd = [
                    sys.executable,
                    str(DATAGEN_RUNNER),
                    "--image",
                    image,
                    "--task",
                    args.task,
                    "--num-envs",
                    str(args.num_envs),
                    "--num-demos",
                    str(args.num_demos),
                    "--run-id",
                    args.run_id,
                    "--wait-timeout",
                    str(args.wait_timeout),
                    "--poll-interval",
                    str(args.poll_interval),
                    "--repo-root",
                    BYOF_REPO_MOUNT,
                ]
            elif args.workload in {"container-verify", "solution-smoke"}:
                cmd = [
                    sys.executable,
                    str(CONTAINER_VERIFY_RUNNER),
                    "--image",
                    image,
                    "--run-id",
                    args.run_id,
                    "--wait-timeout",
                    str(min(args.wait_timeout, 3600)),
                    "--poll-interval",
                    str(args.poll_interval),
                    "--repo-root",
                    BYOF_REPO_MOUNT,
                ]
                if args.smoke_command:
                    cmd.extend(["--smoke-command", args.smoke_command])
                if args.solution_name:
                    cmd.extend(["--solution-name", args.solution_name])
                if args.capability_name:
                    cmd.extend(["--capability-name", args.capability_name])
                if args.smoke_artifact_name:
                    cmd.extend(["--smoke-artifact-name", args.smoke_artifact_name])
            else:
                cmd = [
                    sys.executable,
                    str(ISAAC_RUNNER),
                    "--image",
                    image,
                    "--task",
                    args.task,
                    "--iterations",
                    str(args.iterations),
                    "--run-id",
                    args.run_id,
                    "--wait-timeout",
                    str(args.wait_timeout),
                    "--poll-interval",
                    str(args.poll_interval),
                ]
            if args.yaml:
                cmd.extend(["--yaml", args.yaml])
            if args.output_root:
                cmd.extend(["--output-root", args.output_root])
            if args.sky_bin:
                cmd.extend(["--sky-bin", args.sky_bin])
            if args.config_path:
                cmd.extend(["--config-path", args.config_path])
            if args.cleanup:
                cmd.append("--cleanup")
            if args.workload in {"container-verify", "solution-smoke"}:
                _refresh_registry_pull_secrets(image, args.project)
            run_proc = _run(cmd, capture=True, env=_live_runner_env(args.project))
            sys.stdout.write(run_proc.stdout)
            if run_proc.stderr:
                sys.stderr.write(run_proc.stderr)
            summary["run"] = _parse_last_json(run_proc.stdout) or {"status": "submitted"}
        else:
            summary["run"] = {"skipped": True}
        summary["status"] = "ok"
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        message = str(exc)
        summary["status"] = "failed"
        summary["error"] = message
        if "403 Forbidden" in message and ("BYOF_BASE_IMAGE" in message or "ISAAC_BASE_IMAGE" in message):
            summary["hint"] = (
                "Registry pull for the base image was denied. "
                "Pass --base-image from an accessible registry (e.g. ubuntu:22.04), "
                "or use --base-profile isaac-lab with registry access to the sim image."
            )
        elif "docker push" in message and "403 Forbidden" in message:
            summary["hint"] = (
                "Registry push was denied for the target image. "
                "Grant write access to the target repository, or use --skip-push "
                "with an already-published image."
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        hint = str(summary.get("hint") or "").strip()
        if hint:
            print(f"HINT: {hint}", file=sys.stderr)
        return 1
    finally:
        if docker_config_dir:
            shutil.rmtree(docker_config_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
