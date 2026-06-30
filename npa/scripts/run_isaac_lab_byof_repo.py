#!/usr/bin/env python3
"""Build/push a BYOF Isaac Lab image from an OSS repo and launch a live run."""

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
from npa.deploy.images import container_image_for_tool

SCRIPT_DIR = Path(__file__).resolve().parent
ISAAC_RUNNER = SCRIPT_DIR / "run_isaac_lab_rl.py"

DEFAULT_REPO_URL = "https://github.com/LightwheelAI/leisaac.git"
DEFAULT_REPO_REF = "main"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _registry_path(image_ref: str) -> str:
    ref = image_ref.removeprefix("docker:")
    without_digest = ref.split("@", 1)[0]
    last_slash = without_digest.rfind("/")
    if last_slash <= 0:
        return ""
    return without_digest[:last_slash]


def _docker_login_nebius(server: str, *, env: dict[str, str] | None = None) -> None:
    token_proc = _run(["nebius", "iam", "get-access-token"], capture=True)
    token = token_proc.stdout.strip()
    if not token:
        raise RuntimeError("nebius iam get-access-token returned empty token")
    _run(["docker", "login", "-u", "iam", "--password-stdin", server], stdin=token, env=env)


def _dockerfile_text() -> str:
    return (
        "ARG ISAAC_BASE_IMAGE\n"
        "FROM ${ISAAC_BASE_IMAGE}\n"
        "ARG OSS_REPO_URL\n"
        "ARG OSS_REPO_REF\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \\\n"
        "  && rm -rf /var/lib/apt/lists/*\n"
        "RUN git clone --depth 1 --branch \"${OSS_REPO_REF}\" \"${OSS_REPO_URL}\" /opt/leisaac\n"
        "RUN printf '{\\n  \"source\": \"oss-byof\",\\n  \"repo\": \"%s\",\\n  \"ref\": \"%s\"\\n}\\n' \\\n"
        "  \"${OSS_REPO_URL}\" \"${OSS_REPO_REF}\" > /opt/leisaac/npa_source_metadata.json\n"
        "LABEL npa.byof.repo=\"${OSS_REPO_URL}\" npa.byof.ref=\"${OSS_REPO_REF}\"\n"
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


def _base_image_candidates(*, image: str, registry: str, explicit_base: str) -> list[str]:
    if explicit_base.strip():
        return [explicit_base.strip()]
    candidates: list[str] = []
    derived_registry = _registry_path(image) or registry
    # Try canonical resolver first (may point at a public/default registry).
    try:
        canonical = str(container_image_for_tool("isaac-lab")).strip()
        if canonical:
            candidates.append(canonical)
    except TypeError:
        # Older call paths in tests may require explicit registry.
        pass
    # Then try explicit registry choices used by this run.
    for candidate_registry in (registry, derived_registry):
        candidate = str(container_image_for_tool("isaac-lab", registry=candidate_registry)).strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    # Public fallbacks keep BYOF unblocked when private base repos are inaccessible.
    for public_candidate in ("nvcr.io/nvidia/isaac-lab:2.3.2", "nvcr.io/nvidia/isaac-sim:4.5.0"):
        if public_candidate not in candidates:
            candidates.append(public_candidate)
    return candidates


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--repo-ref", default=DEFAULT_REPO_REF)
    parser.add_argument("--project", default="", help="Project alias used for container-registry resolution.")
    parser.add_argument("--registry", default="", help="Override registry host/path.")
    parser.add_argument("--image", default="", help="Fully-qualified image ref to build/push and run.")
    parser.add_argument("--base-image", default="", help="Override base Isaac Lab image.")
    parser.add_argument("--run-id", default=f"leisaac-byof-{_utc_stamp()}")
    parser.add_argument("--task", default="Isaac-Cartpole-v0")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--yaml", default="", help="Optional SkyPilot YAML override for run_isaac_lab_rl.py.")
    parser.add_argument("--output-root", default="", help="Override run_isaac_lab_rl.py output root.")
    parser.add_argument("--wait-timeout", type=int, default=21600)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--sky-bin", default="")
    parser.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    registry = args.registry.strip() or resolve_container_registry(args.project or None)
    image = args.image.strip() or f"{registry.rstrip('/')}/npa-isaac-lab-leisaac:{args.run_id}"
    base_candidates = _base_image_candidates(image=image, registry=registry, explicit_base=args.base_image)
    if not base_candidates:
        raise RuntimeError("unable to resolve an Isaac Lab base image candidate")
    base_image = base_candidates[0]
    base_registry = _registry_path(base_image) or (_registry_path(image) or registry)

    summary: dict[str, Any] = {
        "repo_url": args.repo_url,
        "repo_ref": args.repo_ref,
        "registry": registry,
        "base_registry": base_registry,
        "image": image,
        "base_image": base_image,
        "base_image_candidates": base_candidates,
        "run_id": args.run_id,
    }

    docker_config_dir: str | None = None
    docker_env: dict[str, str] = {}
    try:
        if not args.skip_build:
            if not args.skip_push:
                docker_config_dir = tempfile.mkdtemp(prefix="npa-docker-auth-")
                docker_env = {"DOCKER_CONFIG": docker_config_dir}
                _docker_login_nebius(_registry_server(image), env=docker_env)
            with tempfile.TemporaryDirectory(prefix="npa-byof-isaac-") as tmp:
                context = Path(tmp)
                (context / "Dockerfile").write_text(_dockerfile_text(), encoding="utf-8")
                last_build_error: Exception | None = None
                for idx, candidate_base in enumerate(base_candidates):
                    base_image = candidate_base
                    summary["base_image"] = base_image
                    summary["base_registry"] = _registry_path(base_image) or (_registry_path(image) or registry)
                    try:
                        _run(
                            # Capture build output so 403 errors can trigger fallback candidates.
                            [
                                "docker",
                                "build",
                                "--platform",
                                "linux/amd64",
                                "--build-arg",
                                f"ISAAC_BASE_IMAGE={base_image}",
                                "--build-arg",
                                f"OSS_REPO_URL={args.repo_url}",
                                "--build-arg",
                                f"OSS_REPO_REF={args.repo_ref}",
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
                            "ISAAC_BASE_IMAGE" in message
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
                _run(["docker", "push", image], env=docker_env or None)
                try:
                    _run(["docker", "buildx", "imagetools", "inspect", image], env=docker_env or None)
                except Exception:
                    # Optional inspect path; image was already pushed.
                    pass
            summary["build"] = {"ok": True, "pushed": not args.skip_push}
        else:
            summary["build"] = {"ok": True, "skipped": True}

        if not args.skip_run:
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
            if args.cleanup:
                cmd.append("--cleanup")
            run_proc = _run(cmd, capture=True)
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
        if "403 Forbidden" in message and "ISAAC_BASE_IMAGE" in message:
            summary["hint"] = (
                "Registry pull for the base Isaac image was denied. "
                "Pass --base-image from an accessible registry, or grant pull access "
                "with an accessible parent image."
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1
    finally:
        if docker_config_dir:
            shutil.rmtree(docker_config_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
