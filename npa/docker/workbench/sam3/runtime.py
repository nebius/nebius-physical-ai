#!/usr/bin/env python3
"""Fetch pinned SAM 3.1 into the operator cache after exact checkpoint access."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "pins.json").read_text())
TOKEN_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _token() -> str:
    token = next((os.environ[key] for key in TOKEN_KEYS if os.environ.get(key)), "")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required. Request access at https://huggingface.co/facebook/sam3.1 "
            "and grant token read access at https://huggingface.co/settings/tokens."
        )
    return token


def _access(token: str) -> None:
    import requests

    url = (
        f"https://huggingface.co/{PINS['model']}/resolve/"
        f"{PINS['model_revision']}/{PINS['checkpoint']}"
    )
    try:
        with requests.get(
            url,
            headers={"Authorization": f"Bearer {token}", "Range": "bytes=0-0"},
            stream=True,
            timeout=30,
        ) as response:
            ready = response.status_code in (200, 206)
            if not ready or not next(response.iter_content(1), b""):
                raise RuntimeError(
                    f"SAM 3.1 checkpoint access denied or unavailable (HTTP {response.status_code}). "
                    "Request access at https://huggingface.co/facebook/sam3.1."
                )
    except requests.RequestException:
        raise RuntimeError("SAM 3.1 checkpoint access could not be verified.") from None


def _child_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in TOKEN_KEYS}
    env.update(GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1", UV_NO_CACHE="1")
    return env


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, env=_child_env())


def _fetch_source(source: Path) -> None:
    _run(["git", "init", "--quiet", str(source)])
    _run(["git", "-C", str(source), "remote", "add", "origin", PINS["source_url"]])
    _run(
        [
            "git",
            "-C",
            str(source),
            "fetch",
            "--depth=1",
            "origin",
            PINS["source_revision"],
        ]
    )
    _run(["git", "-C", str(source), "checkout", "--quiet", "--detach", "FETCH_HEAD"])
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, env=_child_env()
    ).strip()
    if actual != PINS["source_revision"]:
        raise RuntimeError("Fetched SAM source revision does not match the pin")


def _install_packages(destination: Path) -> None:
    python = str(destination / "venv/bin/python")
    _run(
        [
            "uv",
            "venv",
            "--python",
            sys.executable,
            "--relocatable",
            str(destination / "venv"),
        ]
    )
    _run(
        [
            "uv",
            "pip",
            "sync",
            "--python",
            python,
            "--require-hashes",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu128",
            "--index-strategy",
            "unsafe-best-match",
            str(HERE / "requirements.lock"),
        ]
    )
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--no-deps",
            "--no-build-isolation",
            str(destination / "source"),
        ]
    )


def _install(destination: Path) -> None:
    _fetch_source(destination / "source")
    _install_packages(destination)
    (destination / "receipt.json").write_text(
        json.dumps(
            {
                **PINS,
                "runtime_lock_sha256": _sha256(HERE / "requirements.lock"),
            },
            indent=2,
        )
        + "\n"
    )


def _runtime(cache: Path) -> Path:
    identity = hashlib.sha256(
        (HERE / "pins.json").read_bytes() + (HERE / "requirements.lock").read_bytes()
    ).hexdigest()
    runtime = cache / identity
    if (runtime / "receipt.json").is_file():
        return runtime
    pending = Path(tempfile.mkdtemp(prefix=".install-", dir=cache))
    try:
        _install(pending)
        pending.rename(runtime)
    finally:
        if pending.exists():
            shutil.rmtree(pending)
    return runtime


def _checkpoint(cache: Path, token: str) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    try:
        path = Path(
            hf_hub_download(
                repo_id=PINS["model"],
                filename=PINS["checkpoint"],
                revision=PINS["model_revision"],
                token=token,
                cache_dir=cache / "weights",
            )
        )
    except HfHubHTTPError:
        raise RuntimeError(
            "SAM 3.1 checkpoint download failed; verify model and token access."
        ) from None
    if _sha256(path) != PINS["checkpoint_sha256"]:
        raise RuntimeError("SAM 3.1 checkpoint SHA-256 does not match the upstream pin")
    return path


def _ensure() -> tuple[Path, Path]:
    token = _token()
    _access(token)
    cache = Path(os.environ.get("NPA_SAM3_CACHE", "/workspace/.cache/npa/sam3"))
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        checkpoint = _checkpoint(cache, token)
        runtime = _runtime(cache)
    return runtime, checkpoint


def _refusal() -> None:
    with tempfile.TemporaryDirectory() as directory:
        env = _child_env()
        cache = Path(directory) / "cache"
        env["NPA_SAM3_CACHE"] = str(cache)
        result = subprocess.run(
            [sys.executable, str(Path(__file__)), "ensure"],
            env=env,
            capture_output=True,
            text=True,
        )
        if (
            result.returncode != 1
            or "HF_TOKEN is required" not in result.stderr
            or cache.exists()
        ):
            raise RuntimeError(
                "Missing-token refusal did not leave the runtime cache empty"
            )
    print("NPA_SAM3_BOOTSTRAP_REFUSES_WITHOUT_TOKEN_OK")


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "health",
            "version",
            "access",
            "ensure",
            "segment",
            "exec",
            "assert-refusal",
        ],
    )
    args, remainder = parser.parse_known_args()
    if args.mode in {"health", "version"}:
        print(json.dumps({"status": "bootstrap-only", **PINS}))
        return 0
    if args.mode == "assert-refusal":
        _refusal()
        return 0
    if args.mode == "access":
        _access(_token())
        print(
            json.dumps({"status": "Ready", "scope": "exact checkpoint access", **PINS})
        )
        return 0
    runtime, checkpoint = _ensure()
    if args.mode == "ensure":
        print(json.dumps({"runtime": str(runtime), "checkpoint": str(checkpoint)}))
        return 0
    env = _child_env()
    env.update(HF_HUB_OFFLINE="1", SAM3_CHECKPOINT=str(checkpoint))
    command = [str(runtime / "venv/bin/python")]
    if args.mode == "segment":
        command.append(str(HERE / "segment.py"))
    os.execve(command[0], command + remainder, env)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
