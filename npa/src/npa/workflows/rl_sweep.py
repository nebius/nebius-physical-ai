"""Real stage implementations for the Isaac Lab RL sweep blueprint.

These back the ``run.shell`` stages of
``npa/workflows/workbench/npa-workflows/isaac-lab-rl-sweep.yaml`` — the
``npa.workflow`` port of the one ``execution: parallel`` SkyPilot template, which is
now retired (the spec is live-verified on four GPUs; see ``EVIDENCE.md`` §R3):

- :func:`train_variant` runs **one** RSL-RL training variant in-pod (the Isaac Lab
  image's ``train.py`` with Hydra overrides), then uploads the checkpoint, the
  training log and a metrics JSON for that variant.
- :func:`select_best` is the **barrier** stage: it reads every variant's metrics
  from the sweep prefix and writes the ranked best-variant report.

Why ``run.shell`` and not a ``toolRef``: ``npa workbench isaac-lab train`` is a
*launcher* (it provisions a VM / serverless job), so calling it from inside a
SkyPilot task would nest infrastructure. The in-pod contract is the upstream
training script, exactly like the SkyPilot template it replaces. Logic lives here
(unit-tested) instead of being inlined into YAML.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

DEFAULT_TRAIN_SCRIPT = (
    "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py"
)
DEFAULT_PYTHON = "/isaac-sim/python.sh"
METRICS_FILENAME = "npa_rl_sweep_metrics.json"
SUMMARY_FILENAME = "npa_rl_sweep_summary.json"
REPORT_FILENAME = "npa_rl_sweep_best.json"


# --------------------------------------------------------------------- storage


def _storage():
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment()


def _split(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _list_keys(uri: str) -> list[str]:
    bucket, prefix = _split(uri if uri.endswith("/") else uri + "/")
    s3 = _storage().s3
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        keys.extend(item["Key"] for item in page.get("Contents", []) if item.get("Key"))
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return keys


def _upload_json(payload: dict[str, Any], uri: str) -> str:
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if not uri.startswith("s3://"):
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        Path(uri).write_text(body, encoding="utf-8")
        return uri
    with tempfile.TemporaryDirectory(prefix="npa-rl-sweep-") as tmp:
        path = Path(tmp) / "out.json"
        path.write_text(body, encoding="utf-8")
        return _storage().upload_file(str(path), uri)


def _download_json(uri: str) -> dict[str, Any]:
    if not uri.startswith("s3://"):
        return json.loads(Path(uri).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="npa-rl-sweep-") as tmp:
        local = _storage().download_path(uri, tmp)
        path = Path(local)
        if path.is_dir():
            candidates = sorted(path.rglob(uri.rstrip("/").split("/")[-1]))
            if not candidates:
                raise FileNotFoundError(uri)
            path = candidates[0]
        return json.loads(path.read_text(encoding="utf-8"))


def _upload_file(local: Path, uri: str) -> str:
    if not uri.startswith("s3://"):
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        Path(uri).write_bytes(local.read_bytes())
        return uri
    return _storage().upload_file(str(local), uri)


# ---------------------------------------------------------------------- train


def resolve_python_bin(candidates: Sequence[str] = ()) -> str:
    """Pick the interpreter that can import Isaac Lab (image-dependent)."""

    import shutil

    for candidate in [
        *candidates,
        os.environ.get("ISAAC_LAB_PYTHON", ""),
        DEFAULT_PYTHON,
    ]:
        if candidate and Path(candidate).exists() and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("python3") or shutil.which("python") or "python3"


def parse_overrides(overrides: str | Iterable[str]) -> list[str]:
    """Split a Hydra override string (``a=1 b=2``) into argv items."""

    if isinstance(overrides, str):
        return shlex.split(overrides)
    return [str(item) for item in overrides]


def _parse_reward(log_text: str) -> float | None:
    """Best-effort extraction of the last mean reward printed by RSL-RL."""

    import re

    matches = re.findall(r"Mean reward:\s*(-?\d+(?:\.\d+)?)", log_text)
    if not matches:
        matches = re.findall(r"mean_reward[\"']?\s*[:=]\s*(-?\d+(?:\.\d+)?)", log_text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def ensure_training_entrypoint(
    interpreter: str,
    script: str,
    *,
    executor: Any = subprocess.run,
) -> None:
    """Trigger the cold runtime fetch before requiring the source entrypoint."""

    if Path(script).is_file():
        return
    result = executor(
        [interpreter, "-c", "pass"],
        capture_output=True,
        text=True,
        check=False,
    )
    if int(getattr(result, "returncode", 1)) != 0:
        stderr = str(getattr(result, "stderr", "") or "")[-1000:]
        raise RuntimeError(f"Isaac runtime bootstrap failed: {stderr}")
    if not Path(script).is_file():
        raise FileNotFoundError(
            f"pinned upstream Isaac Lab training entrypoint not found after bootstrap: {script}"
        )


def train_variant(
    *,
    variant: str,
    output_uri: str,
    task: str = "Isaac-Cartpole-v0",
    iterations: str | int = 10,
    num_envs: str | int = 64,
    overrides: str = "",
    run_id: str = "",
    train_script: str = "",
    python_bin: str = "",
    runner: Any | None = None,
) -> dict[str, Any]:
    """Train one sweep variant in-pod and publish its artifacts + metrics.

    ``runner`` is injected in unit tests; by default the upstream Isaac Lab
    training script is executed with the variant's Hydra overrides.
    """

    script = (
        train_script
        or os.environ.get("ISAAC_LAB_TRAIN_SCRIPT", "")
        or DEFAULT_TRAIN_SCRIPT
    )
    interpreter = python_bin or resolve_python_bin()
    if runner is None:
        ensure_training_entrypoint(interpreter, script)
    argv = [
        interpreter,
        script,
        "--task",
        str(task),
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(iterations),
        "--visualizer",
        "none",
        "--experiment_name",
        "npa_rl_sweep",
        "--run_name",
        f"{run_id or 'run'}-{variant}",
        *parse_overrides(overrides),
    ]
    started = time.time()
    execute = runner or _default_runner
    result = execute(argv)
    duration = round(time.time() - started, 3)

    log_text = str(getattr(result, "stdout", "") or "") + str(
        getattr(result, "stderr", "") or ""
    )
    returncode = int(getattr(result, "returncode", 0) or 0)
    status = "success" if returncode == 0 else "failed"

    metrics = {
        "schema": "npa.rl_sweep.variant_metrics.v1",
        "status": status,
        "tool": "isaac_lab",
        "stage": "sweep-train",
        "variant": variant,
        "run_id": run_id,
        "task": str(task),
        "num_envs": int(num_envs),
        "max_iterations": int(iterations),
        "hydra_overrides": overrides,
        "returncode": returncode,
        "duration_seconds": duration,
        "mean_reward": _parse_reward(log_text),
        "isaac_lab_version": os.environ.get("ISAAC_LAB_VERSION", ""),
        "isaac_sim_version": os.environ.get("ISAAC_SIM_VERSION", ""),
        "source_commit": os.environ.get("NPA_ISAAC_LAB_SRC_COMMIT", ""),
        "image_source_sha": os.environ.get("NPA_IMAGE_SOURCE_SHA", ""),
    }

    base = output_uri.rstrip("/")
    with tempfile.TemporaryDirectory(prefix="npa-rl-sweep-log-") as tmp:
        log_path = Path(tmp) / "train.log"
        log_path.write_text(log_text[-1_000_000:], encoding="utf-8")
        metrics["log_uri"] = _upload_file(log_path, f"{base}/train.log")
    checkpoint = _publish_checkpoint(base)
    if checkpoint:
        metrics["checkpoint_uri"] = checkpoint
    metrics["metrics_uri"] = _upload_json(metrics, f"{base}/{METRICS_FILENAME}")
    _upload_json(metrics, f"{base}/{SUMMARY_FILENAME}")

    print(json.dumps(metrics))
    if returncode != 0:
        raise RuntimeError(f"variant {variant} training failed with exit {returncode}")
    return metrics


def _default_runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(shlex.quote(part) for part in argv), flush=True)
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def _publish_checkpoint(base_uri: str, search_root: str = "logs/rsl_rl") -> str:
    """Upload the newest RSL-RL checkpoint produced in the working directory."""

    root = Path(search_root)
    if not root.exists():
        return ""
    checkpoints = sorted(root.rglob("model_*.pt"))
    if not checkpoints:
        return ""
    return _upload_file(checkpoints[-1], f"{base_uri}/checkpoint.pt")


# --------------------------------------------------------------------- barrier


def select_best(
    *,
    sweep_uri: str,
    report_uri: str,
    metric: str = "mean_reward",
    run_id: str = "",
    expected_variants: str | Iterable[str] = (),
) -> dict[str, Any]:
    """Barrier stage: rank every variant's metrics and publish the winner.

    Runs only after **all** fan-out members finished (the group's ``needs:``
    barrier), so a partial sweep can never silently produce a "best" variant.
    """

    keys = [key for key in _list_keys(sweep_uri) if key.endswith(METRICS_FILENAME)]
    bucket, _ = _split(sweep_uri if sweep_uri.endswith("/") else sweep_uri + "/")
    variants: list[dict[str, Any]] = []
    for key in sorted(keys):
        try:
            variants.append(_download_json(f"s3://{bucket}/{key}"))
        except Exception as exc:  # noqa: BLE001 - one unreadable variant must not hide the rest
            variants.append(
                {"variant": key, "status": "unreadable", "error": str(exc)[:200]}
            )

    expected = set(parse_overrides(expected_variants))
    observed = {str(item.get("variant") or "") for item in variants}
    successful = [item for item in variants if item.get("status") == "success"]
    scored = [item for item in successful if isinstance(item.get(metric), (int, float))]
    best = max(scored, key=lambda item: float(item[metric])) if scored else None
    errors: list[str] = []
    if expected and observed != expected:
        errors.append(
            f"variant set mismatch: expected={sorted(expected)!r} observed={sorted(observed)!r}"
        )
    if any(item.get("status") != "success" for item in variants):
        errors.append("one or more variants did not report success")
    if len(scored) != len(variants):
        errors.append(f"one or more variants omitted numeric {metric}")
    if any(not item.get("checkpoint_uri") for item in variants):
        errors.append("one or more variants omitted checkpoint_uri")
    report = {
        "schema": "npa.rl_sweep.report.v1",
        "run_id": run_id,
        "sweep_uri": sweep_uri,
        "metric": metric,
        "variant_count": len(variants),
        "status": "success" if variants and not errors else "failed",
        "succeeded": len(successful),
        "expected_variants": sorted(expected),
        "errors": errors,
        "variants": sorted(variants, key=lambda item: str(item.get("variant") or "")),
        "best_variant": (best or {}).get("variant", ""),
        "best_value": (best or {}).get(metric),
    }
    report["report_uri"] = _upload_json(
        report,
        report_uri
        if report_uri.endswith(".json")
        else f"{report_uri.rstrip('/')}/{REPORT_FILENAME}",
    )
    print(json.dumps(report))
    if report["status"] != "success":
        raise RuntimeError(
            "incomplete Isaac Lab sweep: " + "; ".join(errors or ["no variants"])
        )
    return report


__all__ = [
    "METRICS_FILENAME",
    "REPORT_FILENAME",
    "SUMMARY_FILENAME",
    "ensure_training_entrypoint",
    "parse_overrides",
    "resolve_python_bin",
    "select_best",
    "train_variant",
]
