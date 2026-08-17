"""Decision-quality Cosmos 3 still-image checkpoint evaluation.

The evaluator deliberately reuses the runtime baked into ``npa-cosmos3``.  It
loads one checkpoint per subprocess, supplies all prompts as one JSONL batch so
the model is loaded once, keeps guardrails enabled, and publishes each completed
arm before moving to the next checkpoint.  Model and guardrail weights remain
runtime downloads in the operator's Hugging Face cache; only generated media and
small provenance manifests are uploaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from npa.clients.storage import StorageClient
from npa.workbench.cosmos.generate import (
    DEFAULT_PARALLELISM_PRESET,
    _artifact_for,
    _venv_python,
    build_cosmos3_inference_args,
    build_generate_spec,
    check_xet_pin,
    cosmos3_generate_available,
    cosmos3_repo,
    require_model_access,
    resolve_hf_token,
)

CAMPAIGN_CONFIG_SCHEMA = "npa.cosmos3.checkpoint-eval.config.v1"
ARM_SCHEMA = "npa.cosmos3.checkpoint-eval.arm.v1"
PHASE_SCHEMA = "npa.cosmos3.checkpoint-eval.phase.v1"
SUPPORTED_PHASES = ("primary", "consistency")
SUPPORTED_CHECKPOINTS = (
    "Cosmos3-Edge",
    "Cosmos3-Nano",
    "Cosmos3-Super",
    "Cosmos3-Super-Text2Image",
    "Cosmos3-Super-Text2Image-4Step",
)
DEFAULT_WORK_DIR = "/tmp/npa-cosmos3-checkpoint-eval"
_SAFE_ID = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

# The pinned upstream framework enables the prompt blocklist and Qwen3Guard,
# and applies RetinaFace postprocessing to generated media.  Its generated-
# media content-safety list is intentionally empty at the pinned commit because
# the upstream filter is commented out as producing too many false positives.
# Keep this distinction in every arm manifest: a bare ``guardrails_enabled``
# flag would otherwise imply output-content filtering that did not occur.
PINNED_GUARDRAIL_POSTURE: dict[str, Any] = {
    "framework_commit": "5e67049cd94acb667786f1e6dd0dab821cb90c97",
    "prompt_input": {
        "enabled": True,
        "safety_models": ["Blocklist", "Qwen3Guard"],
        "blocked_prompt_behavior": "fail-arm",
    },
    "generated_media": {
        "content_safety_models": [],
        "content_safety_behavior": "fail-open-no-models",
        "upstream_reason": "video content filter disabled because of excessive false positives",
        "postprocessors": ["RetinaFaceFilter"],
    },
}


class Cosmos3CheckpointEvalError(RuntimeError):
    """Raised when a campaign contract or live evaluation arm fails."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _s3_join(prefix: str, *parts: object) -> str:
    if not str(prefix).startswith("s3://"):
        raise Cosmos3CheckpointEvalError(
            f"checkpoint evaluation output must be an s3:// prefix, got {prefix!r}"
        )
    return "/".join(
        [str(prefix).rstrip("/"), *(str(part).strip("/") for part in parts)]
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_slug(checkpoint: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", checkpoint.lower()).strip("-")


def _load_config_path(
    source: str,
    *,
    work_dir: Path,
    storage_client: Any = None,
) -> tuple[dict[str, Any], Path]:
    value = str(source or "").strip()
    if not value:
        raise Cosmos3CheckpointEvalError("--campaign-config is required")
    if value.startswith("s3://"):
        local = work_dir / "campaign-config.json"
        local.parent.mkdir(parents=True, exist_ok=True)
        client = storage_client or StorageClient.from_environment()
        client.download_file(value, str(local))
    else:
        local = Path(value).expanduser().resolve()
    if not local.is_file():
        raise Cosmos3CheckpointEvalError(f"campaign config does not exist: {local}")
    try:
        payload = json.loads(local.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Cosmos3CheckpointEvalError(f"invalid campaign config {local}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Cosmos3CheckpointEvalError("campaign config must be a JSON object")
    return payload, local


def validate_campaign_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the public evaluation contract."""

    config = dict(payload)
    if config.get("schema") != CAMPAIGN_CONFIG_SCHEMA:
        raise Cosmos3CheckpointEvalError(
            f"campaign schema must be {CAMPAIGN_CONFIG_SCHEMA!r}"
        )
    if config.get("mode") != "text2image":
        raise Cosmos3CheckpointEvalError("checkpoint evaluation supports text2image only")
    if config.get("guardrails_enabled") is not True:
        raise Cosmos3CheckpointEvalError(
            "campaign config must keep guardrails enabled; there is no silent opt-out"
        )

    checkpoints = config.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise Cosmos3CheckpointEvalError("campaign checkpoints must be a non-empty list")
    names: list[str] = []
    model_ids: dict[str, str] = {}
    for entry in checkpoints:
        if not isinstance(entry, dict):
            raise Cosmos3CheckpointEvalError("each checkpoint entry must be an object")
        name = str(entry.get("name") or "").strip()
        model_id = str(entry.get("model_id") or "").strip()
        if name not in SUPPORTED_CHECKPOINTS:
            raise Cosmos3CheckpointEvalError(f"unsupported checkpoint in campaign: {name!r}")
        if model_id != f"nvidia/{name}":
            raise Cosmos3CheckpointEvalError(
                f"checkpoint {name} must declare model_id nvidia/{name}"
            )
        if name in names:
            raise Cosmos3CheckpointEvalError(f"duplicate checkpoint: {name}")
        names.append(name)
        model_ids[name] = model_id
    config["checkpoint_names"] = names
    config["checkpoint_model_ids"] = model_ids

    prompts = config.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise Cosmos3CheckpointEvalError("campaign prompts must be a non-empty list")
    prompt_ids: set[str] = set()
    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise Cosmos3CheckpointEvalError("each prompt entry must be an object")
        prompt_id = str(prompt.get("id") or "").strip()
        text = str(prompt.get("text") or "").strip()
        if not _SAFE_ID.fullmatch(prompt_id):
            raise Cosmos3CheckpointEvalError(f"invalid prompt id: {prompt_id!r}")
        if prompt_id in prompt_ids:
            raise Cosmos3CheckpointEvalError(f"duplicate prompt id: {prompt_id}")
        if not text:
            raise Cosmos3CheckpointEvalError(f"prompt {prompt_id} is empty")
        prompt_ids.add(prompt_id)

    primary_seed = config.get("primary_seed")
    additional = config.get("additional_seeds")
    if not isinstance(primary_seed, int) or primary_seed < 0:
        raise Cosmos3CheckpointEvalError("primary_seed must be a non-negative integer")
    if (
        not isinstance(additional, list)
        or len(additional) != 2
        or any(not isinstance(seed, int) or seed < 0 for seed in additional)
    ):
        raise Cosmos3CheckpointEvalError(
            "additional_seeds must contain exactly two non-negative integers"
        )
    if len({primary_seed, *additional}) != 3:
        raise Cosmos3CheckpointEvalError("primary and additional seeds must be distinct")

    expected_ref = str(config.get("framework_commit") or "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", expected_ref):
        raise Cosmos3CheckpointEvalError("framework_commit must be an exact 40-character SHA")
    pinned_ref = str(PINNED_GUARDRAIL_POSTURE["framework_commit"])
    if expected_ref != pinned_ref:
        raise Cosmos3CheckpointEvalError(
            "framework_commit does not match the audited guardrail posture"
        )
    runtime_digest = str(config.get("runtime_image_digest") or "").strip()
    if not _SHA256_DIGEST.fullmatch(runtime_digest):
        raise Cosmos3CheckpointEvalError(
            "runtime_image_digest must be an exact sha256 content digest"
        )
    return config


def phase_arms(
    config: Mapping[str, Any],
    *,
    phase: str,
    top_checkpoints: Sequence[str] = (),
) -> list[tuple[str, int]]:
    """Resolve work without ever repeating the primary seed in consistency."""

    if phase not in SUPPORTED_PHASES:
        raise Cosmos3CheckpointEvalError(
            f"phase must be one of {', '.join(SUPPORTED_PHASES)}"
        )
    checkpoint_names = list(config["checkpoint_names"])
    if phase == "primary":
        if any(str(value).strip() for value in top_checkpoints):
            raise Cosmos3CheckpointEvalError(
                "--top-checkpoint is valid only for the consistency phase"
            )
        return [(name, int(config["primary_seed"])) for name in checkpoint_names]

    selected = [str(value).strip() for value in top_checkpoints if str(value).strip()]
    if len(selected) != 2 or len(set(selected)) != 2:
        raise Cosmos3CheckpointEvalError(
            "consistency phase requires exactly two distinct --top-checkpoint values"
        )
    unknown = sorted(set(selected) - set(checkpoint_names))
    if unknown:
        raise Cosmos3CheckpointEvalError(
            f"consistency checkpoint is not in the campaign: {', '.join(unknown)}"
        )
    return [
        (checkpoint, int(seed))
        for checkpoint in selected
        for seed in config["additional_seeds"]
    ]


def parse_nvidia_smi_inventory(text: str) -> list[dict[str, Any]]:
    """Parse the no-header CSV emitted by the B200 preflight."""

    inventory: list[dict[str, Any]] = []
    for line in str(text or "").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        name, uuid, memory_total, driver_version = fields
        try:
            memory_mib = int(memory_total)
        except ValueError:
            continue
        inventory.append(
            {
                "name": name,
                "uuid": uuid,
                "memory_total_mib": memory_mib,
                "driver_version": driver_version,
            }
        )
    return inventory


def require_b200_gpu(*, runner: Any = None) -> list[dict[str, Any]]:
    """Fail before any weight download unless every visible GPU is a B200."""

    run = runner or subprocess.run
    try:
        completed = run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Cosmos3CheckpointEvalError(
            "nvidia-smi is unavailable; checkpoint evaluation requires a visible B200"
        ) from exc
    if int(getattr(completed, "returncode", 1)) != 0:
        raise Cosmos3CheckpointEvalError("nvidia-smi B200 preflight failed")
    inventory = parse_nvidia_smi_inventory(getattr(completed, "stdout", ""))
    if not inventory:
        raise Cosmos3CheckpointEvalError("no visible GPU passed the B200 preflight")
    wrong = [gpu["name"] for gpu in inventory if "B200" not in str(gpu["name"]).upper()]
    if wrong:
        raise Cosmos3CheckpointEvalError(
            "checkpoint evaluation is B200-only; non-B200 GPU visible: " + ", ".join(wrong)
        )
    return inventory


class _GpuMemorySampler:
    """Best-effort nvidia-smi peak-memory sampler around one inference process."""

    def __init__(self, *, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.samples = 0
        self.max_by_uuid: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=5)
        return {
            "available": bool(self.samples),
            "samples": self.samples,
            "peak_memory_mib_by_uuid": dict(sorted(self.max_by_uuid.items())),
            "peak_memory_mib_sum": sum(self.max_by_uuid.values()),
            "sampling_interval_seconds": self.interval_seconds,
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=uuid,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                if completed.returncode == 0:
                    found = False
                    for line in completed.stdout.splitlines():
                        fields = [field.strip() for field in line.split(",")]
                        if len(fields) != 2:
                            continue
                        try:
                            used = int(fields[1])
                        except ValueError:
                            continue
                        found = True
                        self.max_by_uuid[fields[0]] = max(
                            used, self.max_by_uuid.get(fields[0], 0)
                        )
                    if found:
                        self.samples += 1
            except (OSError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval_seconds)


def _framework_commit(repo: Path, *, runner: Any = None) -> str:
    run = runner or subprocess.run
    try:
        completed = run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if int(getattr(completed, "returncode", 1)) != 0:
        return ""
    return str(getattr(completed, "stdout", "")).strip()


def _verify_framework_provenance(
    *,
    repo: Path,
    expected_commit: str,
    runtime_image: str,
    expected_image_digest: str,
) -> dict[str, str]:
    """Bind the framework commit to either Git or the released image digest.

    The published ``npa-cosmos3`` image intentionally drops ``.git`` after
    checking out the pinned source.  A source checkout can therefore prove the
    commit directly, while the released runtime proves the same packaging
    contract through its immutable multi-arch index digest.
    """

    actual_commit = _framework_commit(repo)
    if actual_commit:
        if actual_commit != expected_commit:
            raise Cosmos3CheckpointEvalError(
                "framework commit mismatch: "
                f"expected {expected_commit}, found {actual_commit}"
            )
        return {
            "framework_commit": actual_commit,
            "framework_commit_verification": "git-rev-parse",
        }

    digest_match = re.search(r"@(sha256:[0-9a-f]{64})(?:$|\s)", runtime_image)
    actual_digest = digest_match.group(1) if digest_match else ""
    if actual_digest != expected_image_digest:
        raise Cosmos3CheckpointEvalError(
            "the packaged Cosmos3 source has no .git metadata; run the released "
            f"image by its expected digest {expected_image_digest}, found "
            f"{actual_digest or '<no digest>'}"
        )
    return {
        "framework_commit": expected_commit,
        "framework_commit_verification": "released-image-digest-contract",
        "runtime_image_digest": actual_digest,
    }


def _hf_cache_refs(hf_home: Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    hub = hf_home / "hub"
    if not hub.is_dir():
        return refs
    for path in sorted(hub.glob("models--*--*/refs/*")):
        if not path.is_file():
            continue
        repo_part = path.parent.parent.name.removeprefix("models--").replace("--", "/")
        try:
            revision = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if revision:
            refs[f"{repo_part}@{path.name}"] = revision
    return refs


def _image_metadata(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.load()
            extrema = image.convert("RGB").getextrema()
            width, height = image.size
            image_format = str(image.format or path.suffix.lstrip(".")).upper()
    except (OSError, ValueError) as exc:
        raise Cosmos3CheckpointEvalError(
            f"generated artifact is not a decodable image: {path}: {exc}"
        ) from exc
    channel_spans = [int(high) - int(low) for low, high in extrema]
    return {
        "width": int(width),
        "height": int(height),
        "format": image_format,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "health_check": {
            "decodable": True,
            "nonblank": any(span > 0 for span in channel_spans),
            "rgb_channel_spans": channel_spans,
            "quality_judgment": False,
        },
    }


def _benchmark_latencies(
    benchmark: Mapping[str, Any], sample_count: int
) -> tuple[str, list[float]]:
    all_values = benchmark.get("all")
    if not isinstance(all_values, dict):
        return "", []
    candidates: list[tuple[str, list[float]]] = []
    for key, values in all_values.items():
        if str(key).startswith("[warmup]") or not str(key).endswith(".generate_batch"):
            continue
        if not isinstance(values, list) or len(values) != sample_count:
            continue
        try:
            parsed = [float(value) for value in values]
        except (TypeError, ValueError):
            continue
        candidates.append((str(key), parsed))
    return min(candidates, key=lambda item: len(item[0])) if candidates else ("", [])


def _checkpoint_cache_dir(hf_home: Path, model_id: str) -> Path:
    hub = (hf_home / "hub").resolve()
    candidate = (hub / ("models--" + model_id.replace("/", "--"))).resolve()
    if candidate.parent != hub or not candidate.name.startswith("models--nvidia--Cosmos3-"):
        raise Cosmos3CheckpointEvalError(
            f"refusing to resolve unsafe checkpoint cache target for {model_id!r}"
        )
    return candidate


def evict_checkpoint_cache(hf_home: Path, model_id: str) -> bool:
    """Remove only one known runtime-downloaded checkpoint cache directory."""

    target = _checkpoint_cache_dir(hf_home, model_id)
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True


def _publish_json(client: Any, path: Path, uri: str, payload: Mapping[str, Any]) -> str:
    _write_json(path, payload)
    return str(client.upload_file(str(path), uri))


def _publish_arm(
    arm: dict[str, Any],
    *,
    output_uri: str,
    local_root: Path,
    storage_client: Any,
) -> dict[str, Any]:
    checkpoint_slug = _checkpoint_slug(str(arm["checkpoint"]))
    seed = int(arm["seed"])
    base = _s3_join(output_uri, str(arm["phase"]), checkpoint_slug, f"seed-{seed}")
    for sample in arm.get("samples", []):
        artifact = Path(str(sample.pop("local_artifact_path")))
        artifact_uri = _s3_join(base, str(sample["prompt_id"]), artifact.name)
        storage_client.upload_file(str(artifact), artifact_uri)
        sample["artifact_uri"] = artifact_uri
    arm["arm_uri"] = _s3_join(base, "arm.json")
    arm_path = local_root / str(arm["phase"]) / checkpoint_slug / f"seed-{seed}" / "arm.json"
    _publish_json(storage_client, arm_path, str(arm["arm_uri"]), arm)
    return arm


def run_checkpoint_arm(
    *,
    config: Mapping[str, Any],
    checkpoint: str,
    seed: int,
    phase: str,
    output_uri: str,
    work_dir: Path,
    run_id: str,
    runtime_image: str,
    gpu_inventory: Sequence[Mapping[str, Any]],
    environ: Mapping[str, str] | None = None,
    runner: Any = None,
    sampler_factory: Callable[[], Any] | None = None,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Run, verify, and immediately publish one checkpoint/seed arm."""

    env = dict(environ if environ is not None else os.environ)
    repo = cosmos3_repo(env)
    if not cosmos3_generate_available(env):
        raise Cosmos3CheckpointEvalError(
            f"Cosmos 3 runtime is absent at {repo}; use the pinned npa-cosmos3 image"
        )
    expected_commit = str(config["framework_commit"])
    framework_provenance = _verify_framework_provenance(
        repo=repo,
        expected_commit=expected_commit,
        runtime_image=runtime_image,
        expected_image_digest=str(config["runtime_image_digest"]),
    )
    access = require_model_access(checkpoint=checkpoint, guardrails=True, environ=env)
    warning = check_xet_pin(repo, environ=env)
    if warning:
        print(f"WARNING: {warning}", file=sys.stderr)

    checkpoint_slug = _checkpoint_slug(checkpoint)
    arm_root = work_dir / phase / checkpoint_slug / f"seed-{seed}"
    output_root = arm_root / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    input_jsonl = arm_root / "inputs.jsonl"
    prompt_entries = list(config["prompts"])
    with input_jsonl.open("w", encoding="utf-8") as stream:
        for prompt in prompt_entries:
            spec = build_generate_spec(
                mode="text2image",
                prompt=str(prompt["text"]),
                name=str(prompt["id"]),
            )
            stream.write(json.dumps(spec, sort_keys=True) + "\n")

    hf_home = Path(env.get("HF_HOME") or (work_dir / "hf-home")).resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    token = resolve_hf_token(env)
    if token:
        env.setdefault("HF_TOKEN", token)
        env.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    env["HF_HOME"] = str(hf_home)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    argv = [
        str(_venv_python(repo)),
        "-m",
        "cosmos_framework.scripts.inference",
        *build_cosmos3_inference_args(
            input_json=str(input_jsonl),
            output_dir=str(output_root),
            checkpoint_path=checkpoint,
            seed=seed,
            no_guardrails=False,
            parallelism_preset=str(
                config.get("parallelism_preset") or DEFAULT_PARALLELISM_PRESET
            ),
            benchmark=True,
        ),
    ]
    started_at = _utc_now()
    started = time.perf_counter()
    sampler = sampler_factory() if sampler_factory else _GpuMemorySampler()
    sampler.start()
    run = runner or subprocess.run
    try:
        completed = run(argv, cwd=str(repo), env=env, check=False)
    finally:
        gpu_memory = sampler.stop()
    wall_seconds = time.perf_counter() - started
    returncode = int(getattr(completed, "returncode", 0) or 0)
    if returncode != 0:
        raise Cosmos3CheckpointEvalError(
            f"cosmos-framework inference failed for {checkpoint} seed {seed} (exit {returncode})"
        )

    benchmark_path = output_root / "benchmark.json"
    try:
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Cosmos3CheckpointEvalError(
            f"framework benchmark evidence is missing or invalid: {benchmark_path}"
        ) from exc
    timer_key, latencies = _benchmark_latencies(benchmark, len(prompt_entries))

    samples: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompt_entries):
        prompt_id = str(prompt["id"])
        sample_dir = output_root / prompt_id
        artifact, kind = _artifact_for(sample_dir, expected_kind="image")
        if artifact is None or kind != "image":
            raise Cosmos3CheckpointEvalError(
                f"no image artifact for {checkpoint} seed {seed} prompt {prompt_id}"
            )
        metadata = _image_metadata(artifact)
        if not metadata["health_check"]["nonblank"]:
            raise Cosmos3CheckpointEvalError(
                f"blank image for {checkpoint} seed {seed} prompt {prompt_id}"
            )
        sample_outputs_path = sample_dir / "sample_outputs.json"
        sample_outputs: dict[str, Any] = {}
        if sample_outputs_path.is_file():
            try:
                loaded = json.loads(sample_outputs_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    sample_outputs = loaded
            except (OSError, ValueError):
                pass
        sample: dict[str, Any] = {
            "prompt_id": prompt_id,
            "prompt": str(prompt["text"]),
            "local_artifact_path": str(artifact),
            "framework_sample_outputs": sample_outputs,
            **metadata,
        }
        if index < len(latencies):
            sample["framework_latency_seconds"] = latencies[index]
        samples.append(sample)

    arm: dict[str, Any] = {
        "schema": ARM_SCHEMA,
        "status": "succeeded",
        "run_id": run_id,
        "phase": phase,
        "checkpoint": checkpoint,
        "model_id": config["checkpoint_model_ids"][checkpoint],
        "seed": seed,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "wall_latency_seconds": wall_seconds,
        "framework_timer_key": timer_key,
        "framework_benchmark": benchmark,
        "samples": samples,
        "gpu_inventory": [dict(gpu) for gpu in gpu_inventory],
        "gpu_memory": gpu_memory,
        "provenance": {
            "runtime_image": runtime_image,
            **framework_provenance,
            "checkpoint_refs": _hf_cache_refs(hf_home),
            "guardrails_enabled": True,
            "guardrail_posture": PINNED_GUARDRAIL_POSTURE,
            "declared_runtime_assets": list(config.get("required_runtime_assets") or []),
            "hf_auth": access["hf_auth"],
            "ngc_auth": access["ngc_auth"],
            "weights_baked": False,
            "runtime_checkpoint_download": True,
            "campaign_config_sha256": _json_sha256(config),
        },
    }
    client = storage_client or StorageClient.from_environment()
    return _publish_arm(
        arm,
        output_uri=output_uri,
        local_root=work_dir,
        storage_client=client,
    )


def execute_phase(
    *,
    campaign_config: str,
    phase: str,
    output_uri: str,
    top_checkpoints: Sequence[str] = (),
    work_dir: str | Path = DEFAULT_WORK_DIR,
    run_id: str = "",
    runtime_image: str = "",
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
    storage_client: Any = None,
    gpu_probe_runner: Any = None,
    arm_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute a primary or consistency phase and publish durable progress."""

    root = Path(work_dir).expanduser().resolve()
    client = storage_client
    raw, config_path = _load_config_path(
        campaign_config,
        work_dir=root,
        storage_client=client,
    )
    config = validate_campaign_config(raw)
    arms = phase_arms(config, phase=phase, top_checkpoints=top_checkpoints)
    plan = {
        "schema": PHASE_SCHEMA,
        "status": "planned" if dry_run else "running",
        "run_id": run_id,
        "phase": phase,
        "campaign": str(config.get("campaign") or ""),
        "campaign_config_sha256": _json_sha256(config),
        "runtime_image": runtime_image,
        "guardrails_enabled": True,
        "weights_baked": False,
        "arms": [
            {"checkpoint": checkpoint, "seed": seed, "status": "planned"}
            for checkpoint, seed in arms
        ],
    }
    if dry_run:
        return plan

    gpu_inventory = require_b200_gpu(runner=gpu_probe_runner)
    if client is None:
        client = StorageClient.from_environment()
    root.mkdir(parents=True, exist_ok=True)
    config_uri = _s3_join(output_uri, "campaign-config.json")
    client.upload_file(str(config_path), config_uri)
    plan.update(
        {
            "started_at": _utc_now(),
            "campaign_config_uri": config_uri,
            "gpu_inventory": gpu_inventory,
        }
    )
    phase_uri = _s3_join(output_uri, f"{phase}.json")
    phase_path = root / f"{phase}.json"
    _publish_json(client, phase_path, phase_uri, plan)

    failed_arms = 0
    cache_eviction_failures = 0
    execute_arm = arm_runner or run_checkpoint_arm
    current_checkpoint = ""
    for index, (checkpoint, seed) in enumerate(arms):
        current_checkpoint = checkpoint
        try:
            result = execute_arm(
                config=config,
                checkpoint=checkpoint,
                seed=seed,
                phase=phase,
                output_uri=output_uri,
                work_dir=root,
                run_id=run_id,
                runtime_image=runtime_image,
                gpu_inventory=gpu_inventory,
                environ=environ,
                storage_client=client,
            )
        except Exception as exc:  # noqa: BLE001 - persist the exact failed arm and continue
            failed_arms += 1
            result = {
                "schema": ARM_SCHEMA,
                "status": "failed",
                "checkpoint": checkpoint,
                "model_id": config["checkpoint_model_ids"][checkpoint],
                "seed": seed,
                "phase": phase,
                "error": str(exc),
                "completed_at": _utc_now(),
            }
        plan["arms"][index] = result
        plan["completed_arms"] = index + 1
        plan["failed_arms"] = failed_arms
        plan["cache_eviction_failures"] = cache_eviction_failures
        _publish_json(client, phase_path, phase_uri, plan)

        next_checkpoint = arms[index + 1][0] if index + 1 < len(arms) else ""
        if next_checkpoint != current_checkpoint:
            hf_home = Path(
                (environ if environ is not None else os.environ).get("HF_HOME")
                or (root / "hf-home")
            ).resolve()
            model_id = str(config["checkpoint_model_ids"][checkpoint])
            try:
                removed = evict_checkpoint_cache(hf_home, model_id)
            except Exception as exc:  # noqa: BLE001 - disk safety evidence is a phase failure
                cache_eviction_failures += 1
                plan.setdefault("cache_eviction_errors", []).append(
                    {"checkpoint": checkpoint, "error": str(exc)}
                )
            else:
                plan.setdefault("cache_evictions", []).append(
                    {"checkpoint": checkpoint, "model_id": model_id, "removed": removed}
                )
            plan["failed_arms"] = failed_arms
            plan["cache_eviction_failures"] = cache_eviction_failures
            _publish_json(client, phase_path, phase_uri, plan)

    plan["completed_at"] = _utc_now()
    total_failures = failed_arms + cache_eviction_failures
    plan["status"] = "succeeded" if total_failures == 0 else "failed"
    plan["phase_uri"] = phase_uri
    _publish_json(client, phase_path, phase_uri, plan)
    if total_failures:
        raise Cosmos3CheckpointEvalError(
            f"{phase} phase failed with {failed_arms} failed generation arm(s) and "
            f"{cache_eviction_failures} cache-eviction failure(s); evidence: {phase_uri}"
        )
    return plan


__all__ = [
    "ARM_SCHEMA",
    "CAMPAIGN_CONFIG_SCHEMA",
    "PHASE_SCHEMA",
    "Cosmos3CheckpointEvalError",
    "evict_checkpoint_cache",
    "execute_phase",
    "parse_nvidia_smi_inventory",
    "phase_arms",
    "require_b200_gpu",
    "run_checkpoint_arm",
    "validate_campaign_config",
]
