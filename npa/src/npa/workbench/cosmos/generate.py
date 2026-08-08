"""Real Cosmos 3 generation runner (NVIDIA cosmos-framework omni model).

Shared by ``npa workbench cosmos3 generate``, the SDK wrapper, and the
``cosmos3-generate`` SkyPilot workflow so all three drive the actual model
instead of emitting a descriptor manifest.

The runtime lives in the ``npa-cosmos3`` image at ``/opt/cosmos3/cosmos-framework``
(framework source at a pinned commit + torch cu130 inference venv). This module
shells out to that venv's ``cosmos_framework.scripts.inference`` so it stays
import-safe on the default interpreter — no torch/CUDA import here — exactly like
:mod:`npa.workbench.cosmos.transfer` does for Cosmos Transfer 2.5.

No model weights ship in the image. Checkpoints resolve through the framework's
checkpoint database to gated Hugging Face repos and download on first use, so
:func:`require_model_access` fails fast unless the operator supplied their own
Hugging Face token (and NGC key, when NGC-hosted artifacts are required).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from npa.workbench.cosmos.cosmos3 import build_cosmos3_inference_args

DEFAULT_REPO = "/opt/cosmos3/cosmos-framework"
DEFAULT_REPO_ENV = "COSMOS3_REPO"
DEFAULT_CHECKPOINT = "Cosmos3-Nano"
DEFAULT_CHECKPOINT_ENV = "NPA_COSMOS3_CHECKPOINT"
# Gated, and fetched from Hugging Face whenever guardrails are enabled, so it
# drives the credential preflight independently of the checkpoint.
GUARDRAIL_MODEL_ID = "nvidia/Cosmos-Guardrail1"
DEFAULT_MODE = "text2image"
DEFAULT_NAME = "npa-generate"
DEFAULT_OUTPUT_DIR = "/tmp/npa-cosmos3-generate"
DEFAULT_OUTPUT_DIR_ENV = "NPA_COSMOS3_OUTPUT_DIR"
DEFAULT_PARALLELISM_PRESET = "latency"
GENERATE_SCHEMA = "npa.cosmos3.generate.v1"

# Generation modes of the omni model. Reasoning ("reasoner") and the
# action/dynamics modes are deliberately out of scope for this runner.
IMAGE_MODES = frozenset({"text2image", "image2image"})
VIDEO_MODES = frozenset({"text2video", "image2video", "video2video"})
GENERATE_MODES = tuple(sorted(IMAGE_MODES | VIDEO_MODES))
# Modes conditioned on an input image/video, so a vision asset is mandatory.
VISION_REQUIRED_MODES = frozenset({"image2image", "image2video", "video2video"})

HF_TOKEN_ENVS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
HF_TOKEN_ENV_OVERRIDE = "NPA_COSMOS3_HF_TOKEN_ENV"
NGC_API_KEY_ENV_OVERRIDE = "NPA_COSMOS3_NGC_API_KEY_ENV"
DEFAULT_NGC_API_KEY_ENV = "NGC_API_KEY"
REQUIRE_NGC_ENV = "NPA_COSMOS3_REQUIRE_NGC"

ACCESS_PREFLIGHT_DOC = "docs/workbench/cosmos3-access-preflight.md"

# huggingface/xet-core#895: a gated-repo download fails with "Unable to parse
# string as hex hash value" on exactly this pin pair. Fixed in later releases
# of both packages, so this is a warning naming the known-bad pair, not an
# unconditional env default. See ACCESS_PREFLIGHT_DOC.
XET_AFFECTED_HUGGINGFACE_HUB_VERSION = "1.23.0"
XET_AFFECTED_HF_XET_VERSION = "1.5.1"

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
_VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".mov")


class Cosmos3GenerateError(RuntimeError):
    """Raised when Cosmos 3 generation cannot run or produced no artifact."""


def cosmos3_repo(environ: Mapping[str, str] | None = None) -> Path:
    """Return the framework checkout that backs generation."""

    env = environ if environ is not None else os.environ
    return Path(env.get(DEFAULT_REPO_ENV, "") or DEFAULT_REPO)


def _venv_python(repo: Path) -> Path:
    return repo / ".venv" / "bin" / "python"


def cosmos3_generate_available(environ: Mapping[str, str] | None = None) -> bool:
    """True when the real Cosmos 3 inference runtime is present and runnable."""

    repo = cosmos3_repo(environ)
    if not (repo / "cosmos_framework" / "scripts" / "inference.py").is_file():
        return False
    python = _venv_python(repo)
    try:
        return python.is_file() and os.access(python, os.X_OK)
    except OSError:
        return False


def resolve_hf_token(environ: Mapping[str, str] | None = None) -> str:
    """Return the operator's Hugging Face token, or "" when none is configured."""

    env = environ if environ is not None else os.environ
    names = list(HF_TOKEN_ENVS)
    override = str(env.get(HF_TOKEN_ENV_OVERRIDE, "") or "").strip()
    if override:
        names.insert(0, override)
    for name in names:
        value = str(env.get(name, "") or "").strip()
        if value:
            return value
    return ""


def _env_truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_local_checkpoint(checkpoint: str) -> bool:
    """True when the checkpoint is a path/URI the operator already staged.

    Named checkpoints (``Cosmos3-Nano``) resolve to gated Hugging Face repos, so
    they need a token; a local directory or ``s3://`` URI does not.
    """

    value = str(checkpoint or "").strip()
    if not value:
        return False
    return value.startswith(("/", "./", "~", "s3://"))


def require_model_access(
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
    guardrails: bool = True,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Verify the operator supplied credentials for the gated weights.

    The image ships no weights, so a run must fetch them under the operator's own
    Hugging Face license acceptance. A run pulls from Hugging Face for more than
    the checkpoint: with guardrails on (the default) it also fetches the gated
    ``nvidia/Cosmos-Guardrail1``. Staging a checkpoint therefore only removes the
    token requirement when guardrails are off as well — otherwise the preflight
    would pass and the run would still die mid-inference fetching the guardrail
    models, which is exactly the failure this check exists to prevent.

    Raises :class:`Cosmos3GenerateError` when the token is missing, or when NGC
    access is demanded (``NPA_COSMOS3_REQUIRE_NGC=1``) without an NGC key.
    """

    env = environ if environ is not None else os.environ
    ngc_env = (
        str(env.get(NGC_API_KEY_ENV_OVERRIDE, "") or "").strip()
        or DEFAULT_NGC_API_KEY_ENV
    )

    needs: list[str] = []
    if not _is_local_checkpoint(checkpoint):
        needs.append(f"the {checkpoint} checkpoint")
    if guardrails:
        needs.append(f"the gated guardrail models ({GUARDRAIL_MODEL_ID})")

    hf_auth = "skipped"
    if needs:
        if not resolve_hf_token(env):
            raise Cosmos3GenerateError(
                "Cosmos 3 weights are not baked into this image and this run must "
                f"download {' and '.join(needs)} from Hugging Face. Set HF_TOKEN "
                f"(or {HF_TOKEN_ENV_OVERRIDE}) to a token that has accepted those "
                "licenses. A staged local/s3 --checkpoint only removes this "
                "requirement when --no-guardrails is also passed. With no token "
                "set, a fetch of a gated repo is anonymous and fails with 401; "
                "if you set a token and then see 403 instead, the token itself "
                "is reaching Hugging Face and the account behind it still has to "
                f"accept the repo's license. See {ACCESS_PREFLIGHT_DOC}."
            )
        hf_auth = "configured"

    ngc_auth = "skipped"
    if _env_truthy(str(env.get(REQUIRE_NGC_ENV, ""))):
        if not str(env.get(ngc_env, "") or "").strip():
            raise Cosmos3GenerateError(
                f"{REQUIRE_NGC_ENV} is set but {ngc_env} is empty; NGC-hosted "
                "Cosmos 3 artifacts need the operator's own NGC API key."
            )
        ngc_auth = "configured"

    return {"hf_auth": hf_auth, "ngc_auth": ngc_auth}


def _installed_package_versions(
    python: Path, packages: tuple[str, ...], *, runner: Any = None
) -> dict[str, str]:
    """Best-effort read of ``packages`` versions inside ``python``'s environment.

    Returns an empty dict on any failure (missing interpreter, package not
    installed, non-zero exit): this backs a warning, not a preflight gate, so
    it must never block or fail a run on its own.
    """

    probe = (
        "import importlib.metadata as m, json, sys\n"
        "out = {}\n"
        "for name in sys.argv[1:]:\n"
        "    try:\n"
        "        out[name] = m.version(name)\n"
        "    except m.PackageNotFoundError:\n"
        "        pass\n"
        "print(json.dumps(out))\n"
    )
    run = runner or subprocess.run
    try:
        completed = run(
            [str(python), "-c", probe, *packages],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if getattr(completed, "returncode", 1) != 0:
        return {}
    try:
        versions = json.loads(getattr(completed, "stdout", "") or "{}")
    except ValueError:
        return {}
    if not isinstance(versions, dict):
        return {}
    return {k: str(v) for k, v in versions.items() if isinstance(k, str)}


def check_xet_pin(repo: Path, *, runner: Any = None) -> str:
    """Warn when the runtime venv has the known-bad Xet-download pin pair.

    huggingface/xet-core#895 makes a gated-repo download fail with
    "Unable to parse string as hex hash value" on exactly
    ``huggingface_hub==1.23.0`` plus ``hf-xet==1.5.1``. Returns a warning
    string naming the ``HF_HUB_DISABLE_XET=1`` workaround when that exact
    pair is installed, or "" when the pair cannot be confirmed (package
    missing, versions differ, or the check itself failed). This never sets
    the environment variable itself: newer releases fix the issue, so
    disabling Xet unconditionally would cost every unaffected environment a
    faster download path for no reason.
    """

    versions = _installed_package_versions(
        _venv_python(repo), ("huggingface_hub", "hf-xet"), runner=runner
    )
    if (
        versions.get("huggingface_hub") == XET_AFFECTED_HUGGINGFACE_HUB_VERSION
        and versions.get("hf-xet") == XET_AFFECTED_HF_XET_VERSION
    ):
        return (
            "huggingface_hub "
            f"{XET_AFFECTED_HUGGINGFACE_HUB_VERSION} + hf-xet "
            f"{XET_AFFECTED_HF_XET_VERSION} is affected by "
            "huggingface/xet-core#895 (gated-repo download fails with "
            "'Unable to parse string as hex hash value'). Set "
            f"HF_HUB_DISABLE_XET=1 and retry; see {ACCESS_PREFLIGHT_DOC}."
        )
    return ""


def build_generate_spec(
    *,
    mode: str = DEFAULT_MODE,
    prompt: str,
    name: str = DEFAULT_NAME,
    vision_path: str = "",
    negative_prompt: str = "",
    num_steps: int = 0,
    guidance: float = 0.0,
) -> dict[str, Any]:
    """Build one cosmos-framework input sample for a generation mode.

    The framework reads a JSON (or JSONL) sample file whose ``model_mode`` selects
    the pipeline and whose remaining keys override that mode's ``sample_args``
    defaults. Unset overrides are omitted so upstream defaults apply.
    """

    resolved_mode = str(mode or DEFAULT_MODE).strip()
    if resolved_mode not in GENERATE_MODES:
        raise Cosmos3GenerateError(
            f"unsupported generate mode {resolved_mode!r}; choose one of: "
            f"{', '.join(GENERATE_MODES)}"
        )
    text = str(prompt or "").strip()
    if not text:
        raise Cosmos3GenerateError("prompt must not be empty")
    sample_name = str(name or DEFAULT_NAME).strip() or DEFAULT_NAME
    vision = str(vision_path or "").strip()
    if resolved_mode in VISION_REQUIRED_MODES and not vision:
        raise Cosmos3GenerateError(
            f"mode {resolved_mode!r} conditions on an input image/video; pass "
            "--input-path"
        )

    spec: dict[str, Any] = {
        "model_mode": resolved_mode,
        "name": sample_name,
        "prompt": text,
    }
    if vision:
        spec["vision_path"] = vision
    if negative_prompt:
        spec["negative_prompt"] = str(negative_prompt)
    if num_steps and int(num_steps) > 0:
        spec["num_steps"] = int(num_steps)
    if guidance and float(guidance) > 0:
        spec["guidance"] = float(guidance)
    return spec


def _resolve_checkpoint(
    checkpoint: str, environ: Mapping[str, str] | None = None
) -> str:
    env = environ if environ is not None else os.environ
    value = (
        str(checkpoint or "").strip()
        or str(env.get(DEFAULT_CHECKPOINT_ENV, "") or "").strip()
        or DEFAULT_CHECKPOINT
    )
    # A staged checkpoint may be written with ~; upstream does not expand it, so
    # resolve it here rather than handing the framework a path it cannot open.
    if value.startswith("~"):
        value = str(Path(value).expanduser())
    return value


def _resolve_output_dir(
    output_dir: str | Path | None, environ: Mapping[str, str] | None = None
) -> Path:
    env = environ if environ is not None else os.environ
    value = str(output_dir or "").strip() or str(
        env.get(DEFAULT_OUTPUT_DIR_ENV, "") or ""
    ).strip()
    return Path(value or DEFAULT_OUTPUT_DIR)


def generate_plan(
    *,
    mode: str = DEFAULT_MODE,
    prompt: str,
    output_dir: str | Path | None = None,
    name: str = DEFAULT_NAME,
    checkpoint: str = "",
    vision_path: str = "",
    negative_prompt: str = "",
    seed: int = 0,
    num_steps: int = 0,
    guidance: float = 0.0,
    no_guardrails: bool = False,
    parallelism_preset: str = DEFAULT_PARALLELISM_PRESET,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve the input sample and inference argv without running the model.

    Lets a caller (or a workflow author) inspect exactly what would run on GPU,
    including whether guardrails stay enabled, from a CPU host.
    """

    spec = build_generate_spec(
        mode=mode,
        prompt=prompt,
        name=name,
        vision_path=vision_path,
        negative_prompt=negative_prompt,
        num_steps=num_steps,
        guidance=guidance,
    )
    resolved_output = _resolve_output_dir(output_dir, environ)
    resolved_checkpoint = _resolve_checkpoint(checkpoint, environ)
    input_json = resolved_output / f"{spec['name']}.json"
    args = build_cosmos3_inference_args(
        input_json=str(input_json),
        output_dir=str(resolved_output),
        checkpoint_path=resolved_checkpoint,
        seed=int(seed),
        no_guardrails=bool(no_guardrails),
        parallelism_preset=str(parallelism_preset or DEFAULT_PARALLELISM_PRESET),
    )
    repo = cosmos3_repo(environ)
    return {
        "schema": GENERATE_SCHEMA,
        "mode": spec["model_mode"],
        "name": spec["name"],
        "checkpoint": resolved_checkpoint,
        "guardrails": not bool(no_guardrails),
        "seed": int(seed),
        "output_dir": str(resolved_output),
        "input_json": str(input_json),
        "input_spec": spec,
        "repo": str(repo),
        "argv": [
            str(_venv_python(repo)),
            "-m",
            "cosmos_framework.scripts.inference",
            *args,
        ],
    }


def _artifact_for(sample_dir: Path, *, expected_kind: str = "") -> tuple[Path | None, str]:
    """Return the generated media file in ``sample_dir`` and its kind.

    The framework writes the sample under ``<output_dir>/<name>/`` and copies any
    conditioning asset into ``<name>/inputs/``, which is excluded so a
    video2video run never reports its own input as the result.

    Selection prefers the largest file whose kind matches ``expected_kind``, so a
    video run that also emits a poster frame still reports the clip rather than
    the (possibly larger) still. Only when the mode's own kind is absent does it
    fall back to the largest media file of any kind, which lets the caller detect
    the mismatch instead of silently succeeding with the wrong artifact.
    """

    candidates: list[tuple[int, Path, str]] = []
    for path in sorted(sample_dir.rglob("*")):
        if not path.is_file():
            continue
        if "inputs" in path.relative_to(sample_dir).parts:
            continue
        suffix = path.suffix.lower()
        if suffix in _VIDEO_SUFFIXES:
            candidate_kind = "video"
        elif suffix in _IMAGE_SUFFIXES:
            candidate_kind = "image"
        else:
            continue
        candidates.append((path.stat().st_size, path, candidate_kind))

    if not candidates:
        return None, ""
    preferred = [c for c in candidates if c[2] == expected_kind] if expected_kind else []
    size, best, kind = max(preferred or candidates, key=lambda item: item[0])
    del size
    return best, kind


def run_cosmos3_generate(
    *,
    mode: str = DEFAULT_MODE,
    prompt: str,
    output_dir: str | Path | None = None,
    name: str = DEFAULT_NAME,
    checkpoint: str = "",
    vision_path: str = "",
    negative_prompt: str = "",
    seed: int = 0,
    num_steps: int = 0,
    guidance: float = 0.0,
    no_guardrails: bool = False,
    parallelism_preset: str = DEFAULT_PARALLELISM_PRESET,
    environ: Mapping[str, str] | None = None,
    runner: Any = None,
    version_probe_runner: Any = None,
) -> dict[str, Any]:
    """Run a real Cosmos 3 generation; return the artifact plus its metadata.

    Guardrails stay on unless ``no_guardrails`` is explicitly requested. Raises
    :class:`Cosmos3GenerateError` when the runtime is absent, the operator's
    Hugging Face credentials are missing, inference fails, or no media artifact
    was produced. ``version_probe_runner`` overrides only the xet pin check's
    subprocess seam, so a test can drive that check without also mocking the
    inference call or the check itself; production leaves it ``None``.
    """

    env = dict(environ if environ is not None else os.environ)
    plan = generate_plan(
        mode=mode,
        prompt=prompt,
        output_dir=output_dir,
        name=name,
        checkpoint=checkpoint,
        vision_path=vision_path,
        negative_prompt=negative_prompt,
        seed=seed,
        num_steps=num_steps,
        guidance=guidance,
        no_guardrails=no_guardrails,
        parallelism_preset=parallelism_preset,
        environ=env,
    )
    if not cosmos3_generate_available(env):
        raise Cosmos3GenerateError(
            "the Cosmos 3 inference runtime is not present at "
            f"{plan['repo']}; run inside the npa-cosmos3 image on a GPU "
            f"(or point {DEFAULT_REPO_ENV} at a framework checkout)."
        )
    access = require_model_access(
        checkpoint=plan["checkpoint"],
        guardrails=bool(plan["guardrails"]),
        environ=env,
    )
    if access["hf_auth"] == "configured":
        # Scoped to "this run downloads from Hugging Face", not to guardrails:
        # xet-core#895 is a transfer bug, so a guardrails-off run that still
        # pulls its checkpoint from HF hits it too.
        xet_warning = check_xet_pin(Path(plan["repo"]), runner=version_probe_runner)
        if xet_warning:
            print(f"WARNING: {xet_warning}", file=sys.stderr)

    output_root = Path(plan["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    input_json = Path(plan["input_json"])
    input_json.write_text(
        json.dumps(plan["input_spec"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    token = resolve_hf_token(env)
    if token:
        env.setdefault("HF_TOKEN", token)
        env.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    env.setdefault("HF_HOME", str(Path(output_root).parent / "hf_home"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    run = runner or subprocess.run
    completed = run(list(plan["argv"]), cwd=str(plan["repo"]), env=env, check=False)
    returncode = int(getattr(completed, "returncode", 0) or 0)
    if returncode != 0:
        raise Cosmos3GenerateError(
            f"cosmos-framework inference failed (exit {returncode}) for mode "
            f"{plan['mode']}"
        )

    sample_dir = output_root / str(plan["name"])
    expected = "image" if plan["mode"] in IMAGE_MODES else "video"
    artifact, kind = _artifact_for(sample_dir, expected_kind=expected)
    if artifact is None:
        raise Cosmos3GenerateError(
            f"cosmos-framework produced no image/video artifact in {sample_dir}"
        )
    if kind != expected:
        # Reporting success with the wrong medium (e.g. a poster frame standing in
        # for a video) would let a broken run look complete downstream.
        raise Cosmos3GenerateError(
            f"mode {plan['mode']} should produce a {expected} artifact but only a "
            f"{kind} was found in {sample_dir}: {artifact.name}"
        )

    result = dict(plan)
    result.pop("argv", None)
    result.update(
        {
            "status": "executed",
            "output_kind": kind,
            "expected_output_kind": expected,
            "output_path": str(artifact),
            "output_bytes": artifact.stat().st_size,
            "sample_dir": str(sample_dir),
            "prompt": plan["input_spec"]["prompt"],
            "hf_auth": access["hf_auth"],
            "ngc_auth": access["ngc_auth"],
            "weights_baked": False,
        }
    )
    sample_outputs = sample_dir / "sample_outputs.json"
    if sample_outputs.is_file():
        try:
            result["sample_outputs"] = json.loads(
                sample_outputs.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            result["sample_outputs"] = {}
    return result


def publish_generation_to_s3(
    result: Mapping[str, Any],
    output_uri: str,
    *,
    run_id: str = "",
    storage_client: Any = None,
) -> dict[str, Any]:
    """Upload one generation artifact plus its manifest under ``output_uri``."""

    if not str(output_uri).startswith("s3://"):
        raise ValueError(f"output_uri must be an s3:// prefix, got: {output_uri!r}")
    import tempfile

    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    base = output_uri if output_uri.endswith("/") else output_uri + "/"
    artifact = Path(str(result["output_path"]))
    artifact_uri = f"{base}{artifact.name}"
    client.upload_file(str(artifact), artifact_uri)

    manifest = dict(result)
    manifest.update(
        {
            "run_id": run_id,
            "output_uri": base,
            "artifact_uri": artifact_uri,
        }
    )
    with tempfile.TemporaryDirectory(prefix="npa-cosmos3-gen-") as tmp:
        path = Path(tmp) / "generate.json"
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        client.upload_file(str(path), f"{base}generate.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for raw workflow YAMLs (mirrors the Typer command)."""

    import argparse

    parser = argparse.ArgumentParser(description="Run a real Cosmos 3 generation.")
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=list(GENERATE_MODES))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--input-path", dest="input_path", default="")
    parser.add_argument("--output-path", dest="output_path", default="")
    parser.add_argument("--negative-prompt", dest="negative_prompt", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", dest="num_steps", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--no-guardrails", dest="no_guardrails", action="store_true")
    parser.add_argument("--parallelism-preset", default=DEFAULT_PARALLELISM_PRESET)
    parser.add_argument("--run-id", dest="run_id", default="")
    parser.add_argument("--output-json", dest="output_json", default="")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = parser.parse_args(argv)

    payload = generate_and_publish(
        mode=args.mode,
        prompt=args.prompt,
        name=args.name,
        checkpoint=args.checkpoint,
        input_path=args.input_path,
        output_path=args.output_path,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        num_steps=args.num_steps,
        guidance=args.guidance,
        no_guardrails=args.no_guardrails,
        parallelism_preset=args.parallelism_preset,
        run_id=args.run_id,
        dry_run=args.dry_run,
    )
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def materialize_vision_input(source: str) -> str:
    """Resolve a conditioning asset to something the framework can read.

    Local paths and ``http(s)`` URLs pass through (upstream downloads URLs
    itself); an ``s3://`` URI is fetched to a temp dir first.
    """

    value = str(source or "").strip()
    if not value or value.startswith(("http://", "https://")):
        return value
    if not value.startswith("s3://"):
        return value
    import tempfile

    from npa.clients.storage import StorageClient

    client = StorageClient.from_environment()
    tmp = tempfile.mkdtemp(prefix="npa-cosmos3-input-")
    name = Path(value).name or "vision-input"
    return client.download_path(value, str(Path(tmp) / name))


def generate_and_publish(
    *,
    mode: str = DEFAULT_MODE,
    prompt: str,
    name: str = DEFAULT_NAME,
    checkpoint: str = "",
    input_path: str = "",
    output_path: str = "",
    negative_prompt: str = "",
    seed: int = 0,
    num_steps: int = 0,
    guidance: float = 0.0,
    no_guardrails: bool = False,
    parallelism_preset: str = DEFAULT_PARALLELISM_PRESET,
    run_id: str = "",
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Generate, then publish to S3 when ``output_path`` is an ``s3://`` prefix.

    One implementation for the Typer command, the SDK, and the raw-YAML entry
    point: ``output_path`` may be a local directory or an ``s3://`` prefix, and
    ``dry_run`` resolves the plan without touching a GPU.
    """

    target = str(output_path or "").strip()
    is_s3 = target.startswith("s3://")
    local_dir = None if is_s3 else (target or None)
    vision = "" if dry_run else materialize_vision_input(input_path)
    resolved_vision = vision or str(input_path or "").strip()
    if dry_run:
        plan = generate_plan(
            mode=mode,
            prompt=prompt,
            output_dir=local_dir,
            name=name,
            checkpoint=checkpoint,
            vision_path=resolved_vision,
            negative_prompt=negative_prompt,
            seed=seed,
            num_steps=num_steps,
            guidance=guidance,
            no_guardrails=no_guardrails,
            parallelism_preset=parallelism_preset,
            environ=environ,
        )
        plan.update({"status": "planned", "run_id": run_id, "weights_baked": False})
        if is_s3:
            plan["output_uri"] = target
        return plan

    result = run_cosmos3_generate(
        mode=mode,
        prompt=prompt,
        output_dir=local_dir,
        name=name,
        checkpoint=checkpoint,
        vision_path=resolved_vision,
        negative_prompt=negative_prompt,
        seed=seed,
        num_steps=num_steps,
        guidance=guidance,
        no_guardrails=no_guardrails,
        parallelism_preset=parallelism_preset,
        environ=environ,
    )
    result["run_id"] = run_id
    if is_s3:
        return publish_generation_to_s3(result, target, run_id=run_id)
    return result


__all__ = [
    "Cosmos3GenerateError",
    "GENERATE_MODES",
    "GENERATE_SCHEMA",
    "IMAGE_MODES",
    "VIDEO_MODES",
    "VISION_REQUIRED_MODES",
    "build_generate_spec",
    "check_xet_pin",
    "cosmos3_generate_available",
    "cosmos3_repo",
    "generate_and_publish",
    "generate_plan",
    "main",
    "materialize_vision_input",
    "publish_generation_to_s3",
    "require_model_access",
    "resolve_hf_token",
    "run_cosmos3_generate",
]


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
