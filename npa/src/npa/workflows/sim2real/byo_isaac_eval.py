"""BYO held-out eval: roll the TRAINED Isaac policy for a real success_rate.

Wired in via ``sim2real run --byo-eval-command 'python3 -m
npa.workflows.sim2real.byo_isaac_eval'``. Satisfies ``run_heldout_eval``'s
contract: write ``NPA_SIM2REAL_OUTPUT_JSON`` with a ``per_env`` list of
``{env_id, score, success}``; the engine's ``_normalize_heldout_report``
computes ``success_rate`` from it.

Unlike the reference/stub held-out payload (which scores synthetic rollouts and
does NOT load any trained policy), this loads the **trained checkpoint** (from
the inner-loop evidence's ``update.checkpoint_path``) and rolls it in Isaac on
``Isaac-Lift-Cube-Franka-v0``, deriving per-env success from the task's own
object-to-goal metric.

Runs in the orchestrator pod (no Isaac), so it submits an Isaac sibling k8s Job
that downloads the checkpoint, plays the policy, writes per-env scores to S3;
this process reads them back and writes the output JSON.

``NPA_BYO_ISAAC_DRYRUN=1`` skips the Kubernetes API/S3 and emits a deterministic per-env
report for unit tests / wiring checks.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from npa.clients.storage import StorageClient
from npa.workflows.sim2real.camera_views import camera_metadata, camera_views_json
from npa.workflows.sim2real.capture import capture_settings
from npa.workflows.sim2real.isaac_job_payload import compressed_bash_launch

DEFAULT_ISAAC_TASK = "Isaac-Lift-Cube-Franka-v0"
DEFAULT_GPU_PRODUCT = "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
# Object-to-goal distance (metres) under which a Lift episode counts as success.
DEFAULT_SUCCESS_DIST_M = 0.05
_MAX_EMBEDDED_SCENARIOS_BYTES = 32_000

# Set by main() so run_isaac_eval_job can sync rendered frames to the heldout
# renders dir + surface the render manifest into the report (for Rerun viz).
_RENDERS_LOCAL_DIR = ""
_RENDER_MANIFEST: dict[str, Any] = {}
_LAST_GPU_PROVENANCE: dict[str, Any] = {}
_CHECKPOINT_PROVENANCE: dict[str, Any] = {}
_APPLIED_SCENARIO_AUDIT: dict[str, Any] = {}
_SCENARIO_INPUT_PROVENANCE: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested without a cluster)
# --------------------------------------------------------------------------- #
def first_episode_masks(completed: Any, done: Any) -> tuple[Any, Any, Any]:
    """Return active/newly-terminal masks while sealing the first episode.

    Isaac vector environments auto-reset an environment inside ``step`` before
    returning control to the caller.  Consequently, state read after a true
    ``done`` belongs to the next episode.  The live evaluator uses these masks
    to retain the last pre-step sample for newly terminal environments and to
    prevent every later auto-reset episode from overwriting that snapshot.

    The operands intentionally use NumPy-compatible bitwise operations without
    importing NumPy in this controller module; the baked Isaac runtime supplies
    boolean arrays.
    """

    active = ~completed
    newly_terminal = active & done
    return active, newly_terminal, completed | done


def extract_checkpoint_uri(inner_evidence: dict[str, Any]) -> str:
    """Pull the trained-policy checkpoint S3 URI from inner-loop evidence.

    Looks at the latest iteration's ``update.checkpoint_path``. Returns "" when
    no real checkpoint is present (e.g. reference trainer).
    """

    selected = str(
        inner_evidence.get("selected_checkpoint_uri")
        or inner_evidence.get("final_checkpoint_uri")
        or ""
    ).strip()
    if selected.startswith("s3://"):
        return selected
    iterations = inner_evidence.get("iterations") or []
    for record in reversed(iterations):
        update = (record or {}).get("update") or {}
        ckpt = str(update.get("checkpoint_path") or "").strip()
        if ckpt.startswith("s3://"):
            return ckpt
    return ""


def policy_inference_provenance(
    *, checkpoint_uri: str, checkpoint: dict[str, Any]
) -> dict[str, Any]:
    """Describe exact learned-actor inference without scripted control."""

    return {
        "backend": "isaac_rsl_rl_inference",
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_sha256": str(checkpoint.get("sha256") or ""),
        "checkpoint_size_bytes": int(checkpoint.get("size_bytes") or 0),
        "loaded_for_inference": bool(checkpoint),
        "stock_or_scripted_policy": False,
        "actor_is_learned": True,
        "scripted_post_actor_controller": False,
        "policy_composition": "learned_actor_only",
        "post_actor_controller": None,
    }


def build_heldout_report(
    per_env: list[dict[str, Any]],
    *,
    isaac_task: str,
    checkpoint_uri: str,
    source: str,
) -> dict[str, Any]:
    """Build the payload _normalize_heldout_report consumes (per_env list)."""

    # Success at multiple object->goal distance thresholds: a single strict
    # threshold hides real progress (a policy that lifts + roughly places scores
    # 0 at 0.05m but high at 0.15m). Report the curve so accuracy improvement is
    # visible even before the policy is pinpoint-accurate.
    dists = [
        r["details"]["object_goal_distance_m"]
        for r in per_env
        if "object_goal_distance_m" in r.get("details", {})
    ]
    success_summary = {}
    if dists:
        for thr in (0.05, 0.10, 0.15, 0.20):
            success_summary[f"success@{thr:.2f}"] = round(
                sum(1 for d in dists if d < thr) / len(dists), 4
            )
        success_summary["mean_object_goal_distance_m"] = round(
            sum(dists) / len(dists), 6
        )
        success_summary["min_object_goal_distance_m"] = round(min(dists), 6)
        closest: list[float] = []
        for row in per_env:
            value = (row.get("details") or {}).get("closest_object_goal_distance_m")
            if value is not None:
                closest.append(float(value))
        if closest:
            success_summary["mean_closest_object_goal_distance_m"] = round(
                sum(closest) / len(closest), 6
            )
            success_summary["min_closest_object_goal_distance_m"] = round(
                min(closest), 6
            )

    n = len(per_env)
    strict_count = sum(bool(row.get("success")) for row in per_env)

    def _wilson(k: int, total: int) -> list[float]:
        if total <= 0:
            return [0.0, 0.0]
        z = 1.959963984540054
        p = k / total
        denom = 1.0 + z * z / total
        center = (p + z * z / (2 * total)) / denom
        half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denom
        return [round(max(0.0, center - half), 6), round(min(1.0, center + half), 6)]

    decomposed = {}
    for key in ("reach", "contact", "stable_grasp", "lift", "place"):
        count = sum(bool((row.get("details") or {}).get(key)) for row in per_env)
        decomposed[key] = {
            "count": count,
            "rate": round(count / n, 6) if n else 0.0,
            "wilson_95": _wilson(count, n),
        }
    strata: dict[str, dict[str, Any]] = {}
    for row in per_env:
        difficulty = str((row.get("details") or {}).get("difficulty") or "unknown")
        entry = strata.setdefault(difficulty, {"count": 0, "strict_success_count": 0})
        entry["count"] += 1
        entry["strict_success_count"] += int(bool(row.get("success")))
    for entry in strata.values():
        entry["strict_success_rate"] = round(
            entry["strict_success_count"] / entry["count"], 6
        )
        entry["wilson_95"] = _wilson(entry["strict_success_count"], entry["count"])

    return {
        "schema": "npa.sim2real.heldout_eval.v1",
        "source": source,
        "sim_backend": "isaac",
        "isaac_task": isaac_task,
        "policy_checkpoint": checkpoint_uri,
        "deployable_policy_eval": bool(checkpoint_uri),
        "success_summary": success_summary,
        "strict_success": {
            "distance_m": DEFAULT_SUCCESS_DIST_M,
            "count": strict_count,
            "episodes": n,
            "rate": round(strict_count / n, 6) if n else 0.0,
            "wilson_95": _wilson(strict_count, n),
        },
        "success_rate": round(strict_count / n, 6) if n else 0.0,
        "decomposed_metrics": decomposed,
        "per_difficulty": strata,
        "per_env": per_env,
    }


def read_generated_envs(envs_dir: str, *, limit: int = 0) -> list[dict[str, Any]]:
    """Read complete generated environment records without relabelling them.

    The envgen stage emits one record per generated env with a per-env ``seed``
    and scene composition. We use those seeds to drive the Isaac eval so the
    trained policy is tested on the generated env distribution (not just stock
    copies), and label results by the real generated ``env_id``.
    """

    path = Path(envs_dir) / "envs.jsonl"
    envs: list[dict[str, Any]] = []
    if not path.is_file():
        return envs
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        envs.append(rec)
        if limit and len(envs) >= limit:
            break
    return envs


def read_durable_generated_envs(
    envs_uri: str,
    *,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    """Hydrate one exact scenario-record object into the replacement controller.

    Directory/prefix discovery is deliberately forbidden here: validation and
    gold lineage must identify the exact immutable ``envs.jsonl`` object.
    """

    if not envs_uri.startswith("s3://") or envs_uri.endswith("/"):
        raise ValueError(
            "NPA_SIM2REAL_HELDOUT_ENVS_URI must be an exact s3:// object URI"
        )
    local_path = cache_dir / "envs.jsonl"
    client = StorageClient.from_environment(
        endpoint_url=_env("AWS_ENDPOINT_URL") or _env("S3_ENDPOINT_URL")
    )
    client.download_file(envs_uri, str(local_path))
    rows = read_generated_envs(str(cache_dir))
    if not rows:
        raise ValueError(f"durable scenario-record object is empty: {envs_uri}")
    return rows


def publish_eval_scenarios(
    rows: list[dict[str, Any]],
    *,
    destination_prefix: str,
    storage: StorageClient | None = None,
) -> dict[str, Any]:
    """Publish the exact selected eval set under its content digest.

    The Isaac process must not receive scenario distributions as command-line
    arguments: Linux limits each argument independently, so a moderately sized
    held-out set can fail before Python starts.  The content-addressed object is
    restart-safe, and the Isaac sibling verifies its bytes before evaluation.
    """

    if not rows:
        raise ValueError("evaluation scenario distribution must not be empty")
    if not destination_prefix.startswith("s3://"):
        raise ValueError("evaluation scenario destination must be an s3:// prefix")
    payload = (
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
        )
        + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    uri = f"{destination_prefix.rstrip('/')}/{digest}.jsonl"
    resolved_storage = storage or StorageClient.from_environment(
        endpoint_url=_env("AWS_ENDPOINT_URL") or _env("S3_ENDPOINT_URL")
    )
    with tempfile.TemporaryDirectory(prefix="npa-eval-scenarios-") as directory:
        path = Path(directory) / f"{digest}.jsonl"
        path.write_bytes(payload)
        resolved_storage.upload_file(str(path), uri)
    return {
        "uri": uri,
        "sha256": digest,
        "size_bytes": len(payload),
        "scenario_count": len(rows),
        "transport": "s3_sha256",
        "content_addressed": True,
    }


def select_stratified_eval_envs(
    rows: list[dict[str, Any]], *, count: int, split: str
) -> list[dict[str, Any]]:
    """Choose a deterministic balanced fixed validation/gold scenario set."""

    if count <= 0 or len(rows) < count:
        raise ValueError(f"{split} evaluation needs {count} rows; found {len(rows)}")
    buckets: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("easy", "medium", "hard")
    }
    for row in rows:
        difficulty = str(row.get("difficulty") or "")
        if difficulty not in buckets:
            raise ValueError(f"unsupported {split} difficulty: {difficulty!r}")
        buckets[difficulty].append(row)
    for difficulty, bucket in buckets.items():
        if not bucket:
            raise ValueError(f"{split} split contains no {difficulty} scenarios")
        bucket.sort(
            key=lambda row: hashlib.sha256(
                f"{split}:{row.get('scenario_config_digest')}".encode()
            ).hexdigest()
        )
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        for difficulty in ("easy", "medium", "hard"):
            if buckets[difficulty] and len(selected) < count:
                selected.append(buckets[difficulty].pop(0))
        if not any(buckets.values()) and len(selected) < count:
            break
    if len(selected) != count:
        raise ValueError(f"could not select {count} balanced {split} scenarios")
    digests = [str(row.get("scenario_config_digest") or "") for row in selected]
    if not all(digests) or len(set(digests)) != len(digests):
        raise ValueError(f"{split} evaluation rows need unique config digests")
    return selected


def per_env_from_distances(
    distances: list[float],
    *,
    success_dist_m: float,
    env_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    generated_envs: list[dict[str, Any]] | None = None,
    runtime_metrics: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert per-env final object-to-goal distances into scored per-env rows.

    score = clamp(1 - dist/(2*success_dist), 0, 1); success = dist < threshold.
    A genuine measurement of the trained policy, grounded in the task metric.
    When provided, rows are labelled by the GENERATED env_id/seed they came from.
    """

    rows: list[dict[str, Any]] = []
    for index, dist in enumerate(distances):
        d = max(0.0, float(dist))
        score = max(0.0, min(1.0, 1.0 - d / (2.0 * success_dist_m)))
        env_id = (
            env_ids[index]
            if env_ids and index < len(env_ids)
            else f"heldout-{index:04d}"
        )
        details: dict[str, Any] = {"object_goal_distance_m": round(d, 6)}
        if seeds and index < len(seeds):
            details["generated_env_seed"] = int(seeds[index])
        if generated_envs and index < len(generated_envs):
            generated = generated_envs[index]
            details.update(
                {
                    "generated_env_seed": int(generated.get("seed") or 0),
                    "difficulty": generated.get("difficulty"),
                    "scenario_config_digest": generated.get("scenario_config_digest"),
                    "applied_config_provenance": "isaac_runtime_reset_event",
                }
            )
        if runtime_metrics and index < len(runtime_metrics):
            details.update(runtime_metrics[index])
        metrics = (
            runtime_metrics[index]
            if runtime_metrics and index < len(runtime_metrics)
            else {}
        )
        stable_place = bool(
            metrics.get("placement_stable", metrics.get("place", False))
        )
        # A final 5 cm distance without a stable placement is not task success.
        # Runtime-free callers (unit/dry-run fixtures) retain distance-only behavior.
        strict_success = d < success_dist_m and (
            stable_place if runtime_metrics is not None else True
        )
        rows.append(
            {
                "env_id": env_id,
                "success": strict_success,
                "score": round(score, 6),
                "details": details,
            }
        )
    return rows


# In-Isaac rollout script (runs in the sibling Job). Defensive: tries the
# standard Isaac Lab + rsl_rl play API, derives per-env final object-to-goal
# distance, and writes per_env_distances.json. Verbose so the first run reveals
# the exact API if anything mismatches.
ISAAC_EVAL_SCRIPT = r"""
import hashlib, json, os, sys, traceback
import numpy as np
N = int(os.environ.get("EVAL_NUM_ENVS", "4"))
STEPS = int(os.environ.get("EVAL_MAX_STEPS", "300"))
TASK = os.environ["EVAL_TASK"]
CKPT = os.environ["EVAL_CKPT_LOCAL"]
OUT = os.environ["EVAL_OUT_JSON"]
SEED = int(os.environ.get("EVAL_SEED", "0"))  # generated-env seed (envgen envs.jsonl)
CAMERA_VIEWS = json.loads(os.environ.get("EVAL_CAMERA_VIEWS_JSON", "[]") or "[]")
CAPTURE_WIDTH = int(os.environ.get("EVAL_CAPTURE_WIDTH", "640"))
CAPTURE_HEIGHT = int(os.environ.get("EVAL_CAPTURE_HEIGHT", "480"))
CAPTURE_STRIDE = max(1, int(os.environ.get("EVAL_CAPTURE_STRIDE", "20")))
PNG_COMPRESS_LEVEL = int(os.environ.get("EVAL_PNG_COMPRESS_LEVEL", "3"))
CAPTURE_FPS = float(os.environ.get("EVAL_CAPTURE_FPS", "10"))
CKPT_URI = os.environ.get("EVAL_CKPT_URI", "").strip()
def checkpoint_provenance():
    if not os.path.isfile(CKPT):
        return {"uri": CKPT_URI, "sha256": "", "size_bytes": 0}
    h = hashlib.sha256()
    with open(CKPT, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return {"uri": CKPT_URI, "sha256": h.hexdigest(), "size_bytes": os.path.getsize(CKPT)}
def dump(distances, note, episodes=None, metrics=None, applied=None):
    json.dump({"object_goal_distances": list(distances), "note": note,
               "render_episodes": episodes or [],
               "per_env_metrics": metrics or [],
               "applied_scenarios": applied or {},
               "camera_views": [view["name"] for view in CAMERA_VIEWS],
               "camera_metadata": CAMERA_VIEWS,
               "capture": {"width": CAPTURE_WIDTH, "height": CAPTURE_HEIGHT,
                           "heldout_stride": CAPTURE_STRIDE,
                           "png_compress_level": PNG_COMPRESS_LEVEL, "fps": CAPTURE_FPS},
               "policy_checkpoint": checkpoint_provenance()},
              open(OUT, "w"))
    print("EVAL_WROTE", OUT, note, "episodes", len(episodes or []), flush=True)
try:
    from isaaclab.app import AppLauncher
    app = AppLauncher(
        headless=True,
        enable_cameras=True,
        kit_args=os.environ.get(
            "NPA_ISAAC_KIT_ARGS", "--portable-root /tmp/npa-isaac-kit"
        ),
    ).app
    import gymnasium as gym, torch
    import isaaclab_tasks  # noqa: F401  registers tasks
    # Opt-in BYO-robot variant: register a Lift variant that swaps in the customer
    # robot articulation and eval against it. No-op (TASK unchanged, byte-for-byte
    # stock behavior) when NPA_BYO_ROBOT_SPEC_JSON is unset or the spec is stock.
    if os.environ.get("NPA_BYO_ROBOT_SPEC_JSON"):
        sys.path.insert(0, os.environ.get("NPA_ROBOT_MODULE_DIR", "/tmp/evalwork"))
        try:
            import isaac_byo_robot_task as _robotmod
            _byo_task = _robotmod.register()
            if _byo_task:
                TASK = _byo_task
                print("EVAL_BYO_ROBOT_TASK", TASK, flush=True)
            else:
                raise RuntimeError(
                    "task/scenario contract did not register an evaluation task"
                )
        except Exception as _e:
            print("EVAL_BYO_ROBOT_REGISTER_FAILED", repr(_e), flush=True)
            traceback.print_exc()
            raise
    from isaaclab_tasks.utils import parse_env_cfg
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    except Exception:
        from omni.isaac.lab_rl.rsl_rl import RslRlVecEnvWrapper  # older layout
    from rsl_rl.runners import OnPolicyRunner
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
    # CUSTOM asset: override the manipuland USD so eval scores the policy on the
    # same custom object it trained on (physically simulated, not the stock cube).
    OBJECT_USD = os.environ.get("EVAL_OBJECT_USD", "").strip()
    if OBJECT_USD:
        try:
            env_cfg.scene.object.spawn.usd_path = OBJECT_USD
            print("EVAL_OBJECT_USD_APPLIED", OBJECT_USD, flush=True)
        except Exception as e:
            raise RuntimeError("could not apply task-contract object USD: %r" % (e,)) from e
    # Drive randomization from the GENERATED env seed so the trained policy is
    # tested on the envgen-produced env distribution, not stock defaults.
    # Zero is a valid and intentional fixed-validation seed.  Do not use a
    # truthiness guard here: skipping seed 0 makes otherwise fixed checkpoint
    # comparisons depend on Isaac's process-global RNG state.
    env_cfg.seed = SEED
    torch.manual_seed(SEED)
    np.random.seed(SEED % (2**32))
    print("EVAL_SEED_APPLIED", SEED, flush=True)
    # Capture synchronized primary, side, and overhead views. Isaac Lab's
    # ``world`` camera convention looks along +X; the orchestrator serializes
    # validated wxyz poses into CAMERA_VIEWS. ``heldout_cam`` remains the primary
    # sensor key for backward compatibility with existing real-run tooling.
    def _camera_key(name):
        return "heldout_cam" if name == "primary" else "heldout_cam_" + name
    for view in CAMERA_VIEWS:
        setattr(
            env_cfg.scene,
            _camera_key(view["name"]),
            TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/heldout_cam_" + view["name"],
                offset=TiledCameraCfg.OffsetCfg(
                    pos=tuple(view["position"]),
                    rot=tuple(view["rotation"]),
                    convention="world",
                ),
                data_types=["rgb", "distance_to_image_plane"],
                width=CAPTURE_WIDTH,
                height=CAPTURE_HEIGHT,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=float(view.get("focal_length_mm", 24.0)),
                    horizontal_aperture=float(view.get("horizontal_aperture_mm", 20.955)),
                    clipping_range=(0.05, 20.0),
                ),
            ),
        )
    print("EVAL_CAMERA_VIEWS", [view["name"] for view in CAMERA_VIEWS], flush=True)
    env = gym.make(TASK, cfg=env_cfg)
    if OBJECT_USD:
        got_object_usd = getattr(env.unwrapped.scene["object"].cfg.spawn, "usd_path", None)
        if got_object_usd != OBJECT_USD:
            raise RuntimeError("evaluation object USD mismatch; refusing stock fallback")
    env = RslRlVecEnvWrapper(env)
    # Load the COMPLETE rsl_rl agent cfg from the task registry (has save_interval,
    # network dims, etc.) — a hand-built cfg is missing keys OnPolicyRunner needs.
    agent_cfg = None
    for loader in ("isaaclab_tasks.utils", "omni.isaac.lab_tasks.utils"):
        try:
            mod = __import__(loader, fromlist=["load_cfg_from_registry"])
            agent_cfg = mod.load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
            print("loaded agent cfg via", loader, flush=True)
            break
        except Exception as e:
            print("cfg loader", loader, "failed:", repr(e), flush=True)
    if agent_cfg is None:
        raise RuntimeError("could not load rsl_rl_cfg_entry_point for task")
    acfg = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else dict(agent_cfg)
    print("AGENT_CFG_KEYS", sorted(acfg.keys()), flush=True)
    runner = OnPolicyRunner(env, acfg, log_dir=None, device="cuda:0")
    runner.load(CKPT)
    policy = runner.get_inference_policy(device="cuda:0")
    # The ACTUAL env count is the single source of truth for per-env sizing.
    realN = int(getattr(env.unwrapped, "num_envs", N) or N)
    # Reset FIRST to force a fully-batched [realN, obs_dim] observation. Calling
    # get_observations() before any reset can hand back a stale/collapsed
    # single-env buffer — that batch-1-vs-num_envs mismatch is what pinned earlier
    # eval runs to num_envs=1. Reset gives a properly batched obs for N>1.
    try:
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    except Exception:
        obs, _ = env.get_observations()
    print("OBS_TYPE", type(obs).__name__, "realN", realN, flush=True)
    N = realN

    def _to_batched(v):
        # Ensure a group tensor is [realN, feat]. A 1-D tensor is either a single
        # env's obs (-> [1, feat]) or a flattened realN*feat batch (-> [realN, feat]).
        if not torch.is_tensor(v) or v.ndim != 1:
            return v
        n = int(v.shape[0])
        if realN > 1 and n % realN == 0:
            return v.reshape(realN, n // realN)
        return v.unsqueeze(0)
    def _batched_obs(o):
        # rsl_rl act_inference does obs[group] internally, so pass the WHOLE obs
        # (Tensor)Dict — but ensure each group tensor is [realN, feat].
        if torch.is_tensor(o):
            return _to_batched(o)
        try:
            for k in list(o.keys()):
                o[k] = _to_batched(o[k])
        except Exception:
            pass
        return o
    def _policy_tensor(o):
        if torch.is_tensor(o):
            return o
        for k in ("policy", "obs", "policy_obs"):
            try:
                v = o[k]
                if torch.is_tensor(v):
                    return v
            except Exception:
                pass
        return o
    obs = _batched_obs(obs)
    _pt = _policy_tensor(obs)
    _pb = int(_pt.shape[0]) if torch.is_tensor(_pt) and _pt.ndim >= 1 else realN
    # N stays the true env count; only WARN if the policy obs batch disagrees so a
    # genuine multi-env mismatch is visible in logs rather than silently collapsing.
    print("STEP0 policy_obs_shape", tuple(getattr(_pt, "shape", ())),
          "env.num_envs", getattr(env.unwrapped, "num_envs", "?"), "N", N,
          "policy_batch", _pb, flush=True)
    if _pb != N:
        print("WARN policy_obs_batch %d != env_count %d" % (_pb, N), flush=True)
    # Per-env render dirs (labelled by generated env_id when provided).
    import json as _json
    env_ids = _json.loads(os.environ.get("EVAL_ENV_IDS", "[]") or "[]")
    rend_root = os.environ.get("EVAL_RENDERS_DIR", "/tmp/evalwork/renders")
    def _env_id(i):
        return env_ids[i] if i < len(env_ids) else f"heldout-{i:04d}"
    frame_names = {
        i: {view["name"]: [] for view in CAMERA_VIEWS}
        for i in range(N)
    }
    frame_metadata = {
        i: {view["name"]: [] for view in CAMERA_VIEWS}
        for i in range(N)
    }
    try:
        from PIL import Image as _PILImage
        _have_pil = True
    except Exception:
        _have_pil = False
    # GPU depth -> colored point clouds (world frame) for env 0. Capturing every
    # angle fills the rear/side occlusions that made the original single-camera
    # point cloud look incomplete when the Rerun 3D view was orbited.
    def _pc_fn():
        for mod in ("isaaclab.sensors.camera.utils", "omni.isaac.lab.sensors.camera.utils"):
            try:
                m = __import__(mod, fromlist=["create_pointcloud_from_rgbd"])
                return m.create_pointcloud_from_rgbd
            except Exception:
                continue
        return None
    _create_pc = _pc_fn()
    _pc_count = {view["name"]: 0 for view in CAMERA_VIEWS}
    def capture_pointcloud():
        if _create_pc is None:
            return
        for view in CAMERA_VIEWS:
            name = view["name"]
            try:
                cam = env.unwrapped.scene[_camera_key(name)]
                intr = cam.data.intrinsic_matrices
                depth = cam.data.output.get("distance_to_image_plane")
                rgb = cam.data.output["rgb"]
                if depth is None:
                    continue
                pts, cols = _create_pc(
                    intrinsic_matrix=intr[0], depth=depth[0], rgb=rgb[0],
                    position=cam.data.pos_w[0], orientation=cam.data.quat_w_ros[0],
                    device="cuda:0", num_channels=3,
                )
                xyz = pts.detach().cpu().numpy().reshape(-1, 3).astype(np.float32)
                col = cols.detach().cpu().numpy().reshape(-1, 3)
                if col.dtype != np.uint8:
                    col = (np.clip(col, 0.0, 1.0) * 255).astype(np.uint8) if col.max() <= 1.0 else col.astype(np.uint8)
                good = np.isfinite(xyz).all(axis=1)
                xyz, col = xyz[good], col[good]
                try:
                    cam_pos = cam.data.pos_w[0].detach().cpu().numpy().reshape(3)
                    range_m = float(os.environ.get("NPA_SIM2REAL_POINTCLOUD_RANGE_M", "5.0"))
                    near = np.linalg.norm(xyz - cam_pos[None, :], axis=1) <= range_m
                    if int(near.sum()) >= 200:
                        xyz, col = xyz[near], col[near]
                except Exception as _pe:
                    print("pc_clip_err", name, repr(_pe), flush=True)
                if xyz.shape[0] > 6000:
                    sel = np.random.default_rng(0).choice(xyz.shape[0], 6000, replace=False)
                    xyz, col = xyz[sel], col[sel]
                if xyz.shape[0] == 0:
                    continue
                pc_dir = os.path.join(rend_root, "_pointcloud", _env_id(0), name)
                os.makedirs(pc_dir, exist_ok=True)
                np.savez_compressed(
                    os.path.join(pc_dir, f"cloud-{_pc_count[name]:04d}.npz"),
                    xyz=xyz,
                    rgb=col,
                )
                _pc_count[name] += 1
            except Exception as e:
                print("pc_capture_err", name, repr(e), flush=True)
    def capture(step, eligible=None):
        if not _have_pil:
            return
        for view in CAMERA_VIEWS:
            view_name = view["name"]
            try:
                rgb = env.unwrapped.scene[_camera_key(view_name)].data.output["rgb"]
                arr = rgb.detach().cpu().numpy()
                for i in range(min(N, arr.shape[0])):
                    if eligible is not None and not bool(eligible[i]):
                        continue
                    d = os.path.join(rend_root, _env_id(i)); os.makedirs(d, exist_ok=True)
                    index = len(frame_names[i][view_name])
                    name = (
                        f"camera-{index:04d}.png"
                        if view_name == "primary"
                        else f"camera-{view_name}-{index:04d}.png"
                    )
                    _PILImage.fromarray(arr[i, :, :, :3].astype(np.uint8)).save(
                        os.path.join(d, name), compress_level=PNG_COMPRESS_LEVEL
                    )
                    frame_names[i][view_name].append(name)
                    frame_metadata[i][view_name].append({
                        "path": name,
                        "view_name": view_name,
                        "frame_index": index,
                        "sim_step": int(step),
                        "timestamp_seconds": round(float(step) / CAPTURE_FPS, 6),
                        "episode_id": _env_id(i),
                        "isaac_env_index": i,
                        "width": CAPTURE_WIDTH,
                        "height": CAPTURE_HEIGHT,
                        "policy_checkpoint": CKPT_URI,
                    })
            except Exception as e:
                print("capture_err", view_name, repr(e), flush=True)
        capture_pointcloud()
    min_dist = np.full(N, 1e9)
    final_dist = np.full(N, 1e9)
    reach = np.zeros(N, dtype=bool)
    contact = np.zeros(N, dtype=bool)
    grasp = np.zeros(N, dtype=bool)
    lift = np.zeros(N, dtype=bool)
    place = np.zeros(N, dtype=bool)
    final_place = np.zeros(N, dtype=bool)
    stable_grasp_steps = np.zeros(N, dtype=np.int64)
    stable_place_steps = np.zeros(N, dtype=np.int64)
    max_stable_place_steps = np.zeros(N, dtype=np.int64)
    min_speed_in_strict_basin = np.full(N, 1e9)
    termination = np.array(["max_steps"] * N, dtype=object)
    completed = np.zeros(N, dtype=bool)
    initial_obj_z = None
    for _step in range(STEPS):
        # Isaac auto-resets done environments inside env.step(). Preserve the
        # last sample from the evaluated episode so the returned reset state can
        # never replace terminal metrics or render lineage.
        prior = {
            "min_dist": min_dist.copy(),
            "final_dist": final_dist.copy(),
            "reach": reach.copy(),
            "contact": contact.copy(),
            "grasp": grasp.copy(),
            "lift": lift.copy(),
            "place": place.copy(),
            "final_place": final_place.copy(),
            "stable_grasp_steps": stable_grasp_steps.copy(),
            "stable_place_steps": stable_place_steps.copy(),
            "max_stable_place_steps": max_stable_place_steps.copy(),
            "min_speed_in_strict_basin": min_speed_in_strict_basin.copy(),
        }
        with torch.inference_mode():
            actions = policy(_batched_obs(obs))
        if _step == 0:
            print("STEP0 act_shape", tuple(getattr(actions, "shape", ())), flush=True)
        if hasattr(actions, "ndim") and actions.ndim == 1:
            actions = actions.reshape(N, -1)
        obs, _, dones, extras = env.step(actions)
        try:
            done_np = dones.detach().cpu().numpy().astype(bool)
        except Exception:
            done_np = np.zeros(N, dtype=bool)
        from npa.workflows.sim2real.byo_isaac_eval import first_episode_masks
        active, newly_terminal, completed = first_episode_masks(completed, done_np)
        if _step % CAPTURE_STRIDE == 0:
            capture(_step, active & ~newly_terminal)
        # object-to-goal distance: prefer an explicit metric, else infer.
        d = None
        log = (extras or {}).get("log") or {}
        for k, v in log.items():
            if "object" in k.lower() and ("dist" in k.lower() or "error" in k.lower()):
                try:
                    d = float(v);
                except Exception:
                    d = None
                break
        try:
            uenv = env.unwrapped
            if hasattr(uenv, "command_manager"):
                cmd = uenv.command_manager.get_command("object_pose")
                obj = uenv.scene["object"].data.root_pos_w[:, :3]
                goal = cmd[:, :3] + uenv.scene.env_origins[:, :3]
                per_t = torch.linalg.norm(obj - goal, dim=1)
                per = per_t.detach().cpu().numpy()
                final_dist = np.where(active, per, final_dist)
                min_dist = np.where(active, np.minimum(min_dist, per), min_dist)
                ee = uenv.scene["ee_frame"].data.target_pos_w[..., 0, :]
                ee_dist = torch.linalg.norm(obj - ee, dim=1).detach().cpu().numpy()
                reach |= active & (ee_dist < 0.05)
                if initial_obj_z is None:
                    initial_obj_z = obj[:, 2].detach().cpu().numpy()
                height = obj[:, 2].detach().cpu().numpy() - initial_obj_z
                lift |= active & (height >= 0.05)
                contact_now = ee_dist < 0.035
                try:
                    forces = uenv.scene["object_contact"].data.net_forces_w_history
                    contact_now = np.linalg.norm(
                        forces.detach().cpu().numpy(), axis=-1
                    ).reshape(N, -1).max(axis=1) > 1.0e-3
                except Exception:
                    pass
                contact |= active & contact_now
                stable_grasp_steps = np.where(
                    active,
                    np.where(contact_now & (height > 0.015), stable_grasp_steps + 1, 0),
                    stable_grasp_steps,
                )
                grasp |= active & (stable_grasp_steps >= 3)
                try:
                    obj_speed = torch.linalg.norm(
                        uenv.scene["object"].data.root_lin_vel_w[:, :3], dim=1
                    ).detach().cpu().numpy()
                except Exception:
                    obj_speed = np.full(N, 1.0)
                in_strict_basin = per < 0.05
                min_speed_in_strict_basin = np.where(
                    active & in_strict_basin,
                    np.minimum(min_speed_in_strict_basin, obj_speed),
                    min_speed_in_strict_basin,
                )
                stable_place_steps = np.where(
                    active,
                    np.where(
                        in_strict_basin & (obj_speed < 0.03),
                        stable_place_steps + 1,
                        0,
                    ),
                    stable_place_steps,
                )
                max_stable_place_steps = np.maximum(
                    max_stable_place_steps, stable_place_steps
                )
                final_place = stable_place_steps >= 3
                place |= active & final_place
                if np.any(newly_terminal):
                    # The state above is already the reset state. Restore the
                    # exact pre-step sample for this episode and seal it.
                    min_dist[newly_terminal] = prior["min_dist"][newly_terminal]
                    final_dist[newly_terminal] = prior["final_dist"][newly_terminal]
                    reach[newly_terminal] = prior["reach"][newly_terminal]
                    contact[newly_terminal] = prior["contact"][newly_terminal]
                    grasp[newly_terminal] = prior["grasp"][newly_terminal]
                    lift[newly_terminal] = prior["lift"][newly_terminal]
                    place[newly_terminal] = prior["place"][newly_terminal]
                    stable_grasp_steps[newly_terminal] = prior[
                        "stable_grasp_steps"
                    ][newly_terminal]
                    stable_place_steps[newly_terminal] = prior[
                        "stable_place_steps"
                    ][newly_terminal]
                    max_stable_place_steps[newly_terminal] = prior[
                        "max_stable_place_steps"
                    ][newly_terminal]
                    min_speed_in_strict_basin[newly_terminal] = prior[
                        "min_speed_in_strict_basin"
                    ][newly_terminal]
                    final_place[newly_terminal] = prior["final_place"][
                        newly_terminal
                    ]
                    termination[newly_terminal] = "task_or_timeout"
                continue
        except Exception:
            pass
        if d is not None:
            min_dist = np.where(
                active,
                np.minimum(min_dist, np.full(N, d)),
                min_dist,
            )
    capture(STEPS, ~completed)  # final frame only for a still-live first episode
    episodes = [
        {
            "env_id": _env_id(i),
            "frames": frame_names[i].get("primary", []),
            "camera_views": frame_names[i],
            "camera_frame_metadata": frame_metadata[i],
        }
        for i in range(N)
        if any(frame_names[i].values())
    ]
    import isaac_scenario_task as _scenarios
    applied = _scenarios.runtime_audit(env.unwrapped)
    metrics = [
        {
            "closest_object_goal_distance_m": float(min_dist[i] if min_dist[i] < 1e8 else 0.5),
            "final_object_goal_distance_m": float(final_dist[i] if final_dist[i] < 1e8 else 0.5),
            "reach": bool(reach[i]), "contact": bool(contact[i]),
            "stable_grasp": bool(grasp[i]), "lift": bool(lift[i]),
            "place": bool(place[i]), "placement_stable": bool(final_place[i]),
            "max_consecutive_strict_stable_steps": int(max_stable_place_steps[i]),
            "min_speed_in_strict_basin_mps": (
                float(min_speed_in_strict_basin[i])
                if min_speed_in_strict_basin[i] < 1e8
                else None
            ),
            "terminal_snapshot": "first_episode_last_pre_reset",
            "termination_reason": (
                "success" if final_place[i] else str(termination[i])
            ),
        }
        for i in range(N)
    ]
    # Strict success uses FINAL stable placement distance. Closest distance is a
    # diagnostic only and can never promote a policy that moved away again.
    dump([m["final_object_goal_distance_m"] for m in metrics], "rollout_ok", episodes, metrics, applied)
except Exception as e:
    traceback.print_exc()
    dump([0.5]*N, "rollout_failed:%s" % e)
# With enable_cameras the Isaac app hangs on exit (even app.close() blocks), so the
# post-script bash upload never runs. Upload distances + renders HERE from boto3,
# then hard-exit the process so nothing hangs.
try:
    import boto3
    from urllib.parse import urlparse
    s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)
    ou = urlparse(os.environ["EVAL_OUT_S3"])
    s3.upload_file(OUT, ou.netloc, ou.path.lstrip("/"))
    print("UPLOADED_DISTANCES", os.environ["EVAL_OUT_S3"], flush=True)
    ru = urlparse(os.environ.get("EVAL_RENDERS_S3", ""))
    if ru.netloc:
        import glob
        base = ru.path.lstrip("/").rstrip("/")
        n = 0
        for pat in ("**/*.png", "**/*.npz"):
            for p in glob.glob(os.environ["EVAL_RENDERS_DIR"] + "/" + pat, recursive=True):
                rel = os.path.relpath(p, os.environ["EVAL_RENDERS_DIR"])
                s3.upload_file(p, ru.netloc, base + "/" + rel); n += 1
        print("UPLOADED_RENDERS", n, os.environ.get("EVAL_RENDERS_S3"), flush=True)
    print("BYO_EVAL_DONE", flush=True)
except Exception as _e:
    print("inproc_upload_err", repr(_e), flush=True)
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)
"""


def _isaac_eula_env_entries() -> list[dict[str, str]]:
    """Kubernetes ``env`` entries carrying the operator's NVIDIA licence acceptance.

    The Isaac image ships no Isaac Sim and refuses to fetch it (exit 78) unless
    OMNI_KIT_ACCEPT_EULA and ISAACSIM_ACCEPT_EULA are set. These jobs invoke
    /isaac-sim/python.sh, so without forwarding they cannot run at all.

    Read from the submitting process's environment and never defaulted to "YES": the
    operator driving the pipeline is the one consenting, and hardcoding acceptance here
    would put us in the position of accepting on their behalf. Unset stays unset, and the
    job then fails with the bootstrap's actionable refusal instead of silently consenting.
    """

    return [
        {"name": name, "value": os.environ[name]}
        for name in ("OMNI_KIT_ACCEPT_EULA", "ISAACSIM_ACCEPT_EULA")
        if os.environ.get(name)
    ]


def build_isaac_eval_job_manifest(
    *,
    job_name: str,
    run_id: str,
    image: str,
    task: str,
    num_envs: int,
    checkpoint_uri: str,
    per_env_s3_uri: str,
    s3_endpoint: str,
    namespace: str,
    service_account: str,
    gpu_product: str,
    gpu_resource: str = "nvidia.com/gpu",
    image_pull_policy: str = "Always",
    seed: int = 0,
    object_usd: str = "",
    env_ids_json: str = "[]",
    renders_s3_prefix: str = "",
    robot_spec: dict[str, Any] | None = None,
    robot_usd_uri: str = "",
    task_config: dict[str, Any] | None = None,
    camera_views: str = "",
    capture: dict[str, Any] | None = None,
    scenarios_jsonl: str = "",
    scenarios_uri: str = "",
    scenarios_sha256: str = "",
) -> dict[str, Any]:
    """Isaac eval Job: download checkpoint, roll trained policy, upload distances.

    ``seed`` (from the generated env spec) drives the env randomization so the
    policy is evaluated on the envgen-produced env distribution. ``object_usd``
    overrides the manipuland so eval scores the policy on the same CUSTOM asset
    it was trained on. RGB frames of the (custom) object are rendered and, when
    ``renders_s3_prefix`` is set, uploaded for Rerun visualization.
    """

    import shlex as _shlex
    import json as _json

    capture = dict(capture or capture_settings({}))
    scenarios_uri = scenarios_uri.strip()
    scenarios_sha256 = scenarios_sha256.strip().lower()
    if scenarios_uri and scenarios_jsonl:
        raise ValueError("provide scenarios_uri or scenarios_jsonl, not both")
    if scenarios_uri and not scenarios_uri.startswith("s3://"):
        raise ValueError("scenarios_uri must be an s3:// URI")
    if scenarios_uri and not re.fullmatch(r"[0-9a-f]{64}", scenarios_sha256):
        raise ValueError("scenarios_uri requires its exact SHA-256 digest")
    if (
        scenarios_jsonl
        and len(scenarios_jsonl.encode("utf-8")) > _MAX_EMBEDDED_SCENARIOS_BYTES
    ):
        raise ValueError(
            "large evaluation scenario distributions require scenarios_uri; "
            "refusing an oversized process argument"
        )

    # Opt-in BYO-robot eval uses the module baked into the exact Isaac image and
    # passes the spec via NPA_BYO_ROBOT_SPEC_JSON.
    # Empty when robot_spec is None -> byte-for-byte the stock eval.
    robot_block = ""
    if robot_spec:
        spec_json = _json.dumps(robot_spec, sort_keys=True)
        usd_dest = str(robot_spec.get("usd_path") or "").strip()
        robot_stage = ""
        if robot_usd_uri and usd_dest:
            robot_stage = (
                '"$PY" -m npa.workflows.sim2real.isaac_job_io download '
                f"--uri {_shlex.quote(robot_usd_uri)} "
                f"--destination {_shlex.quote(usd_dest)}\n"
            )
        # Matched task config (placement / gripper targets) so the held-out eval
        # rolls the policy on the SAME scaled task distribution it trained on
        # (object_init_range + goal_range). register() picks it up via
        # task_config_from_env(). Empty -> stock placement (Franka path unchanged).
        task_cfg_export = ""
        if task_config:
            task_cfg_export = (
                "export NPA_BYO_TASK_CONFIG_JSON="
                + _shlex.quote(_json.dumps(task_config, sort_keys=True))
                + "\n"
            )
        robot_block = (
            robot_stage
            + "export NPA_ROBOT_MODULE_DIR=/opt/npa/isaac-runtime\n"
            + "export NPA_BYO_ROBOT_SPEC_JSON="
            + _shlex.quote(spec_json)
            + "\n"
            + task_cfg_export
        )
    scenario_block = ""
    if scenarios_uri:
        scenario_block = (
            '"$PY" -m npa.workflows.sim2real.isaac_job_io download '
            f"--uri {_shlex.quote(scenarios_uri)} "
            "--destination /tmp/evalwork/scenarios.jsonl "
            f"--sha256 {_shlex.quote(scenarios_sha256)}\n"
        )
    elif scenarios_jsonl:
        encoded_scenarios = base64.b64encode(scenarios_jsonl.encode()).decode()
        scenario_block = (
            '"$PY" -m npa.workflows.sim2real.isaac_job_io write-base64 '
            f"--payload {_shlex.quote(encoded_scenarios)} "
            "--destination /tmp/evalwork/scenarios.jsonl\n"
        )
    if scenario_block:
        scenario_block += (
            "export NPA_SIM2REAL_SCENARIOS_JSONL=/tmp/evalwork/scenarios.jsonl\n"
            "export NPA_SIM2REAL_TASK_CONTRACT_DIGEST="
            + _shlex.quote(_env("NPA_SIM2REAL_TASK_CONTRACT_DIGEST"))
            + "\n"
            "export NPA_SIM2REAL_SCENARIO_ROTATE_ON_RESET=0\n"
        )

    render_upload = ""
    if renders_s3_prefix:
        render_upload = (
            '"$PY" -m npa.workflows.sim2real.isaac_job_io upload-tree '
            "--root /tmp/evalwork/renders "
            f"--uri {_shlex.quote(renders_s3_prefix)}\n"
        )
    script = (
        "set -euo pipefail\n"
        "exec > >(tee -a /tmp/byo-eval.log) 2>&1\n"
        "PY=/isaac-sim/python.sh\n"
        '[ -x "$PY" ] || { echo "MISSING_PINNED_ISAAC_RUNTIME"; exit 127; }\n'
        '"$PY" -m npa.workflows.sim2real.runtime_attestation\n'
        "mkdir -p /tmp/evalwork/renders; cd /tmp/evalwork\n"
        f'export EVAL_TASK="{task}" EVAL_NUM_ENVS="{num_envs}" EVAL_SEED="{seed}" '
        f'EVAL_OBJECT_USD="{object_usd}" EVAL_ENV_IDS={_shlex.quote(env_ids_json)} '
        f"EVAL_CAMERA_VIEWS_JSON={_shlex.quote(camera_views or camera_views_json())} "
        f'EVAL_CAPTURE_WIDTH="{capture["width"]}" '
        f'EVAL_CAPTURE_HEIGHT="{capture["height"]}" '
        f'EVAL_CAPTURE_STRIDE="{capture["heldout_stride"]}" '
        f'EVAL_PNG_COMPRESS_LEVEL="{capture["png_compress_level"]}" '
        f'EVAL_CAPTURE_FPS="{capture["fps"]}" '
        f"EVAL_CKPT_URI={_shlex.quote(checkpoint_uri)} "
        f"EVAL_OUT_S3={_shlex.quote(per_env_s3_uri)} "
        f"EVAL_RENDERS_S3={_shlex.quote(renders_s3_prefix)} "
        "EVAL_RENDERS_DIR=/tmp/evalwork/renders "
        "EVAL_CKPT_LOCAL=/tmp/evalwork/policy.pt "
        "EVAL_OUT_JSON=/tmp/evalwork/per_env_distances.json\n"
        '"$PY" -m npa.workflows.sim2real.isaac_job_io download '
        f"--uri {_shlex.quote(checkpoint_uri)} "
        "--destination /tmp/evalwork/policy.pt\n"
        + robot_block
        + scenario_block
        + '"$PY" /opt/npa/isaac-runtime/isaac_eval.py\n'
        '"$PY" -m npa.workflows.sim2real.isaac_job_io upload '
        "--source /tmp/evalwork/per_env_distances.json "
        f"--uri {_shlex.quote(per_env_s3_uri)}\n"
        + render_upload
        + 'echo "BYO_EVAL_DONE"\n'
    )
    command, args = compressed_bash_launch(script)
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {"app": "sim2real-byo-isaac-eval", "run-id": run_id},
            "annotations": {
                "sim2real.npa.dev/scenarios-uri": scenarios_uri,
                "sim2real.npa.dev/scenarios-sha256": scenarios_sha256,
            }
            if scenarios_uri
            else {},
        },
        "spec": {
            "backoffLimit": 1,
            "template": {
                "metadata": {
                    "labels": {"app": "sim2real-byo-isaac-eval", "run-id": run_id}
                },
                "spec": {
                    "restartPolicy": "Never",
                    "serviceAccountName": service_account,
                    # Match the hardened image's uid/gid 1000 runtime contract. The
                    # image now owns its workspace and installs a traversable Isaac
                    # shim, so retained standalone jobs no longer need a root override.
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "runAsGroup": 1000,
                        "fsGroup": 1000,
                        "fsGroupChangePolicy": "OnRootMismatch",
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "imagePullSecrets": [
                        {"name": "agent-sa"},
                        {"name": "ngc-nvcr-imagepullsecret"},
                        {"name": "npa-nebius-registry"},
                    ],
                    "containers": [
                        {
                            "name": "eval",
                            "image": image,
                            "imagePullPolicy": image_pull_policy,
                            "securityContext": {
                                "runAsNonRoot": True,
                                "runAsUser": 1000,
                                "runAsGroup": 1000,
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "resources": {
                                "limits": {gpu_resource: "1"},
                                "requests": {gpu_resource: "1"},
                            },
                            "envFrom": [
                                {"secretRef": {"name": "hf-ngc-tokens"}},
                                {"secretRef": {"name": "npa-storage-credentials"}},
                            ],
                            "env": [
                                {"name": "AWS_ENDPOINT_URL", "value": s3_endpoint},
                                {
                                    "name": "NPA_SIM2REAL_SOURCE_SHA",
                                    "value": os.environ.get(
                                        "NPA_SIM2REAL_SOURCE_SHA", ""
                                    ),
                                },
                                {
                                    "name": "NPA_SIM2REAL_RUNTIME_IMAGE",
                                    "value": image.removeprefix("docker:"),
                                },
                            ]
                            + _isaac_eula_env_entries(),
                            "command": command,
                            "args": args,
                        }
                    ],
                    "nodeSelector": {f"{gpu_resource}.product": gpu_product},
                },
            },
        },
    }


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _download_json(uri: str) -> dict[str, Any]:
    import boto3
    from urllib.parse import urlparse

    u = urlparse(uri)
    s3 = boto3.client("s3", endpoint_url=_env("AWS_ENDPOINT_URL") or None)
    local = "/tmp/byo_eval_per_env.json"
    s3.download_file(u.netloc, u.path.lstrip("/"), local)
    return json.loads(Path(local).read_text())


def run_isaac_eval_job(
    run_id: str,
    *,
    checkpoint_uri: str,
    num_envs: int,
    generated_envs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    global _LAST_GPU_PROVENANCE, _SCENARIO_INPUT_PROVENANCE

    task = _env("NPA_SIM2REAL_ISAAC_TASK", DEFAULT_ISAAC_TASK)
    image = _env("NPA_SIM2REAL_ISAAC_IMAGE") or _env("ISAAC_IMAGE")
    from npa.workflows.sim2real.k8s_components import _image_pull_policy

    bucket = (
        _env("NPA_SIM2REAL_BUCKET")
        or _env("S3_BUCKET")
        or _env("NPA_SIM2REAL_S3_BUCKET")
    )
    s3_prefix = _env("NPA_SIM2REAL_PREFIX", "sim2real-b").strip("/")
    namespace = _env("NPA_SIM2REAL_K8S_NAMESPACE", "default")
    sa = _env("NPA_SIM2REAL_K8S_SERVICE_ACCOUNT", "agent-sa")
    gpu_product = _env("NPA_SIM2REAL_K8S_GPU_PRODUCT", DEFAULT_GPU_PRODUCT)
    success_dist = float(
        _env("NPA_BYO_ISAAC_SUCCESS_DIST_M", str(DEFAULT_SUCCESS_DIST_M))
        or DEFAULT_SUCCESS_DIST_M
    )
    timeout_s = int(_env("NPA_BYO_ISAAC_JOB_TIMEOUT_S", "0") or 0)
    capture = capture_settings()
    from npa.workflows.sim2real.byo_isaac_trainer import artifact_tag, k8s_job_name

    eval_tag = artifact_tag(_env("NPA_SIM2REAL_EVAL_TAG"))
    job_name = k8s_job_name("s2r-byo-isaac-eval", run_id, eval_tag)
    per_env_uri = (
        f"s3://{bucket}/{s3_prefix}/{run_id}/byo-eval/{job_name}/per_env_distances.json"
    )

    gen = generated_envs or []
    env_ids = [e["env_id"] for e in gen] or None
    seeds = [e["seed"] for e in gen] or None
    seed = int(_env("NPA_SIM2REAL_SEED", "0") or 0)
    if len(gen) != num_envs:
        raise RuntimeError(
            f"evaluation requires {num_envs} complete scenario records; got {len(gen)}"
        )
    digests = [str(row.get("scenario_config_digest") or "") for row in gen]
    if not all(digests) or len(set(digests)) != len(digests):
        raise RuntimeError("evaluation scenarios lack unique config digests")
    _SCENARIO_INPUT_PROVENANCE = publish_eval_scenarios(
        gen,
        destination_prefix=(
            f"s3://{bucket}/{s3_prefix}/{run_id}/byo-eval/{job_name}/scenario-input"
        ),
    )
    # Eval must spawn the SAME manipuland the policy trained on (default: the
    # proven rigid-ready USD, not the stock primitive cube).
    from npa.workflows.sim2real.byo_isaac_trainer import resolve_object_usd

    object_usd = resolve_object_usd(_env("NPA_BYO_ISAAC_OBJECT_USD"))
    renders_prefix = f"s3://{bucket}/{s3_prefix}/{run_id}/byo-eval/{job_name}/renders"

    # Opt-in BYO-robot eval path (guarded; default unchanged): eval the policy on
    # the same robot-swapped Lift variant it was trained on. Reuses the trainer's
    # spec resolution + payload so train and eval agree on the variant.
    robot_spec_dict = None
    robot_usd_uri = ""
    task_config_dict = None
    if _env("NPA_BYO_ROBOT_TASK") == "1":
        from npa.workflows.sim2real import byo_isaac_trainer as _trainer
        from npa.workflows.sim2real import isaac_byo_robot_task as _robotmod

        spec = _trainer._resolve_byo_robot_spec()
        usd_dest = ""
        if (
            spec is not None
            and str(getattr(spec, "robot_source", "")) != "stock_franka"
        ):
            robot_uri = str(getattr(spec, "robot_uri", "") or "")
            if robot_uri.startswith("s3://"):
                robot_usd_uri = robot_uri
                usd_dest = _trainer.ROBOT_USD_CONTAINER_PATH
            elif robot_uri:
                usd_dest = robot_uri
        robot_spec_dict = _trainer.robot_spec_payload(spec, usd_container_path=usd_dest)
        # Matched robot-aware task config (object scale / placement / gripper targets)
        # so the held-out eval spawns and rolls the policy on the SAME scaled task the
        # trainer used — most importantly the SAME manipuland size. Without this a
        # shrunk-object policy is scored on the stock ~5 cm cube and reports a false
        # zero. register() in the eval sibling consumes it via task_config_from_env().
        task_config_dict = _robotmod.task_config_from_env()
        print(
            f"byo_isaac_eval: BYO-ROBOT eval path "
            f"{'ON' if robot_spec_dict else 'OFF (no spec)'} -> {robot_spec_dict}",
            flush=True,
        )
        print(
            f"byo_isaac_eval: BYO task config "
            f"{'-> ' + json.dumps(task_config_dict, sort_keys=True) if task_config_dict else 'none'}",
            flush=True,
        )
    if robot_spec_dict is None:
        robot_spec_dict = {"robot_source": "stock_franka", "name": "franka"}

    manifest = build_isaac_eval_job_manifest(
        job_name=job_name,
        run_id=run_id,
        image=image,
        task=task,
        num_envs=num_envs,
        checkpoint_uri=checkpoint_uri,
        per_env_s3_uri=per_env_uri,
        s3_endpoint=_env("AWS_ENDPOINT_URL"),
        namespace=namespace,
        service_account=sa,
        gpu_product=gpu_product,
        image_pull_policy=_image_pull_policy(image),
        seed=seed,
        object_usd=object_usd,
        env_ids_json=json.dumps([e["env_id"] for e in gen]),
        renders_s3_prefix=renders_prefix,
        robot_spec=robot_spec_dict,
        robot_usd_uri=robot_usd_uri,
        task_config=task_config_dict,
        camera_views=json.dumps(
            camera_metadata(
                _env("NPA_SIM2REAL_CAMERA_VIEWS"),
                width=int(capture["width"]),
                height=int(capture["height"]),
            ),
            separators=(",", ":"),
        ),
        capture=capture,
        scenarios_uri=str(_SCENARIO_INPUT_PROVENANCE["uri"]),
        scenarios_sha256=str(_SCENARIO_INPUT_PROVENANCE["sha256"]),
    )
    if _env("NPA_SIM2REAL_INLINE_TASK") == "1":
        from npa.workflows.sim2real.isaac_job_payload import (
            execute_manifest_container_inline,
        )

        provenance = execute_manifest_container_inline(manifest)
    else:
        from npa.workflows.sim2real.gpu_fallback import run_gpu_job_with_fallback
        from npa.workflows.sim2real.k8s_client import KubernetesJobClient

        def manifest_factory(product: str, candidate_job_name: str) -> dict[str, Any]:
            candidate = copy.deepcopy(manifest)
            candidate.setdefault("metadata", {})["name"] = candidate_job_name
            pod_spec = (
                candidate.setdefault("spec", {})
                .setdefault("template", {})
                .setdefault("spec", {})
            )
            pod_spec["nodeSelector"] = {"nvidia.com/gpu.product": product}
            return candidate

        provenance = run_gpu_job_with_fallback(
            client=KubernetesJobClient.from_environment(namespace=namespace),
            manifest_factory=manifest_factory,
            base_job_name=job_name,
            namespace=namespace,
            image=image,
            preferred_product=gpu_product,
            explicit_candidates=_env("NPA_SIM2REAL_K8S_GPU_CANDIDATES"),
            workload="isaac",
            gpu_resource=_env("NPA_SIM2REAL_K8S_GPU_RESOURCE", "nvidia.com/gpu"),
            gpu_count=1,
            timeout_s=timeout_s,
        )
    job_name = str(provenance["job_name"])
    _LAST_GPU_PROVENANCE = provenance
    out = _download_json(per_env_uri)
    note = str(out.get("note") or "")
    if note != "rollout_ok":
        raise RuntimeError(
            f"real Isaac held-out inference failed closed: {note or 'missing status'}"
        )
    distances = out.get("object_goal_distances", [])
    if len(distances) != num_envs:
        raise RuntimeError(
            "real Isaac held-out inference returned "
            f"{len(distances)} distances for {num_envs} environments"
        )
    runtime_metrics = list(out.get("per_env_metrics") or [])
    if len(runtime_metrics) != num_envs:
        raise RuntimeError(
            "real Isaac held-out inference did not return decomposed metrics for "
            f"all {num_envs} environments"
        )
    checkpoint_provenance = dict(out.get("policy_checkpoint") or {})
    if (
        checkpoint_provenance.get("uri") != checkpoint_uri
        or not checkpoint_provenance.get("sha256")
        or int(checkpoint_provenance.get("size_bytes") or 0) <= 0
    ):
        raise RuntimeError(
            "held-out inference did not prove the loaded candidate checkpoint bytes"
        )
    global _APPLIED_SCENARIO_AUDIT, _CHECKPOINT_PROVENANCE
    _CHECKPOINT_PROVENANCE = checkpoint_provenance
    applied = dict(out.get("applied_scenarios") or {})
    applied_records = applied.get("records") or []
    actually_applied = {
        str(row.get("scenario_config_digest"))
        for row in applied_records
        if int(row.get("applied_count") or 0) > 0
    }
    if actually_applied != set(digests):
        raise RuntimeError(
            "Isaac runtime applied-scenario digests do not exactly match reported evaluation rows"
        )
    _APPLIED_SCENARIO_AUDIT = {
        **applied,
        "expected_config_digests": digests,
        "exact_digest_match": True,
        "applied_record_count": len(applied_records),
        "scenario_input_provenance": _SCENARIO_INPUT_PROVENANCE,
    }
    # Pull the rendered frames of the (custom) object down to the local heldout
    # renders dir so stage-14 Rerun viz logs them under heldout/camera/**.
    episodes = out.get("render_episodes") or []
    if episodes and _RENDERS_LOCAL_DIR:
        try:
            import boto3
            from urllib.parse import urlparse

            u = urlparse(renders_prefix)
            s3 = boto3.client("s3", endpoint_url=_env("AWS_ENDPOINT_URL") or None)
            base = u.path.lstrip("/").rstrip("/")
            for ep in episodes:
                eid = ep["env_id"]
                names = list(ep.get("frames", []))
                for view_names in (ep.get("camera_views") or {}).values():
                    names.extend(view_names or [])
                for name in dict.fromkeys(names):
                    dst = Path(_RENDERS_LOCAL_DIR) / eid / name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(u.netloc, f"{base}/{eid}/{name}", str(dst))
            pointcloud_count = 0
            pointcloud_prefix = f"{base}/_pointcloud/"
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=u.netloc, Prefix=pointcloud_prefix):
                for item in page.get("Contents", []) or []:
                    key = str(item.get("Key") or "")
                    if not key.endswith(".npz") or not key.startswith(
                        pointcloud_prefix
                    ):
                        continue
                    relative = key[len(base) + 1 :]
                    dst = Path(_RENDERS_LOCAL_DIR) / relative
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(u.netloc, key, str(dst))
                    pointcloud_count += 1
            total = sum(
                len(
                    {
                        name
                        for names in (e.get("camera_views") or {}).values()
                        for name in names
                    }
                )
                or len(e.get("frames", []))
                for e in episodes
            )
            print(
                f"byo_isaac_eval: synced {total} multi-view frames and "
                f"{pointcloud_count} point clouds",
                flush=True,
            )
        except Exception as e:
            print("byo_isaac_eval: render sync failed:", repr(e), flush=True)
    global _RENDER_MANIFEST
    _RENDER_MANIFEST = {
        "schema": "npa.sim2real.heldout_renders.v1",
        "sim_backend": "isaac",
        "isaac_task": task,
        "camera_views": out.get("camera_views") or [],
        "camera_metadata": out.get("camera_metadata") or [],
        "capture": out.get("capture") or capture,
        "policy_checkpoint": checkpoint_provenance,
        "renders_s3_uri": renders_prefix,
        "episodes": episodes,
    }
    return per_env_from_distances(
        distances,
        success_dist_m=success_dist,
        env_ids=env_ids,
        seeds=seeds,
        generated_envs=gen,
        runtime_metrics=runtime_metrics,
    )


def main() -> int:
    global \
        _APPLIED_SCENARIO_AUDIT, \
        _CHECKPOINT_PROVENANCE, \
        _LAST_GPU_PROVENANCE, \
        _RENDER_MANIFEST, \
        _SCENARIO_INPUT_PROVENANCE
    _LAST_GPU_PROVENANCE = {}
    _CHECKPOINT_PROVENANCE = {}
    _APPLIED_SCENARIO_AUDIT = {}
    _RENDER_MANIFEST = {}
    _SCENARIO_INPUT_PROVENANCE = {}
    output_json = _env("NPA_SIM2REAL_OUTPUT_JSON")
    if not output_json:
        print("byo_isaac_eval: NPA_SIM2REAL_OUTPUT_JSON not set", file=sys.stderr)
        return 2
    run_id = _env("NPA_SIM2REAL_RUN_ID") or _env("RUN_ID") or "byo-isaac"
    task = _env("NPA_SIM2REAL_ISAAC_TASK", DEFAULT_ISAAC_TASK)
    num_envs = int(_env("NPA_SIM2REAL_HELDOUT_ENV_COUNT", "4") or 4)
    # Heldout renders live next to the report so stage-14 viz finds them.
    global _RENDERS_LOCAL_DIR
    _RENDERS_LOCAL_DIR = str(Path(output_json).parent / "renders")
    success_dist = float(
        _env("NPA_BYO_ISAAC_SUCCESS_DIST_M", str(DEFAULT_SUCCESS_DIST_M))
        or DEFAULT_SUCCESS_DIST_M
    )

    inner_evidence = {}
    ev_path = _env("NPA_SIM2REAL_INNER_EVIDENCE_JSON")
    if ev_path and Path(ev_path).is_file():
        inner_evidence = json.loads(Path(ev_path).read_text())
    checkpoint_uri = extract_checkpoint_uri(inner_evidence)

    # GENERATED held-out env specs (env_id + seed) — drive eval on the envgen
    # distribution and label results by the real generated env_id.
    envs_uri = _env("NPA_SIM2REAL_HELDOUT_ENVS_URI")
    envs_dir = _env("NPA_SIM2REAL_HELDOUT_ENVS_DIR")
    if envs_uri:
        generated_envs = read_durable_generated_envs(
            envs_uri,
            cache_dir=Path(output_json).parent / "durable-scenario-input",
        )
        scenario_records_source = "durable_s3_object"
    else:
        generated_envs = read_generated_envs(envs_dir) if envs_dir else []
        scenario_records_source = "local_ephemeral"
    if generated_envs:
        generated_envs = select_stratified_eval_envs(
            generated_envs,
            count=num_envs,
            split=_env("NPA_SIM2REAL_EVALUATION_SPLIT", "heldout"),
        )

    if not generated_envs and checkpoint_uri and _env("NPA_BYO_ISAAC_DRYRUN") != "1":
        raise ValueError(
            "real Isaac evaluation requires complete durable generated scenarios"
        )

    if _env("NPA_BYO_ISAAC_DRYRUN") == "1":
        gids = [e["env_id"] for e in generated_envs] or None
        seeds = [e["seed"] for e in generated_envs] or None
        per_env = per_env_from_distances(
            [0.02, 0.04, 0.08, 0.12][:num_envs],
            success_dist_m=success_dist,
            env_ids=gids,
            seeds=seeds,
        )
    elif not checkpoint_uri:
        print(
            "byo_isaac_eval: no trained checkpoint in inner evidence — refusing to fake success",
            file=sys.stderr,
        )
        return 3
    else:
        per_env = run_isaac_eval_job(
            run_id,
            checkpoint_uri=checkpoint_uri,
            num_envs=num_envs,
            generated_envs=generated_envs,
        )

    report = build_heldout_report(
        per_env,
        isaac_task=task,
        checkpoint_uri=checkpoint_uri,
        source="byo_isaac_eval_dryrun"
        if _env("NPA_BYO_ISAAC_DRYRUN") == "1"
        else "byo_isaac_eval",
    )
    report["generated_envs_tested"] = len(generated_envs)
    report["generated_env_ids"] = [e["env_id"] for e in generated_envs]
    report["scenario_records_uri"] = envs_uri
    report["scenario_records_source"] = scenario_records_source
    report["component_invocation"] = {
        "mode": str(_LAST_GPU_PROVENANCE.get("mode") or "kubernetes_job")
        if _LAST_GPU_PROVENANCE
        else "dryrun",
        "gpu_provenance": _LAST_GPU_PROVENANCE,
    }
    report["success_distance_m"] = success_dist
    capture = capture_settings()
    report["capture"] = capture
    report["camera_metadata"] = list(
        _RENDER_MANIFEST.get("camera_metadata")
        or camera_metadata(
            _env("NPA_SIM2REAL_CAMERA_VIEWS"),
            width=int(capture["width"]),
            height=int(capture["height"]),
        )
    )
    report["policy_checkpoint_sha256"] = str(_CHECKPOINT_PROVENANCE.get("sha256") or "")
    report["policy_checkpoint_size_bytes"] = int(
        _CHECKPOINT_PROVENANCE.get("size_bytes") or 0
    )
    report["policy_inference_provenance"] = policy_inference_provenance(
        checkpoint_uri=checkpoint_uri,
        checkpoint=_CHECKPOINT_PROVENANCE,
    )
    report["applied_scenario_proof"] = _APPLIED_SCENARIO_AUDIT
    report["scenario_input_provenance"] = _SCENARIO_INPUT_PROVENANCE
    if _RENDER_MANIFEST.get("episodes"):
        report["render_manifest"] = _RENDER_MANIFEST
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    passed = sum(1 for r in per_env if r["success"])
    print(
        f"byo_isaac_eval: wrote {output_json} per_env={len(per_env)} passed={passed} "
        f"checkpoint={checkpoint_uri or 'NONE'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
