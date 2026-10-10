"""Render the standalone RSL-RL trainer and checkpoint-validity protocol."""

from collections.abc import Mapping
import re

_TRAIN_SHELL = """\
	export PYTHONUNBUFFERED=1
	@TRAINING_ENV@
	TASK=@TASK@
	NUM_ENVS=@NUM_ENVS@
	MAX_ITERATIONS=@ITERATIONS@
OUTPUT_DIR=@OUTPUT_DIR@
RUN_NAME=@RUN_NAME@
EXPERIMENT_NAME=npa_isaac_lab
PYTHON_BIN=@PYTHON_BIN@
TRAIN_REL=@TRAIN_REL@
if [ "$PYTHON_BIN" = '${NPA_PYTHON_BIN}' ]; then
  PYTHON_BIN="${NPA_PYTHON_BIN:-python3}"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [ ! -x "$PYTHON_BIN" ]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 127
fi

mkdir -p "$OUTPUT_DIR"
if [ -z "${ISAACLAB_PKG:-}" ]; then
  ISAACLAB_PKG=$("$PYTHON_BIN" <<'PY' 2>/dev/null || true
from pathlib import Path

try:
    import isaaclab
except Exception:
    raise SystemExit(0)

print(Path(isaaclab.__file__).resolve().parent)
PY
)
fi
if [ -n "${ISAACLAB_PKG:-}" ] && [ -d "$ISAACLAB_PKG/source" ]; then
  export ISAACLAB_PKG
  export PYTHONPATH="$ISAACLAB_PKG/source/isaaclab:$ISAACLAB_PKG/source/isaaclab_tasks:$ISAACLAB_PKG/source/isaaclab_rl:$ISAACLAB_PKG/source/isaaclab_assets:$ISAACLAB_PKG/source/isaaclab_mimic:$ISAACLAB_PKG/source/isaaclab_contrib:${PYTHONPATH:-}"
fi

TRAIN_SCRIPT=""
TRAIN_ROOT=""
for root in "${ISAACLAB_PATH:-}" /workspace/isaaclab /opt/isaac-lab "${ISAAC_LAB_HOME:-}" "${ISAACLAB_PKG:-}"; do
  [ -n "$root" ] || continue
  if [ -f "$root/$TRAIN_REL" ]; then
    TRAIN_ROOT="$root"
    TRAIN_SCRIPT="$root/$TRAIN_REL"
    break
  fi
done
if [ -z "$TRAIN_SCRIPT" ]; then
  found=$(
    find /workspace /opt -path "*/$TRAIN_REL" -type f       ! -path "*/runs/*"       ! -path "*/npa_isaac_lab_generated/*"       -print -quit 2>/dev/null || true
  )
  if [ -n "$found" ]; then
    TRAIN_SCRIPT="$found"
    TRAIN_ROOT="${found%/$TRAIN_REL}"
  fi
fi
if [ -z "$TRAIN_SCRIPT" ]; then
  echo "Isaac Lab source checkout is missing the real RSL-RL entrypoint: $TRAIN_REL" >&2
  echo "Refusing to generate or run a compatibility trainer." >&2
  exit 66
fi
cd "$OUTPUT_DIR"
export NPA_ISAAC_LAB_RUN_DIR="$OUTPUT_DIR"
export NPA_ISAAC_LAB_TASK="$TASK"
export NPA_ISAAC_LAB_NUM_ENVS="$NUM_ENVS"
export NPA_ISAAC_LAB_MAX_ITERATIONS="$MAX_ITERATIONS"
export NPA_ISAAC_LAB_RUN_NAME="$RUN_NAME"
export NPA_ISAAC_LAB_EXPERIMENT_NAME="$EXPERIMENT_NAME"
export NPA_ISAAC_LAB_TRAIN_SCRIPT="$TRAIN_SCRIPT"
export NPA_ISAAC_LAB_TRAIN_ROOT="$TRAIN_ROOT"

cmd=(
  "$PYTHON_BIN"
  "$TRAIN_SCRIPT"
  --task "$TASK"
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERATIONS"
	  --visualizer none
	  --experiment_name "$EXPERIMENT_NAME"
	  --run_name "$RUN_NAME"
	@EXTRA_ARGUMENTS@
	)
printf 'ISAAC_LAB_RSL_RL_COMMAND'
printf ' %q' "${cmd[@]}"
printf '\\n'
echo "ISAAC_LAB_RSL_RL_TRAIN_START task=$TASK num_envs=$NUM_ENVS max_iterations=$MAX_ITERATIONS output_dir=$OUTPUT_DIR"

set +e
"${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/isaac_lab_train.log"
train_rc=${PIPESTATUS[0]}
export NPA_ISAAC_LAB_TRAIN_RC="$train_rc"
"$PYTHON_BIN" <<'PY'
@VALIDITY_SOURCE@
import json
import os
import shutil
import sys
import time
from pathlib import Path

root = Path(os.environ["NPA_ISAAC_LAB_RUN_DIR"])
train_rc = int(os.environ.get("NPA_ISAAC_LAB_TRAIN_RC", "1"))
physics = inspect_training_log(root / "isaac_lab_train.log")
if not physics["physics_valid"]:
    train_rc = train_rc or 78
log_root = root / "logs" / "rsl_rl"
checkpoints = sorted(
    log_root.rglob("model_*.pt"),
    key=lambda path: (path.stat().st_mtime, str(path)),
)
latest = checkpoints[-1] if checkpoints else None
stable_checkpoint = root / "npa_isaac_lab_checkpoint.pt"
if not physics["physics_valid"]:
    stable_checkpoint.unlink(missing_ok=True)
elif latest is not None:
    shutil.copy2(latest, stable_checkpoint)
manifest = {
    **physics,
    "format": "npa_isaac_lab_rsl_rl_checkpoint_v1",
    "tool": "isaac_lab",
    "framework": "rsl_rl",
    "data_path": os.environ.get("NPA_TRAINING_DATA_PATH", ""),
    "overrides": json.loads(os.environ.get("NPA_TRAINING_OVERRIDES_JSON", "[]")),
    "wandb": {
        "enabled": os.environ.get("NPA_TRAINING_WANDB_ENABLED", "0") == "1",
        "project": os.environ.get("NPA_TRAINING_WANDB_PROJECT", ""),
        "run_name": os.environ.get("NPA_TRAINING_WANDB_RUN_NAME", ""),
        "mode": os.environ.get("WANDB_MODE", ""),
    },
    "checkpoint_s3_uri": os.environ.get("NPA_CHECKPOINT_S3_URI", ""),
    "task": os.environ["NPA_ISAAC_LAB_TASK"],
    "num_envs": int(os.environ["NPA_ISAAC_LAB_NUM_ENVS"]),
    "max_iterations": int(os.environ["NPA_ISAAC_LAB_MAX_ITERATIONS"]),
    "run_name": os.environ["NPA_ISAAC_LAB_RUN_NAME"],
    "experiment_name": os.environ["NPA_ISAAC_LAB_EXPERIMENT_NAME"],
    "train_script": os.environ["NPA_ISAAC_LAB_TRAIN_SCRIPT"],
    "train_root": os.environ["NPA_ISAAC_LAB_TRAIN_ROOT"],
    "checkpoint_path": str(latest) if latest is not None else "",
    "stable_checkpoint_path": str(stable_checkpoint) if latest is not None and physics["physics_valid"] else "",
    "checkpoint_count": len(checkpoints),
    "created_unix": round(time.time(), 3),
}
summary = {
    **physics,
    "status": "success" if train_rc == 0 and latest is not None else "failed",
    "exit_code": train_rc,
    "tool": "isaac_lab",
    "framework": "rsl_rl",
    "task": manifest["task"],
    "num_envs": manifest["num_envs"],
    "steps": manifest["max_iterations"],
    "max_iterations": manifest["max_iterations"],
    "run_name": manifest["run_name"],
    "experiment_name": manifest["experiment_name"],
    "train_script": manifest["train_script"],
    "log_root": str(log_root),
    "checkpoint_path": manifest["checkpoint_path"],
    "stable_checkpoint_path": manifest["stable_checkpoint_path"],
    "checkpoint_count": manifest["checkpoint_count"],
}
(root / "npa_isaac_lab_checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2))
(root / "npa_isaac_lab_train_summary.json").write_text(json.dumps(summary, indent=2))
print("ISAAC_LAB_TRAIN_COMPLETE" if summary["status"] == "success" else "ISAAC_LAB_TRAIN_FAILED", flush=True)
print(json.dumps(summary, indent=2), flush=True)
if not physics["physics_valid"]:
    sys.exit(train_rc)
if train_rc == 0 and latest is None:
    sys.exit(3)
PY
summary_rc=$?
set -e
if [ "$train_rc" -ne 0 ]; then
  exit "$train_rc"
fi
	exit "$summary_rc"
	"""


def render_training_shell(values: Mapping[str, str]) -> str:
    """Substitute quoted trainer settings once without reinterpreting their values.

    Args:
        values: Shell-quoted settings, environment exports, and validator source.

    Returns:
        The complete standalone training shell script.

    Raises:
        KeyError: A required template setting is absent.
    """
    return re.sub(r"@([A-Z_]+)@", lambda match: values[match.group(1)], _TRAIN_SHELL)
