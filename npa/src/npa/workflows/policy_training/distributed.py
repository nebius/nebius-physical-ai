"""Render a fixed-size Slurm allocation with one torchrun launcher per node."""

from __future__ import annotations

import re
import shlex


def slurm_script(recipe: dict) -> str:
    """Render an Enroot/Pyxis job without evaluating recipe strings as shell code.

    Args:
        recipe: Immutable container image, shared paths, trainer argv and topology.
    Returns:
        A shell script for submission from the Soperator shared jail.
    Raises:
        ValueError: Topology, mode, command or required shared paths are invalid.
    """
    _validate_recipe(recipe)
    q = shlex.quote
    nodes, gpus = recipe["nodes"], recipe["gpus_per_node"]
    trainer = shlex.join(recipe["trainer_argv"])
    output = q(recipe["output_path"])
    image = q(enroot_image(recipe["image"]))
    mounts = q(recipe["shared_path"] + ":" + recipe["shared_path"])
    return f"""#!/usr/bin/env bash
# Fixed membership: Slurm allocates nodes; torchrun launches ranks inside them.
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node={gpus}
#SBATCH --cpus-per-task={recipe["cpus_per_node"]}
#SBATCH --mem={recipe["memory_gib"]}G
set -euo pipefail
mkdir -p {output}
cd {q(recipe["working_directory"])}
export NPA_MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')"
export NPA_MASTER_PORT={recipe["master_port"]}
export NPA_GPUS_PER_NODE={gpus}
srun --kill-on-bad-exit=1 --container-image={image} \\
  --container-mounts={mounts} --container-workdir={q(recipe["working_directory"])} \\
  bash -c 'exec torchrun --nnodes="$SLURM_NNODES" \\
    --nproc-per-node="$NPA_GPUS_PER_NODE" --node-rank="$SLURM_PROCID" \\
    --master-addr="$NPA_MASTER_ADDR" --master-port="$NPA_MASTER_PORT" \\
    "$@"' foundation-trainer {trainer}
"""


def enroot_image(image: str) -> str:
    """Translate a pinned OCI reference into Enroot's registry and digest syntax.

    Args:
        image: Validated OCI reference ending in @sha256:<digest>.
    Returns:
        Pyxis image URI compatible with Enroot 3.x and current releases.
    Raises:
        ValueError: The reference lacks an immutable digest.
    """
    name, marker, digest = image.removeprefix("docker://").partition("@sha256:")
    if not marker or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Enroot image requires a SHA256 digest")
    registry, separator, path = name.partition("/")
    if "#" not in name and separator and ("." in registry or ":" in registry):
        name = registry + "#" + path
    return name + ":sha256:" + digest


def _validate_recipe(recipe):
    mode = recipe.get("mode")
    if mode not in {"pretrain", "continued-pretrain", "finetune"}:
        raise ValueError("training mode must distinguish pretraining and fine-tuning")
    for name in (
        "nodes",
        "gpus_per_node",
        "cpus_per_node",
        "memory_gib",
        "master_port",
    ):
        value = recipe.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if recipe["master_port"] > 65535:
        raise ValueError("master_port must be a valid TCP port")
    if not re.fullmatch(r"[\w./:#-]+@sha256:[a-f0-9]{64}", recipe.get("image", "")):
        raise ValueError("training image must have an immutable SHA256 digest")
    shared = recipe.get("shared_path", "")
    if not shared.startswith("/") or shared == "/" or any(c in shared for c in ",:\n"):
        raise ValueError("shared_path must be an absolute unambiguous mount path")
    for name in ("working_directory", "output_path"):
        path = recipe.get(name, "")
        if not path.startswith(shared.rstrip("/") + "/") or ".." in path.split("/"):
            raise ValueError(f"{name} must be inside shared_path")
    argv = recipe.get("trainer_argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(v, str) or "\0" in v for v in argv)
    ):
        raise ValueError("trainer_argv must be a nonempty argument vector")
