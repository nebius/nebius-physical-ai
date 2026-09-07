"""Validated opt-in CPU RayCluster policy for the existing mk8s backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

RAY_VERSION = "2.58.0"
RAY_IMAGE = (
    "docker.io/rayproject/ray@sha256:"
    "507464fe56b3d24cec2e812a25850db91b97d52752d63119fecf6914f7b0a37a"
)
KUBERAY_STATE_FILES = frozenset({"terraform.tfstate", "terraform.tfstate.backup"})


@dataclass(frozen=True)
class KubeRaySpec:
    """A fixed CPU worker group; native Ray APIs own application execution."""

    enabled: bool = False
    worker_replicas: int = 1
    worker_cpus: int = 2
    worker_memory_gib: int = 4

    def validate(self, *, cpu_nodes: Any | None) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("kuberay.enabled must be a boolean")
        for name in ("worker_replicas", "worker_cpus", "worker_memory_gib"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"kuberay.{name} must be a positive integer")
        if not self.enabled:
            if self != KubeRaySpec():
                raise ValueError("kuberay worker settings require enabled: true")
            return
        if cpu_nodes is None or cpu_nodes.count <= 0 or cpu_nodes.is_gpu():
            raise ValueError("kuberay requires a nonempty CPU node pool")
        if not re.fullmatch(r"cpu-[a-z0-9-]+", cpu_nodes.platform) or not re.fullmatch(
            r"[1-9][0-9]*vcpu-[1-9][0-9]*gb", cpu_nodes.preset
        ):
            raise ValueError("kuberay requires an explicit CPU platform and preset")

    def plan(self) -> dict[str, Any]:
        return {**asdict(self), "ray_version": RAY_VERSION, "image": RAY_IMAGE}


def kuberay_spec_from_mapping(value: Any) -> KubeRaySpec | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("kuberay must be a mapping")
    unknown = set(value) - set(KubeRaySpec.__dataclass_fields__)
    if unknown:
        raise ValueError(
            "kuberay has unsupported fields; only enabled, worker_replicas, "
            "worker_cpus and worker_memory_gib are supported (CPU RayCluster only)"
        )
    if value.get("enabled") is not True and set(value) - {"enabled"}:
        raise ValueError("kuberay worker settings require enabled: true")
    return KubeRaySpec(**value)


def validate_kuberay(cluster: Any) -> None:
    policy = cluster.kuberay
    if policy is None:
        return
    if not isinstance(policy, KubeRaySpec):
        raise ValueError("kuberay must be a KubeRaySpec")
    if cluster.backend_name() != "mk8s":
        raise ValueError("kuberay is supported only by the mk8s backend")
    policy.validate(cpu_nodes=cluster.cpu_nodes)


def validate_recipe_kuberay_compatibility(cluster: Any, recipe_dir: Path) -> None:
    """Require reviewed module wiring and safety bytes before any cloud mutation.

    Alternate recipe revisions are deliberately unsupported unless their KubeRay
    complete source inventory matches the reviewed vendored recipe. In particular,
    an additional Terraform override or auto-loaded variable file can change the
    effective configuration without changing an existing file.
    """

    validate_kuberay(cluster)
    if cluster.kuberay is None or not cluster.kuberay.enabled:
        return
    validate_kuberay_execution_inputs(cluster)
    validate_kuberay_recipe_inventory(recipe_dir)


def _source_inventory(root: Path, *, materialized: bool = False) -> dict[str, str]:
    """Hash regular source files only, rejecting special entries before reads."""

    actual = {}
    excluded = {
        *(f"k8s-training/{name}" for name in KUBERAY_STATE_FILES),
        "k8s-training/.terraform.lock.hcl", "k8s-training/.terraform.tfstate.lock.info",
    }
    runtime_dirs = {"k8s-training/.terraform", "k8s-training/filesystem-csi-validation/.state"}

    def unreadable(error: OSError) -> None:
        raise ValueError("Unreadable source cannot honor the reviewed CPU KubeRay contract") from error

    for subtree in (root / "k8s-training", root / "modules"):
        try:
            mode = subtree.lstat().st_mode
        except FileNotFoundError as exc:
            raise ValueError("Missing source root cannot honor the reviewed CPU KubeRay contract") from exc
        if not stat.S_ISDIR(mode):
            raise ValueError("Source roots must be directories for the reviewed CPU KubeRay contract")
        for directory, dirs, files in os.walk(subtree, followlinks=False, onerror=unreadable):
            for name in [*dirs, *files]:
                path = Path(directory) / name
                relative = path.relative_to(root).as_posix()
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode):
                    raise ValueError("Symlinks cannot honor the reviewed CPU KubeRay contract")
                if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                    raise ValueError("Special source entries cannot honor the reviewed CPU KubeRay contract")
                if materialized and relative in runtime_dirs and stat.S_ISDIR(mode):
                    dirs.remove(name)
                elif stat.S_ISREG(mode) and not (materialized and relative in excluded):
                    actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return actual


def validate_kuberay_recipe_inventory(recipe_dir: Path) -> None:
    contract = json.loads(
        Path(__file__).with_name("kuberay_recipe_contract.json").read_text()
    )
    actual = _source_inventory(recipe_dir.parent)
    if actual != contract:
        raise ValueError(
            "Selected k8s-training recipe cannot honor the reviewed CPU "
            "KubeRay contract. Use the pristine vendored recipe; unsupported "
            "pinned, upstream or local recipes are rejected before provisioning."
        )


def kuberay_materialized_digest(install_dir: Path) -> str:
    inventory = _source_inventory(install_dir, materialized=True)
    return hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()


def validate_kuberay_destroyed_state(workdir: Path) -> None:
    """A successful command cannot justify discarding nonempty recovery state."""

    state = workdir / "terraform.tfstate"
    if not stat.S_ISREG(state.lstat().st_mode):
        raise ValueError("KubeRay teardown requires a regular canonical state file")
    data = json.loads(state.read_text())
    if not isinstance(data, dict) or data.get("version") != 4 or not isinstance(data.get("resources"), list):
        raise ValueError("KubeRay teardown state is missing or malformed; recovery state retained")
    for resource in data["resources"]:
        if not isinstance(resource, dict) or resource.get("mode") != "data":
            raise ValueError("KubeRay teardown still has managed or unrecognized resources; recovery state retained")


def validate_kuberay_execution_inputs(
    cluster: Any, *, workdir: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Bind opt-in execution to generated inputs and the owned local state."""

    if cluster.kuberay is None or not cluster.kuberay.enabled:
        return
    env = os.environ if environ is None else environ
    if any(value.strip() for key, value in env.items()
           if key == "TF_CLI_ARGS" or key.startswith("TF_CLI_ARGS_")):
        raise ValueError("KubeRay does not support inherited TF_CLI_ARGS overrides; unset them before deployment")
    if env.get("TF_DATA_DIR", "") or env.get("TF_WORKSPACE", "") not in ("", "default"):
        raise ValueError("KubeRay requires the default workspace and local Terraform data directory")
    if workdir is None:
        return
    modules = workdir.parent / "modules"
    if workdir.is_symlink() or modules.is_symlink():
        raise ValueError("KubeRay installation destinations must not be symlinks")
    if not workdir.exists():
        return
    if not workdir.is_dir():
        raise ValueError("KubeRay Terraform workdir must be a directory")
    contract = json.loads(Path(__file__).with_name("kuberay_recipe_contract.json").read_text())
    allowed = {Path(name).name for name in contract if Path(name).parent == Path("k8s-training")}
    allowed.add("terraform.tfvars")
    for path in workdir.iterdir():
        if path.is_symlink():
            raise ValueError("KubeRay installation entries must not be symlinks")
        name = path.name
        if name == "errored.tfstate":
            raise ValueError("KubeRay requires recovery of errored.tfstate before continuing")
        if not (path.is_file() or path.is_dir()):
            raise ValueError("KubeRay installation entries must be regular files or directories")
        if name.startswith("terraform.tfstate"):
            if name not in KUBERAY_STATE_FILES or not path.is_file():
                raise ValueError("KubeRay requires exact regular Terraform state and backup files")
        if name.endswith((".tf", ".tf.json", ".tfvars", ".tfvars.json")) and name not in allowed:
            raise ValueError("KubeRay installation contains unsupported effective Terraform inputs")
        if name == ".terraform" and not path.is_dir():
            raise ValueError("KubeRay Terraform data path must be a directory")
    backend = workdir / ".terraform/terraform.tfstate"
    if backend.is_symlink() or backend.exists():
        raise ValueError("KubeRay requires implicit local Terraform state without retained backend metadata")
    workspace = workdir / ".terraform/environment"
    if workspace.is_symlink() or (workspace.exists() and (
        not workspace.is_file() or workspace.read_text() != "default"
    )):
        raise ValueError("KubeRay requires the default retained Terraform workspace")
    modules_cache = workdir / ".terraform/modules"
    if modules_cache.is_symlink() or (modules_cache.exists() and not modules_cache.is_dir()):
        raise ValueError("KubeRay Terraform module cache must be a local directory")
