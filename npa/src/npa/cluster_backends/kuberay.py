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
_CPU_NODE_PRESET = re.compile(
    r"(?P<cpus>[1-9][0-9]*)vcpu-(?P<memory_gib>[1-9][0-9]*)gb"
)
_HEAD_CPUS = 1
_HEAD_MEMORY_GIB = 4


@dataclass(frozen=True)
class KubeRaySpec:
    """Configure a fixed CPU worker group for native Ray application execution.

    Args:
        enabled: Deploy the cluster when true; defaults to false.
        worker_replicas: Fixed worker count, defaulting to one.
        worker_cpus: CPU request and limit per worker, defaulting to two.
        worker_memory_gib: Memory request and limit per worker, defaulting to four.
    Returns:
        None.
    Raises:
        None. Call validate to check configuration before use.
    """

    enabled: bool = False
    worker_replicas: int = 1
    worker_cpus: int = 2
    worker_memory_gib: int = 4

    def validate(self, *, cpu_nodes: Any | None) -> None:
        """Require typed positive settings and an explicit CPU pool when enabled.

        Args:
            cpu_nodes: Target node pool, or None when no CPU pool is configured.
        Returns:
            None.
        Raises:
            ValueError: Settings or the target node pool violate the CPU policy.
        """
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
            _CPU_NODE_PRESET, cpu_nodes.preset
        ):
            raise ValueError("kuberay requires an explicit CPU platform and preset")
        self._validate_fixed_pod_capacity(cpu_nodes)

    def _validate_fixed_pod_capacity(self, cpu_nodes: Any) -> None:
        """Reject Ray pod requests that exceed nominal declared CPU capacity."""

        match = _CPU_NODE_PRESET.fullmatch(cpu_nodes.preset)
        if match is None:  # The caller validates the preset before this helper.
            raise ValueError("kuberay requires an explicit CPU platform and preset")
        node_cpus = int(match.group("cpus"))
        node_memory_gib = int(match.group("memory_gib"))
        if _HEAD_CPUS > node_cpus or _HEAD_MEMORY_GIB > node_memory_gib:
            raise ValueError(
                f"kuberay head pod requests {_HEAD_CPUS} vCPU/{_HEAD_MEMORY_GIB} GiB, "
                f"which exceeds one declared {node_cpus} vCPU/{node_memory_gib} GiB CPU node"
            )
        if self.worker_cpus > node_cpus or self.worker_memory_gib > node_memory_gib:
            raise ValueError(
                f"kuberay worker pod requests {self.worker_cpus} vCPU/"
                f"{self.worker_memory_gib} GiB, which exceeds one declared "
                f"{node_cpus} vCPU/{node_memory_gib} GiB CPU node"
            )
        pool_cpus = cpu_nodes.count * node_cpus
        pool_memory_gib = cpu_nodes.count * node_memory_gib
        requested_cpus = _HEAD_CPUS + self.worker_replicas * self.worker_cpus
        requested_memory_gib = (
            _HEAD_MEMORY_GIB + self.worker_replicas * self.worker_memory_gib
        )
        if requested_cpus > pool_cpus or requested_memory_gib > pool_memory_gib:
            raise ValueError(
                f"kuberay fixed pod requests {requested_cpus} vCPU/"
                f"{requested_memory_gib} GiB, which exceeds the declared CPU pool's "
                f"nominal {pool_cpus} vCPU/{pool_memory_gib} GiB capacity"
            )

    def plan(self) -> dict[str, Any]:
        """Describe the policy and its pinned Ray runtime.

        Args:
            None.
        Returns:
            Configuration fields, Ray version and immutable image reference.
        Raises:
            None.
        """
        return {**asdict(self), "ray_version": RAY_VERSION, "image": RAY_IMAGE}


def kuberay_spec_from_mapping(value: Any) -> KubeRaySpec | None:
    """Parse optional fleet configuration without accepting unsupported fields.

    Args:
        value: The kuberay mapping, or None when omitted.
    Returns:
        Parsed policy, or None when omitted; validate it against the target pool.
    Raises:
        ValueError: Input is not a supported mapping or sets disabled workers.
    """
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
    """Validate the optional policy against its fleet backend and CPU pool.

    Args:
        cluster: Fleet cluster declaration carrying the optional kuberay policy.
    Returns:
        None.
    Raises:
        ValueError: The policy, backend or node pool is unsupported.
    """
    policy = cluster.kuberay
    if policy is None:
        return
    if not isinstance(policy, KubeRaySpec):
        raise ValueError("kuberay must be a KubeRaySpec")
    if cluster.backend_name() != "mk8s":
        raise ValueError("kuberay is supported only by the mk8s backend")
    policy.validate(cpu_nodes=cluster.cpu_nodes)


def validate_recipe_kuberay_compatibility(cluster: Any, recipe_dir: Path) -> None:
    """Require the complete reviewed recipe before any cloud mutation.

    Args:
        cluster: Target fleet cluster declaration.
        recipe_dir: Selected k8s-training directory, beside its modules directory.
    Returns:
        None.
    Raises:
        ValueError: Policy, execution environment or recipe violates the contract.
        OSError: Recipe entries or the recorded contract cannot be read.
    """

    validate_kuberay(cluster)
    if cluster.kuberay is None or not cluster.kuberay.enabled:
        return
    validate_kuberay_execution_inputs(cluster)
    validate_kuberay_recipe_inventory(recipe_dir)


def _walk_recipe_sources(root: Path):
    def unreadable(error: OSError) -> None:
        raise ValueError("Unreadable source cannot honor the reviewed CPU KubeRay contract") from error

    for subtree in (root / "k8s-training", root / "modules"):
        try:
            mode = subtree.lstat().st_mode
        except FileNotFoundError as exc:
            raise ValueError("Missing source root cannot honor the reviewed CPU KubeRay contract") from exc
        if not stat.S_ISDIR(mode):
            raise ValueError("Source roots must be directories for the reviewed CPU KubeRay contract")
        yield from os.walk(subtree, followlinks=False, onerror=unreadable)


def _validate_source_entry(path: Path, relative: str, allowed_directories: set[str] | None) -> int:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise ValueError("Symlinks cannot honor the reviewed CPU KubeRay contract")
    if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
        raise ValueError("Special source entries cannot honor the reviewed CPU KubeRay contract")
    if stat.S_ISDIR(mode) and allowed_directories is not None and relative not in allowed_directories:
        raise ValueError("Unexpected source directories cannot honor the reviewed CPU KubeRay contract")
    return mode


def _source_inventory(
    root: Path, *, materialized: bool = False,
    allowed_directories: set[str] | None = None,
) -> dict[str, str]:
    """Hash regular source files only, rejecting special entries before reads."""
    actual = {}
    excluded = {
        *(f"k8s-training/{name}" for name in KUBERAY_STATE_FILES),
        "k8s-training/.terraform.lock.hcl", "k8s-training/.terraform.tfstate.lock.info",
    }
    runtime_directories = {"k8s-training/.terraform", "k8s-training/filesystem-csi-validation/.state"}
    for directory, directories, files in _walk_recipe_sources(root):
        for name in [*directories, *files]:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            mode = _validate_source_entry(path, relative, allowed_directories)
            if materialized and relative in runtime_directories and stat.S_ISDIR(mode):
                directories.remove(name)
            elif stat.S_ISREG(mode) and not (materialized and relative in excluded):
                actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return actual


def validate_kuberay_recipe_inventory(recipe_dir: Path) -> None:
    """Reject changed or additional recipe entries, including empty directories.

    Args:
        recipe_dir: Selected k8s-training directory, beside its modules directory.
    Returns:
        None.
    Raises:
        ValueError: The source inventory differs from the reviewed contract.
        OSError: Recipe entries or the recorded contract cannot be read.
    """
    contract = json.loads(
        Path(__file__).with_name("kuberay_recipe_contract.json").read_text()
    )
    directories = {
        parent.as_posix()
        for filename in contract
        for parent in Path(filename).parents
        if parent != Path(".")
    }
    actual = _source_inventory(recipe_dir.parent, allowed_directories=directories)
    if actual != contract:
        raise ValueError(
            "Selected k8s-training recipe cannot honor the reviewed CPU "
            "KubeRay contract. Use the pristine vendored recipe; unsupported "
            "pinned, upstream or local recipes are rejected before provisioning."
        )


def kuberay_materialized_digest(install_dir: Path) -> str:
    """Hash installed source and generated inputs while excluding runtime state.

    Args:
        install_dir: Installation containing k8s-training and modules directories.
    Returns:
        SHA-256 digest of the ordered file inventory.
    Raises:
        ValueError: Source roots or entries are unsupported or unreadable.
        OSError: An entry cannot be inspected or read.
    """
    inventory = _source_inventory(install_dir, materialized=True)
    return hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()


def validate_kuberay_destroyed_state(workdir: Path) -> None:
    """Require empty canonical managed state before discarding recovery files.

    Args:
        workdir: Terraform working directory containing terraform.tfstate.
    Returns:
        None.
    Raises:
        ValueError: State is malformed, nonregular or still holds managed resources.
        OSError: Canonical state cannot be inspected or read.
    """

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
    """Bind opt-in execution to generated inputs and owned local state.

    Args:
        cluster: Cluster declaration carrying the optional kuberay policy.
        workdir: Existing or intended Terraform directory; None checks environment only.
        environ: Explicit environment, or None to inspect the current process.
    Returns:
        None.
    Raises:
        ValueError: Execution overrides, installation entries or state are unsupported.
        OSError: Existing installation entries cannot be inspected or read.
    """

    if cluster.kuberay is None or not cluster.kuberay.enabled:
        return
    environment = os.environ if environ is None else environ
    _validate_execution_environment(environment)
    if workdir is None:
        return
    _validate_installation_entries(workdir)
    if workdir.exists():
        _validate_retained_backend(workdir)


def _validate_execution_environment(environment: Mapping[str, str]) -> None:
    if any(value.strip() for key, value in environment.items()
           if key == "TF_CLI_ARGS" or key.startswith("TF_CLI_ARGS_")):
        raise ValueError("KubeRay does not support inherited TF_CLI_ARGS overrides; unset them before deployment")
    if environment.get("TF_DATA_DIR", "") or environment.get("TF_WORKSPACE", "") not in ("", "default"):
        raise ValueError("KubeRay requires the default workspace and local Terraform data directory")


def _validate_installation_entries(workdir: Path) -> None:
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


def _validate_retained_backend(workdir: Path) -> None:
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
