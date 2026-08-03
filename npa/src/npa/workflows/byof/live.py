"""Resolve customer/project infra for BYOF live validation (no operator-host assumptions)."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from npa.clients.config import ConfigError, _load_yaml, _resolve_project_section, default_project_name, list_projects
from npa.cluster.state import kubeconfig_file, load_cluster_state

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
# BYOF-owned SkyPilot **resource profiles**: the pod shape a BYOF workload runs in
# (accelerator, cpu/memory floors, image placeholder, smoke command). They are not
# workflow templates — the workflow surface is the npa.workflow spec `byof.yaml`, whose
# `workbench.byof.repo` toolRef passes one of these through
# `--yaml {{config.resource_profile_yaml}}`.
#
# They live beside this runner rather than in the retired raw SkyPilot workflow
# catalog so BYOF live runs keep their pod-shape profiles without reopening a
# workflow authoring surface. Resolved package-relative so a repo checkout and an
# installed wheel behave the same. See profiles/README.md.
BYOF_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
DEFAULT_TRAIN_YAML = BYOF_PROFILES_DIR / "isaac-lab-rl-train.yaml"
RTXPRO_TRAIN_YAML = BYOF_PROFILES_DIR / "isaac-lab-rl-train-rtxpro.yaml"
RTXPRO_SMOKE_TRAIN_YAML = BYOF_PROFILES_DIR / "isaac-lab-rl-train-rtxpro-smoke.yaml"
BYOF_DATAGEN_SMOKE_YAML = BYOF_PROFILES_DIR / "byof-datagen-rtxpro-smoke.yaml"
BYOF_CONTAINER_SMOKE_YAML = BYOF_PROFILES_DIR / "byof-container-smoke-rtxpro.yaml"
RTXPRO_SKYPILOT_CONFIG = BYOF_PROFILES_DIR / "skypilot-kubernetes-rtxpro.yaml"
BYOF_ONBOARD_SKILL = WORKSPACE_ROOT / "skills" / "workflows" / "byof-onboard" / "SKILL.md"

DEFAULT_VALIDATION_REPO_URL = "https://github.com/LightwheelAI/leisaac.git"
DEFAULT_VALIDATION_REPO_REF = "main"
DEFAULT_UBUNTU_VALIDATION_REPO_URL = "https://github.com/githubtraining/hellogitworld.git"
DEFAULT_UBUNTU_VALIDATION_REPO_REF = "master"


class ByofProfileError(ValueError):
    """Raised when a ``--yaml`` resource-profile value cannot be resolved."""


def available_byof_profiles() -> tuple[str, ...]:
    """Return the packaged resource-profile names, without the ``.yaml`` suffix."""

    if not BYOF_PROFILES_DIR.is_dir():  # pragma: no cover - packaging failure
        return ()
    return tuple(sorted(path.stem for path in BYOF_PROFILES_DIR.glob("*.yaml")))


def resolve_byof_profile_path(value: str | Path) -> Path:
    """Resolve a ``--yaml`` resource-profile value to a real file.

    Accepts, in order:

    1. an absolute path, or any path that exists relative to the current directory —
       the operator-host case, unchanged;
    2. a bare profile **name** (with or without ``.yaml``) packaged under
       ``BYOF_PROFILES_DIR``;
    3. a path whose *basename* matches a packaged profile — so a value written as a repo
       path still resolves from an installed package.

    Cases 2 and 3 exist because a BYOF stage in an npa.workflow spec runs in a pod where
    the repo is **not** checked out: five specs passed
    ``npa/src/npa/workflows/byof/profiles/byof-solution-smoke-rtxpro-gpu.yaml``, which can
    only resolve on a developer's machine. The packaged profile always resolves, so specs
    name the profile instead of a path (guarded by
    ``tests/guardrails/test_spec_paths_are_not_repo_relative.py``).
    """

    raw = str(value).strip()
    if not raw:
        raise ByofProfileError("resource profile value is empty")

    candidate = Path(raw)
    if candidate.is_absolute() or candidate.exists():
        return candidate

    named = BYOF_PROFILES_DIR / (raw if raw.endswith(".yaml") else f"{raw}.yaml")
    if named.is_file():
        return named

    packaged = BYOF_PROFILES_DIR / candidate.name
    if packaged.is_file():
        return packaged

    raise ByofProfileError(
        f"resource profile not found: {raw}. Pass an existing path or one of the packaged "
        f"profile names: {', '.join(available_byof_profiles())}"
    )


@dataclass(frozen=True)
class ByofKubernetesTarget:
    context: str
    kubeconfig: str
    namespace: str = "default"


def resolve_byof_project() -> str:
    """Return the project alias for BYOF live runs (env override, then ~/.npa/config.yaml)."""

    for key in ("NPA_E2E_PROJECT", "NPA_AGENT_PROJECT", "NPA_PROJECT"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    projects = list_projects()
    if not projects:
        return ""
    preferred = default_project_name()
    if preferred in projects:
        return preferred
    return sorted(projects)[0]


def byof_validation_repo() -> tuple[str, str]:
    """Return the OSS repo used to validate BYOF container layout on live infra."""

    url = os.environ.get("NPA_BYOF_REPO_URL", "").strip() or os.environ.get(
        "NPA_BYOF_VALIDATION_REPO_URL", DEFAULT_VALIDATION_REPO_URL
    ).strip()
    ref = os.environ.get("NPA_BYOF_REPO_REF", "").strip() or os.environ.get(
        "NPA_BYOF_VALIDATION_REPO_REF", DEFAULT_VALIDATION_REPO_REF
    ).strip()
    return url, ref


def byof_ubuntu_validation_repo() -> tuple[str, str]:
    """Return a small public OSS repo for Ubuntu BYOF container smokes."""

    if os.environ.get("NPA_BYOF_UBUNTU_VALIDATION_REPO_URL", "").strip():
        url = os.environ["NPA_BYOF_UBUNTU_VALIDATION_REPO_URL"].strip()
        ref = os.environ.get("NPA_BYOF_UBUNTU_VALIDATION_REPO_REF", DEFAULT_UBUNTU_VALIDATION_REPO_REF).strip()
        return url, ref or DEFAULT_UBUNTU_VALIDATION_REPO_REF
    url, ref = byof_validation_repo()
    if "leisaac" in url.lower():
        return DEFAULT_UBUNTU_VALIDATION_REPO_URL, DEFAULT_UBUNTU_VALIDATION_REPO_REF
    return url, ref


def byof_onboard_skill_path() -> str:
    return str(BYOF_ONBOARD_SKILL.relative_to(WORKSPACE_ROOT))


def load_byof_onboard_skill_text() -> str:
    if BYOF_ONBOARD_SKILL.is_file():
        return BYOF_ONBOARD_SKILL.read_text(encoding="utf-8")
    return ""


def resolve_byof_base_profile() -> str:
    profile = os.environ.get("NPA_BYOF_BASE_PROFILE", "").strip()
    return profile or "ubuntu"


def _project_kubernetes_block(project: str | None) -> dict[str, object]:
    try:
        proj = _resolve_project_section(_load_yaml(), project)
    except ConfigError:
        return {}
    if not isinstance(proj, dict):
        return {}
    block = proj.get("kubernetes")
    return block if isinstance(block, dict) else {}


def _project_storage_block(project: str | None) -> dict[str, object]:
    try:
        proj = _resolve_project_section(_load_yaml(), project)
    except ConfigError:
        return {}
    if not isinstance(proj, dict):
        return {}
    for key in ("object-storage", "object_storage", "storage"):
        block = proj.get(key)
        if isinstance(block, dict):
            return block
    return {}


def resolve_byof_kubernetes_target(project: str | None = None) -> ByofKubernetesTarget:
    """Resolve kube context/config from env, ~/.npa/config.yaml, or npa cluster state."""

    namespace = (
        os.environ.get("NPA_BYOF_K8S_NAMESPACE")
        or os.environ.get("NPA_K8S_NAMESPACE")
        or "default"
    ).strip() or "default"
    context = (
        os.environ.get("NPA_BYOF_K8S_CONTEXT")
        or os.environ.get("NPA_K8S_CONTEXT")
        or os.environ.get("KUBECONTEXT")
        or ""
    ).strip()
    kubeconfig = (
        os.environ.get("NPA_BYOF_KUBECONFIG")
        or os.environ.get("NPA_KUBECONFIG")
        or os.environ.get("KUBECONFIG")
        or ""
    ).strip()

    k8s = _project_kubernetes_block(project)
    storage = _project_storage_block(project)

    cluster_name = str(
        k8s.get("cluster_name")
        or k8s.get("cluster")
        or os.environ.get("NPA_BYOF_CLUSTER_NAME")
        or ""
    ).strip()

    if not context:
        context = str(k8s.get("context") or k8s.get("k8s_context") or storage.get("k8s_context") or "").strip()
    if not kubeconfig:
        kubeconfig = str(k8s.get("kubeconfig") or k8s.get("kubeconfig_path") or "").strip()

    if cluster_name:
        state = load_cluster_state(cluster_name)
        if state and state.kubeconfig_path:
            kubeconfig = kubeconfig or state.kubeconfig_path
        default_kcfg = kubeconfig_file(cluster_name)
        if not kubeconfig and default_kcfg.is_file():
            kubeconfig = str(default_kcfg)
        if not context:
            context = cluster_name

    if not context:
        yml = _load_yaml()
        global_storage = yml.get("storage")
        if isinstance(global_storage, dict):
            context = str(global_storage.get("k8s_context") or "").strip()

    if not kubeconfig:
        # Fall back to the operator host kubeconfig when project config only
        # records a context name (common on shared Nebius operator VMs).
        default_kube = Path.home() / ".kube" / "config"
        if default_kube.is_file():
            kubeconfig = str(default_kube)

    return ByofKubernetesTarget(context=context, kubeconfig=kubeconfig, namespace=namespace)


def _uses_rtxpro_profile(project: str | None) -> bool:
    k8s = _project_kubernetes_block(project)
    accel = str(
        k8s.get("gpu_accelerator")
        or k8s.get("accelerators")
        or k8s.get("gpu_product")
        or ""
    ).upper()
    if "RTXPRO" in accel or "RTX-PRO" in accel or "BLACKWELL" in accel:
        return True
    profile = str(k8s.get("byof_profile") or k8s.get("gpu_profile") or "").strip().lower()
    return profile in {"rtxpro", "rtx6000", "rtx-pro"}


def resolve_byof_resource_yaml(
    project: str | None,
    *,
    smoke: bool = False,
    workload: str = "rl-train",
) -> str:
    """Pick a SkyPilot train/datagen YAML from project config or env overrides."""

    override = os.environ.get("NPA_BYOF_RESOURCE_YAML", "").strip()
    if override:
        return override

    normalized_workload = (workload or os.environ.get("NPA_BYOF_WORKLOAD", "rl-train")).strip().lower()
    k8s = _project_kubernetes_block(project)
    if normalized_workload in {"container-verify", "solution-smoke"}:
        key = "byof_container_smoke_yaml" if smoke else "byof_container_yaml"
        configured = str(k8s.get(key) or "").strip()
        if configured:
            path = Path(configured)
            return str(path if path.is_absolute() else REPO_ROOT / configured)
        if BYOF_CONTAINER_SMOKE_YAML.is_file():
            return str(BYOF_CONTAINER_SMOKE_YAML)
    if normalized_workload == "datagen":
        key = "byof_datagen_smoke_yaml" if smoke else "byof_datagen_yaml"
        configured = str(k8s.get(key) or "").strip()
        if configured:
            path = Path(configured)
            return str(path if path.is_absolute() else REPO_ROOT / configured)
        if _uses_rtxpro_profile(project) and BYOF_DATAGEN_SMOKE_YAML.is_file():
            return str(BYOF_DATAGEN_SMOKE_YAML)

    key = "byof_train_smoke_yaml" if smoke else "byof_train_yaml"
    configured = str(k8s.get(key) or "").strip()
    if configured:
        path = Path(configured)
        return str(path if path.is_absolute() else REPO_ROOT / configured)

    if _uses_rtxpro_profile(project):
        candidate = RTXPRO_SMOKE_TRAIN_YAML if smoke else RTXPRO_TRAIN_YAML
        if candidate.is_file():
            return str(candidate)

    return str(DEFAULT_TRAIN_YAML)


def skypilot_config_for_project(project: str | None) -> str:
    override = os.environ.get("NPA_BYOF_SKYPILOT_CONFIG", "").strip()
    if override:
        return override
    k8s = _project_kubernetes_block(project)
    configured = str(k8s.get("skypilot_config") or k8s.get("byof_skypilot_config") or "").strip()
    if configured:
        path = Path(configured)
        return str(path if path.is_absolute() else REPO_ROOT / configured)
    if _uses_rtxpro_profile(project) and RTXPRO_SKYPILOT_CONFIG.is_file():
        return str(RTXPRO_SKYPILOT_CONFIG)
    return ""


def resolve_skypilot_bin() -> str:
    explicit = os.environ.get("NPA_SKYPILOT_BIN", "").strip()
    if explicit:
        return explicit.rstrip("/")
    found = shutil.which("sky")
    return found or ""
