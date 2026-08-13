"""Deploy, reconcile, destroy, and repair npa-managed Soperator clusters.

NPA wraps an immutable, runtime-asserted ``nebius-solutions-library`` Soperator
Terraform contract. It applies the monitoring prerequisites, stalled-dashboard
repair, ``ncclInspectorPreConf`` CRD compatibility patch, prefixed Slurm scripts
configmap, Ubuntu user-namespace configuration, best-effort worker recovery,
and direct creation-time CUDA checks needed by that contract. REST is explicit
at the NPA surface, while runtime validation accounts for the pinned operator's
remaining REST/accounting limitation.

Reuses the terraform subprocess helpers from ``npa.cli.cluster.terraform_lifecycle``.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, replace
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from npa.cli.cluster.terraform_lifecycle import (
    _require_bin,
    _run_capture,
    _run_stream,
    _terraform_env,
)
from npa.clients.config import resolve_environment
from npa.soperator.spec import (
    DEFAULT_SOLUTIONS_LIBRARY_REF,
    DEFAULT_SLURM_OPERATOR_VERSION,
    SoperatorSpec,
    validate_ssh_public_key_record,
)
from npa.soperator.tfvars import render_tfvars

_SOLUTIONS_LIBRARY_REPO = "https://github.com/nebius/nebius-solutions-library.git"
_IMMUTABLE_GIT_REF_RE = re.compile(r"^[0-9a-f]{40}$")
_PROMETHEUS_CRD_BASE = (
    "https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/"
    "v0.76.0/example/prometheus-operator-crd"
)
_PROMETHEUS_CRDS = (
    "monitoring.coreos.com_servicemonitors.yaml",
    "monitoring.coreos.com_podmonitors.yaml",
    "monitoring.coreos.com_probes.yaml",
)
_MONITORING_NAMESPACE = "monitoring-system"
_MONITORING_RELEASE_SUFFIX = "-monitoring-dashboards"
_ACTIVECHECKS_RELEASE_SUFFIX = "-soperator-activechecks"

# Sidecar written next to the generated tfvars so ``destroy`` can rebuild the
# same TF_VAR_* env the recipe requires. region/tenant/project/subnet/o11y are
# passed as env vars at apply time (not persisted in terraform.tfvars), so a
# later ``terraform destroy`` would fail on "No value for required variable"
# without these.
_ENV_SIDECAR = ".npa-soperator-env.json"


class UpstreamContractError(ValueError):
    """Raised before provider mutation when the pinned recipe contract differs."""


@dataclass(frozen=True)
class ResolvedRootLoginSSHKey:
    """One validated public key that explicitly grants login-node root access."""

    value: str
    source: str
    fingerprint: str


def _write_env_sidecar(
    install_dir: Path,
    *,
    region: str,
    tenant_id: str,
    project_id: str,
    subnet_id: str,
    o11y_profile: str,
) -> None:
    (install_dir / _ENV_SIDECAR).write_text(
        json.dumps(
            {
                "region": region,
                "tenant_id": tenant_id,
                "project_id": project_id,
                "subnet_id": subnet_id,
                "o11y_profile": o11y_profile,
            },
            indent=2,
        )
    )


def _load_env_sidecar(install_dir: Path) -> dict[str, str] | None:
    path = install_dir / _ENV_SIDECAR
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _log(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)


def _api_domain(region: str) -> str:
    """Nebius API domain for a region (the recipe hardcodes the EU domain)."""

    return "api.eu.nebius.cloud:443" if region.startswith("eu") else "api.nebius.cloud:443"


def _root_login_key_fingerprint(value: str) -> str:
    blob = base64.b64decode(value.split(maxsplit=2)[1], validate=True)
    digest = base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
    return f"SHA256:{digest}"


def _read_root_login_key_file(path: Path, *, source: str) -> ResolvedRootLoginSSHKey:
    try:
        value = path.expanduser().read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read {source} root-login SSH public-key file") from exc
    normalized = validate_ssh_public_key_record(value)
    return ResolvedRootLoginSSHKey(
        value=normalized,
        source=source,
        fingerprint=_root_login_key_fingerprint(normalized),
    )


def _resolve_root_login_ssh_public_key(
    spec: SoperatorSpec,
    *,
    explicit_file: Path | None = None,
    environ: dict[str, str] | None = None,
    home: Path | None = None,
) -> ResolvedRootLoginSSHKey:
    """Resolve the one key granting root access to the public login node.

    Precedence is: explicit CLI/SDK file, canonical or legacy spec field,
    Soperator-specific inline environment value, Soperator-specific environment
    file, generic ``NPA_SSH_PUBLIC_KEY`` file, then conventional operator-home
    key discovery. Only a single OpenSSH public-key record is accepted.
    """

    if explicit_file is not None:
        return _read_root_login_key_file(explicit_file, source="explicit argument")

    explicit = spec.explicit_root_login_ssh_public_key()
    if explicit:
        normalized = validate_ssh_public_key_record(explicit)
        source = (
            "spec root_login_ssh_public_key"
            if spec.root_login_ssh_public_key
            else "legacy spec ssh_public_keys"
        )
        return ResolvedRootLoginSSHKey(
            value=normalized,
            source=source,
            fingerprint=_root_login_key_fingerprint(normalized),
        )

    env = os.environ if environ is None else environ
    inline = env.get("NPA_SOPERATOR_ROOT_LOGIN_SSH_PUBLIC_KEY", "").strip()
    if inline:
        normalized = validate_ssh_public_key_record(inline)
        return ResolvedRootLoginSSHKey(
            value=normalized,
            source="NPA_SOPERATOR_ROOT_LOGIN_SSH_PUBLIC_KEY",
            fingerprint=_root_login_key_fingerprint(normalized),
        )

    for env_name in (
        "NPA_SOPERATOR_ROOT_LOGIN_SSH_PUBLIC_KEY_FILE",
        "NPA_SSH_PUBLIC_KEY",
    ):
        configured_path = env.get(env_name, "").strip()
        if configured_path:
            return _read_root_login_key_file(Path(configured_path), source=env_name)

    ssh_dir = (home or Path.home()).expanduser() / ".ssh"
    for name in ("id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub"):
        candidate = ssh_dir / name
        if candidate.is_file():
            return _read_root_login_key_file(candidate, source=f"operator default {name}")
    raise ValueError(
        "soperator login-node root access requires one SSH public key: set "
        "root_login_ssh_public_key in the spec, pass "
        "--root-login-ssh-public-key-file, set "
        "NPA_SOPERATOR_ROOT_LOGIN_SSH_PUBLIC_KEY[_FILE], or create "
        "~/.ssh/id_ed25519.pub"
    )


def _with_resolved_ssh_public_keys(
    spec: SoperatorSpec, *, home: Path | None = None
) -> SoperatorSpec:
    """Compatibility wrapper returning *spec* with its canonical root-login key."""

    resolved = _resolve_root_login_ssh_public_key(spec, home=home)
    if (
        spec.root_login_ssh_public_key == resolved.value
        and not spec.ssh_public_keys
    ):
        return spec
    return replace(
        spec,
        root_login_ssh_public_key=resolved.value,
        ssh_public_keys=[],
    )


def _validate_immutable_solutions_library_ref(ref: str) -> str:
    normalized = ref.strip().lower()
    if not _IMMUTABLE_GIT_REF_RE.fullmatch(normalized):
        raise ValueError(
            "solutions_library_ref must be an immutable 40-character commit SHA; "
            "branches and moving tags are not accepted"
        )
    return normalized


def _resolve_solutions_library(terraform_dir: Path | None, work_root: Path, ref: str) -> Path:
    """Resolve a checkout of one immutable solutions-library commit."""

    ref = _validate_immutable_solutions_library_ref(ref)

    if terraform_dir is not None:
        path = terraform_dir.expanduser().resolve()
        if not (path / "installations" / "example").exists():
            raise ValueError(
                f"{path} is not a soperator recipe dir (missing installations/example)"
            )
        return path
    # Preserve the historical default location when it exists so an already
    # deployed installation keeps using its Terraform state. The contract
    # assertion immediately after resolution still requires its HEAD to equal
    # the requested immutable commit and permits only NPA's two known patches.
    legacy_clone = work_root / "nebius-solutions-library"
    if (legacy_clone / "soperator" / "installations" / "example").exists():
        return legacy_clone / "soperator"
    clone_dir = work_root / f"nebius-solutions-library-{ref[:12]}"
    if not (clone_dir / "soperator" / "installations" / "example").exists():
        work_root.mkdir(parents=True, exist_ok=True)
        git = _require_bin("git")
        _run_stream(
            [git, "clone", "--filter=blob:none", "--no-checkout", _SOLUTIONS_LIBRARY_REPO, str(clone_dir)],
            timeout=600,
        )
        _run_stream(
            [git, "checkout", "--detach", ref],
            cwd=clone_dir,
            timeout=600,
        )
    return clone_dir / "soperator"


def _nebius_cli_env() -> dict[str, str]:
    """Environment for direct ``nebius`` CLI calls (pre-flight / cleanup).

    A stale ambient ``NEBIUS_IAM_TOKEN`` (e.g. an expired cloud-env token left in
    the parent process) is used by the CLI in preference to the active profile's
    exec-plugin, so pre-flight calls like ``vpc subnet list`` fail Unauthenticated
    even though the profile can mint a fresh token. Drop it so the CLI falls back
    to the auto-refreshing profile credential -- unless the caller explicitly opts
    into reuse (NPA_REUSE_IAM_TOKEN, e.g. CI injecting a short-lived token).
    """

    env = os.environ.copy()
    if env.get("NPA_NEBIUS_PROFILE", "").strip() and not env.get(
        "NEBIUS_PROFILE", ""
    ).strip():
        env["NEBIUS_PROFILE"] = env["NPA_NEBIUS_PROFILE"].strip()
    reuse = env.get("NPA_REUSE_IAM_TOKEN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not reuse:
        env.pop("NEBIUS_IAM_TOKEN", None)
    return env


def _resolve_subnet(nebius_bin: str, project_id: str, env: dict[str, str]) -> str:
    result = _run_capture(
        [nebius_bin, "vpc", "subnet", "list", "--parent-id", project_id, "--format", "json"],
        env=env,
    )
    payload = json.loads(result.stdout or "{}")
    items = payload.get("items") or []
    if not items:
        raise ValueError(f"no VPC subnet found in project {project_id}")
    return str(items[0].get("metadata", {}).get("id") or "")


_ESSENTIAL_HEALTHY_NODES_MARKER = "# npa: CPU-only clusters disable the GPU checks"
_ESSENTIAL_HEALTHY_NODES_OVERRIDE = (
    "      " + _ESSENTIAL_HEALTHY_NODES_MARKER + "\n"
    "      # that ensure-healthy-nodes dependsOn, so its creation run never fires\n"
    "      # and wait-for-active-checks (which gates the activechecks HelmRelease,\n"
    "      # and thus terraform apply) deadlocks. Skip it at creation time.\n"
    "      ensure-healthy-nodes = {\n"
    "        runAfterCreation = false\n"
    "      }\n"
)
_NODECONFIGURATOR_USERNS_MARKER = "# npa: allow Enroot user namespaces on Ubuntu hosts"
_NODECONFIGURATOR_USERNS_VALUES = (
    "              " + _NODECONFIGURATOR_USERNS_MARKER + "\n"
    "              # Preserve the chart's complete default init-container list; Helm\n"
    "              # replaces arrays rather than merging individual list entries.\n"
    "              initContainers:\n"
    "                - name: node-sysctl-params\n"
    "                  image: cr.eu-north1.nebius.cloud/soperator/busybox\n"
    "                  securityContext:\n"
    "                    privileged: true\n"
    "                    runAsUser: 0\n"
    "                    runAsGroup: 0\n"
    "                    readOnlyRootFilesystem: false\n"
    "                    allowPrivilegeEscalation: true\n"
    "                  command:\n"
    "                    - /bin/sh\n"
    "                    - -c\n"
    "                    - |-\n"
    "                      sysctl -w kernel.unprivileged_userns_clone=1\n"
    "                      if [ -e /proc/sys/kernel/apparmor_restrict_unprivileged_userns ] && [ \"${apparmor_enabled}\" = \"false\" ]; then\n"
    "                        sysctl -w kernel.apparmor_restrict_unprivileged_userns=0\n"
    "                      fi\n"
    "                      sysctl -w net.core.rmem_max=536870912\n"
    "                      sysctl -w net.core.wmem_max=536870912\n"
    "                      sysctl -w net.ipv4.tcp_rmem=\"4096 131072 536870912\"\n"
    "                      sysctl -w net.ipv4.tcp_wmem=\"4096 16384 536870912\"\n"
)


def _patch_active_checks_text(text: str) -> tuple[str, bool]:
    if _ESSENTIAL_HEALTHY_NODES_MARKER in text:
        return text, False
    marker = "    essential = {\n"
    idx = text.find(marker)
    if idx == -1:
        raise UpstreamContractError(
            "pinned active-checks contract lacks the essential scope"
        )
    insert_at = idx + len(marker)
    return (
        text[:insert_at] + _ESSENTIAL_HEALTHY_NODES_OVERRIDE + text[insert_at:],
        True,
    )


def _patch_active_checks_locals(recipe_dir: Path) -> bool:
    """Ensure the ``essential`` active-checks scope skips ``ensure-healthy-nodes``.

    On a CPU-only cluster npa selects the ``essential`` scope, which sets
    ``runAfterCreation = false`` on every GPU/NCCL/IB/perf check. But
    ``ensure-healthy-nodes`` (a slurmJob check) ``dependsOn`` those very checks,
    so with them disabled its creation run never triggers and its status stays
    empty. ``wait-for-active-checks`` -- the Helm hook that gates the activechecks
    HelmRelease, and therefore ``terraform apply`` -- waits for every
    ``runAfterCreation = true`` check to reach a terminal state, so it hangs until
    the 2h Helm timeout. Add ``ensure-healthy-nodes = { runAfterCreation = false }``
    to the ``essential`` scope (mirroring the recipe's own gb300 handling) so the
    hook no longer waits on it. Idempotent; returns True if a patch was applied.
    """

    locals_tf = recipe_dir / "modules" / "slurm" / "locals_active_checks.tf"
    if not locals_tf.exists():
        return False
    text = locals_tf.read_text()
    patched, changed = _patch_active_checks_text(text)
    if changed:
        locals_tf.write_text(patched)
    return changed


def _yaml_mapping_block_bounds(text: str, key: str) -> tuple[int, int, int]:
    """Return byte bounds and indentation for one YAML mapping block.

    The Terraform template is YAML with interpolation expressions; parsing it as
    ordinary YAML is unreliable. Indentation is nevertheless structural, so a
    bounded mapping walk prevents a missing child from matching a later chart.
    """

    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    matches: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        offsets.append(offset)
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if stripped.rstrip("\r\n") == f"{key}:":
            matches.append((index, indent))
        offset += len(line)
    if len(matches) != 1:
        raise UpstreamContractError(
            f"pinned Helm template must contain exactly one {key!r} mapping; "
            f"found {len(matches)}"
        )
    start_index, indent = matches[0]
    end_index = len(lines)
    for index in range(start_index + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("%{"):
            continue
        candidate = lines[index].lstrip(" ")
        candidate_indent = len(lines[index]) - len(candidate)
        if candidate_indent <= indent:
            end_index = index
            break
    start = offsets[start_index]
    end = offsets[end_index] if end_index < len(offsets) else len(text)
    return start, end, indent


def _patch_nodeconfigurator_text(text: str) -> tuple[str, bool]:
    section_start, section_end, section_indent = _yaml_mapping_block_bounds(
        text, "nodeConfigurator"
    )
    section = text[section_start:section_end]
    marker_inside = _NODECONFIGURATOR_USERNS_MARKER in section
    marker_anywhere = _NODECONFIGURATOR_USERNS_MARKER in text
    if marker_anywhere and not marker_inside:
        raise UpstreamContractError(
            "nodeConfigurator user-namespace marker exists outside its chart block"
        )
    if marker_inside:
        return text, False

    values_line = " " * (section_indent + 2) + "values:\n"
    matches = [match.start() for match in re.finditer(re.escape(values_line), section)]
    if len(matches) != 1:
        raise UpstreamContractError(
            "pinned nodeConfigurator block must contain its own values: mapping; "
            "refusing to inject into a sibling chart"
        )
    insert_at = section_start + matches[0] + len(values_line)
    return (
        text[:insert_at] + _NODECONFIGURATOR_USERNS_VALUES + text[insert_at:],
        True,
    )


def _patch_nodeconfigurator_userns(recipe_dir: Path) -> bool:
    """Teach the upstream node configurator about Ubuntu's AppArmor userns gate.

    The verified chart enables ``kernel.unprivileged_userns_clone`` but Ubuntu's
    newer host images independently deny unprivileged user namespaces through
    ``kernel.apparmor_restrict_unprivileged_userns``. Enroot/Pyxis image startup
    then fails even though the worker container itself is AppArmor-unconfined.
    Override the chart's full init-container list and disable the second gate
    only when Soperator's default AppArmor profile is intentionally disabled.
    """

    template = (
        recipe_dir
        / "modules"
        / "slurm"
        / "templates"
        / "helm_values"
        / "terraform_fluxcd_values.yaml.tftpl"
    )
    if not template.exists():
        return False
    text = template.read_text()
    patched, changed = _patch_nodeconfigurator_text(text)
    if changed:
        template.write_text(patched)
    return changed


def _git_checkout_text(repo_root: Path, ref: str, relative_path: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{ref}:{relative_path}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise UpstreamContractError(
            f"could not read {relative_path} from solutions-library {ref[:12]}"
        )
    return proc.stdout


def _assert_solutions_library_contract(
    recipe_dir: Path,
    *,
    ref: str,
) -> None:
    """Assert the complete pinned source/mutation contract before cloud writes."""

    ref = _validate_immutable_solutions_library_ref(ref)
    repo_root = recipe_dir.parent
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if head.returncode != 0 or head.stdout.strip().lower() != ref:
        actual = head.stdout.strip()[:12] if head.returncode == 0 else "not-a-git-checkout"
        raise UpstreamContractError(
            f"solutions-library checkout mismatch: expected {ref[:12]}, got {actual}"
        )

    critical_fragments = {
        "soperator/installations/example/terraform.tfvars": (
            f'slurm_operator_version = "{DEFAULT_SLURM_OPERATOR_VERSION}"',
            "k8s_version = 1.34",
            "node_group_version = 72",
        ),
        "soperator/installations/example/main.tf": (
            "rest_enabled                    = var.slurm_rest_enabled",
            "accounting_enabled              = var.accounting_enabled",
            "sizing_tier_override = module.sizing.sizing_tier",
        ),
        "soperator/installations/example/variables.tf": (
            'variable "slurm_rest_enabled"',
            'variable "accounting_enabled"',
            'variable "sizing_tier_override"',
        ),
        "soperator/modules/sizing_tier/main.tf": (
            'var.worker_count < 10 ? "XS"',
            'var.worker_count < 100 ? "S"',
            'var.worker_count < 500 ? "M"',
            'var.worker_count < 2000 ? "L" : "XL"',
            'L  = "32vcpu-128gb"',
            'XL = "64vcpu-256gb"',
        ),
    }
    for relative, fragments in critical_fragments.items():
        pristine = _git_checkout_text(repo_root, ref, relative)
        current_path = repo_root / relative
        try:
            current = current_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise UpstreamContractError(f"missing pinned contract file {relative}") from exc
        if current != pristine:
            raise UpstreamContractError(
                f"unexpected local mutation in pinned contract file {relative}"
            )
        missing = [fragment for fragment in fragments if fragment not in current]
        if missing:
            raise UpstreamContractError(
                f"pinned runtime contract is incompatible in {relative}: "
                f"missing {missing[0]!r}"
            )

    patch_contracts = (
        (
            "soperator/modules/slurm/locals_active_checks.tf",
            _patch_active_checks_text,
        ),
        (
            "soperator/modules/slurm/templates/helm_values/"
            "terraform_fluxcd_values.yaml.tftpl",
            _patch_nodeconfigurator_text,
        ),
    )
    for relative, transform in patch_contracts:
        pristine = _git_checkout_text(repo_root, ref, relative)
        expected_patched, _ = transform(pristine)
        current_path = repo_root / relative
        try:
            current = current_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise UpstreamContractError(f"missing pinned patch target {relative}") from exc
        if current not in (pristine, expected_patched):
            raise UpstreamContractError(
                f"unexpected local mutation in pinned patch target {relative}"
            )


def _prepare_installation(recipe_dir: Path, spec: SoperatorSpec, region: str) -> Path:
    """Create installations/<name> with the recipe files + generated tfvars."""

    _patch_active_checks_locals(recipe_dir)
    if not spec.use_default_apparmor_profile:
        _patch_nodeconfigurator_userns(recipe_dir)
    example = recipe_dir / "installations" / "example"
    install_dir = recipe_dir / "installations" / spec.name
    install_dir.mkdir(parents=True, exist_ok=True)
    for item in ("main.tf", "variables.tf", "terraform.tf", "driver_presets.tf"):
        src = example / item
        if src.exists():
            shutil.copy2(src, install_dir / item)
    assets = example / "assets"
    if assets.exists():
        shutil.copytree(assets, install_dir / "assets", dirs_exist_ok=True)

    # Patch the hardcoded provider domain for the target region.
    terraform_tf = install_dir / "terraform.tf"
    if terraform_tf.exists():
        text = terraform_tf.read_text()
        text = text.replace("api.eu.nebius.cloud:443", _api_domain(region))
        terraform_tf.write_text(text)

    (install_dir / "terraform.tfvars").write_text(render_tfvars(spec))
    return install_dir


def _soperator_tf_env(
    nebius_bin: str,
    *,
    region: str,
    tenant_id: str,
    project_id: str,
    subnet_id: str,
) -> dict[str, str]:
    profile = (
        os.environ.get("NPA_NEBIUS_PROFILE", "").strip()
        or os.environ.get("NEBIUS_PROFILE", "").strip()
    )
    env = _terraform_env(nebius_bin, profile=profile)
    if profile:
        # Terraform local-exec and kubeconfig generation invoke the bare CLI;
        # keep them on the same explicitly selected cross-tenant principal.
        env["NEBIUS_PROFILE"] = profile
    env["TF_VAR_region"] = region
    env["TF_VAR_iam_tenant_id"] = tenant_id
    env["TF_VAR_iam_project_id"] = project_id
    # o11y is disabled in tfvars, but the variables are required to parse.
    env["TF_VAR_o11y_iam_tenant_id"] = tenant_id
    env["TF_VAR_o11y_profile"] = profile or "default"
    env["TF_VAR_vpc_subnet_id"] = subnet_id
    return env


def _terraform_cluster_id(terraform_bin: str, install_dir: Path, env: dict[str, str]) -> str:
    """Return the mk8s cluster id from Terraform state (empty if not found)."""

    result = _run_capture(
        [terraform_bin, "state", "pull"], cwd=install_dir, env=env, check=False
    )
    if result.returncode != 0 or not result.stdout.strip():
        return ""
    try:
        state = json.loads(result.stdout)
    except json.JSONDecodeError:
        return ""
    for resource in state.get("resources", []):
        if resource.get("type") != "nebius_mk8s_v1_cluster":
            continue
        for instance in resource.get("instances", []):
            cid = instance.get("attributes", {}).get("id")
            if cid:
                return str(cid)
    return ""


def _find_cluster_id_by_name(
    nebius_bin: str, project_id: str, cluster_name: str, env: dict[str, str]
) -> str:
    """Return the mk8s cluster id matching *cluster_name* (empty if none)."""

    result = _run_capture(
        [nebius_bin, "mk8s", "cluster", "list", "--parent-id", project_id, "--format", "json"],
        env=env,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return ""
    try:
        items = json.loads(result.stdout).get("items", [])
    except json.JSONDecodeError:
        return ""
    for item in items:
        meta = item.get("metadata", {})
        if meta.get("name") == cluster_name and meta.get("id"):
            return str(meta["id"])
    return ""


def _refresh_kube_credentials(
    nebius_bin: str, cluster_id: str, context: str, env: dict[str, str]
) -> None:
    """Write an admin kubeconfig context for the cluster (recipe writes a limited SA)."""

    argv = [nebius_bin]
    profile = (
        env.get("NPA_NEBIUS_PROFILE", "").strip()
        or env.get("NEBIUS_PROFILE", "").strip()
    )
    if profile:
        argv.extend(["--profile", profile])
    _run_capture(
        [
            *argv, "mk8s", "cluster", "get-credentials",
            "--id", cluster_id, "--external", "--force", "--context-name", context,
        ],
        env=env,
        check=False,
    )


def _install_monitoring_crds(
    kubectl_bin: str, context: str, *, on_status: Callable[[str], None] | None = None
) -> None:
    """Install prometheus-operator CRDs the soperator operator chart requires.

    The operator chart creates a ServiceMonitor unconditionally; with telemetry
    off the recipe never installs its CRD, so the operator HelmRelease cannot
    install. These must be present before the operator reconciles.

    kubectl runs with the ambient ``NEBIUS_IAM_TOKEN`` stripped (via
    ``_nebius_cli_env``): a stale token shadows the kubeconfig exec-plugin and
    makes the apply fail Unauthenticated. Each apply is retried, and the
    ServiceMonitor CRD is confirmed registered before returning -- swallowing a
    failure here otherwise surfaces only ~an hour later as an operator
    HelmRelease InstallFailed and a ``wait_for_slurm_cluster_hr`` timeout.
    """

    kube_env = _nebius_cli_env()
    _ensure_monitoring_namespace(kubectl_bin, context, env=kube_env)
    _log(on_status, "installing prometheus-operator CRDs (ServiceMonitor/PodMonitor/Probe)")
    for crd in _PROMETHEUS_CRDS:
        last: subprocess.CompletedProcess[str] | None = None
        for _attempt in range(3):
            last = _run_capture(
                [kubectl_bin, "--context", context, "apply", "--server-side", "-f",
                 f"{_PROMETHEUS_CRD_BASE}/{crd}"],
                env=kube_env,
                check=False,
            )
            if last.returncode == 0:
                break
            time.sleep(5)
        if last is None or last.returncode != 0:
            detail = (last.stderr or last.stdout).strip() if last else ""
            raise RuntimeError(
                f"failed to install prometheus-operator CRD {crd} after 3 attempts"
                + (f": {detail}" if detail else "")
            )
    # Confirm the ServiceMonitor CRD is actually registered: the operator chart
    # renders a ServiceMonitor and cannot install without it, so a no-op apply
    # (wrong context / swallowed auth error) must fail loudly here, not later.
    check = _run_capture(
        [kubectl_bin, "--context", context, "get", "crd",
         "servicemonitors.monitoring.coreos.com", "-o", "name"],
        env=kube_env,
        check=False,
    )
    if check.returncode != 0 or not check.stdout.strip():
        detail = (check.stderr or check.stdout).strip()
        raise RuntimeError(
            "prometheus-operator ServiceMonitor CRD not present after install"
            + (f": {detail}" if detail else "")
        )
    reset = _reset_stalled_monitoring_releases(kubectl_bin, context, env=kube_env)
    if reset:
        _log(on_status, f"reset {reset} stalled monitoring HelmRelease(s)")


def _ensure_monitoring_namespace(
    kubectl_bin: str, context: str, *, env: dict[str, str] | None = None
) -> None:
    """Ensure the namespace required by the unconditional dashboards chart.

    The pinned Soperator contract still reconciles monitoring dashboards when observability is
    disabled, but that mode does not create ``monitoring-system``.  Creating the
    namespace is idempotent and lets Flux install the chart instead of leaving a
    permanently failed HelmRelease in an otherwise healthy cluster.
    """

    kube_env = env or _nebius_cli_env()
    get = _run_capture(
        [
            kubectl_bin,
            "--context",
            context,
            "get",
            "namespace",
            _MONITORING_NAMESPACE,
            "-o",
            "name",
        ],
        env=kube_env,
        check=False,
    )
    if get.returncode == 0 and get.stdout.strip():
        return
    create = _run_capture(
        [
            kubectl_bin,
            "--context",
            context,
            "create",
            "namespace",
            _MONITORING_NAMESPACE,
        ],
        env=kube_env,
        check=False,
    )
    if create.returncode == 0:
        return

    # Another reconciler may create the namespace between our read and write.
    # Confirm the desired end state before treating a failed create as fatal.
    confirm = _run_capture(
        [
            kubectl_bin,
            "--context",
            context,
            "get",
            "namespace",
            _MONITORING_NAMESPACE,
            "-o",
            "name",
        ],
        env=kube_env,
        check=False,
    )
    if confirm.returncode == 0 and confirm.stdout.strip():
        return

    detail = (create.stderr or create.stdout).strip()
    raise RuntimeError(
        f"failed to ensure {_MONITORING_NAMESPACE} namespace"
        + (f": {detail}" if detail else "")
    )


def _reset_stalled_monitoring_releases(
    kubectl_bin: str, context: str, *, env: dict[str, str] | None = None
) -> int:
    """Reset failed dashboards releases after repairing their prerequisites.

    Flux stops retrying a HelmRelease after its remediation budget is exhausted.
    A rerun of NPA can therefore repair ``monitoring-system`` and the CRDs yet
    remain blocked on the old failure. Flux's paired ``requestedAt``/``resetAt``
    annotations reset that counter. Clean installs have no HelmRelease at this
    point, and healthy releases are left untouched.
    """

    kube_env = env or _nebius_cli_env()
    listed = _run_capture(
        [
            kubectl_bin,
            "--context",
            context,
            "-n",
            "flux-system",
            "get",
            "helmreleases",
            "-o",
            "json",
        ],
        env=kube_env,
        check=False,
    )
    if listed.returncode != 0:
        raw_detail = (listed.stderr or listed.stdout).strip()
        detail = raw_detail.lower()
        if (
            "not found" in detail
            or "doesn't have a resource type" in detail
            or "the server could not find the requested resource" in detail
        ):
            return 0
        raise RuntimeError(
            "failed to inspect monitoring HelmReleases"
            + (f": {raw_detail}" if raw_detail else "")
        )

    try:
        payload = json.loads(listed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("failed to inspect monitoring HelmReleases: invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("failed to inspect monitoring HelmReleases: invalid JSON object")

    reset = 0
    for item in payload.get("items") or []:
        metadata = item.get("metadata") or {}
        name = str(metadata.get("name") or "")
        if not name.endswith(_MONITORING_RELEASE_SUFFIX):
            continue
        conditions = (item.get("status") or {}).get("conditions") or []
        stalled = any(
            condition.get("status") == "True" and condition.get("type") == "Stalled"
            for condition in conditions
        )
        retries_exhausted = any(
            condition.get("type") == "Ready"
            and condition.get("status") == "False"
            and condition.get("reason") == "RetriesExceeded"
            for condition in conditions
        )
        if not (stalled or retries_exhausted):
            continue
        token = str(time.time_ns())
        annotated = _run_capture(
            [
                kubectl_bin,
                "--context",
                context,
                "-n",
                "flux-system",
                "annotate",
                "helmrelease",
                name,
                f"reconcile.fluxcd.io/requestedAt={token}",
                f"reconcile.fluxcd.io/resetAt={token}",
                "--overwrite",
            ],
            env=kube_env,
            check=False,
        )
        if annotated.returncode != 0:
            detail = (annotated.stderr or annotated.stdout).strip()
            raise RuntimeError(
                f"failed to reset monitoring HelmRelease {name}"
                + (f": {detail}" if detail else "")
            )
        reset += 1
    return reset


def _abort_superseded_activechecks_upgrade(
    kubectl_bin: str,
    context: str,
    *,
    namespace: str = "soperator",
    env: dict[str, str] | None = None,
) -> list[str]:
    """Unblock a newer ActiveChecks generation from an older Helm action.

    An upgrade can remain in its generated ``wait-for-active-checks`` hook when
    an old REST-backed generation cannot observe Slurm job status. Flux cannot
    start an already-rendered newer generation until that action exits. Only
    when ``lastAttemptedGeneration`` is older than metadata.generation and the
    release is actively Progressing, delete the Helm-owned hook Job and request
    a reset/reconcile. Current-generation installs are never interrupted.
    """

    kube_env = env or _nebius_cli_env()
    listed = _run_capture(
        [
            kubectl_bin,
            "--context",
            context,
            "-n",
            "flux-system",
            "get",
            "helmreleases",
            "-o",
            "json",
        ],
        env=kube_env,
        check=False,
    )
    if listed.returncode != 0:
        return []
    try:
        items = json.loads(listed.stdout or "{}").get("items", [])
    except (AttributeError, json.JSONDecodeError):
        return []

    reset: list[str] = []
    for item in items:
        metadata = item.get("metadata") or {}
        status = item.get("status") or {}
        name = str(metadata.get("name") or "")
        generation = int(metadata.get("generation") or 0)
        attempted = int(status.get("lastAttemptedGeneration") or 0)
        progressing = any(
            condition.get("type") == "Reconciling"
            and condition.get("status") == "True"
            and condition.get("reason") == "Progressing"
            for condition in (status.get("conditions") or [])
        )
        if (
            not name.endswith(_ACTIVECHECKS_RELEASE_SUFFIX)
            or not progressing
            or attempted <= 0
            or attempted >= generation
        ):
            continue
        deleted = _run_capture(
            [
                kubectl_bin,
                "--context",
                context,
                "-n",
                namespace,
                "delete",
                "job",
                "wait-for-active-checks",
                "--ignore-not-found=true",
                "--wait=false",
            ],
            env=kube_env,
            check=False,
        )
        if deleted.returncode != 0:
            continue
        token = str(time.time_ns())
        annotated = _run_capture(
            [
                kubectl_bin,
                "--context",
                context,
                "-n",
                "flux-system",
                "annotate",
                "helmrelease",
                name,
                f"reconcile.fluxcd.io/requestedAt={token}",
                f"reconcile.fluxcd.io/resetAt={token}",
                "--overwrite",
            ],
            env=kube_env,
            check=False,
        )
        if annotated.returncode == 0:
            reset.append(name)
    return reset


def _patch_slurmcluster_crd(kubectl_bin: str, context: str) -> bool:
    """Patch the SlurmCluster CRD to accept plugStackConfig.ncclInspectorPreConf.

    Idempotent. Returns True once the CRD exists and the patch is applied. The
    CRD is created by the operator, so this only succeeds after the operator
    installs -- callers should retry until it returns True.
    """

    kube_env = _nebius_cli_env()
    got = _run_capture(
        [kubectl_bin, "--context", context, "get", "crd",
         "slurmclusters.slurm.nebius.ai", "-o", "name"],
        env=kube_env,
        check=False,
    )
    if got.returncode != 0 or not got.stdout.strip():
        return False
    _run_capture(
        [kubectl_bin, "--context", context, "patch", "crd",
         "slurmclusters.slurm.nebius.ai", "--type=json", "-p",
         '[{"op":"add","path":"/spec/versions/0/schema/openAPIV3Schema/'
         'properties/spec/properties/plugStackConfig/'
         'x-kubernetes-preserve-unknown-fields","value":true}]'],
        env=kube_env,
        check=False,
    )
    return True


def _ensure_scripts_configmap(kubectl_bin: str, context: str, namespace: str) -> bool:
    """Create the cluster-name-prefixed <ns>-slurm-scripts configmap.

    The nodesets chart mounts ``<ns>-slurm-scripts`` while the slurm-cluster
    chart creates the unprefixed ``slurm-scripts`` (a chart naming skew).
    Idempotent; returns True once the prefixed copy exists.
    """

    kube_env = _nebius_cli_env()
    target = f"{namespace}-slurm-scripts"
    exists = _run_capture(
        [kubectl_bin, "--context", context, "get", "cm", target, "-n", namespace, "-o", "name"],
        env=kube_env,
        check=False,
    )
    if exists.returncode == 0 and exists.stdout.strip():
        return True
    src = _run_capture(
        [kubectl_bin, "--context", context, "get", "cm", "slurm-scripts",
         "-n", namespace, "-o", "json"],
        env=kube_env,
        check=False,
    )
    if src.returncode != 0 or not src.stdout.strip():
        return False
    try:
        cm = json.loads(src.stdout)
    except json.JSONDecodeError:
        return False
    cm["metadata"] = {"name": target, "namespace": namespace}
    subprocess.run(
        [kubectl_bin, "--context", context, "apply", "-f", "-"],
        input=json.dumps(cm),
        text=True,
        env=kube_env,
        check=False,
    )
    return True


def _mid_apply_fix_loop(
    kubectl_bin: str,
    context: str,
    name: str,
    *,
    namespace: str = "soperator",
    stop: "threading.Event | None" = None,
    on_status: Callable[[str], None] | None = None,
) -> None:
    """Apply mid-apply fixes while phase 2 blocks on the slurm-cluster HelmRelease.

    The operator creates the SlurmCluster CRD and the slurm-scripts configmap
    *during* phase 2, and the slurm-cluster / nodesets HelmReleases then block on
    the CRD patch + the prefixed configmap. Poll and apply both as soon as they
    appear so phase 2 can converge unattended.
    """

    crd_done = False
    cm_done = False
    logged_crd = False
    logged_cm = False
    while stop is None or not stop.is_set():
        if not crd_done and _patch_slurmcluster_crd(kubectl_bin, context):
            crd_done = True
            if not logged_crd:
                _log(on_status, "mid-apply: patched SlurmCluster CRD (ncclInspectorPreConf)")
                logged_crd = True
        if not cm_done and _ensure_scripts_configmap(kubectl_bin, context, namespace):
            cm_done = True
            if not logged_cm:
                _log(on_status, f"mid-apply: ensured {namespace}-slurm-scripts configmap")
                logged_cm = True
        if crd_done and cm_done:
            return
        if stop is not None:
            stop.wait(15)
        else:
            time.sleep(15)


def deploy_cluster(
    spec: SoperatorSpec,
    *,
    terraform_dir: Path | None = None,
    work_root: Path | None = None,
    solutions_library_ref: str = DEFAULT_SOLUTIONS_LIBRARY_REF,
    root_login_ssh_public_key_file: Path | None = None,
    project: str | None = None,
    timeout_minutes: int = 90,
    apply_fixes: bool = True,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Deploy or reconcile *spec* after pinned-contract and key preflight."""

    spec.validate()
    root_login_key = _resolve_root_login_ssh_public_key(
        spec, explicit_file=root_login_ssh_public_key_file
    )
    spec = replace(
        spec,
        root_login_ssh_public_key=root_login_key.value,
        ssh_public_keys=[],
    )
    spec.validate()
    _log(
        on_status,
        "login-node root SSH access enabled: "
        f"source={root_login_key.source}; fingerprint={root_login_key.fingerprint}",
    )
    envcfg = resolve_environment(
        project=project,
        project_id=spec.project_id or None,
        tenant_id=spec.tenant_id or None,
        region=spec.region or None,
    )
    region = spec.region or envcfg.region
    tenant_id = spec.tenant_id or envcfg.tenant_id
    project_id = spec.project_id or envcfg.project_id
    if not (region and tenant_id and project_id):
        raise ValueError(
            "region, tenant_id and project_id must be resolvable from the spec or ~/.npa config"
        )

    terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")

    work_root = (work_root or Path.home() / ".npa" / "soperator").expanduser()
    recipe_dir = _resolve_solutions_library(terraform_dir, work_root, solutions_library_ref)
    _assert_solutions_library_contract(recipe_dir, ref=solutions_library_ref)
    _log(on_status, f"verified solutions-library contract {solutions_library_ref[:12]}")
    install_dir = _prepare_installation(recipe_dir, spec, region)
    _log(on_status, f"Installation dir: {install_dir}")

    subnet_id = spec.subnet_id or _resolve_subnet(nebius_bin, project_id, _nebius_cli_env())
    env = _soperator_tf_env(
        nebius_bin,
        region=region,
        tenant_id=tenant_id,
        project_id=project_id,
        subnet_id=subnet_id,
    )
    # Persist the env the recipe requires so ``destroy`` can reconstruct it.
    _write_env_sidecar(
        install_dir,
        region=region,
        tenant_id=tenant_id,
        project_id=project_id,
        subnet_id=subnet_id,
        o11y_profile=env["TF_VAR_o11y_profile"],
    )

    context = f"nebius-{spec.name}-slurm"
    _log(on_status, "terraform init")
    _run_stream([terraform_bin, "init"], cwd=install_dir, env=env, timeout=900)

    if apply_fixes:
        # Two-phase apply: the soperator operator HelmRelease is reconciled inside
        # the full apply and blocks on the prometheus ServiceMonitor CRD. With
        # telemetry off, that CRD is not installed by the recipe, so a single
        # apply times out waiting for the operator. Phase 1 brings up the mk8s
        # cluster + node groups (and writes the kube context); we then refresh
        # admin credentials and install the monitoring CRDs so the operator can
        # install cleanly in phase 2.
        kubectl_bin = _require_bin(os.environ.get("NPA_KUBECTL_BIN") or "kubectl")
        _log(on_status, "terraform apply (phase 1: k8s cluster + node groups)")
        _run_stream(
            [terraform_bin, "apply", "-target=module.k8s", "-auto-approve"],
            cwd=install_dir,
            env=env,
            timeout=timeout_minutes * 60,
        )
        cluster_id = _terraform_cluster_id(terraform_bin, install_dir, env)
        if cluster_id:
            _log(on_status, "refreshing kube admin credentials")
            _refresh_kube_credentials(nebius_bin, cluster_id, context, env)
        _log(on_status, "installing monitoring CRDs (before operator reconcile)")
        _install_monitoring_crds(kubectl_bin, context, on_status=on_status)
        _log(on_status, f"terraform apply (phase 2: operator + Slurm; {len(spec.workers)} worker pool(s))")
        # The SlurmCluster CRD is created by the operator *during* phase 2, and the
        # slurm-cluster HelmRelease then blocks on it accepting
        # plugStackConfig.ncclInspectorPreConf (a chart/CRD skew); the nodesets
        # chart likewise mounts a cluster-name-prefixed slurm-scripts configmap.
        # Both must be fixed mid-apply, so run a concurrent fixer while phase 2
        # blocks on wait_for_slurm_cluster_hr.
        stop = threading.Event()
        fixer = threading.Thread(
            target=_mid_apply_fix_loop,
            args=(kubectl_bin, context, spec.name),
            kwargs={"stop": stop, "on_status": on_status},
            daemon=True,
        )
        fixer.start()
        try:
            _run_stream(
                [terraform_bin, "apply", "-auto-approve"],
                cwd=install_dir,
                env=env,
                timeout=timeout_minutes * 60,
            )
        finally:
            stop.set()
            fixer.join(timeout=10)
    else:
        _log(on_status, f"terraform apply ({len(spec.workers)} worker pool(s))")
        _run_stream(
            [terraform_bin, "apply", "-auto-approve"],
            cwd=install_dir,
            env=env,
            timeout=timeout_minutes * 60,
        )

    result: dict[str, Any] = {
        "name": spec.name,
        "region": region,
        "project_id": project_id,
        "install_dir": str(install_dir),
        "kube_context": context,
        "worker_pools": [p.name for p in spec.workers],
        "docker_cache_pools": [p.name for p in spec.workers if p.docker_cache],
    }

    if apply_fixes:
        kubectl_bin = _require_bin(os.environ.get("NPA_KUBECTL_BIN") or "kubectl")
        warnings = apply_post_deploy_fixes(context, kubectl_bin, on_status=on_status)
        result["post_deploy_fixes"] = "applied"
        result["post_deploy_fix_warnings"] = warnings

    # GPU validation is a deploy contract, not an optional repair. Keep it
    # active even when an operator deliberately selects --skip-fixes.
    kubectl_bin = _require_bin(os.environ.get("NPA_KUBECTL_BIN") or "kubectl")
    result["gpu_creation_checks"] = _run_gpu_creation_checks(
        spec,
        context,
        kubectl_bin,
        on_status=on_status,
    )

    return result


def destroy_cluster(
    name: str,
    *,
    terraform_dir: Path | None = None,
    work_root: Path | None = None,
    solutions_library_ref: str = DEFAULT_SOLUTIONS_LIBRARY_REF,
    project: str | None = None,
    timeout_minutes: int = 90,
    on_status: Callable[[str], None] | None = None,
) -> None:
    """Destroy an npa-managed soperator cluster by name."""

    terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")
    work_root = (work_root or Path.home() / ".npa" / "soperator").expanduser()
    recipe_dir = _resolve_solutions_library(terraform_dir, work_root, solutions_library_ref)
    _assert_solutions_library_contract(recipe_dir, ref=solutions_library_ref)
    install_dir = recipe_dir / "installations" / name
    if not install_dir.exists():
        raise ValueError(f"no installation found for cluster {name!r} at {install_dir}")

    # ``terraform destroy`` still parses the config, so the region/tenant/project/
    # subnet/o11y variables (passed as env at apply time, never written to
    # terraform.tfvars) must be set or destroy fails on "No value for required
    # variable". Prefer the sidecar written at deploy time; fall back to
    # re-resolving from ~/.npa for installs predating the sidecar.
    saved = _load_env_sidecar(install_dir)
    if saved and saved.get("region") and saved.get("tenant_id") and saved.get("project_id"):
        env = _soperator_tf_env(
            nebius_bin,
            region=str(saved["region"]),
            tenant_id=str(saved["tenant_id"]),
            project_id=str(saved["project_id"]),
            subnet_id=str(saved.get("subnet_id") or ""),
        )
        if saved.get("o11y_profile"):
            env["TF_VAR_o11y_profile"] = str(saved["o11y_profile"])
    else:
        envcfg = resolve_environment(project=project)
        region = envcfg.region
        tenant_id = envcfg.tenant_id
        project_id = envcfg.project_id
        if not (region and tenant_id and project_id):
            raise ValueError(
                "cannot resolve region/tenant/project to destroy "
                f"{name!r}: no env sidecar at {install_dir / _ENV_SIDECAR} and "
                "~/.npa config is incomplete (pass --project)"
            )
        subnet_id = _resolve_subnet(nebius_bin, project_id, _nebius_cli_env())
        env = _soperator_tf_env(
            nebius_bin,
            region=region,
            tenant_id=tenant_id,
            project_id=project_id,
            subnet_id=subnet_id,
        )
    _log(on_status, f"terraform destroy: {name}")
    _run_stream([terraform_bin, "init"], cwd=install_dir, env=env, timeout=900)
    cluster_id = _terraform_cluster_id(terraform_bin, install_dir, env)
    project_id = str(
        (saved or {}).get("project_id")
        or env.get("TF_VAR_iam_project_id")
        or ""
    )
    # An interrupted deploy can leave the cloud cluster running while local
    # Terraform state is empty, so cluster_id is blank here. Fall back to finding
    # the mk8s cluster by its recipe name (soperator-<name>) so destroy can still
    # tear it down instead of silently no-op'ing.
    if not cluster_id and project_id:
        cluster_id = _find_cluster_id_by_name(nebius_bin, project_id, f"soperator-{name}", env)
        if cluster_id:
            _log(on_status, f"terraform state empty; found cluster {cluster_id} by name")

    # Reclaim CSI-provisioned PVC disks (NFS + any dynamic volumes) BEFORE the
    # cluster is torn down. Deleting the mk8s cluster does NOT cascade-delete the
    # NETWORK_SSD_IO_M3 disks backing PVCs, so they leak against the (small) IO_M3
    # quota across deploy/destroy cycles. Delete the PVCs while the cluster is
    # still reachable so the CSI provisioner releases their backing disks.
    if cluster_id:
        context = f"nebius-{name}-slurm"
        _refresh_kube_credentials(nebius_bin, cluster_id, context, env)
        kubectl_bin = shutil.which(os.environ.get("NPA_KUBECTL_BIN") or "kubectl")
        if kubectl_bin:
            _log(on_status, "reclaiming CSI PVC disks before teardown")
            _run_capture(
                [kubectl_bin, "--context", context, "delete", "pvc", "--all",
                 "--all-namespaces", "--wait=false", "--timeout=60s"],
                env=env,
                check=False,
                timeout=120,
            )
            time.sleep(20)  # give the CSI provisioner a moment to delete disks

    # Best-effort terraform destroy. The recipe's disk_cleanup local-exec and
    # occasional node-group deletion races can fail even when the cluster itself
    # is removable, so don't hard-fail here -- fall through to a direct delete +
    # state reset so the install dir is reusable and quota is freed.
    destroy = _run_capture(
        [terraform_bin, "destroy", "-auto-approve"],
        cwd=install_dir,
        env=env,
        timeout=timeout_minutes * 60,
        check=False,
    )
    if destroy.returncode != 0:
        _log(on_status, "terraform destroy reported errors; falling back to direct cleanup")

    # Ensure the mk8s cluster is actually gone (cascades node groups + instances).
    if cluster_id:
        still = _run_capture(
            [nebius_bin, "mk8s", "cluster", "get", "--id", cluster_id, "--format", "json"],
            env=env,
            check=False,
        )
        if still.returncode == 0 and still.stdout.strip():
            _log(on_status, f"deleting mk8s cluster {cluster_id} directly")
            _run_capture(
                [nebius_bin, "mk8s", "cluster", "delete", "--id", cluster_id],
                env=env,
                check=False,
                timeout=timeout_minutes * 60,
            )
            # Wait for the cluster to actually disappear before cleaning up VPC
            # allocations below. The delete call can return while the cluster (and
            # its cloud-controller-manager) still exists; if we delete the static-IP
            # allocation while the CCM is alive it will re-create a same-named orphan
            # that isn't in terraform state, and the next deploy fails with
            # "Allocation ... already exists" (AlreadyExists). Poll get until gone.
            deadline = time.monotonic() + timeout_minutes * 60
            while time.monotonic() < deadline:
                gone = _run_capture(
                    [nebius_bin, "mk8s", "cluster", "get", "--id", cluster_id, "--format", "json"],
                    env=env,
                    check=False,
                )
                if gone.returncode != 0 or not gone.stdout.strip():
                    break
                time.sleep(15)

    # Best-effort delete filesystems this cluster created (jail / controller-spool
    # / accounting are named ``soperator-<name>-*``) so they don't linger against
    # quota. NOTE: the recipe prefixes every filesystem with ``soperator-`` -- the
    # match below must use that full prefix, otherwise orphans survive the destroy
    # and the next deploy fails with "filesystem ... already exists" (AlreadyExists).
    if project_id:
        fs_list = _run_capture(
            [nebius_bin, "compute", "filesystem", "list", "--parent-id", project_id, "--format", "json"],
            env=env,
            check=False,
        )
        try:
            items = json.loads(fs_list.stdout or "{}").get("items", [])
        except json.JSONDecodeError:
            items = []
        for item in items:
            meta = item.get("metadata", {})
            fs_name = str(meta.get("name") or "")
            fs_id = str(meta.get("id") or "")
            if fs_id and fs_name.startswith(f"soperator-{name}-"):
                _log(on_status, f"deleting orphaned filesystem {fs_name}")
                _run_capture(
                    [nebius_bin, "compute", "filesystem", "delete", "--id", fs_id],
                    env=env,
                    check=False,
                )

    # Best-effort delete orphaned VPC allocations this cluster created. The recipe
    # provisions a static public IP named ``soperator-<name>-public-static-ip``
    # for the login LoadBalancer. The Nebius cloud-controller-manager can also
    # *re-create* a same-named allocation mid-teardown (a LoadBalancer-service
    # race) after terraform has already deleted the in-state copy, leaving an
    # orphan that isn't in state -- so a later ``terraform apply`` fails with
    # "Allocation with name 'soperator-<name>-public-static-ip' already exists".
    # These are safe to remove once the cluster (and its CCM) is gone. Runs after
    # the direct cluster delete above so the CCM can't re-create them again.
    if project_id:
        alloc_list = _run_capture(
            [nebius_bin, "vpc", "allocation", "list", "--parent-id", project_id, "--format", "json"],
            env=env,
            check=False,
        )
        try:
            items = json.loads(alloc_list.stdout or "{}").get("items", [])
        except json.JSONDecodeError:
            items = []
        for item in items:
            meta = item.get("metadata", {})
            alloc_name = str(meta.get("name") or "")
            alloc_id = str(meta.get("id") or "")
            if alloc_id and alloc_name.startswith(f"soperator-{name}-"):
                _log(on_status, f"deleting orphaned VPC allocation {alloc_name}")
                _run_capture(
                    [nebius_bin, "vpc", "allocation", "delete", "--id", alloc_id],
                    env=env,
                    check=False,
                )

    # Reset local terraform state so the install dir is clean for a redeploy.
    for stale in install_dir.glob("terraform.tfstate*"):
        try:
            stale.unlink()
        except OSError:
            pass
    _log(on_status, f"destroy complete: {name}")


def apply_post_deploy_fixes(
    context: str,
    kubectl_bin: str,
    *,
    namespace: str = "soperator",
    on_status: Callable[[str], None] | None = None,
    timeout_minutes: int = 20,
) -> list[str]:
    """Apply idempotent repairs after a successful Terraform reconciliation.

    Monitoring namespace/CRD/dashboard repair is best-effort here: RBAC or a
    transient Flux error is retained in the returned diagnostics but cannot turn
    an already healthy Terraform apply into a reported deploy failure. A
    superseded ActiveChecks wait hook is reset only when its attempted generation
    is older than the desired generation. The CRD and scripts compatibility
    fixes remain polled/best-effort, followed by worker address/RESUME recovery.
    Returns non-secret warning strings.
    """

    warnings: list[str] = []
    try:
        _install_monitoring_crds(kubectl_bin, context, on_status=on_status)
    except RuntimeError as exc:
        warning = f"monitoring repair skipped after successful apply: {exc}"
        warnings.append(warning)
        _log(on_status, f"post-deploy warning: {warning}")

    reset_activechecks = _abort_superseded_activechecks_upgrade(
        kubectl_bin,
        context,
        namespace=namespace,
    )
    if reset_activechecks:
        _log(
            on_status,
            "post-deploy: aborted a superseded ActiveChecks Helm action so "
            "the newer generation can reconcile",
        )

    _log(on_status, "post-deploy: patching SlurmCluster CRD + ensuring scripts configmap")
    deadline = time.monotonic() + timeout_minutes * 60
    crd_done = False
    cm_done = False
    while time.monotonic() < deadline and not (crd_done and cm_done):
        crd_done = crd_done or _patch_slurmcluster_crd(kubectl_bin, context)
        cm_done = cm_done or _ensure_scripts_configmap(kubectl_bin, context, namespace)
        if crd_done and cm_done:
            break
        time.sleep(15)
    if not crd_done:
        _log(on_status, "post-deploy: SlurmCluster CRD not present yet; skipped CRD patch")
    if not cm_done:
        _log(on_status, "post-deploy: slurm-scripts configmap not present yet; skipped")

    _register_slurm_workers(kubectl_bin, context, namespace, on_status=on_status)
    _log(on_status, "post-deploy: fixes applied" + (" with warnings" if warnings else ""))
    return warnings


def _register_slurm_workers(
    kubectl_bin: str,
    context: str,
    namespace: str,
    *,
    on_status: Callable[[str], None] | None = None,
    wait_minutes: int = 10,
) -> None:
    """Best-effort: bring DOWN worker nodes to IDLE after registration races.

    Dynamic-node registration can race worker readiness and leave slurmctld
    resolving a bare short name. Set
    the FQDN NodeAddr and RESUME any node that is down. Idempotent and non-fatal.
    """

    kube_env = _nebius_cli_env()
    ctl = ["exec", "-n", namespace, "controller-0", "-c", "slurmctld", "--"]

    def slurmctl(args: list[str]) -> subprocess.CompletedProcess[str]:
        return _run_capture(
            [kubectl_bin, "--context", context, *ctl, *args], env=kube_env, check=False
        )

    deadline = time.monotonic() + wait_minutes * 60
    while time.monotonic() < deadline:
        info = slurmctl(["sinfo", "-h", "-N", "-o", "%N %t"])
        if info.returncode != 0 or not info.stdout.strip():
            time.sleep(15)
            continue
        down = [
            line.split()[0]
            for line in info.stdout.splitlines()
            if line.split() and (line.split()[1].endswith("*") or "down" in line.split()[1].lower())
        ]
        if not down:
            _log(on_status, "post-deploy: all Slurm worker nodes are responding")
            return
        for node in sorted(set(down)):
            fqdn = f"{node}.soperator-nodeset-svc.{namespace}.svc.cluster.local"
            slurmctl(["scontrol", "update", f"NodeName={node}", f"NodeAddr={fqdn}"])
            slurmctl(["scontrol", "update", f"NodeName={node}", "State=RESUME"])
        _log(on_status, f"post-deploy: registered worker node(s): {', '.join(sorted(set(down)))}")
        time.sleep(15)


def _gpu_count_from_preset(preset: str) -> int:
    """Return the leading GPU count from an upstream GPU preset name."""

    match = re.match(r"^([1-9][0-9]*)gpu-", preset)
    if match is None:
        raise RuntimeError(
            f"GPU creation check cannot derive a GPU count from preset {preset!r}"
        )
    return int(match.group(1))


def _run_gpu_creation_checks(
    spec: SoperatorSpec,
    context: str,
    kubectl_bin: str,
    *,
    namespace: str = "soperator",
    on_status: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run real CUDA samples on every GPU worker through the login jail.

    The pinned 4.1.6 Terraform surface exposes REST separately from accounting,
    but the exact operator implementation skips REST reconciliation when
    accounting is disabled. Its REST-backed ActiveCheck controller therefore
    cannot provide creation-time GPU validation for that combination. This
    direct Slurm check is the safe runtime contract: one exclusive task per
    worker receives every GPU on that worker and requires deviceQuery,
    vectorAdd, simpleMultiGPU, and p2pBandwidthLatencyTest to all report PASS.
    Any failed/missing node or CUDA result fails the deploy.
    """

    checks: list[dict[str, Any]] = []
    kube_env = _nebius_cli_env()
    for pool in (worker for worker in spec.workers if worker.is_gpu()):
        gpu_count = _gpu_count_from_preset(pool.preset)
        nodes = [f"{pool.name}-{index}" for index in range(pool.size)]
        node_list = ",".join(nodes)
        task_script = f"""
set -uo pipefail
gpu_count=$(nvidia-smi -L | wc -l)
if [ "$gpu_count" -ne {gpu_count} ]; then
  echo "NPA_GPU_CREATION_CHECK_RESULT host=$(hostname) status=FAIL expected_gpus={gpu_count} actual_gpus=$gpu_count"
  exit 1
fi
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)
case "$gpu_name" in
  "NVIDIA H100") platform=8xH100 ;;
  "NVIDIA H200") platform=8xH200 ;;
  "NVIDIA B200") platform=8xB200 ;;
  "NVIDIA B300") platform=8xB300 ;;
  "NVIDIA GB300") platform=4xGB300 ;;
  *) echo "NPA_GPU_CREATION_CHECK_RESULT host=$(hostname) status=FAIL unsupported_gpu=$gpu_name"; exit 1 ;;
esac
echo "NPA_GPU_CREATION_CHECK_START host=$(hostname) gpu_count=$gpu_count platform=$platform"
out=$(health-checker run -e soperator -p "$platform" \
  -n deviceQuery,vectorAdd,simpleMultiGPU,p2pBandwidthLatencyTest \
  -f json-partial \
  --tests-stdout-path /opt/soperator-outputs/health_checker_cmd_stdout)
command_rc=$?
status=$(printf '%s\n' "$out" | awk '/^[[:space:]]*{{/,/^[[:space:]]*}}/' | jq -r '.status // empty')
echo "NPA_GPU_CREATION_CHECK_RESULT host=$(hostname) status=$status command_rc=$command_rc"
test "$command_rc" -eq 0
test "$status" = PASS
""".strip()
        command = [
            kubectl_bin,
            "--context",
            context,
            "exec",
            "-n",
            namespace,
            "login-0",
            "--",
            "chroot",
            "/mnt/jail",
            "srun",
            "--label",
            f"--nodes={pool.size}",
            f"--ntasks={pool.size}",
            "--ntasks-per-node=1",
            f"--gpus-per-node={gpu_count}",
            "--exclusive",
            f"--nodelist={node_list}",
            "bash",
            "-lc",
            task_script,
        ]
        _log(
            on_status,
            "GPU creation check: "
            f"pool={pool.name}; nodes={pool.size}; GPUs/node={gpu_count}; "
            "tests=deviceQuery,vectorAdd,simpleMultiGPU,p2pBandwidthLatencyTest",
        )
        completed = _run_capture(command, env=kube_env, check=False)
        passes = completed.stdout.count("NPA_GPU_CREATION_CHECK_RESULT")
        passes_with_status = completed.stdout.count("status=PASS")
        if (
            completed.returncode != 0
            or passes != pool.size
            or passes_with_status != pool.size
        ):
            diagnostic = "\n".join(
                (completed.stderr or completed.stdout).strip().splitlines()[-20:]
            )
            raise RuntimeError(
                f"GPU creation check failed for pool {pool.name!r} "
                f"({passes_with_status}/{pool.size} workers reported PASS)"
                + (f":\n{diagnostic}" if diagnostic else "")
            )
        checks.append(
            {
                "pool": pool.name,
                "nodes": pool.size,
                "gpus_per_node": gpu_count,
                "tests": [
                    "deviceQuery",
                    "vectorAdd",
                    "simpleMultiGPU",
                    "p2pBandwidthLatencyTest",
                ],
                "status": "PASS",
            }
        )
        _log(on_status, f"GPU creation check passed: pool={pool.name}; workers={pool.size}")
    return checks
