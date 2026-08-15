"""Render a solutions-library ``terraform.tfvars`` from a :class:`SoperatorSpec`.

The solutions-library recipe defaults to a 128x H100 production cluster, so
every nodeset must be set explicitly. This renderer produces a minimal, working
tfvars: a trimmed control plane, one worker nodeset per spec pool (mixed presets
supported), optional per-pool node-local Docker/Enroot image cache
(``NETWORK_SSD_IO_M3``), NFS-in-k8s, accounting/telemetry/backups off by
default, REST resolved through the pinned operator compatibility contract,
upstream-derived control-plane sizing, and ``use_default_apparmor_profile``
wired from the spec. Mandatory direct GPU creation checks live in the lifecycle.
"""

from __future__ import annotations

import json

from npa.soperator.spec import SoperatorSpec, WorkerPoolSpec


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _tfstr(value: str) -> str:
    """Render a Terraform/HCL string with single-pass template escaping."""

    # HCL quoted strings still evaluate ${...} interpolation and %{...}
    # directives after JSON-compatible escaping. Scan once so literal forms that
    # are already escaped ($${...} / %%{...}) remain unchanged instead of
    # becoming $$${...} / %%%{...} and changing Terraform template semantics.
    escaped_parts: list[str] = []
    index = 0
    while index < len(value):
        if value.startswith("$${", index):
            escaped_parts.append("$${")
            index += 3
        elif value.startswith("${", index):
            escaped_parts.append("$${")
            index += 2
        elif value.startswith("%%{", index):
            escaped_parts.append("%%{")
            index += 3
        elif value.startswith("%{", index):
            escaped_parts.append("%%{")
            index += 2
        else:
            escaped_parts.append(value[index])
            index += 1
    escaped = "".join(escaped_parts)
    return json.dumps(escaped, ensure_ascii=True)


def _nullable_tfstr(value: str | None) -> str:
    return "null" if value is None else _tfstr(value)


def _render_worker(pool: WorkerPoolSpec) -> str:
    lines: list[str] = []
    lines.append("  {")
    lines.append(f"    name = {_tfstr(pool.name)}")
    lines.append(f"    size = {pool.size}")
    lines.append("    autoscaling = {")
    lines.append("      enabled = false")
    lines.append("    }")
    lines.append("    resource = {")
    lines.append(f"      platform = {_tfstr(pool.platform)}")
    lines.append(f"      preset   = {_tfstr(pool.preset)}")
    lines.append("    }")
    lines.append("    boot_disk = {")
    lines.append('      type                 = "NETWORK_SSD"')
    lines.append(f"      size_gibibytes       = {pool.boot_disk_gib}")
    lines.append("      block_size_kibibytes = 4")
    lines.append("    }")
    if pool.is_gpu():
        lines.append("    gpu_cluster = {")
        lines.append(f"      infiniband_fabric = {_tfstr(pool.fabric)}")
        lines.append("    }")
    reservation_id = (
        pool.resolved_capacity_block_group_id or pool.capacity_block_group
    )
    if pool.capacity_block_group_name and not reservation_id:
        raise ValueError(
            f"worker pool {pool.name}: capacity block name must be provider-resolved "
            "before rendering Terraform"
        )
    lines.append(f"    preemptible = {'{}' if pool.preemptible else 'null'}")
    if reservation_id:
        # STRICT is the only safe reservation policy for an explicit capacity
        # block. AUTO could silently fall back to ordinary on-demand capacity.
        lines.append("    reservation_policy = {")
        lines.append('      policy          = "STRICT"')
        lines.append(f"      reservation_ids = [{_tfstr(reservation_id)}]")
        lines.append("    }")
    else:
        lines.append("    reservation_policy = null")
    lines.append("    features = null")
    lines.append("    create_partition = null")
    lines.append("    ephemeral_nodes                = false")
    lines.append("    initial_number_ephemeral_nodes = 0")
    lines.append("    persistent_volume_claim_retention_policy = {")
    lines.append('      when_deleted = "Delete"')
    lines.append('      when_scaled  = "Delete"')
    lines.append("    }")
    lines.append("    max_pods = 32")
    lines.append("    node_local_jail_submounts = []")
    if pool.docker_cache:
        # Node-local Docker/Enroot image cache disk (IO_M3 by default).
        lines.append("    node_local_image_disk = {")
        lines.append("      enabled = true")
        lines.append("      spec = {")
        lines.append(f"        size_gibibytes  = {pool.docker_cache_gib}")
        lines.append('        filesystem_type = "ext4"')
        lines.append(f"        disk_type       = {_tfstr(pool.docker_cache_disk_type)}")
        lines.append("      }")
        lines.append("    }")
    else:
        lines.append("    node_local_image_disk = {")
        lines.append("      enabled = false")
        lines.append("    }")
    lines.append("  },")
    return "\n".join(lines)


def render_tfvars(spec: SoperatorSpec) -> str:
    """Return the full ``terraform.tfvars`` content for *spec*."""

    workers_block = "\n".join(_render_worker(p) for p in spec.workers)
    root_login_key = spec.explicit_root_login_ssh_public_key()
    ssh_keys = f"  {_tfstr(root_login_key)}," if root_login_key else ""
    telemetry = _bool(spec.telemetry)
    # ActiveChecks in the "dev" scope run NCCL all-reduce + InfiniBand + GPU perf
    # jobs that Error on a CPU-only cluster, hanging wait_for_soperator_activechecks_hr
    # (so terraform apply never returns). The recipe requires a valid scope
    # (variables.tf validation rejects ""), so on CPU-only clusters use "essential",
    # which sets runAfterCreation=false for every GPU/NCCL/IB/perf check and only
    # runs CPU-safe checks (ssh-check, etc.).
    # The pinned Terraform surface exposes REST independently, but its 4.1.6
    # operator skips REST reconciliation without accounting. Only select the
    # REST-backed dev checks for the runnable combination. NPA performs direct
    # login-jail CUDA creation checks for every GPU pool, including accounting-
    # disabled clusters, so essential scope no longer means zero GPU validation.
    rest_enabled = spec.effective_slurm_rest_enabled()
    active_checks_scope = (
        "dev"
        if rest_enabled and spec.accounting and any(p.is_gpu() for p in spec.workers)
        else "essential"
    )
    # filestore_accounting must be non-null only when accounting is enabled, but
    # slurm_nodeset_accounting's variable validation dereferences it even when
    # disabled -- so always provide a valid accounting nodeset object.
    filestore_accounting = (
        "filestore_accounting = {\n"
        "  spec = {\n"
        "    size_gibibytes       = 512\n"
        "    block_size_kibibytes = 4\n"
        "    forbid_deletion      = false\n"
        "  }\n"
        "}"
        if spec.accounting
        else "filestore_accounting = null"
    )

    return f"""# Generated by `npa soperator deploy` from an npa.soperator/v0.0.1 spec.
# Cluster: {spec.name}
company_name = {_tfstr(spec.name)}
production   = false

# --- Storage ---
controller_state_on_filestore = false
filestore_controller_spool = {{
  spec = {{
    size_gibibytes       = 128
    block_size_kibibytes = 4
    forbid_deletion      = false
  }}
}}
filestore_jail = {{
  spec = {{
    size_gibibytes       = {spec.jail_size_gib}
    block_size_kibibytes = 4
    forbid_deletion      = false
  }}
}}
filestore_jail_submounts   = []
allow_empty_jail_submounts = true
{filestore_accounting}

nfs_in_k8s = {{
  enabled         = true
  version         = "1.2.0"
  use_stable_repo = true
  size_gibibytes  = 186
  disk_type       = "NETWORK_SSD_IO_M3"
  filesystem_type = "ext4"
  threads         = 8
}}

# --- Slurm ---
slurm_operator_version = {_tfstr(spec.slurm_operator_version)}
slurm_operator_stable  = true

slurm_nodesets_partitions = [
  {{
    name               = "main"
    is_all             = true
    slurm_nodeset_refs = []
    config             = "Default=YES PriorityTier=10 PreemptMode=OFF MaxTime=INFINITE State=UP OverSubscribe=YES"
  }},
  {{
    name               = "hidden"
    is_all             = true
    slurm_nodeset_refs = []
    config             = "Default=NO PriorityTier=10 PreemptMode=OFF Hidden=YES MaxTime=INFINITE State=UP OverSubscribe=YES"
  }},
]
slurm_partition_config_type = "default"

# --- Nodes ---
slurm_nodeset_system = {{
  min_size = {spec.system_min_size}
  max_size = {spec.effective_system_max_size()}
  resource = {{
    platform = "cpu-d3"
    preset   = {_nullable_tfstr(spec.system_preset)}
  }}
  boot_disk = {{
    type                 = "NETWORK_SSD"
    size_gibibytes       = 128
    block_size_kibibytes = 4
  }}
}}

slurm_nodeset_controller = {{
  size = 1
  resource = {{
    platform = "cpu-d3"
    preset   = {_nullable_tfstr(spec.controller_preset)}
  }}
  boot_disk = {{
    type                 = "NETWORK_SSD"
    size_gibibytes       = 128
    block_size_kibibytes = 4
  }}
}}

slurm_nodeset_workers = [
{workers_block}
]

use_preinstalled_gpu_drivers = true

slurm_nodeset_login = {{
  size = 1
  resource = {{
    platform = "cpu-d3"
    preset   = {_tfstr(spec.login_preset)}
  }}
  boot_disk = {{
    type                 = "NETWORK_SSD"
    size_gibibytes       = 256
    block_size_kibibytes = 4
  }}
}}

# accounting_enabled gates whether this nodeset creates an instance; the object
# is required for variable validation regardless.
slurm_nodeset_accounting = {{
  resource = {{
    platform = "cpu-d3"
    preset   = {_nullable_tfstr(spec.accounting_preset)}
  }}
  boot_disk = {{
    type                 = "NETWORK_SSD"
    size_gibibytes       = 128
    block_size_kibibytes = 4
  }}
}}

slurm_nodeset_nfs = null

slurm_login_public_ip = true
slurm_login_ssh_root_public_keys = [
{ssh_keys}
]

slurm_exporter_enabled = {_bool(spec.telemetry)}
active_checks_scope    = "{active_checks_scope}"
slurm_shared_memory_size_gibibytes = 16
slurm_topology_block_size = null
maintenance_ignore_node_groups = ["controller", "nfs"]

telemetry_enabled        = {telemetry}
public_o11y_enabled      = {telemetry}
dcgm_job_mapping_enabled = {telemetry}
opentelemetry_delete_jail_logs_after_read = false
soperator_notifier   = {{ enabled = false }}
nccl_inspector_profiling = {{ enabled = false }}

accounting_enabled = {_bool(spec.accounting)}
slurm_rest_enabled = {_bool(rest_enabled)}

backups_enabled           = "force_disable"
backups_password          = "unused-backups-disabled"
backups_schedule          = "@daily-random"
backups_prune_schedule    = "@daily-random"
backups_retention         = {{ keepDaily = 7 }}
cleanup_bucket_on_destroy = false

k8s_version        = {_tfstr(spec.k8s_version)}
node_group_version = {_tfstr(spec.node_group_version)}
nvidia_config_lines = [
  "options nvidia NVreg_RestrictProfilingToAdminUsers=0",
  "options nvidia NVreg_EnableStreamMemOPs=1",
]

# With the verified chart contract, NPA configures the Ubuntu user-namespace
# sysctls when this is false so Enroot/Pyxis can run unconfined. Overrides are
# version-gated by spec validation.
use_default_apparmor_profile = {_bool(spec.use_default_apparmor_profile)}
"""
