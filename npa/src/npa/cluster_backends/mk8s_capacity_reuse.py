"""Prove zero added capacity when a repair removes only its owned CPU pool."""

import json
from pathlib import Path
import re


def is_cpu_pool_removal(saved: str, desired: str) -> bool:
    """Compare normalized renderings while permitting only complete CPU removal.

    Args:
        saved: Previously rendered, normalized Terraform assignments.
        desired: Requested, normalized Terraform assignments.
    Returns:
        Whether only a positive CPU count and its unused shape are removed.
    Raises:
        None.
    """
    pattern = r"(?m)^cpu_nodes_fixed_count=([1-9][0-9]*)$"
    if len(re.findall(pattern, saved)) != 1:
        return False
    reduced = re.sub(pattern, "cpu_nodes_fixed_count=0", saved)
    unused = {"cpu_nodes_platform", "cpu_nodes_preset", "cpu_disk_size"}
    reduced = "\n".join(
        line for line in reduced.splitlines() if line.split("=", 1)[0] not in unused
    )
    return reduced == desired


def retained_node_groups(
    groups: list, state_path: Path, cluster_id: str
) -> list | None:
    """Exclude only the exact Terraform-owned CPU group slated for removal.

    Args:
        groups: Complete fresh provider node-group inventory.
        state_path: This target's retained Terraform state file.
        cluster_id: Independently verified parent cluster identity.
    Returns:
        Groups requiring normal capacity proof, or None for uncertain ownership.
    Raises:
        None; malformed or unavailable evidence fails closed.
    """
    try:
        state = json.loads(state_path.read_text())
        owned = [
            instance
            for resource in state.get("resources", [])
            if resource.get("mode") == "managed"
            and resource.get("type") == "nebius_mk8s_v1_node_group"
            and resource.get("name") == "cpu-only"
            and not resource.get("module")
            for instance in resource.get("instances", [])
        ]
        if len(owned) != 1 or owned[0].get("deposed"):
            return None
        attributes = owned[0]["attributes"]
        if not attributes.get("id") or attributes.get("parent_id") != cluster_id:
            return None
        return _exclude_cpu_group(groups, attributes, cluster_id)
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return None


def _exclude_cpu_group(groups, attributes, cluster_id):
    matches = [g for g in groups if g.get("metadata", {}).get("id") == attributes["id"]]
    if not matches:
        return groups
    if len(matches) != 1:
        return None
    group = matches[0]
    metadata = group.get("metadata") or {}
    template = group.get("spec", {}).get("template", {})
    resources = template.get("resources") or {}
    if (
        metadata.get("parent_id") != cluster_id
        or metadata.get("name") != attributes.get("name")
        or not str(resources.get("platform", "")).startswith("cpu-")
        or resources != attributes.get("template", {}).get("resources")
        or template.get("reservation_policy")
    ):
        return None
    return [g for g in groups if g is not group]


def is_removed_cpu_taint(resource, instance, pool, cluster_ids, cluster_name):
    """Keep a removed CPU group's taint so Terraform deletes it without adoption.

    Args:
        resource: Managed node-group resource from this target's Terraform state.
        instance: Exact tainted instance within that resource.
        pool: Desired CPU pool, explicitly retaining its shape with count zero.
        cluster_ids: Exact managed cluster identities from the same state.
        cluster_name: Desired cluster name used by the recipe.
    Returns:
        Whether the state proves this is only an owned CPU pool being removed.
    Raises:
        None.
    """
    attributes = instance.get("attributes") or {}
    template = attributes.get("template") or {}
    return bool(
        pool is not None
        and pool.count == 0
        and resource.get("name") == "cpu-only"
        and not resource.get("module")
        and not instance.get("deposed")
        and len(cluster_ids) == 1
        and attributes.get("id")
        and attributes.get("parent_id") in cluster_ids
        and attributes.get("name") == cluster_name + "-ng-cpu"
        and pool.platform.startswith("cpu-")
        and template.get("resources")
        == {"platform": pool.platform, "preset": pool.preset}
        and not template.get("reservation_policy")
    )
