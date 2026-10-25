"""Validate explicit Soperator reservations when tenant inventory is forbidden."""

from dataclasses import replace
import json

from npa.cluster_backends.quotas import (
    ReservationRequirement,
    find_reservation_shortfalls,
    parse_capacity_blocks,
    reservation_shortfall_message,
)


def resolve_project_reservations(spec, reader, existing_usage, region, inventory=None):
    """Bind STRICT worker IDs to verified incremental project entitlement.

    Args:
        spec: Selected Soperator desired state.
        reader: Exact profile, tenant and project capacity reader.
        existing_usage: Applied GPU counts by reservation and preset.
        region: Verified project region.
        inventory: Tenant reservation evidence if it was readable.
    Returns:
        Desired state with resolved explicit reservation IDs.
    Raises:
        ValueError: Selectors, compatibility or project entitlement are unverified.
    """
    selectors = _reservation_selectors(spec, inventory)
    requirements = _project_requirements(spec, existing_usage, region, selectors)
    if inventory is not None:
        blocks = parse_capacity_blocks(json.dumps(inventory))
        shortfalls = find_reservation_shortfalls(requirements, blocks, reader.tenant_id)
        if shortfalls:
            raise ValueError(reservation_shortfall_message(shortfalls))
    reader.verify_project_allowances(requirements)
    workers = [
        replace(pool, resolved_capacity_block_group_id=selectors[pool.name])
        if pool.capacity_mode() == "reserved"
        else pool
        for pool in spec.workers
    ]
    return replace(spec, workers=workers)


def _reservation_selectors(spec, inventory):
    selectors = {}
    for pool in spec.workers:
        if pool.capacity_mode() != "reserved":
            continue
        if pool.reservation_selector_kind() == "id":
            selectors[pool.name] = pool.capacity_block_group
            continue
        if inventory is None:
            raise ValueError(
                "Tenant reservation names cannot be resolved with project-only access; "
                "supply the immutable capacity_block_group ID"
            )
        matches = [
            row
            for row in inventory["items"]
            if row.get("metadata", {}).get("name") == pool.capacity_block_group_name
        ]
        if len(matches) != 1 or not matches[0].get("metadata", {}).get("id"):
            raise ValueError(
                "worker reservation name is missing or ambiguous in tenant inventory"
            )
        selectors[pool.name] = matches[0]["metadata"]["id"]
    return selectors


def _project_requirements(spec, existing_usage, region, selectors):
    grouped = {}
    for pool in spec.workers:
        if pool.capacity_mode() != "reserved":
            continue
        reservation_id = selectors[pool.name]
        previous = grouped.get(reservation_id)
        if previous and previous[0] != (pool.platform, pool.fabric, pool.preset):
            raise ValueError(
                "one capacity block cannot back incompatible Soperator worker pools"
            )
        desired = (previous[1] if previous else 0) + pool.size * _gpu_count(pool.preset)
        grouped[reservation_id] = ((pool.platform, pool.fabric, pool.preset), desired)
    return {
        reservation_id: ReservationRequirement(
            reservation_id,
            region,
            platform,
            fabric,
            max(0, desired - existing_usage.get((reservation_id, preset), 0)),
        )
        for reservation_id, ((platform, fabric, preset), desired) in grouped.items()
    }


def _gpu_count(preset):
    from npa.cluster_backends.quotas import _preset_gpus

    count = _preset_gpus(preset)
    if count <= 0:
        raise ValueError("reserved worker preset must identify its GPU count")
    return count
