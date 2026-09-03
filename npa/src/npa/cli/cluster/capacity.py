"""GPU quota and capacity preflight for ``npa cluster up``.

Nebius rejects a GPU node group with ``QuotaFailure`` when the tenant's GPU
allowance is exhausted, but Terraform keeps printing ``Still creating...`` while
the node group retries, so the operator sees a silent multi-hour wait instead of
the reason. Both facts are readable up front:

* ``nebius quotas quota-allowance get-by-name --name compute.instance.gpu.<gpu>``
  gives the tenant's per-region GPU limit and current usage;
* ``nebius capacity resource-advice list`` gives per-platform/preset availability,
  including how much *preemptible* capacity is free when on-demand is not.

Everything here is best-effort: an unreadable quota or advice listing skips the
check rather than blocking a healthy provision.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable

#: Platform prefix stripped to build the quota name: platform ``gpu-rtx6000``
#: maps to quota ``compute.instance.gpu.rtx6000``. Suffixes that name a form
#: factor rather than the GPU model are dropped, since the quota counts GPUs.
_PLATFORM_SUFFIXES = ("-sxm", "-pcie", "-nvl")

CaptureFn = Callable[[list[str]], Any]


def gpu_quota_name(platform: str) -> str:
    """Return the tenant GPU quota name for a Nebius compute *platform*."""
    value = str(platform or "").strip().lower()
    if not value.startswith("gpu-"):
        return ""
    model = value[len("gpu-") :]
    for suffix in _PLATFORM_SUFFIXES:
        if model.endswith(suffix):
            model = model[: -len(suffix)]
            break
    return f"compute.instance.gpu.{model}" if model else ""


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def platform_advice(items: Iterable[dict[str, Any]], *, platform: str, preset: str, region: str) -> dict[str, Any]:
    """Return the ``resource-advice`` entry matching *platform*/*preset*/*region*.

    Falls back to any entry for the platform in the region (a different preset
    still tells the operator whether the GPU model has capacity at all), then to
    ``{}``.
    """
    platform = str(platform or "").strip().lower()
    preset = str(preset or "").strip().lower()
    region = str(region or "").strip().lower()
    fallback: dict[str, Any] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        spec = item.get("spec") or {}
        instance = spec.get("compute_instance") or {}
        if str(spec.get("region", "")).strip().lower() != region:
            continue
        if str(instance.get("platform", "")).strip().lower() != platform:
            continue
        entry_preset = str((instance.get("preset") or {}).get("name", "")).strip().lower()
        if entry_preset == preset:
            return item
        fallback = fallback or item
    return fallback


def _availability(entry: dict[str, Any], key: str) -> str:
    section = (entry.get("status") or {}).get(key) or {}
    level = str(section.get("availability_level", "") or "").strip()
    level = level.removeprefix("AVAILABILITY_LEVEL_") or "UNKNOWN"
    # Nebius reports `available` for the preemptible pool and `limit` for
    # on-demand; conflating them reads as "LIMIT_REACHED (available 2)".
    available = _int_or_none(section.get("available"))
    if available is not None:
        return f"{level} (available {available})"
    limit = _int_or_none(section.get("limit"))
    return f"{level} (limit {limit})" if limit is not None else level


def capacity_summary(entry: dict[str, Any]) -> str:
    """Return a one-line on-demand / preemptible availability summary."""
    if not entry:
        return ""
    instance = (entry.get("spec") or {}).get("compute_instance") or {}
    preset = str((instance.get("preset") or {}).get("name", "") or "?")
    return (
        f"capacity for {instance.get('platform', '?')} / {preset}: "
        f"on-demand {_availability(entry, 'on_demand')}, "
        f"preemptible {_availability(entry, 'preemptible')}, "
        f"reserved {_availability(entry, 'reserved')}"
    )


def preemptible_available(entry: dict[str, Any]) -> int:
    section = ((entry or {}).get("status") or {}).get("preemptible") or {}
    return _int_or_none(section.get("available")) or 0


def _json_payload(capture: CaptureFn, args: list[str]) -> dict[str, Any]:
    result = capture(args)
    if getattr(result, "returncode", 0) != 0:
        return {}
    try:
        payload = json.loads(getattr(result, "stdout", "") or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def gpu_quota_headroom(
    capture: CaptureFn,
    *,
    nebius_bin: str,
    tenant_id: str,
    region: str,
    quota_name: str,
) -> tuple[int, int] | None:
    """Return ``(usage, limit)`` GPUs for *quota_name*, or None when unreadable."""
    payload = _json_payload(
        capture,
        [
            nebius_bin,
            "quotas",
            "quota-allowance",
            "get-by-name",
            "--parent-id",
            tenant_id,
            "--region",
            region,
            "--name",
            quota_name,
            "--format",
            "json",
        ],
    )
    limit = _int_or_none((payload.get("spec") or {}).get("limit"))
    if limit is None:
        return None
    # Nebius omits `status.usage` entirely for a quota with nothing allocated —
    # which is exactly the `limit: "0"` case this gate exists for. Treating the
    # missing field as "unreadable" made the check fail open and let the apply
    # hang on `Still creating...` until the Terraform timeout.
    usage = _int_or_none((payload.get("status") or {}).get("usage")) or 0
    return usage, limit


def capacity_advice_items(
    capture: CaptureFn, *, nebius_bin: str, tenant_id: str
) -> list[dict[str, Any]]:
    payload = _json_payload(
        capture,
        [
            nebius_bin,
            "capacity",
            "resource-advice",
            "list",
            "--parent-id",
            tenant_id,
            "--all",
            "--format",
            "json",
        ],
    )
    items = payload.get("items")
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def capacity_block_group_payload(
    capture: CaptureFn, *, nebius_bin: str, block_group_id: str
) -> dict[str, Any] | None:
    """Read one capacity block group by ID, or None when it cannot be read.

    Read-only inventory used to prove a STRICT reservation is active, in the
    right tenant/region, for the right platform, and has enough free GPUs --
    the reservation-side gate that replaces the ordinary GPU-family quota for a
    bound pool.
    """

    result = capture(
        [
            nebius_bin,
            "capacity",
            "capacity-block-group",
            "get",
            "--id",
            block_group_id,
            "--format",
            "json",
        ]
    )
    raw = getattr(result, "stdout", "") or ""
    if getattr(result, "returncode", 1) != 0 or not raw.strip():
        return None
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001 - unreadable reservation must fail closed
        return None


def capacity_block_group_error(
    capture: CaptureFn,
    *,
    nebius_bin: str,
    block_group_id: str,
    tenant_id: str,
    region: str,
    platform: str,
    required_gpus: int,
) -> str | None:
    """Return an actionable error when the STRICT reservation cannot cover the
    request, else None.

    The ordinary GPU-family quota does not govern a bound pool: reservation
    GPUs come from the named block. This check therefore validates the block
    itself instead -- owner tenant, region, platform, active state, and remaining
    capacity. Fail-closed: an unreadable block blocks (None is only returned for
    a genuinely sufficient, correct reservation).
    """

    payload = capacity_block_group_payload(
        capture, nebius_bin=nebius_bin, block_group_id=block_group_id
    )
    if payload is None:
        return (
            f"Could not read capacity block group {block_group_id}; refusing to "
            "apply a STRICT reservation-backed node group without capacity evidence. "
            "Check `nebius capacity capacity-block-group get --id "
            "<capacity-block-group-id>` (owner/region/platform/state) and that the "
            "block is active and in this tenant."
        )
    metadata = payload.get("metadata") or {}
    status = payload.get("status") or {}
    parent_id = str(metadata.get("parent_id") or "")
    if parent_id and parent_id != tenant_id:
        return (
            f"Capacity block group {block_group_id} belongs to a different tenant "
            f"than {tenant_id}; refusing to apply."
        )
    block_region = str(status.get("region") or "")
    if block_region and block_region != region:
        return (
            f"Capacity block group {block_group_id} is in region {block_region!r}, "
            f"not {region!r}; refusing to apply."
        )
    affinity = (status.get("resource_affinity") or {}).get("compute_v1") or {}
    block_platform = str(affinity.get("platform") or "")
    if block_platform and block_platform != platform:
        return (
            f"Capacity block group {block_group_id} is scoped to platform "
            f"{block_platform!r}, not {platform!r}; refusing to apply."
        )
    state = str(status.get("state") or "")
    if state and state != "STATE_ACTIVE":
        return (
            f"Capacity block group {block_group_id} is not active "
            f"(state {state!r}); refusing to apply."
        )
    limit = _int_or_none(status.get("current_limit"))
    usage = _int_or_none(status.get("usage"))
    if limit is not None:
        used = 0 if usage is None else usage
        available = max(0, limit - used)
        if available < required_gpus:
            return (
                f"Capacity block group {block_group_id} has {available} of "
                f"{limit} GPU(s) free, but this cluster requests {required_gpus}; "
                "refusing to apply a STRICT reservation-backed node group it cannot "
                "satisfy."
            )
    return None


def capacity_advice_reachable(
    capture: CaptureFn, *, nebius_bin: str, tenant_id: str
) -> bool:
    """Whether `capacity resource-advice list` answered at all.

    The service can return ``Unavailable``, which is indistinguishable from "no
    matching entry" once the payload is parsed -- so the remedy would tell an
    operator to run the very command that had just failed.
    """

    result = capture(
        [
            nebius_bin,
            "capacity",
            "resource-advice",
            "list",
            "--parent-id",
            tenant_id,
            "--all",
            "--format",
            "json",
        ]
    )
    return getattr(result, "returncode", 0) == 0


def gpu_capacity_error(
    capture: CaptureFn,
    *,
    nebius_bin: str,
    tenant_id: str,
    region: str,
    platform: str,
    preset: str,
    required_gpus: int,
    preemptible: bool = False,
) -> str | None:
    """Return an actionable error when the tenant cannot get *required_gpus*, else None.

    Preemptible node groups draw on a different pool, so the on-demand GPU quota
    is not the gate for them; the check is skipped in that case.
    """
    if required_gpus <= 0 or preemptible or not tenant_id or not region:
        return None
    quota_name = gpu_quota_name(platform)
    if not quota_name:
        return None
    headroom = gpu_quota_headroom(
        capture,
        nebius_bin=nebius_bin,
        tenant_id=tenant_id,
        region=region,
        quota_name=quota_name,
    )
    if headroom is None:
        return None
    usage, limit = headroom
    free = limit - usage
    if free >= required_gpus:
        return None

    advice = platform_advice(
        capacity_advice_items(capture, nebius_bin=nebius_bin, tenant_id=tenant_id),
        platform=platform,
        preset=preset,
        region=region,
    )
    lines = [
        f"GPU quota is insufficient in {region}: {quota_name} allows {limit} GPU(s) "
        f"with {usage} in use ({free} free), but this cluster requests {required_gpus}.",
        "Nebius rejects the node group with QuotaFailure and Terraform then retries "
        "silently for hours, so this stops before `terraform apply`.",
    ]
    summary = capacity_summary(advice)
    if summary:
        lines.append(f"Reported {summary}.")
    remedies = [
        f"ask a tenant admin to raise the {quota_name} quota for {region}",
    ]
    if preemptible_available(advice) >= required_gpus:
        remedies.append(
            "or run the GPU node group as preemptible: "
            "`gpu_nodes_preemptible = true` in terraform.tfvars "
            "(TF_VAR_gpu_nodes_preemptible=true), which draws on the preemptible pool"
        )
    else:
        remedies.append(
            "or try preemptible capacity (`gpu_nodes_preemptible = true`), a smaller "
            "`gpu_nodes_preset`, or another `gpu_nodes_platform`"
        )
    if advice or capacity_advice_reachable(
        capture, nebius_bin=nebius_bin, tenant_id=tenant_id
    ):
        remedies.append(
            "see what is available with `nebius capacity resource-advice list "
            f"--parent-id {tenant_id} --all`"
        )
    else:
        remedies.append(
            "the capacity advice API did not answer, so there is no availability "
            "hint to act on here -- raise the quota, or try preemptible/another "
            "platform and let the apply tell you"
        )
    lines.append("Fix: " + "; ".join(remedies) + ".")
    return " ".join(lines)
