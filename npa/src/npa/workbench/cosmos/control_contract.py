"""Import-light Cosmos Transfer control and checkpoint contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


class ControlContractError(ValueError):
    """A deterministic control request cannot be executed safely."""


@dataclass(frozen=True)
class ControlCheckpoint:
    modality: str
    upstream_key: str
    repo: str
    revision: str
    filename: str
    control_source: str


COSMOS_TRANSFER_CHECKPOINTS: Mapping[str, ControlCheckpoint] = {
    "edge": ControlCheckpoint(
        "edge",
        "edge",
        "nvidia/Cosmos-Transfer2.5-2B",
        "b67b64abda3801a9aceddbff2bdb86126c06db74",
        "general/edge/61f5694b-0ad5-4ecd-8ad7-c8545627d125_ema_bf16.pt",
        "weight-free Canny control generated from the input",
    ),
    "vis": ControlCheckpoint(
        "vis",
        "blur",
        "nvidia/Cosmos-Transfer2.5-2B",
        "eb5325b77d358944da58a690157dd2b8071bbf85",
        "general/blur/ba2f44f2-c726-4fe7-949f-597069d9b91c_ema_bf16.pt",
        "weight-free bilateral-blur control generated from the input",
    ),
    "depth": ControlCheckpoint(
        "depth",
        "depth",
        "nvidia/Cosmos-Transfer2.5-2B",
        "dea7737ca29dd8d9086413c6dc5724b8250a0bb4",
        "general/depth/626e6618-bfcd-4d9a-a077-1409e2ce353f_ema_bf16.pt",
        "operator-owned precomputed weight-free depth control",
    ),
    "seg": ControlCheckpoint(
        "seg",
        "seg",
        "nvidia/Cosmos-Transfer2.5-2B",
        "23057a4167b89de89a4a397fdbf3887994d115eb",
        "general/seg/5136ef49-6d8d-42e8-8abf-7dac722a304a_ema_bf16.pt",
        "upstream text-driven segmentation control",
    ),
}


def validate_control_request(
    *,
    modality: Any,
    weight: Any,
    control_asset: Any = "",
    control_prompt: Any = "",
    mask_asset: Any = "",
    mask_prompt: Any = "",
) -> tuple[ControlCheckpoint, float]:
    """Validate deterministic control semantics without touching GPU or storage."""

    selected = str(modality or "edge").strip().lower()
    checkpoint = COSMOS_TRANSFER_CHECKPOINTS.get(selected)
    if checkpoint is None:
        raise ControlContractError(
            f"unsupported Cosmos Transfer control modality {selected!r}; expected "
            + ", ".join(COSMOS_TRANSFER_CHECKPOINTS)
        )
    try:
        normalized_weight = float(weight)
    except (TypeError, ValueError) as exc:
        raise ControlContractError(
            f"control weight {weight!r} is not a number"
        ) from exc
    if not math.isfinite(normalized_weight) or not 0.0 <= normalized_weight <= 1.0:
        raise ControlContractError(
            f"control weight {weight!r} is outside Cosmos Transfer's accepted "
            "range 0.0-1.0 (a finite value is required)"
        )
    if str(mask_asset or "").strip() and str(mask_prompt or "").strip():
        raise ControlContractError(
            "mask asset and mask prompt are mutually exclusive; give either a "
            "precomputed region mask or a mask prompt, not both"
        )
    if str(control_prompt or "").strip() and selected != "seg":
        raise ControlContractError(
            f"{selected!r} control is not text-driven, so a control prompt has no "
            "effect; only seg accepts one"
        )
    if selected == "depth" and not str(control_asset or "").strip():
        raise ControlContractError(
            "depth control requires an operator-owned precomputed control asset; "
            "NPA does not download or run Video Depth Anything weights"
        )
    return checkpoint, normalized_weight


__all__ = [
    "COSMOS_TRANSFER_CHECKPOINTS",
    "ControlCheckpoint",
    "ControlContractError",
    "validate_control_request",
]
