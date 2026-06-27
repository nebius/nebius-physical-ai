"""Export a trained rsl_rl actor policy to ONNX for torch-free robot inference.

The sim2real BYO trainer (:mod:`npa.workflows.sim2real.byo_isaac_trainer`) trains
an rsl_rl ``OnPolicyRunner`` and uploads ``model_*.pt`` checkpoints to S3. Those
checkpoints store an ``ActorCritic`` ``model_state_dict`` whose ``actor.*`` keys
are a plain MLP. Running that policy on a robot otherwise requires torch **and**
Isaac Lab on the robot, which is not deployable.

This module re-materializes the actor MLP directly from the checkpoint's
``model_state_dict`` (no Isaac Lab, no env construction) and exports it to ONNX,
alongside a ``policy_contract.json`` sidecar declaring the observation/action
contract the real robot must satisfy. The exported ``policy.onnx`` then runs with
nothing but ``onnxruntime`` (CPU) -- the same runtime contract SONIC eval uses
(see :mod:`npa.workbench.sonic.eval`).

CAVEAT -- ONNX export is necessary, NOT sufficient, for sim-to-real.
It makes the trained policy *portable*; it does not make it *correct on a real
robot*. Export adds no domain randomization, does not validate real-world
dynamics, and does not align the robot's real observation pipeline with the sim
observation manager. Closing the sim-to-real gap (obs alignment, dynamics,
randomization, real-world eval) is out of scope here and remains unsolved by this
step. See ``docs/architecture/policy-export-onnx.md``.

The heavy ``torch``/``onnx`` imports are guarded inside functions so the pure
contract builder (:func:`build_policy_contract`) and dim inference
(:func:`infer_mlp_dims`) are unit-testable without torch installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

POLICY_EXPORT_FORMAT = "npa_sim2real_onnx_export_v1"
DEFAULT_INPUT_NAME = "obs"
DEFAULT_OUTPUT_NAME = "action"
DEFAULT_OPSET = 17
DEFAULT_ACTIVATION = "elu"
ONNX_FILENAME = "policy.onnx"
CONTRACT_FILENAME = "policy_contract.json"

# rsl_rl ``resolve_nn_activation`` names -> torch.nn module attribute names. The
# activation is NOT recoverable from the state dict (activations carry no
# parameters), so it must match the activation used at training time. rsl_rl's
# ``RslRlPpoActorCriticCfg`` defaults to "elu".
_ACTIVATIONS = {
    "elu": "ELU",
    "relu": "ReLU",
    "selu": "SELU",
    "tanh": "Tanh",
    "sigmoid": "Sigmoid",
    "leakyrelu": "LeakyReLU",
    "lrelu": "LeakyReLU",
    "gelu": "GELU",
}

# Best-effort, conservative hints for known Isaac Lab tasks. These describe the
# *declared* action semantics a robot integrator must verify against the task's
# ActionManager -- they are documentation, not a guarantee. Unknown tasks fall
# back to an "opaque" action declaration.
TASK_HINTS: dict[str, dict[str, Any]] = {
    "Isaac-Lift-Cube-Franka": {
        "action_type": "joint_position",
        "note": (
            "Franka manager-based lift: 7 arm joints (JointPositionAction, "
            "scaled offset from default joint pos) + 1 binary gripper command. "
            "Confirm scale/offset against the task ActionManager before deploying."
        ),
    },
    "Isaac-Reach-Franka": {
        "action_type": "joint_position",
        "note": (
            "Franka reach: 7 arm-joint position targets (scaled offset from "
            "default joint pos). Confirm scale/offset against the ActionManager."
        ),
    },
}

SIM_TO_REAL_CAVEAT = (
    "ONNX export makes this policy portable (torch/Isaac-free inference via "
    "onnxruntime); it does NOT bridge sim-to-real. No domain randomization, no "
    "real-dynamics validation, and no real-robot observation alignment are "
    "performed here. The obs vector fed at inference MUST be assembled in the "
    "exact same order/units/frame as the sim observation manager produced during "
    "training, and the action output MUST be applied with the same action-space "
    "semantics (scaling, offset, limits) as the sim ActionManager. Validate on "
    "hardware before trusting outputs."
)


class PolicyExportError(ValueError):
    """Raised when an rsl_rl checkpoint cannot be exported to ONNX."""


# --------------------------------------------------------------------------- #
# Pure helpers (no torch) -- unit-testable without GPU/heavy deps.
# --------------------------------------------------------------------------- #
def actor_weight_shapes(
    state_dict: Mapping[str, Any],
) -> dict[str, tuple[int, ...]]:
    """Return ``{param_name: shape}`` for every ``actor.<n>.weight`` entry.

    Accepts torch tensors or anything exposing ``.shape``; the returned shapes
    are plain int tuples so downstream pure helpers never touch torch.
    """

    shapes: dict[str, tuple[int, ...]] = {}
    for name, value in state_dict.items():
        if re.fullmatch(r"actor\.\d+\.weight", name) is None:
            continue
        shape = getattr(value, "shape", None)
        if shape is None:
            raise PolicyExportError(f"actor weight {name!r} has no shape")
        shapes[name] = tuple(int(dim) for dim in shape)
    return shapes


def infer_mlp_dims(
    weight_shapes: Mapping[str, Sequence[int]],
) -> tuple[int, int, list[int]]:
    """Infer ``(obs_dim, act_dim, hidden_dims)`` from actor linear weight shapes.

    ``weight_shapes`` maps ``actor.<n>.weight`` -> ``(out_features, in_features)``
    (PyTorch ``nn.Linear`` weight layout). The first layer's ``in_features`` is
    the observation dim; the last layer's ``out_features`` is the action dim;
    everything between is a hidden width.
    """

    if not weight_shapes:
        raise PolicyExportError(
            "no actor.<n>.weight entries found in checkpoint; cannot infer the "
            "policy MLP. Is this an rsl_rl ActorCritic model_state_dict?"
        )

    def _index(name: str) -> int:
        match = re.fullmatch(r"actor\.(\d+)\.weight", name)
        if match is None:
            raise PolicyExportError(f"unexpected actor weight key: {name!r}")
        return int(match.group(1))

    ordered = sorted(weight_shapes.items(), key=lambda kv: _index(kv[0]))
    for name, shape in ordered:
        if len(shape) != 2:
            raise PolicyExportError(
                f"actor weight {name!r} is not 2-D (got shape {tuple(shape)}); "
                "only plain MLP actors are supported"
            )
    out_dims = [int(shape[0]) for _, shape in ordered]
    in_dims = [int(shape[1]) for _, shape in ordered]
    obs_dim = in_dims[0]
    act_dim = out_dims[-1]
    hidden_dims = out_dims[:-1]
    # Sanity: each layer's input must equal the previous layer's output.
    for prev_out, cur_in, (name, _) in zip(out_dims, in_dims[1:], ordered[1:]):
        if prev_out != cur_in:
            raise PolicyExportError(
                f"actor layer dim mismatch at {name!r}: expected in_features "
                f"{prev_out}, got {cur_in}"
            )
    return obs_dim, act_dim, hidden_dims


def build_policy_contract(
    *,
    obs_dim: int,
    act_dim: int,
    isaac_task: str = "",
    checkpoint: Mapping[str, Any] | None = None,
    hidden_dims: Sequence[int] | None = None,
    activation: str = DEFAULT_ACTIVATION,
    opset: int = DEFAULT_OPSET,
    input_name: str = DEFAULT_INPUT_NAME,
    output_name: str = DEFAULT_OUTPUT_NAME,
    obs_terms: Sequence[Mapping[str, Any]] | None = None,
    action_type: str | None = None,
    action_scaling: Any = None,
    action_limits: Any = None,
    normalization: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the ``policy_contract.json`` payload (pure -- no torch).

    Declares the obs/action contract a real robot must satisfy to consume the
    exported ONNX policy. ``obs_terms`` is an ordered list of
    ``{"name", "dim", ...}`` describing the observation layout when recoverable
    from the task's observation manager; when ``None`` the layout is marked
    ``"opaque"`` (the contract only guarantees the flat ``obs_dim`` ordering used
    at training time). ``checkpoint`` carries provenance fields (source path/URI,
    sha256, train iter).
    """

    if obs_dim <= 0:
        raise PolicyExportError(f"obs_dim must be positive, got {obs_dim}")
    if act_dim <= 0:
        raise PolicyExportError(f"act_dim must be positive, got {act_dim}")

    hint = _task_hint(isaac_task)
    resolved_action_type = action_type or hint.get("action_type") or "opaque"

    if obs_terms is not None:
        terms = [dict(term) for term in obs_terms]
        declared = sum(int(term.get("dim", 0)) for term in terms)
        if declared != obs_dim:
            raise PolicyExportError(
                f"obs_terms dims sum to {declared} but obs_dim is {obs_dim}"
            )
        obs_layout: dict[str, Any] = {"kind": "ordered_terms", "terms": terms}
    else:
        obs_layout = {
            "kind": "opaque",
            "note": (
                "Ordered observation-term names were not recovered (requires the "
                "Isaac task observation manager). The contract only guarantees a "
                f"flat float32 vector of length {obs_dim} in the SAME term order "
                "the sim observation manager emitted during training."
            ),
        }

    action_spec: dict[str, Any] = {
        "dim": act_dim,
        "dtype": "float32",
        "type": resolved_action_type,
        "scaling": action_scaling,
        "limits": action_limits,
        "output_name": output_name,
    }
    if hint.get("note"):
        action_spec["note"] = hint["note"]

    norm_spec = (
        dict(normalization)
        if normalization is not None
        else {"type": "none", "note": "actor consumes raw observations"}
    )

    payload: dict[str, Any] = {
        "format": POLICY_EXPORT_FORMAT,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "isaac_task": isaac_task,
        "input_name": input_name,
        "output_name": output_name,
        "opset": opset,
        "deterministic_action_path": "actor_mean",
        "network": {
            "type": "mlp",
            "activation": activation,
            "hidden_dims": list(hidden_dims) if hidden_dims is not None else None,
            "framework": "rsl_rl.modules.ActorCritic",
        },
        "obs": {
            "dim": obs_dim,
            "dtype": "float32",
            "shape": [1, obs_dim],
            "input_name": input_name,
            "layout": obs_layout,
        },
        "action": action_spec,
        "normalization": norm_spec,
        "checkpoint": dict(checkpoint) if checkpoint is not None else {},
        "sim_to_real_caveat": SIM_TO_REAL_CAVEAT,
    }
    return payload


def _task_hint(isaac_task: str) -> dict[str, Any]:
    if not isaac_task:
        return {}
    for key, hint in TASK_HINTS.items():
        if key.lower() in isaac_task.lower():
            return hint
    return {}


# --------------------------------------------------------------------------- #
# Torch-backed export.
# --------------------------------------------------------------------------- #
def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised via env without torch
        raise PolicyExportError(
            "ONNX policy export requires torch. Install it in the export "
            "environment (e.g. the Isaac container python, or `pip install torch`)."
        ) from exc
    return torch


def _resolve_activation(torch: Any, activation: str) -> Any:
    attr = _ACTIVATIONS.get(activation.lower())
    if attr is None:
        raise PolicyExportError(
            f"unsupported activation {activation!r}; supported: "
            f"{sorted(_ACTIVATIONS)}"
        )
    return getattr(torch.nn, attr)


def load_state_dict_from_checkpoint(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the ActorCritic state dict from an rsl_rl checkpoint payload.

    rsl_rl ``OnPolicyRunner.save`` writes ``{"model_state_dict": ..., ...}``; a
    bare state dict (``actor.*`` keys at top level) is also accepted.
    """

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        if not isinstance(state, Mapping):
            raise PolicyExportError("model_state_dict is not a mapping")
        return state
    if any(re.fullmatch(r"actor\.\d+\.weight", key) for key in checkpoint):
        return checkpoint
    raise PolicyExportError(
        "checkpoint has neither a 'model_state_dict' nor top-level 'actor.*' "
        "keys; not an rsl_rl ActorCritic checkpoint"
    )


def _build_actor_module(torch: Any, state_dict: Mapping[str, Any], activation: str) -> Any:
    """Rebuild the actor MLP (nn.Sequential) from ``actor.<n>.{weight,bias}``."""

    nn = torch.nn
    indices = sorted(
        {
            int(m.group(1))
            for key in state_dict
            if (m := re.fullmatch(r"actor\.(\d+)\.weight", key))
        }
    )
    if not indices:
        raise PolicyExportError("no actor linear layers found in state dict")
    activation_cls = _resolve_activation(torch, activation)
    modules: list[Any] = []
    for position, idx in enumerate(indices):
        weight = state_dict[f"actor.{idx}.weight"]
        bias = state_dict[f"actor.{idx}.bias"]
        out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
        linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        modules.append(linear)
        if position < len(indices) - 1:
            modules.append(activation_cls())
    return nn.Sequential(*modules)


def _detect_normalization(
    torch: Any, checkpoint: Mapping[str, Any], obs_dim: int
) -> tuple[Any | None, dict[str, Any]]:
    """Detect an rsl_rl empirical observation normalizer in the checkpoint.

    Returns ``(mean_var_or_None, normalization_spec)``. When present, the
    normalizer is baked into the exported graph so the ONNX consumer always
    feeds RAW observations.
    """

    state = checkpoint.get("obs_norm_state_dict")
    if not isinstance(state, Mapping):
        return None, {"type": "none", "note": "actor consumes raw observations"}
    mean = state.get("mean")
    var = state.get("var")
    if mean is None or var is None:
        return None, {"type": "none", "note": "actor consumes raw observations"}
    mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
    var_t = torch.as_tensor(var, dtype=torch.float32).reshape(-1)
    if mean_t.numel() != obs_dim:
        raise PolicyExportError(
            f"obs normalizer mean has {mean_t.numel()} elements, expected {obs_dim}"
        )
    spec = {
        "type": "empirical_baked",
        "eps": 1e-8,
        "note": (
            "rsl_rl EmpiricalNormalization detected and BAKED into the ONNX graph "
            "as (obs - mean) / sqrt(var + eps). Feed RAW observations at inference."
        ),
        "mean": [float(v) for v in mean_t.tolist()],
        "var": [float(v) for v in var_t.tolist()],
    }
    return (mean_t, var_t), spec


def _make_forward_module(torch: Any, actor: Any, norm: Any | None) -> Any:
    nn = torch.nn

    class _PolicyForward(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.actor = actor
            if norm is not None:
                mean_t, var_t = norm
                self.register_buffer("_obs_mean", mean_t)
                self.register_buffer("_obs_var", var_t)
                self._normalize = True
            else:
                self._normalize = False

        def forward(self, obs: Any) -> Any:
            if self._normalize:
                obs = (obs - self._obs_mean) / torch.sqrt(self._obs_var + 1e-8)
            return self.actor(obs)

    module = _PolicyForward()
    module.eval()
    return module


def export_policy_onnx(
    checkpoint_path: str,
    *,
    out_dir: str,
    isaac_task: str = "",
    obs_dim: int | None = None,
    act_dim: int | None = None,
    activation: str = DEFAULT_ACTIVATION,
    opset: int = DEFAULT_OPSET,
    input_name: str = DEFAULT_INPUT_NAME,
    output_name: str = DEFAULT_OUTPUT_NAME,
    obs_terms: Sequence[Mapping[str, Any]] | None = None,
    action_type: str | None = None,
    action_scaling: Any = None,
    action_limits: Any = None,
    dynamic_batch: bool = True,
    checkpoint_source: str | None = None,
) -> dict[str, Any]:
    """Export an rsl_rl ``model_*.pt`` actor to ``policy.onnx`` + contract.

    Loads the checkpoint, rebuilds the actor MLP from ``model_state_dict``, runs
    ``torch.onnx.export`` (input ``[1, obs_dim]`` -> output ``[1, act_dim]``),
    and writes ``policy.onnx`` and ``policy_contract.json`` into ``out_dir``.

    ``obs_dim``/``act_dim`` are inferred from the actor's first/last layer when
    not given; when given they are cross-checked against the checkpoint and a
    mismatch is an error. Returns a result dict with output paths and dims.
    """

    torch = _import_torch()
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise PolicyExportError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise PolicyExportError(
            f"checkpoint did not deserialize to a mapping (got {type(checkpoint)})"
        )
    state_dict = load_state_dict_from_checkpoint(checkpoint)

    inferred_obs, inferred_act, hidden_dims = infer_mlp_dims(
        actor_weight_shapes(state_dict)
    )
    if obs_dim is not None and obs_dim != inferred_obs:
        raise PolicyExportError(
            f"--obs-dim {obs_dim} disagrees with checkpoint actor input "
            f"dimension {inferred_obs}"
        )
    if act_dim is not None and act_dim != inferred_act:
        raise PolicyExportError(
            f"--act-dim {act_dim} disagrees with checkpoint actor output "
            f"dimension {inferred_act}"
        )
    obs_dim = inferred_obs
    act_dim = inferred_act

    actor = _build_actor_module(torch, state_dict, activation)
    norm, norm_spec = _detect_normalization(torch, checkpoint, obs_dim)
    forward_model = _make_forward_module(torch, actor, norm)

    sample = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        sample_action = forward_model(sample)
    if int(sample_action.shape[-1]) != act_dim:
        raise PolicyExportError(
            f"rebuilt actor produced action dim {int(sample_action.shape[-1])}, "
            f"expected {act_dim}"
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    onnx_path = out_path / ONNX_FILENAME
    contract_path = out_path / CONTRACT_FILENAME

    dynamic_axes = (
        {input_name: {0: "batch"}, output_name: {0: "batch"}}
        if dynamic_batch
        else None
    )
    # Prefer the legacy TorchScript exporter: it embeds the (small) MLP weights
    # directly into a single self-contained ``policy.onnx`` (the newer dynamo
    # exporter spills weights into a ``policy.onnx.data`` sidecar, which breaks
    # the single-file portability contract the robot consumer expects). Fall back
    # to the default exporter on older torch builds that lack the ``dynamo`` kwarg.
    export_kwargs: dict[str, Any] = {
        "input_names": [input_name],
        "output_names": [output_name],
        "dynamic_axes": dynamic_axes,
        "opset_version": opset,
    }
    with torch.no_grad():
        try:
            torch.onnx.export(
                forward_model, sample, str(onnx_path), dynamo=False, **export_kwargs
            )
        except TypeError:
            torch.onnx.export(forward_model, sample, str(onnx_path), **export_kwargs)
    # Guard the single-file contract: external-weight sidecars are not portable.
    sidecar = onnx_path.with_name(onnx_path.name + ".data")
    if sidecar.exists():
        raise PolicyExportError(
            f"ONNX export produced an external-data sidecar {sidecar.name}; the "
            "policy must be a single self-contained file. Re-run with an exporter "
            "that embeds weights."
        )

    provenance = _checkpoint_provenance(
        ckpt_path, checkpoint, source=checkpoint_source
    )
    contract = build_policy_contract(
        obs_dim=obs_dim,
        act_dim=act_dim,
        isaac_task=isaac_task,
        checkpoint=provenance,
        hidden_dims=hidden_dims,
        activation=activation,
        opset=opset,
        input_name=input_name,
        output_name=output_name,
        obs_terms=obs_terms,
        action_type=action_type,
        action_scaling=action_scaling,
        action_limits=action_limits,
        normalization=norm_spec,
    )
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "status": "success",
        "onnx_path": str(onnx_path),
        "contract_path": str(contract_path),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden_dims": hidden_dims,
        "activation": activation,
        "opset": opset,
        "input_name": input_name,
        "output_name": output_name,
        "normalization": norm_spec.get("type"),
        "isaac_task": isaac_task,
        "checkpoint": provenance,
    }


def _checkpoint_provenance(
    ckpt_path: Path, checkpoint: Mapping[str, Any], *, source: str | None
) -> dict[str, Any]:
    import hashlib

    digest = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    train_iter = checkpoint.get("iter")
    return {
        "source": source or str(ckpt_path),
        "filename": ckpt_path.name,
        "size_bytes": ckpt_path.stat().st_size,
        "sha256": digest,
        "train_iter": int(train_iter) if isinstance(train_iter, int) else None,
    }
