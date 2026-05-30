"""SONIC ONNX export helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import copy
from importlib import import_module
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_EXPORT_OPSET = 17
DEFAULT_EXPORT_AXES = "dynamic"
DEFAULT_NORMALIZE_MODE = "baked"
DEFAULT_METADATA_MODE = "sidecar"
DEFAULT_INPUT_NAME = "obs"
DEFAULT_OUTPUT_NAME = "action"
EXPORT_METADATA_FORMAT = "npa_sonic_onnx_export_v1"

AXES_MODES = {"dynamic", "static"}
NORMALIZE_MODES = {"baked", "sidecar", "none"}
METADATA_MODES = {"sidecar", "embedded"}


class SonicExportError(ValueError):
    """Raised when a SONIC policy cannot be exported."""


@dataclass(frozen=True)
class SonicExportResult:
    """Result from exporting a SONIC policy to ONNX."""

    status: str
    checkpoint: str
    onnx_path: str
    metadata_path: str
    opset: int
    axes: str
    normalize: str
    metadata: str
    input_name: str
    output_name: str
    obs_dim: int
    action_dim: int
    parity: dict[str, Any] | None = None


@dataclass(frozen=True)
class SonicParityResult:
    """Numerical parity result for an exported ONNX policy."""

    passed: bool
    atol: float
    max_abs_diff: float
    mean_abs_diff: float
    samples: int


def export_onnx(
    *,
    checkpoint: str,
    output: str,
    opset: int = DEFAULT_EXPORT_OPSET,
    axes: str = DEFAULT_EXPORT_AXES,
    normalize: str = DEFAULT_NORMALIZE_MODE,
    metadata: str = DEFAULT_METADATA_MODE,
    obs_spec: str | dict[str, Any] | None = None,
    action_spec: str | dict[str, Any] | None = None,
    config: str | dict[str, Any] | None = None,
    control_dt: float | None = None,
    policy: Any | None = None,
    sample_observation: Any | None = None,
    verify: bool = False,
    parity_atol: float = 1e-4,
) -> SonicExportResult:
    """Export a deterministic SONIC policy action path as ONNX.

    The ONNX graph has one float32 input named ``obs`` and one float32 output
    named ``action``. Policy loading is lazy so importing the SDK does not
    require torch, onnx, or onnxruntime.
    """

    axes = _validate_choice("axes", axes, AXES_MODES)
    normalize = _validate_choice("normalize", normalize, NORMALIZE_MODES)
    metadata = _validate_choice("metadata", metadata, METADATA_MODES)
    if opset <= 0:
        raise SonicExportError(f"opset must be positive, got {opset}")

    torch = _import_torch()
    config_payload = _load_structured(config)
    policy_model = (
        policy
        if policy is not None
        else _load_policy_from_checkpoint(checkpoint, torch, config_payload)
    )
    policy_model = copy.deepcopy(policy_model).to("cpu")
    policy_model.eval()

    obs_spec_payload = _coerce_spec(
        explicit=_load_structured(obs_spec),
        config=config_payload,
        keys=("obs_spec", "observation_spec", "observations"),
        policy=policy_model,
        attr_names=("obs_spec", "observation_spec", "observations"),
        default_name="obs",
    )
    action_spec_payload = _coerce_spec(
        explicit=_load_structured(action_spec),
        config=config_payload,
        keys=("action_spec", "actions", "action"),
        policy=policy_model,
        attr_names=("action_spec", "actions"),
        default_name="action",
    )

    normalization_stats = _normalization_stats(policy_model, config_payload)
    if normalize == "none":
        normalization_stats = None
    export_policy_model = policy_model
    if normalize == "sidecar" and normalization_stats is not None:
        export_policy_model = _without_internal_normalizer(policy_model)

    output_path = _resolve_onnx_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    obs_dim = _resolve_dim(
        spec=obs_spec_payload,
        policy=export_policy_model,
        attr_names=("observation_dim", "obs_dim", "input_dim", "num_observations"),
        label="observation",
    )
    if obs_dim <= 0:
        raise SonicExportError("observation dimension could not be resolved")

    sample = _sample_observation(sample_observation, obs_dim, torch)
    call_kind = _select_policy_call(export_policy_model, sample, torch)
    forward_model = _make_policy_forward(
        export_policy_model,
        call_kind=call_kind,
        normalization=normalization_stats if normalize == "baked" else None,
    )
    forward_model.eval()
    with torch.no_grad():
        action_sample = forward_model(sample)
    action_dim = _resolve_action_dim(
        action_spec_payload, export_policy_model, action_sample
    )

    dynamic_axes = (
        {DEFAULT_INPUT_NAME: {0: "batch"}, DEFAULT_OUTPUT_NAME: {0: "batch"}}
        if axes == "dynamic"
        else None
    )
    with torch.no_grad():
        torch.onnx.export(
            forward_model,
            sample,
            str(output_path),
            input_names=[DEFAULT_INPUT_NAME],
            output_names=[DEFAULT_OUTPUT_NAME],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
        )

    control_dt_value = (
        control_dt
        if control_dt is not None
        else _control_dt(policy_model, config_payload)
    )
    metadata_payload = _build_metadata(
        checkpoint=checkpoint,
        onnx_path=str(output_path),
        opset=opset,
        axes=axes,
        normalize=normalize,
        metadata=metadata,
        obs_spec=obs_spec_payload,
        action_spec=action_spec_payload,
        obs_dim=obs_dim,
        action_dim=action_dim,
        normalization_stats=normalization_stats if normalize == "sidecar" else None,
        control_dt=control_dt_value,
    )
    metadata_path = ""
    if metadata == "sidecar":
        metadata_path = str(_write_sidecar_metadata(output_path, metadata_payload))
    else:
        _embed_metadata(output_path, metadata_payload)

    parity_payload = None
    if verify:
        parity = validate_onnx_parity(
            policy=policy_model,
            onnx_path=str(output_path),
            observations=sample,
            normalize=normalize,
            normalization_stats=normalization_stats,
            atol=parity_atol,
        )
        parity_payload = asdict(parity)
        if not parity.passed:
            raise SonicExportError(
                "ONNX parity check failed: "
                f"max_abs_diff={parity.max_abs_diff:.6g}, atol={parity.atol:.6g}"
            )

    return SonicExportResult(
        status="exported",
        checkpoint=checkpoint,
        onnx_path=str(output_path),
        metadata_path=metadata_path,
        opset=opset,
        axes=axes,
        normalize=normalize,
        metadata=metadata,
        input_name=DEFAULT_INPUT_NAME,
        output_name=DEFAULT_OUTPUT_NAME,
        obs_dim=obs_dim,
        action_dim=action_dim,
        parity=parity_payload,
    )


def validate_onnx_parity(
    *,
    policy: Any,
    onnx_path: str,
    observations: Any,
    normalize: str = DEFAULT_NORMALIZE_MODE,
    normalization_stats: dict[str, Any] | None = None,
    atol: float = 1e-4,
) -> SonicParityResult:
    """Compare a policy's deterministic action path with an exported ONNX graph."""

    torch = _import_torch()
    ort = _import_onnxruntime()
    policy_model = copy.deepcopy(policy).to("cpu")
    policy_model.eval()
    sample = torch.as_tensor(observations, dtype=torch.float32, device="cpu")
    if sample.ndim == 1:
        sample = sample.unsqueeze(0)
    call_kind = _select_policy_call(policy_model, sample, torch)
    reference_model = _make_policy_forward(
        policy_model,
        call_kind=call_kind,
        normalization=normalization_stats if normalize == "baked" else None,
    )
    with torch.no_grad():
        reference = reference_model(sample).detach().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output = session.run(
        [DEFAULT_OUTPUT_NAME], {DEFAULT_INPUT_NAME: sample.cpu().numpy()}
    )[0]
    diff = abs(reference - output)
    max_diff = float(diff.max()) if diff.size else 0.0
    mean_diff = float(diff.mean()) if diff.size else 0.0
    return SonicParityResult(
        passed=max_diff <= atol,
        atol=atol,
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        samples=int(sample.shape[0]),
    )


def load_export_metadata(path: str) -> dict[str, Any]:
    """Load SONIC ONNX sidecar metadata from JSON."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _make_policy_forward(
    policy: Any,
    *,
    call_kind: str,
    normalization: dict[str, Any] | None,
) -> Any:
    torch = _import_torch()

    class PolicyForward(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy = policy

        def forward(self, obs: Any) -> Any:
            if normalization is not None:
                obs = _apply_normalization(obs, normalization)
            if call_kind == "act_pure_inference":
                output = self.policy.act_pure_inference({"actor_obs": obs.unsqueeze(1)})
            elif call_kind == "act_inference":
                if hasattr(self.policy, "init_rollout"):
                    self.policy.init_rollout()
                output = self.policy.act_inference({"actor_obs": obs})
            elif call_kind == "dict_call":
                output = self.policy({"actor_obs": obs})
            else:
                output = self.policy(obs)
            return _extract_action(output)

    return PolicyForward()


def _validate_choice(label: str, value: str, choices: set[str]) -> str:
    normalized = str(value).strip().lower()
    if normalized not in choices:
        options = ", ".join(sorted(choices))
        raise SonicExportError(f"{label} must be one of: {options}")
    return normalized


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SonicExportError(
            "SONIC ONNX export requires torch. Install the sonic export dependencies "
            "inside the SONIC container or use the npa[sonic] extra."
        ) from exc
    return torch


def _import_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SonicExportError(
            "ONNX parity validation requires onnxruntime. Install the npa[sonic] extra."
        ) from exc
    return ort


def _load_policy_from_checkpoint(
    checkpoint: str,
    torch: Any,
    config: dict[str, Any],
) -> Any:
    if not checkpoint:
        raise SonicExportError("checkpoint is required")
    path = Path(checkpoint)
    if not path.exists():
        raise SonicExportError(f"checkpoint not found: {checkpoint}")
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(payload, torch.nn.Module):
        return payload
    if isinstance(payload, dict):
        for key in (
            "policy",
            "actor",
            "model",
            "module",
            "policy_model",
            "actor_model",
        ):
            candidate = payload.get(key)
            if isinstance(candidate, torch.nn.Module):
                return candidate
        policy = _instantiate_policy_from_config(config)
        if policy is not None:
            state_dict, state_key = _state_dict_from_checkpoint(payload)
            if state_dict is None:
                raise SonicExportError(
                    "checkpoint does not contain a policy state dict to load into "
                    "the policy class from --config"
                )
            _load_state_dict(policy, state_dict, state_key)
            return policy
    raise SonicExportError(
        "checkpoint does not contain a loadable torch.nn.Module policy. "
        "Provide an SDK `policy=` object, a checkpoint that stores `policy`, "
        "`actor`, or `model` as a module, or --config with policy.class and "
        "policy.kwargs for state-dict checkpoints."
    )


def _instantiate_policy_from_config(config: dict[str, Any]) -> Any | None:
    policy_cfg = config.get("policy") if isinstance(config.get("policy"), dict) else {}
    target = (
        policy_cfg.get("class")
        or policy_cfg.get("target")
        or policy_cfg.get("_target_")
        or config.get("policy_class")
    )
    if target is None:
        actor_cfg: dict[str, Any] = {}
        algo_cfg = config.get("algo")
        if isinstance(algo_cfg, dict):
            algo_config = algo_cfg.get("config")
            if isinstance(algo_config, dict) and isinstance(
                algo_config.get("actor"), dict
            ):
                actor_cfg = algo_config["actor"]
        if isinstance(actor_cfg, dict):
            target = actor_cfg.get("_target_")
            policy_cfg = {"kwargs": actor_cfg.get("kwargs", {})}
    if not target:
        return None
    kwargs = policy_cfg.get("kwargs") or config.get("policy_kwargs") or {}
    if not isinstance(kwargs, dict):
        raise SonicExportError("policy kwargs in --config must be a mapping")
    cls = _import_object(str(target))
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise SonicExportError(
            f"failed to instantiate policy class {target}: {exc}"
        ) from exc


def _import_object(target: str) -> Any:
    module_name, _, attr_name = target.replace(":", ".").rpartition(".")
    if not module_name or not attr_name:
        raise SonicExportError(f"invalid policy class target: {target}")
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise SonicExportError(
            f"failed to import policy module {module_name}: {exc}"
        ) from exc
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise SonicExportError(f"policy class not found: {target}") from exc


def _state_dict_from_checkpoint(
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    for key in (
        "actor_model_state_dict",
        "policy_state_dict",
        "state_dict",
        "model_state_dict",
    ):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate, key
    return None, ""


def _load_state_dict(policy: Any, state_dict: dict[str, Any], state_key: str) -> None:
    target = copy.deepcopy(state_dict)
    policy_state = policy.state_dict()
    if "std" in policy_state and "log_std" in target and "std" not in target:
        torch = _import_torch()
        target["std"] = torch.exp(target.pop("log_std"))
    elif "log_std" in policy_state and "std" in target and "log_std" not in target:
        torch = _import_torch()
        target["log_std"] = torch.log(target.pop("std"))
    try:
        policy.load_state_dict(target)
    except RuntimeError as exc:
        raise SonicExportError(
            f"failed to load {state_key} into policy from --config: {exc}"
        ) from exc


def _load_structured(value: str | dict[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    path = Path(value)
    if not path.exists():
        raise SonicExportError(f"structured input not found: {value}")
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        data = yaml.safe_load(raw)
    if data is None:
        return {}
    if isinstance(data, list):
        return {"fields": data}
    if not isinstance(data, dict):
        raise SonicExportError(f"expected a mapping in {value}")
    return data


def _coerce_spec(
    *,
    explicit: dict[str, Any],
    config: dict[str, Any],
    keys: tuple[str, ...],
    policy: Any,
    attr_names: tuple[str, ...],
    default_name: str,
) -> dict[str, Any]:
    raw = (
        explicit
        or _first_mapping(config, keys)
        or _first_attr_mapping(policy, attr_names)
    )
    if not raw:
        return {"name": default_name}
    if "fields" in raw or "shape" in raw or "dim" in raw:
        return dict(raw)
    if "observations" in raw and isinstance(raw["observations"], list):
        return {"name": default_name, "fields": raw["observations"]}
    if isinstance(raw, dict):
        return dict(raw)
    return {"name": default_name}


def _first_mapping(data: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return {"fields": value}
    for value in data.values():
        if isinstance(value, dict):
            found = _first_mapping(value, keys)
            if found:
                return found
    return {}


def _first_attr_mapping(policy: Any, attr_names: tuple[str, ...]) -> dict[str, Any]:
    for name in attr_names:
        value = getattr(policy, name, None)
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return {"fields": value}
    return {}


def _normalization_stats(policy: Any, config: dict[str, Any]) -> dict[str, Any] | None:
    explicit = _first_mapping(
        config,
        ("normalization", "obs_normalization", "observation_normalization"),
    )
    stats = _stats_from_mapping(explicit)
    if stats is not None:
        return stats
    for attr in (
        "normalization",
        "normalization_stats",
        "obs_normalization",
        "observation_normalization",
    ):
        value = getattr(policy, attr, None)
        if isinstance(value, dict):
            stats = _stats_from_mapping(value)
            if stats is not None:
                return stats
    for attr in ("running_mean_std", "obs_rms", "normalizer"):
        value = getattr(policy, attr, None)
        stats = _stats_from_object(value)
        if stats is not None:
            stats["source"] = f"policy.{attr}"
            return stats
    return _find_stats(config)


def _stats_from_mapping(value: dict[str, Any]) -> dict[str, Any] | None:
    if not value:
        return None
    mean = value.get("mean", value.get("running_mean"))
    var = value.get("var", value.get("variance", value.get("running_var")))
    std = value.get("std", value.get("running_std"))
    if mean is None or (var is None and std is None):
        return None
    return {
        "mean": _jsonable(mean),
        "var": _jsonable(var) if var is not None else None,
        "std": _jsonable(std) if std is not None else None,
        "epsilon": float(value.get("epsilon", value.get("eps", 1e-5))),
        "clip": value.get("clip", 5.0),
        "source": value.get("source", "config"),
    }


def _stats_from_object(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    mean = getattr(value, "running_mean", getattr(value, "mean", None))
    var = getattr(value, "running_var", getattr(value, "var", None))
    std = getattr(value, "running_std", getattr(value, "std", None))
    if mean is None or (var is None and std is None):
        return None
    return {
        "mean": _jsonable(mean),
        "var": _jsonable(var) if var is not None else None,
        "std": _jsonable(std) if std is not None else None,
        "epsilon": float(getattr(value, "epsilon", getattr(value, "eps", 1e-5))),
        "clip": 5.0,
    }


def _find_stats(data: dict[str, Any]) -> dict[str, Any] | None:
    stats = _stats_from_mapping(data)
    if stats is not None:
        return stats
    for value in data.values():
        if isinstance(value, dict):
            stats = _find_stats(value)
            if stats is not None:
                return stats
    return None


def _without_internal_normalizer(policy: Any) -> Any:
    clone = copy.deepcopy(policy)
    for attr in ("running_mean_std", "obs_rms", "normalizer"):
        if hasattr(clone, attr):
            setattr(clone, attr, None)
    return clone


def _resolve_onnx_path(output: str) -> Path:
    if not output:
        raise SonicExportError("output is required")
    path = Path(output)
    if path.suffix.lower() == ".onnx":
        return path
    return path / "sonic_policy.onnx"


def _resolve_dim(
    *,
    spec: dict[str, Any],
    policy: Any,
    attr_names: tuple[str, ...],
    label: str,
) -> int:
    spec_dim = _dim_from_spec(spec)
    if spec_dim:
        return spec_dim
    for attr in attr_names:
        value = getattr(policy, attr, None)
        if isinstance(value, int):
            return value
    raise SonicExportError(
        f"{label} dimension is required. Provide --{label[:3]}-spec or a policy "
        f"with one of: {', '.join(attr_names)}"
    )


def _dim_from_spec(spec: dict[str, Any]) -> int:
    if not spec:
        return 0
    if isinstance(spec.get("dim"), int):
        return int(spec["dim"])
    shape = spec.get("shape")
    if isinstance(shape, list) and shape:
        product = 1
        for item in shape:
            product *= int(item)
        return product
    fields = spec.get("fields")
    if isinstance(fields, list):
        total = 0
        for field in fields:
            if not isinstance(field, dict) or field.get("enabled", True) is False:
                continue
            total += _dim_from_spec(field)
        return total
    return 0


def _resolve_action_dim(spec: dict[str, Any], policy: Any, action_sample: Any) -> int:
    spec_dim = _dim_from_spec(spec)
    if spec_dim:
        return spec_dim
    for attr in ("action_dim", "num_actions", "actions_dim"):
        value = getattr(policy, attr, None)
        if isinstance(value, int):
            return value
    if getattr(action_sample, "ndim", 0) >= 2:
        return int(action_sample.shape[-1])
    raise SonicExportError("action dimension could not be resolved")


def _sample_observation(value: Any, obs_dim: int, torch: Any) -> Any:
    if value is None:
        return torch.randn(1, obs_dim, dtype=torch.float32)
    sample = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if sample.ndim == 1:
        sample = sample.unsqueeze(0)
    if sample.shape[-1] != obs_dim:
        raise SonicExportError(
            f"sample observation last dimension {sample.shape[-1]} does not match {obs_dim}"
        )
    return sample


def _select_policy_call(policy: Any, sample: Any, torch: Any) -> str:
    candidates = []
    if hasattr(policy, "act_pure_inference"):
        candidates.append("act_pure_inference")
    if hasattr(policy, "act_inference"):
        candidates.append("act_inference")
    candidates.extend(["dict_call", "tensor_call"])
    errors: list[str] = []
    for kind in candidates:
        try:
            test_policy = copy.deepcopy(policy).to("cpu")
            test_policy.eval()
            wrapper = _make_policy_forward(
                test_policy, call_kind=kind, normalization=None
            )
            with torch.no_grad():
                output = wrapper(sample)
            if getattr(output, "shape", None) is not None and output.shape[-1] > 0:
                return kind
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{kind}: {type(exc).__name__}: {exc}")
    joined = "; ".join(errors)
    raise SonicExportError(f"could not find a deterministic policy call path: {joined}")


def _extract_action(output: Any) -> Any:
    if isinstance(output, dict):
        for key in ("action_mean", "actions", "action"):
            if key in output:
                return output[key]
    if hasattr(output, "get"):
        for key in ("action_mean", "actions", "action"):
            try:
                value = output.get(key)
            except Exception:  # noqa: BLE001
                value = None
            if value is not None:
                return value
    return output


def _apply_normalization(obs: Any, stats: dict[str, Any]) -> Any:
    torch = _import_torch()
    mean = torch.as_tensor(stats["mean"], dtype=obs.dtype, device=obs.device)
    if stats.get("std") is not None:
        denom = torch.as_tensor(stats["std"], dtype=obs.dtype, device=obs.device)
    else:
        var = torch.as_tensor(stats["var"], dtype=obs.dtype, device=obs.device)
        denom = torch.sqrt(var + float(stats.get("epsilon", 1e-5)))
    normalized = (obs - mean) / denom
    clip = stats.get("clip", 5.0)
    if clip is not None:
        normalized = torch.clamp(normalized, min=-float(clip), max=float(clip))
    return normalized


def _control_dt(policy: Any, config: dict[str, Any]) -> float | None:
    for attr in ("control_dt", "dt"):
        value = getattr(policy, attr, None)
        if isinstance(value, int | float):
            return float(value)
    found = _first_number(config, ("control_dt", "control_timestep", "dt"))
    return float(found) if found is not None else None


def _first_number(data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, int | float):
            return float(value)
    for value in data.values():
        if isinstance(value, dict):
            found = _first_number(value, keys)
            if found is not None:
                return found
    return None


def _build_metadata(
    *,
    checkpoint: str,
    onnx_path: str,
    opset: int,
    axes: str,
    normalize: str,
    metadata: str,
    obs_spec: dict[str, Any],
    action_spec: dict[str, Any],
    obs_dim: int,
    action_dim: int,
    normalization_stats: dict[str, Any] | None,
    control_dt: float | None,
) -> dict[str, Any]:
    obs_payload = dict(obs_spec)
    action_payload = dict(action_spec)
    obs_payload.setdefault("shape", [obs_dim])
    obs_payload.setdefault("dtype", "float32")
    obs_payload.setdefault("input_name", DEFAULT_INPUT_NAME)
    action_payload.setdefault("shape", [action_dim])
    action_payload.setdefault("dtype", "float32")
    action_payload.setdefault("output_name", DEFAULT_OUTPUT_NAME)
    payload = {
        "format": EXPORT_METADATA_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "onnx_model": onnx_path,
        "deterministic_action_path": "mean",
        "input_name": DEFAULT_INPUT_NAME,
        "output_name": DEFAULT_OUTPUT_NAME,
        "opset": opset,
        "axes": axes,
        "normalize": normalize,
        "metadata": metadata,
        "control_dt": control_dt,
        "obs_spec": _jsonable(obs_payload),
        "action_spec": _jsonable(action_payload),
    }
    if normalization_stats is not None:
        payload["normalization"] = _jsonable(normalization_stats)
    return payload


def _write_sidecar_metadata(onnx_path: Path, payload: dict[str, Any]) -> Path:
    metadata_path = onnx_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def _embed_metadata(onnx_path: Path, payload: dict[str, Any]) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise SonicExportError(
            "embedding SONIC export metadata requires onnx. Install the npa[sonic] extra."
        ) from exc
    model = onnx.load(str(onnx_path))
    entries = {
        "npa.sonic.export": json.dumps(payload, sort_keys=True),
        "npa.sonic.format": EXPORT_METADATA_FORMAT,
        "npa.sonic.normalize": str(payload["normalize"]),
        "npa.sonic.opset": str(payload["opset"]),
    }
    existing = {item.key: item for item in model.metadata_props}
    for key, value in entries.items():
        item = existing.get(key)
        if item is None:
            item = model.metadata_props.add()
            item.key = key
        item.value = value
    onnx.save(model, str(onnx_path))


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


from npa.workbench.sonic.eval import evaluate_onnx_policy  # noqa: E402


__all__ = [
    "SonicExportResult",
    "SonicParityResult",
    "evaluate_onnx_policy",
    "export_onnx",
    "load_export_metadata",
    "validate_onnx_parity",
]
