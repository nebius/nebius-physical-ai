"""Load and validate NPA workflow API versions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.predicates import PREDICATES

#: Upper bound for a ``parallel:`` group's declared concurrency. The *effective*
#: concurrency is ``min(maxConcurrency, len(parallel))``, so a large value is not
#: dangerous by itself — but a typo (400 for 40) should not silently ask a shared
#: cluster for hundreds of simultaneous clusters.
MAX_GROUP_CONCURRENCY = 64

#: Upper bound for a resource profile's ``num_nodes``. Multi-node stages are gang
#: scheduled, so a typo (80 for 8) asks a shared cluster for a block it cannot fill and
#: the task sits PENDING instead of failing fast.
MAX_PROFILE_NODES = 32

API_VERSION = "npa.workflow/v0.0.1"
API_VERSION_BETA = "npa.workflow/v0.0.1-beta"
SUPPORTED_API_VERSIONS = frozenset({API_VERSION, API_VERSION_BETA})


@dataclass
class LoopSpec:
    max: Any = None  # int or "config.<attr>"
    until: str | None = None


@dataclass
class TransitionSpec:
    when: str | None = None
    goto: str = ""
    if_config: str | None = None  # config.<attr> truthy


@dataclass
class ArtifactSpec:
    uri: str
    schema: str = ""
    kind: str = ""


@dataclass
class RunSpec:
    shell: str = ""
    argv: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.shell.strip() and not self.argv


@dataclass
class TriggerSpec:
    """Driver-side watch on an object-storage prefix before a state runs."""

    uri: str
    poll_seconds: int = 30
    max_polls: int = 0  # 0 == unbounded (bounded by the runtime deadline)
    min_objects: int = 1
    # Parse provenance for reapplying config overrides; resolved fields above remain
    # the runtime contract. Exclude this metadata from durable workflow identity.
    config_expressions: dict[str, str] = field(
        default_factory=dict, repr=False, compare=False
    )


@dataclass
class StateSpec:
    name: str
    description: str = ""
    needs: list[str] = field(default_factory=list)
    run: RunSpec | None = None
    tool_ref: str = ""
    sequence: list[str] = field(default_factory=list)
    parallel: list[str] = field(default_factory=list)
    parallel_count: Any = None  # int or "{{config.<attr>}}" cardinality assertion
    max_concurrency: Any = None  # int or "{{config.<attr>}}"
    params: dict[str, Any] = field(default_factory=dict)
    trigger: TriggerSpec | None = None
    loop: LoopSpec | None = None
    transitions: list[TransitionSpec] = field(default_factory=list)
    next: str = ""
    inputs: list[ArtifactSpec] = field(default_factory=list)
    outputs: list[ArtifactSpec] = field(default_factory=list)
    resources: str = "default"
    terminal: bool = False
    writes_decision: bool = False
    # Explicit acknowledgement for rank-aware raw commands. Catalog toolRefs
    # normally carry this contract themselves.
    multi_node_mode: str = "forbidden"


@dataclass
class NpaWorkflowSpec:
    api_version: str
    kind: str
    metadata: dict[str, Any]
    config: dict[str, Any]
    run_defaults: dict[str, Any]
    resources: dict[str, Any]
    initial: str
    states: dict[str, StateSpec]

    @property
    def name(self) -> str:
        return str(self.metadata.get("name") or "unnamed")


def load_spec(path: str | Path) -> NpaWorkflowSpec:
    import yaml

    spec_path = Path(path)
    if not spec_path.is_file():
        raise NpaWorkflowError(f"workflow spec not found: {spec_path}")
    try:
        data = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise NpaWorkflowError(f"workflow spec is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise NpaWorkflowError(
            f"workflow spec must be a mapping, got {type(data).__name__}"
        )
    spec = _parse_document(data)
    validate_spec(spec)
    # Parse first so state-specific failures retain their actionable domain
    # messages. The schema pass then catches raw-document constraints that do not
    # affect the normalized runtime model (metadata fields, enums, and so on).
    from npa.orchestration.npa_workflow.schema_validation import validate_document

    validate_document(data)
    return spec


def _parse_document(data: dict[str, Any]) -> NpaWorkflowSpec:
    api_version = str(data.get("apiVersion") or "")
    kind = str(data.get("kind") or "Workflow")
    metadata = _document_mapping(data, "metadata")
    config = _document_mapping(data, "config")
    run_defaults = _document_mapping(data, "run")
    resources = _document_mapping(data, "resources")

    raw_states = data.get("states") or {}
    if isinstance(raw_states, list):
        states_dict = {}
        for entry in raw_states:
            if not isinstance(entry, dict) or "name" not in entry:
                raise NpaWorkflowError(f"each list state needs a name: {entry!r}")
            name = entry["name"]
            if not isinstance(name, str):
                raise NpaWorkflowError(
                    f"state name must be a string, got {type(name).__name__}"
                )
            if name in states_dict:
                raise NpaWorkflowError(f"duplicate state name {name!r}")
            states_dict[name] = entry
        raw_states = states_dict
    if not isinstance(raw_states, dict) or not raw_states:
        raise NpaWorkflowError(
            "workflow spec must declare a non-empty 'states' mapping"
        )

    states: dict[str, StateSpec] = {}
    for name, entry in raw_states.items():
        if not isinstance(name, str):
            raise NpaWorkflowError(
                f"state name must be a string, got {type(name).__name__}"
            )
        if not isinstance(entry, dict):
            raise NpaWorkflowError(f"state {name!r} must be a mapping")
        states[name] = _parse_state(name, entry, config)

    initial = str(data.get("initial") or next(iter(states)))
    return NpaWorkflowSpec(
        api_version=api_version,
        kind=kind,
        metadata=metadata,
        config=config,
        run_defaults=run_defaults,
        resources=resources,
        initial=initial,
        states=states,
    )


def _document_mapping(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    raw = data.get(key)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise NpaWorkflowError(
            f"workflow field {key!r} must be a mapping, got {type(raw).__name__}"
        )
    non_string_keys = [item for item in raw if not isinstance(item, str)]
    if non_string_keys:
        raise NpaWorkflowError(
            f"workflow field {key!r} keys must be strings, got {non_string_keys[0]!r}"
        )
    return dict(raw)


def _parse_state(
    name: str, entry: dict[str, Any], config: dict[str, Any] | None = None
) -> StateSpec:
    loop = None
    loop_raw = entry.get("loop")
    if loop_raw is not None:
        if not isinstance(loop_raw, dict):
            raise NpaWorkflowError(f"state {name}: loop must be a mapping")
        until = _optional_string(name, "loop.until", loop_raw, "until")
        loop = LoopSpec(
            max=loop_raw.get("max"),
            until=until or None,
        )

    transitions: list[TransitionSpec] = []
    transitions_raw = entry.get("transitions")
    if transitions_raw is not None and not isinstance(transitions_raw, list):
        raise NpaWorkflowError(f"state {name}: transitions must be a list")
    for tr in transitions_raw or []:
        if not isinstance(tr, dict):
            raise NpaWorkflowError(f"state {name}: transition must be a mapping")
        goto = _optional_string(name, "transition.goto", tr, "goto")
        if not goto:
            raise NpaWorkflowError(f"state {name}: transition needs goto")
        transitions.append(
            TransitionSpec(
                when=_optional_string(name, "transition.when", tr, "when") or None,
                goto=goto,
                if_config=_optional_string(name, "transition.if", tr, "if") or None,
            )
        )

    run = None
    run_raw = entry.get("run")
    if run_raw is not None:
        if not isinstance(run_raw, dict):
            raise NpaWorkflowError(f"state {name}: run must be a mapping")
        argv = _string_list(
            name, "run.argv", run_raw.get("argv"), present="argv" in run_raw
        )
        run = RunSpec(
            shell=_optional_string(name, "run.shell", run_raw, "shell"),
            argv=argv,
        )

    trigger = None
    trigger_raw = entry.get("trigger")
    if trigger_raw is not None:
        if not isinstance(trigger_raw, dict):
            raise NpaWorkflowError(f"state {name}: trigger must be a mapping")
        trigger = TriggerSpec(
            uri=_optional_string(name, "trigger.uri", trigger_raw, "uri"),
            poll_seconds=_positive_int(
                name, "trigger.pollSeconds", trigger_raw, 30, config=config
            ),
            max_polls=_positive_int(
                name, "trigger.maxPolls", trigger_raw, 0, allow_zero=True, config=config
            ),
            min_objects=_positive_int(
                name, "trigger.minObjects", trigger_raw, 1, config=config
            ),
            config_expressions={
                key: value
                for key, snake in (
                    ("pollSeconds", "poll_seconds"),
                    ("maxPolls", "max_polls"),
                    ("minObjects", "min_objects"),
                )
                if isinstance(
                    value := trigger_raw.get(key, trigger_raw.get(snake)), str
                )
                and "{{" in value
            },
        )

    params_raw = entry["params"] if "params" in entry else {}
    if not isinstance(params_raw, dict):
        raise NpaWorkflowError(f"state {name}: params must be a mapping")
    for param_key, param_value in params_raw.items():
        if not isinstance(param_key, str):
            raise NpaWorkflowError(
                f"state {name}: params keys must be strings, got {param_key!r}"
            )
        # params values become config values for this state, and config values are
        # rendered into commands/URIs by token substitution. A dict/list cannot be
        # rendered, and would otherwise only fail much later (at render or run time).
        if not isinstance(param_value, (str, int, float, bool)) or param_value is None:
            raise NpaWorkflowError(
                f"state {name}: params.{param_key} must be a string, number or bool "
                f"(tokens render scalars), got {type(param_value).__name__}"
            )

    # Keep strict checks in the parser as well as the JSON Schema walker: callers
    # may construct raw state mappings directly, and silent ``str()``/``bool()``
    # coercion changes workflow control flow (for example, ``terminal: "false"``
    # used to become true).
    needs = _string_list(name, "needs", entry.get("needs"), present="needs" in entry)
    sequence = _string_list(
        name, "sequence", entry.get("sequence"), present="sequence" in entry
    )
    parallel = _string_list(
        name, "parallel", entry.get("parallel"), present="parallel" in entry
    )
    inputs = _artifact_list(name, "inputs", entry)
    outputs = _artifact_list(name, "outputs", entry)

    invalid_kinds = [
        artifact.kind
        for artifact in [*inputs, *outputs]
        if artifact.kind not in {"", "file", "directory"}
    ]
    if invalid_kinds:
        raise NpaWorkflowError(
            f"state {name}: artifact kind must be file or directory, got "
            f"{invalid_kinds[0]!r}"
        )

    return StateSpec(
        name=name,
        description=_optional_string(name, "description", entry, "description"),
        needs=needs,
        run=run,
        tool_ref=_aliased_string(name, entry, "toolRef", "tool_ref"),
        sequence=sequence,
        parallel=parallel,
        parallel_count=entry.get("parallelCount", entry.get("parallel_count")),
        max_concurrency=entry.get("maxConcurrency", entry.get("max_concurrency")),
        params=dict(params_raw),
        trigger=trigger,
        loop=loop,
        transitions=transitions,
        next=_optional_string(name, "next", entry, "next"),
        inputs=inputs,
        outputs=outputs,
        resources=_optional_string(
            name, "resources", entry, "resources", default="default"
        ),
        terminal=_aliased_bool(name, entry, "terminal"),
        writes_decision=_aliased_bool(name, entry, "writesDecision", "writes_decision"),
        multi_node_mode=_aliased_string(
            name,
            entry,
            "multiNodeMode",
            "multi_node_mode",
            default="forbidden",
        ),
    )


def _optional_string(
    state_name: str,
    field_name: str,
    entry: Mapping[str, Any],
    key: str,
    *,
    default: str = "",
) -> str:
    if key not in entry:
        return default
    value = entry[key]
    if not isinstance(value, str):
        raise NpaWorkflowError(
            f"state {state_name}: {field_name} must be a string, got "
            f"{type(value).__name__}"
        )
    return value


def _aliased_string(
    state_name: str,
    entry: Mapping[str, Any],
    key: str,
    alias: str,
    *,
    default: str = "",
) -> str:
    selected = key if key in entry else alias
    return _optional_string(state_name, key, entry, selected, default=default)


def _aliased_bool(
    state_name: str,
    entry: Mapping[str, Any],
    key: str,
    alias: str | None = None,
    *,
    default: bool = False,
) -> bool:
    selected = key if key in entry else alias
    if selected is None or selected not in entry:
        return default
    value = entry[selected]
    if type(value) is not bool:
        raise NpaWorkflowError(
            f"state {state_name}: {key} must be a boolean, got {type(value).__name__}"
        )
    return value


def _string_list(
    state_name: str,
    field_name: str,
    raw: Any,
    *,
    present: bool,
) -> list[str]:
    if not present:
        return []
    if not isinstance(raw, list):
        expected = (
            "a list of state names"
            if field_name in {"needs", "sequence", "parallel"}
            else "a list of strings"
        )
        raise NpaWorkflowError(
            f"state {state_name}: {field_name} must be {expected}, got "
            f"{type(raw).__name__}"
        )
    for item in raw:
        if not isinstance(item, str):
            expected = (
                "a state name (string)"
                if field_name in {"needs", "sequence", "parallel"}
                else "a string"
            )
            raise NpaWorkflowError(
                f"state {state_name}: {field_name} member must be {expected}, got "
                f"{item!r}"
            )
    return list(raw)


def _artifact_list(
    state_name: str, field_name: str, entry: Mapping[str, Any]
) -> list[ArtifactSpec]:
    if field_name not in entry:
        return []
    raw = entry[field_name]
    if not isinstance(raw, list):
        raise NpaWorkflowError(
            f"state {state_name}: {field_name} must be a list of artifacts, got "
            f"{type(raw).__name__}"
        )
    artifacts: list[ArtifactSpec] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise NpaWorkflowError(
                f"state {state_name}: {field_name}[{index}] must be a mapping"
            )
        if "uri" not in item:
            raise NpaWorkflowError(
                f"state {state_name}: {field_name}[{index}] requires uri"
            )
        artifacts.append(
            ArtifactSpec(
                uri=_optional_string(
                    state_name, f"{field_name}[{index}].uri", item, "uri"
                ),
                schema=_optional_string(
                    state_name, f"{field_name}[{index}].schema", item, "schema"
                ),
                kind=_optional_string(
                    state_name, f"{field_name}[{index}].kind", item, "kind"
                ),
            )
        )
    return artifacts


def _positive_int(
    state_name: str,
    field_name: str,
    entry: dict[str, Any],
    default: int,
    *,
    allow_zero: bool = False,
    config: dict[str, Any] | None = None,
) -> int:
    """Parse an optional positive integer from a nested mapping key."""

    key = field_name.split(".", 1)[1]
    snake = "".join(f"_{char.lower()}" if char.isupper() else char for char in key)
    if key in entry:
        raw = entry[key]
    elif snake in entry:
        raw = entry[snake]
    else:
        return default
    if isinstance(raw, str) and "{{" in raw:
        # Config-driven knob (e.g. pollSeconds: "{{config.inbox_poll_seconds}}"); the
        # value is resolved against config, like loop.max.
        raw = resolve_config_int(raw, config or {})
    try:
        value = _exact_int(raw)
    except ValueError as exc:
        raise NpaWorkflowError(
            f"state {state_name}: {field_name} must be an integer, got {raw!r}"
        ) from exc
    floor = 0 if allow_zero else 1
    if value < floor:
        raise NpaWorkflowError(
            f"state {state_name}: {field_name} must be >= {floor}, got {value}"
        )
    return value


def resolve_trigger_config(
    state_name: str, trigger: TriggerSpec, config: dict[str, Any]
) -> TriggerSpec:
    """Rebind a parsed trigger's expressions without changing its source instance."""

    values = {
        "pollSeconds": trigger.poll_seconds,
        "maxPolls": trigger.max_polls,
        "minObjects": trigger.min_objects,
        **trigger.config_expressions,
    }
    return TriggerSpec(
        uri=trigger.uri,
        poll_seconds=_positive_int(
            state_name, "trigger.pollSeconds", values, 30, config=config
        ),
        max_polls=_positive_int(
            state_name, "trigger.maxPolls", values, 0, allow_zero=True, config=config
        ),
        min_objects=_positive_int(
            state_name, "trigger.minObjects", values, 1, config=config
        ),
        config_expressions=dict(trigger.config_expressions),
    )


def validate_spec(spec: NpaWorkflowSpec) -> None:
    if spec.api_version not in SUPPORTED_API_VERSIONS:
        raise NpaWorkflowError(
            f"unsupported apiVersion {spec.api_version!r} (expected one of {sorted(SUPPORTED_API_VERSIONS)!r})"
        )
    if spec.kind != "Workflow":
        raise NpaWorkflowError(f"unsupported kind {spec.kind!r} (expected Workflow)")

    if spec.initial not in spec.states:
        raise NpaWorkflowError(f"initial state {spec.initial!r} is not defined")

    for state in spec.states.values():
        if state.loop and state.loop.until and state.loop.until not in PREDICATES:
            raise NpaWorkflowError(
                f"state {state.name}: unknown loop.until {state.loop.until!r}"
            )
        for tr in state.transitions:
            if tr.when and tr.when not in PREDICATES:
                raise NpaWorkflowError(
                    f"state {state.name}: unknown transition.when {tr.when!r}"
                )
            if tr.goto not in spec.states:
                raise NpaWorkflowError(
                    f"state {state.name}: transition goto unknown state {tr.goto!r}"
                )
        if state.next and state.next not in spec.states:
            raise NpaWorkflowError(
                f"state {state.name}: next references unknown state {state.next!r}"
            )
        for dep in state.needs:
            if dep not in spec.states:
                raise NpaWorkflowError(f"state {state.name}: unknown needs {dep!r}")
        for seq in state.sequence:
            if seq not in spec.states:
                raise NpaWorkflowError(f"state {state.name}: unknown sequence {seq!r}")
        _validate_parallel_group(spec, state)
        if state.trigger is not None:
            if not state.trigger.uri.strip():
                raise NpaWorkflowError(f"state {state.name}: trigger.uri is required")
            if not state.tool_ref and state.run is None:
                # A trigger gates real work; a wait-only state would render as an
                # empty scheduler task. Attach the trigger to the state it guards.
                raise NpaWorkflowError(
                    f"state {state.name}: trigger requires run or toolRef on the "
                    "same state (the trigger gates that state's work)"
                )
        if state.tool_ref:
            from npa.orchestration.npa_workflow.catalog import validate_tool_ref

            validate_tool_ref(state.tool_ref)
        if (
            not state.terminal
            and not state.sequence
            and not state.parallel
            and not state.run
            and not state.tool_ref
        ):
            if not state.transitions and not state.next:
                raise NpaWorkflowError(
                    f"state {state.name}: must set run, toolRef, sequence, parallel, "
                    "transitions, next, or terminal"
                )
        if (
            state.run
            and state.run.is_empty()
            and not state.tool_ref
            and not state.sequence
        ):
            raise NpaWorkflowError(f"state {state.name}: empty run block")

        if state.loop:
            _validate_loop_max(state, spec.config)

    _validate_resource_profiles(spec)
    _validate_executable_resource_contracts(spec)
    _validate_optional_sam2_config(spec)
    _validate_appearance_profiles(spec)
    _validate_transfer_rgb_weight(spec)
    _validate_transfer_first_chunk_frames(spec)
    _validate_transfer_cfg_normalization(spec)
    if "transfer_edge_threshold" in spec.config:
        from npa.workbench.cosmos.structural_transfer import edge_thresholds

        try:
            edge_thresholds(spec.config["transfer_edge_threshold"])
        except ValueError as exc:
            raise NpaWorkflowError(str(exc)) from exc
    _assert_acyclic_needs(spec)
    _assert_terminal_exists(spec)
    _assert_bounded_control_flow_cycles(spec)
    _assert_acyclic_expansion(spec)
    _validate_resolvable(spec)


def _validate_transfer_cfg_normalization(spec: NpaWorkflowSpec) -> None:
    key = "transfer_cfg_normalization"
    if key not in spec.config:
        return
    from npa.workbench.cosmos.structural_transfer import cfg_normalization_enabled

    try:
        enabled = cfg_normalization_enabled(spec.config[key])
    except ValueError as exc:
        raise NpaWorkflowError(str(exc)) from exc
    if enabled and spec.config.get("structural_control") != "edge":
        raise NpaWorkflowError(f"{key} requires structural_control=edge")


def _validate_transfer_first_chunk_frames(spec: NpaWorkflowSpec) -> None:
    key = "transfer_first_chunk_conditional_frames"
    if key not in spec.config:
        return
    value = spec.config[key]
    if type(value) not in (int, str) or str(value) not in ("0", "1"):
        raise NpaWorkflowError(f"{key} must be 0 or 1")
    if str(value) == "0" and spec.config.get("structural_control") != "edge":
        raise NpaWorkflowError(f"{key} requires structural_control=edge")


def _validate_transfer_rgb_weight(spec: NpaWorkflowSpec) -> None:
    if "transfer_rgb_weight" not in spec.config:
        return
    from npa.workbench.cosmos.structural_transfer import TransferSettings

    value = spec.config["transfer_rgb_weight"]
    try:
        if isinstance(value, bool):
            raise ValueError("transfer_rgb_weight must be numeric, not boolean")
        weight = float(value)
        TransferSettings(rgb_weight=weight).validate()
        if weight and spec.config.get("structural_control") != "edge":
            raise ValueError("transfer_rgb_weight requires structural_control=edge")
    except (TypeError, ValueError) as exc:
        raise NpaWorkflowError(f"invalid transfer_rgb_weight: {exc}") from exc


def _validate_appearance_profiles(spec: NpaWorkflowSpec) -> None:
    if "appearance_profiles_json" not in spec.config:
        return
    from npa.workflows.data_factory_appearance import parse_appearance_profiles

    try:
        parse_appearance_profiles(spec.config["appearance_profiles_json"])
    except ValueError as exc:
        raise NpaWorkflowError(str(exc)) from exc


def _validate_optional_sam2_config(spec: NpaWorkflowSpec) -> None:
    """Fail before provisioning when a workflow opts into the SAM2 contract."""

    if (
        not any(
            state.tool_ref == "workbench.cosmos2.transfer_execute"
            for state in spec.states.values()
        )
        or "segmentation_mode" not in spec.config
    ):
        return
    mode = str(spec.config.get("segmentation_mode") or "off").strip().lower()
    if mode == "off":
        return
    from npa.workbench.cosmos.sam2_masks import Sam2MaskConfig, Sam2MaskError

    try:
        config = Sam2MaskConfig(
            mode=mode,
            model_id=str(spec.config.get("sam2_model") or ""),
            model_revision=str(spec.config.get("sam2_model_revision") or ""),
            points_per_side=_exact_int(spec.config.get("sam2_points_per_side", 0)),
            predicted_iou_threshold=float(
                spec.config.get("sam2_predicted_iou_threshold") or 0
            ),
            stability_threshold=float(spec.config.get("sam2_stability_threshold") or 0),
            min_area_fraction=float(spec.config.get("sam2_min_area_fraction") or 0),
            max_area_fraction=float(spec.config.get("sam2_max_area_fraction") or 0),
            max_objects=_exact_int(spec.config.get("sam2_max_objects", 0)),
        )
    except (TypeError, ValueError) as exc:
        raise NpaWorkflowError(
            "optional SAM2 settings must use numeric sampling, threshold, area, "
            "and object-count values"
        ) from exc
    try:
        config.validate()
    except Sam2MaskError as exc:
        raise NpaWorkflowError(f"invalid optional SAM2 configuration: {exc}") from exc
    uri = str(spec.config.get("segmentation_uri") or "")
    if not uri.startswith("s3://"):
        raise NpaWorkflowError(
            "optional SAM2 requires segmentation_uri as a versioned s3:// prefix"
        )
    if str(spec.config.get("protected_chroma_regions_json") or "").strip():
        raise NpaWorkflowError(
            "optional SAM2 masks and protected_chroma_regions_json are mutually exclusive"
        )
    for state in spec.states.values():
        if state.tool_ref != "workbench.cosmos2.transfer_execute":
            continue
        profile = spec.resources.get(state.resources, {})
        nodes = profile_num_nodes(
            profile,
            name=state.resources,
            config=spec.config,
            run={"id": "validate-run"},
        )
        if nodes > 1:
            raise NpaWorkflowError(
                "optional SAM2 auto segmentation requires one augment node; use "
                "multi-GPU variant fan-out on that node"
            )
    try:
        luma_delta = _exact_int(spec.config.get("protected_luma_max_delta"))
        feather_pixels = _exact_int(spec.config.get("protected_feather_pixels"))
    except ValueError as exc:
        raise NpaWorkflowError(
            "optional SAM2 protected-luma delta and feather width must be integers"
        ) from exc
    if not 0 <= luma_delta <= 255:
        raise NpaWorkflowError(
            "optional SAM2 protected_luma_max_delta must be within 0..255"
        )
    if feather_pixels < 1:
        raise NpaWorkflowError(
            "optional SAM2 protected_feather_pixels must be positive"
        )


def profile_num_nodes(
    profile: Any,
    *,
    name: str,
    config: Mapping[str, Any] | None = None,
    run: Mapping[str, Any] | None = None,
) -> int:
    """Return a resource profile's ``num_nodes`` (default 1), validating bounds.

    The value may be a literal or a ``{{config.*}}`` token, so one blueprint can be
    submitted with a different block size (``--var augment_nodes=4``) instead of
    being edited. Callers that pass an unresolved profile must also pass the merged
    config; planned steps already carry a fully resolved resource profile.
    """

    if not isinstance(profile, dict):
        raise NpaWorkflowError(f"resource profile {name!r} must be a mapping")
    raw = profile.get("num_nodes")
    if raw in (None, ""):
        return 1
    if isinstance(raw, bool):
        raise NpaWorkflowError(
            f"resource profile {name!r}: num_nodes must be an integer, not a bool"
        )
    if isinstance(raw, str) and "{{" in raw:
        if config is None:
            raise NpaWorkflowError(
                f"resource profile {name!r}: num_nodes token requires merged config"
            )
        from npa.orchestration.npa_workflow.tokens import TokenError, resolve_tokens

        try:
            raw = resolve_tokens(raw, config=config, run=run or {})
        except TokenError as exc:
            raise NpaWorkflowError(
                f"resource profile {name!r}: num_nodes token is unresolvable: {exc}"
            ) from exc
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise NpaWorkflowError(
            f"resource profile {name!r}: num_nodes must be an integer, got {raw!r}"
        ) from exc
    if value < 1:
        raise NpaWorkflowError(
            f"resource profile {name!r}: num_nodes must be >= 1, got {value}"
        )
    if value > MAX_PROFILE_NODES:
        raise NpaWorkflowError(
            f"resource profile {name!r}: num_nodes must be <= {MAX_PROFILE_NODES}, "
            f"got {value} (a gang-scheduled block this large is almost always a typo; "
            "it would sit PENDING rather than fail)"
        )
    return value


def _validate_resource_profiles(spec: NpaWorkflowSpec) -> None:
    """Validate resource-profile fields the renderer will act on."""

    from npa.orchestration.npa_workflow.interpreter import _make_context

    context = _make_context(spec, run_id="validate-run")
    for name, profile in spec.resources.items():
        if not isinstance(profile, dict):
            raise NpaWorkflowError(f"resource profile {name!r} must be a mapping")
        resolved = resolve_resource_profile(
            str(name),
            profile,
            config=context.config,
            run=context.run,
        )
        profile_num_nodes(resolved, name=name)


def _validate_executable_resource_contracts(spec: NpaWorkflowSpec) -> None:
    """Reject unsafe gang reuse and run import-light tool semantic checks."""

    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
    from npa.orchestration.npa_workflow.interpreter import _make_context

    context = _make_context(spec, run_id="validate-run")
    # Executable params may carry a named loop token (for append-only per-attempt
    # output prefixes). Use the same first-iteration sentinel as the general
    # resolvability validator so semantic checks validate the resolved contract.
    context.loop_iterations = {
        state.name: 1 for state in spec.states.values() if state.loop is not None
    }
    for state in spec.states.values():
        if not state.run and not state.tool_ref:
            continue
        profile = spec.resources.get(state.resources)
        if profile is None and state.resources == "default":
            profile = {}
        if not isinstance(profile, Mapping):
            raise NpaWorkflowError(
                f"state {state.name}: unknown resource profile {state.resources!r}"
            )
        resolved = resolve_resource_profile(
            state.resources,
            profile,
            config=context.config,
            run=context.run,
        )
        nodes = profile_num_nodes(resolved, name=state.resources)
        entry = TOOL_CATALOG.get(state.tool_ref) if state.tool_ref else None
        mode = entry.multi_node_mode if entry is not None else state.multi_node_mode
        if mode not in {"forbidden", "sharded"}:
            raise NpaWorkflowError(
                f"state {state.name}: multiNodeMode must be 'sharded', got {mode!r}"
            )
        if nodes > 1 and mode != "sharded":
            raise NpaWorkflowError(
                f"state {state.name}: resource profile {state.resources!r} requests "
                f"num_nodes={nodes}, but this executable is not declared sharded; "
                "identical workers could duplicate or race output writers"
            )
        if entry is None:
            continue
        if not any(
            (
                entry.semantic_contract,
                entry.variant_count_config,
                entry.shard_activation_config,
                entry.shard_output_config,
            )
        ):
            continue
        from npa.orchestration.npa_workflow.tokens import TokenError, resolve_value

        effective_config = dict(context.config)
        try:
            resolved_params = resolve_value(
                state.params,
                config=context.config,
                run=context.run,
                loop_iterations=context.loop_iterations,
            )
        except TokenError as exc:
            raise NpaWorkflowError(f"state {state.name}: {exc}") from exc
        if not isinstance(
            resolved_params, Mapping
        ):  # defensive: params is typed mapping
            raise NpaWorkflowError(
                f"state {state.name}: params must resolve to a mapping"
            )
        effective_config.update(resolved_params)
        if nodes > 1 and entry.shard_activation_config:
            activation = str(
                effective_config.get(entry.shard_activation_config, "") or ""
            ).strip()
            if not activation:
                raise NpaWorkflowError(
                    f"state {state.name}: sharded execution requires non-empty "
                    f"config {entry.shard_activation_config!r}; without it every "
                    "gang member could run the unsharded writer"
                )
        if nodes > 1 and entry.shard_output_config:
            shard_output = str(
                effective_config.get(entry.shard_output_config, "") or ""
            ).strip()
            if not shard_output.startswith("s3://"):
                raise NpaWorkflowError(
                    f"state {state.name}: sharded execution requires config "
                    f"{entry.shard_output_config!r} to be a durable s3:// URI; "
                    "without it workers cannot publish and join fenced shards"
                )
        if entry.semantic_contract == "paidf_direct_translation":
            from npa.workflows.paidf_upstream import (
                validate_direct_generation_model,
                validate_token_factory_endpoint,
            )

            try:
                validate_token_factory_endpoint(
                    str(effective_config.get("vlm_url") or ""), "VLM"
                )
                validate_token_factory_endpoint(
                    str(effective_config.get("llm_url") or ""), "LLM"
                )
                validate_direct_generation_model(
                    str(effective_config.get("paidf_workflow") or ""),
                    str(effective_config.get("generation_model") or ""),
                    str(effective_config.get("generation_revision") or ""),
                )
            except ValueError as exc:
                raise NpaWorkflowError(f"state {state.name}: {exc}") from exc
        elif entry.semantic_contract == "cosmos_transfer_control":
            from npa.workbench.cosmos.control_contract import (
                ControlContractError,
                validate_control_request,
            )

            try:
                validate_control_request(
                    modality=effective_config.get("augment_control", "edge"),
                    weight=effective_config.get("augment_control_weight", "1.0"),
                    control_asset=effective_config.get("augment_control_asset_uri", ""),
                    control_prompt=effective_config.get("augment_control_prompt", ""),
                    mask_asset=effective_config.get("augment_mask_asset_uri", ""),
                    mask_prompt=effective_config.get("augment_mask_prompt", ""),
                )
            except ControlContractError as exc:
                raise NpaWorkflowError(f"state {state.name}: {exc}") from exc
        elif entry.semantic_contract:
            raise NpaWorkflowError(
                f"state {state.name}: unknown semantic contract "
                f"{entry.semantic_contract!r}"
            )
        if entry.variant_count_config:
            raw_variants = effective_config.get(entry.variant_count_config, "1")
            if isinstance(raw_variants, bool):
                raise NpaWorkflowError(
                    f"state {state.name}: {entry.variant_count_config} must be a "
                    "positive integer, not a bool"
                )
            try:
                variants = _exact_int(raw_variants)
            except ValueError as exc:
                raise NpaWorkflowError(
                    f"state {state.name}: {entry.variant_count_config} must be a "
                    f"positive integer, got {raw_variants!r}"
                ) from exc
            if variants < 1:
                raise NpaWorkflowError(
                    f"state {state.name}: {entry.variant_count_config} must be >= 1, "
                    f"got {variants}"
                )
            if nodes > variants:
                raise NpaWorkflowError(
                    f"state {state.name}: num_nodes={nodes} exceeds "
                    f"{entry.variant_count_config}={variants}; surplus GPU workers "
                    "would have empty strides"
                )


def resolve_resource_profile(
    name: str,
    profile: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    run: Mapping[str, Any],
    state_outputs: Mapping[str, Mapping[str, str]] | None = None,
    loop_iterations: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Deep-resolve resource tokens and reject non-positive accelerator counts."""

    from npa.orchestration.npa_workflow.tokens import TokenError, resolve_value

    try:
        resolved = resolve_value(
            dict(profile),
            config=config,
            run=run,
            state_outputs=state_outputs,
            loop_iterations=loop_iterations,
        )
    except TokenError as exc:
        raise NpaWorkflowError(f"resource profile {name!r}: {exc}") from exc
    assert isinstance(resolved, dict)

    accelerators = resolved.get("accelerators")
    raw_accelerators = profile.get("accelerators")
    if isinstance(accelerators, str):
        match = re.fullmatch(r"\s*[^,:\s]+\s*:\s*([+-]?\d+)\s*", accelerators)
        if match:
            count = int(match.group(1))
            if count < 1:
                raise NpaWorkflowError(
                    f"resource profile {name!r}: accelerator count must be >= 1, "
                    f"got {count}"
                )
        elif isinstance(raw_accelerators, str) and "{{" in raw_accelerators:
            raise NpaWorkflowError(
                f"resource profile {name!r}: resolved accelerators must use "
                f"'<type>:<positive-count>', got {accelerators!r}"
            )
    elif isinstance(accelerators, Mapping):
        for accelerator, raw_count in accelerators.items():
            if isinstance(raw_count, bool):
                raise NpaWorkflowError(
                    f"resource profile {name!r}: accelerator {accelerator!r} "
                    "count must be a positive integer"
                )
            try:
                count = _exact_int(raw_count)
            except ValueError as exc:
                raise NpaWorkflowError(
                    f"resource profile {name!r}: accelerator {accelerator!r} "
                    f"count must be a positive integer, got {raw_count!r}"
                ) from exc
            if count < 1:
                raise NpaWorkflowError(
                    f"resource profile {name!r}: accelerator {accelerator!r} "
                    f"count must be >= 1, got {count}"
                )
    return resolved


def _validate_parallel_group(spec: NpaWorkflowSpec, state: StateSpec) -> None:
    """Enforce the v0.0.1 shape of a ``parallel:`` fan-out group.

    Members are leaf states that the group owns: the group declares the barrier
    edge (``next``), so a member may not declare its own ``next``/``transitions``
    and may not itself be a group. This keeps the barrier deterministic — the
    downstream state starts only after every member reaches a terminal state.
    """

    if not state.parallel:
        if state.parallel_count is not None:
            raise NpaWorkflowError(
                f"state {state.name}: parallelCount requires a parallel group"
            )
        if state.max_concurrency is not None:
            raise NpaWorkflowError(
                f"state {state.name}: maxConcurrency requires a parallel group"
            )
        return

    if state.sequence:
        raise NpaWorkflowError(
            f"state {state.name}: set either sequence or parallel, not both"
        )
    if state.loop is not None:
        raise NpaWorkflowError(
            f"state {state.name}: loop is not supported directly on a parallel "
            "group; wrap the group in a sequence state that carries the loop"
        )
    if state.run is not None or state.tool_ref:
        raise NpaWorkflowError(
            f"state {state.name}: a parallel group cannot also declare run/toolRef"
        )
    if len(set(state.parallel)) != len(state.parallel):
        raise NpaWorkflowError(f"state {state.name}: duplicate parallel member")

    if state.parallel_count is not None:
        declared_count = resolve_config_int(state.parallel_count, spec.config)
        actual_count = len(state.parallel)
        if declared_count != actual_count:
            raise NpaWorkflowError(
                f"state {state.name}: parallelCount resolves to {declared_count}, but "
                f"the group declares {actual_count} members; change the member list or "
                "use the supported config value before workflow submission"
            )

    for member in state.parallel:
        if member not in spec.states:
            raise NpaWorkflowError(
                f"state {state.name}: unknown parallel member {member!r}"
            )
        if member == state.name:
            raise NpaWorkflowError(
                f"state {state.name}: parallel member cannot be itself"
            )
        child = spec.states[member]
        if child.sequence or child.parallel or child.loop is not None:
            raise NpaWorkflowError(
                f"state {state.name}: parallel member {member!r} must be a leaf state "
                "(no sequence, parallel, or loop)"
            )
        if child.terminal:
            raise NpaWorkflowError(
                f"state {state.name}: parallel member {member!r} cannot be terminal"
            )
        if child.next or child.transitions:
            raise NpaWorkflowError(
                f"state {state.name}: parallel member {member!r} must not declare "
                "next/transitions; the group owns the barrier edge"
            )

    if state.max_concurrency is not None:
        resolved = resolve_config_int(state.max_concurrency, spec.config)
        if resolved < 1:
            raise NpaWorkflowError(
                f"state {state.name}: maxConcurrency must be >= 1, got {resolved}"
            )
        if resolved > MAX_GROUP_CONCURRENCY:
            raise NpaWorkflowError(
                f"state {state.name}: maxConcurrency must be <= "
                f"{MAX_GROUP_CONCURRENCY}, got {resolved} (the effective value is "
                f"min(maxConcurrency, {len(state.parallel)} members); a larger bound "
                "is almost always a typo)"
            )


def _validate_resolvable(spec: NpaWorkflowSpec) -> None:
    """Resolve tokens and loop bounds so user errors surface at validate time."""

    from npa.orchestration.npa_workflow.interpreter import (
        _make_context,
        _resolved_run,
        state_config,
    )
    from npa.orchestration.npa_workflow.tokens import TokenError, resolve_tokens

    ctx = _make_context(spec, run_id="validate-run")
    # Named loop tokens are execution-time values. Resolve them with a harmless
    # first-iteration sentinel here so validation still catches misspelled loop
    # names while nested-loop specs remain statically checkable.
    ctx.loop_iterations = {
        state.name: 1 for state in spec.states.values() if state.loop is not None
    }
    for state in spec.states.values():
        if state.loop and state.loop.max is not None:
            try:
                resolved = resolve_config_int(state.loop.max, ctx.config)
            except NpaWorkflowError as exc:
                raise NpaWorkflowError(f"state {state.name}: {exc}") from exc
            if resolved < 1:
                raise NpaWorkflowError(
                    f"state {state.name}: loop.max must be >= 1, got {resolved}"
                )
        for key, value in state.params.items():
            if not isinstance(value, str):
                continue
            try:
                resolve_tokens(
                    value,
                    config=ctx.config,
                    run=ctx.run,
                    loop_iterations=ctx.loop_iterations,
                )
            except TokenError as exc:
                raise NpaWorkflowError(
                    f"state {state.name}: params.{key}: {exc}"
                ) from exc
        try:
            _resolved_run(state, ctx)
        except TokenError as exc:
            if not str(exc).startswith("unknown state token:"):
                raise NpaWorkflowError(f"state {state.name}: {exc}") from exc
        # Artifact URIs and trigger prefixes see the same per-state ``params``
        # overlay the command does, so a fan-out member can point its outputs at
        # its own prefix.
        state_scope = state_config(state, ctx)
        trigger_uris = [state.trigger.uri] if state.trigger is not None else []
        for uri in trigger_uris:
            try:
                resolve_tokens(
                    uri,
                    config=state_scope,
                    run=ctx.run,
                    loop_iterations=ctx.loop_iterations,
                )
            except TokenError as exc:
                raise NpaWorkflowError(
                    f"state {state.name}: trigger.uri: {exc}"
                ) from exc
        for artifact in [*state.inputs, *state.outputs]:
            if not artifact.uri:
                continue
            try:
                resolve_tokens(
                    artifact.uri,
                    config=state_scope,
                    run=ctx.run,
                    state_outputs=ctx.state_outputs,
                    loop_iterations=ctx.loop_iterations,
                )
            except TokenError as exc:
                if not str(exc).startswith("unknown state token:"):
                    raise NpaWorkflowError(f"state {state.name}: {exc}") from exc


def _validate_loop_max(state: StateSpec, config: dict[str, Any]) -> None:
    if state.loop is None or state.loop.max is None:
        return
    resolved = resolve_config_int(state.loop.max, config)
    if resolved < 1:
        raise NpaWorkflowError(
            f"state {state.name}: loop.max must be >= 1, got {resolved}"
        )


def _assert_acyclic_needs(spec: NpaWorkflowSpec) -> None:
    """Needs edges must be acyclic (ordering hints only)."""

    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(name: str) -> None:
        if name in visiting:
            raise NpaWorkflowError(f"cycle detected in needs among states (at {name})")
        if name in visited:
            return
        visiting.add(name)
        for dep in spec.states[name].needs:
            dfs(dep)
        visiting.remove(name)
        visited.add(name)

    for name in spec.states:
        dfs(name)


def _assert_terminal_exists(spec: NpaWorkflowSpec) -> None:
    terminals = [name for name, state in spec.states.items() if state.terminal]
    if not terminals:
        raise NpaWorkflowError(
            "workflow must declare at least one terminal: true state"
        )


def _assert_bounded_control_flow_cycles(spec: NpaWorkflowSpec) -> None:
    graph: dict[str, set[str]] = {name: set() for name in spec.states}
    for name, state in spec.states.items():
        if state.next:
            graph[name].add(state.next)
        for transition in state.transitions:
            graph[name].add(transition.goto)

    visited: set[str] = set()
    stack: list[str] = []

    def dfs(node: str) -> None:
        if node in stack:
            cycle = stack[stack.index(node) :] + [node]
            joined = " -> ".join(cycle)
            raise NpaWorkflowError(f"unbounded control-flow cycle detected: {joined}")
        if node in visited:
            return
        stack.append(node)
        for nxt in sorted(graph.get(node, ())):
            dfs(nxt)
        stack.pop()
        visited.add(node)

    for name in spec.states:
        dfs(name)


def _assert_acyclic_expansion(spec: NpaWorkflowSpec) -> None:
    """Reject cycles in the edges followed during recursive group expansion.

    A leaf inside ``sequence`` is expanded with ``follow_transitions=False``:
    transitions become bounded-loop decision signals, but an unconditional
    ``next`` is still followed. Group and loop states always follow ``next``.
    Model that separately from ordinary control flow so valid decision loops are
    accepted while nested groups cannot recurse forever during ``plan-spec``.
    """

    graph: dict[str, set[str]] = {name: set() for name in spec.states}
    for name, state in spec.states.items():
        graph[name].update(state.sequence)
        follows_next_in_sequence = (
            bool(state.sequence)
            or bool(state.parallel)
            or state.loop is not None
            or not state.transitions
        )
        if state.next and follows_next_in_sequence:
            graph[name].add(state.next)

    visiting: list[str] = []
    visited: set[str] = set()

    def dfs(node: str) -> None:
        if node in visiting:
            cycle = visiting[visiting.index(node) :] + [node]
            raise NpaWorkflowError(
                "unbounded control-flow cycle detected during sequence expansion: "
                + " -> ".join(cycle)
            )
        if node in visited:
            return
        visiting.append(node)
        for nxt in sorted(graph[node]):
            dfs(nxt)
        visiting.pop()
        visited.add(node)

    for name in spec.states:
        dfs(name)


def resolve_config_int(value: Any, config: dict[str, Any]) -> int:
    if isinstance(value, bool):
        raise NpaWorkflowError("loop max must be int or config ref, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{{config.") and text.endswith("}}"):
            attr = text[len("{{config.") : -2].strip()
        elif text.startswith("config."):
            attr = text[len("config.") :]
        else:
            attr = ""
        if attr:
            if attr not in config:
                raise NpaWorkflowError(f"config has no attribute {attr!r}")
            try:
                return _exact_int(config[attr])
            except ValueError as exc:
                raise NpaWorkflowError(
                    f"config.{attr} must be an integer loop bound, got {config[attr]!r}"
                ) from exc
        try:
            return _exact_int(text)
        except ValueError:
            pass
    raise NpaWorkflowError(f"cannot resolve loop max from {value!r}")


def _exact_int(value: Any) -> int:
    """Return an integer without accepting booleans or truncating decimals."""

    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    raise ValueError(f"not an exact integer: {value!r}")


def config_truthy(value: Any, config: dict[str, Any]) -> bool:
    if isinstance(value, str) and value.startswith("config."):
        attr = value[len("config.") :]
        return bool(config.get(attr))
    return bool(value)
