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


@dataclass
class StateSpec:
    name: str
    description: str = ""
    needs: list[str] = field(default_factory=list)
    run: RunSpec | None = None
    tool_ref: str = ""
    sequence: list[str] = field(default_factory=list)
    parallel: list[str] = field(default_factory=list)
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
        raise NpaWorkflowError(f"workflow spec must be a mapping, got {type(data).__name__}")
    from npa.orchestration.npa_workflow.schema_validation import validate_document

    validate_document(data)
    spec = _parse_document(data)
    validate_spec(spec)
    return spec


def _parse_document(data: dict[str, Any]) -> NpaWorkflowSpec:
    api_version = str(data.get("apiVersion") or "")
    kind = str(data.get("kind") or "Workflow")
    metadata = dict(data.get("metadata") or {})
    config = dict(data.get("config") or {})
    run_defaults = dict(data.get("run") or {})
    resources = dict(data.get("resources") or {})

    raw_states = data.get("states") or {}
    if isinstance(raw_states, list):
        states_dict = {}
        for entry in raw_states:
            if not isinstance(entry, dict) or "name" not in entry:
                raise NpaWorkflowError(f"each list state needs a name: {entry!r}")
            name = str(entry["name"])
            states_dict[name] = entry
        raw_states = states_dict
    if not isinstance(raw_states, dict) or not raw_states:
        raise NpaWorkflowError("workflow spec must declare a non-empty 'states' mapping")

    states: dict[str, StateSpec] = {}
    for name, entry in raw_states.items():
        if not isinstance(entry, dict):
            raise NpaWorkflowError(f"state {name!r} must be a mapping")
        states[str(name)] = _parse_state(str(name), entry, config)

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


def _parse_state(
    name: str, entry: dict[str, Any], config: dict[str, Any] | None = None
) -> StateSpec:
    loop = None
    loop_raw = entry.get("loop")
    if loop_raw is not None:
        if not isinstance(loop_raw, dict):
            raise NpaWorkflowError(f"state {name}: loop must be a mapping")
        loop = LoopSpec(
            max=loop_raw.get("max"),
            until=str(loop_raw["until"]) if loop_raw.get("until") else None,
        )

    transitions: list[TransitionSpec] = []
    for tr in entry.get("transitions") or []:
        if not isinstance(tr, dict) or not tr.get("goto"):
            raise NpaWorkflowError(f"state {name}: transition needs goto")
        transitions.append(
            TransitionSpec(
                when=str(tr["when"]) if tr.get("when") else None,
                goto=str(tr["goto"]),
                if_config=str(tr["if"]) if tr.get("if") else None,
            )
        )

    run = None
    run_raw = entry.get("run")
    if run_raw is not None:
        if not isinstance(run_raw, dict):
            raise NpaWorkflowError(f"state {name}: run must be a mapping")
        run = RunSpec(
            shell=str(run_raw.get("shell") or ""),
            argv=[str(item) for item in (run_raw.get("argv") or [])],
        )

    trigger = None
    trigger_raw = entry.get("trigger")
    if trigger_raw is not None:
        if not isinstance(trigger_raw, dict):
            raise NpaWorkflowError(f"state {name}: trigger must be a mapping")
        trigger = TriggerSpec(
            uri=str(trigger_raw.get("uri") or ""),
            poll_seconds=_positive_int(
                name, "trigger.pollSeconds", trigger_raw, 30, config=config
            ),
            max_polls=_positive_int(
                name, "trigger.maxPolls", trigger_raw, 0, allow_zero=True, config=config
            ),
            min_objects=_positive_int(
                name, "trigger.minObjects", trigger_raw, 1, config=config
            ),
        )

    params_raw = entry.get("params") or {}
    if not isinstance(params_raw, dict):
        raise NpaWorkflowError(f"state {name}: params must be a mapping")
    for param_key, param_value in params_raw.items():
        # params values become config values for this state, and config values are
        # rendered into commands/URIs by token substitution. A dict/list cannot be
        # rendered, and would otherwise only fail much later (at render or run time).
        if not isinstance(param_value, (str, int, float, bool)) or param_value is None:
            raise NpaWorkflowError(
                f"state {name}: params.{param_key} must be a string, number or bool "
                f"(tokens render scalars), got {type(param_value).__name__}"
            )

    # Type checks live here (not in the JSON Schema): the shipped schema walker in
    # schema_validation.py does not resolve `$ref`/`$defs`, so state bodies are
    # only enforced in Python. Without this an author writing `parallel: shard-a`
    # would get a confusing "duplicate parallel member" error from iterating a
    # string character by character.
    parallel_raw = entry.get("parallel")
    if parallel_raw is not None and not isinstance(parallel_raw, list):
        raise NpaWorkflowError(
            f"state {name}: parallel must be a list of state names, got "
            f"{type(parallel_raw).__name__}"
        )
    for member in parallel_raw or []:
        if not isinstance(member, str):
            raise NpaWorkflowError(
                f"state {name}: parallel member must be a state name (string), got {member!r}"
            )

    inputs = [
        ArtifactSpec(uri=str(item.get("uri") or ""), schema=str(item.get("schema") or ""))
        for item in (entry.get("inputs") or [])
        if isinstance(item, dict)
    ]
    outputs = [
        ArtifactSpec(uri=str(item.get("uri") or ""), schema=str(item.get("schema") or ""))
        for item in (entry.get("outputs") or [])
        if isinstance(item, dict)
    ]

    return StateSpec(
        name=name,
        description=str(entry.get("description") or ""),
        needs=[str(item) for item in (entry.get("needs") or [])],
        run=run,
        tool_ref=str(entry.get("toolRef") or entry.get("tool_ref") or ""),
        sequence=[str(item) for item in (entry.get("sequence") or [])],
        parallel=[str(item) for item in (entry.get("parallel") or [])],
        max_concurrency=entry.get("maxConcurrency", entry.get("max_concurrency")),
        params=dict(params_raw),
        trigger=trigger,
        loop=loop,
        transitions=transitions,
        next=str(entry.get("next") or ""),
        inputs=inputs,
        outputs=outputs,
        resources=str(entry.get("resources") or "default"),
        terminal=bool(entry.get("terminal")),
        writes_decision=bool(entry.get("writesDecision") or entry.get("writes_decision")),
    )


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
    raw = entry.get(key, entry.get(snake))
    if raw is None or raw == "":
        return default
    if isinstance(raw, str) and "{{" in raw:
        # Config-driven knob (e.g. pollSeconds: "{{config.inbox_poll_seconds}}"); the
        # value is resolved against config, like loop.max.
        return resolve_config_int(raw, config or {})
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise NpaWorkflowError(
            f"state {state_name}: {field_name} must be an integer, got {raw!r}"
        ) from exc
    floor = 0 if allow_zero else 1
    if value < floor:
        raise NpaWorkflowError(
            f"state {state_name}: {field_name} must be >= {floor}, got {value}"
        )
    return value


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
        if state.run and state.run.is_empty() and not state.tool_ref and not state.sequence:
            raise NpaWorkflowError(f"state {state.name}: empty run block")

        if state.loop:
            _validate_loop_max(state, spec.config)

    _validate_resource_profiles(spec)
    _assert_acyclic_needs(spec)
    _assert_terminal_exists(spec)
    _assert_bounded_control_flow_cycles(spec)
    _validate_resolvable(spec)


def _validate_resource_profiles(spec: NpaWorkflowSpec) -> None:
    """Validate resource-profile fields the renderer will act on.

    ``num_nodes`` is the only one that changes the *shape* of the rendered task (a
    multi-node gang instead of one pod), so a bad value has to fail at validate time
    rather than at provision time.
    """

    from npa.orchestration.npa_workflow.interpreter import _make_context

    context = _make_context(spec, run_id="validate-run")
    for name, profile in spec.resources.items():
        if not isinstance(profile, dict):
            raise NpaWorkflowError(f"resource profile {name!r} must be a mapping")
        resolve_resource_profile(
            str(name),
            profile,
            config=context.config,
            run=context.run,
        )
        raw = profile.get("num_nodes")
        if raw in (None, ""):
            continue
        if isinstance(raw, bool):
            raise NpaWorkflowError(
                f"resource profile {name!r}: num_nodes must be an integer, not a bool"
            )
        try:
            value = int(raw)
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


def resolve_resource_profile(
    name: str,
    profile: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    run: Mapping[str, Any],
    state_outputs: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Resolve resource tokens and reject non-positive accelerator counts."""

    from npa.orchestration.npa_workflow.tokens import TokenError, resolve_tokens

    resolved: dict[str, Any] = {}
    for key, value in profile.items():
        try:
            resolved[key] = (
                resolve_tokens(
                    value,
                    config=config,
                    run=run,
                    state_outputs=state_outputs,
                )
                if isinstance(value, str)
                else value
            )
        except TokenError as exc:
            raise NpaWorkflowError(f"resource profile {name!r}: {key}: {exc}") from exc

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
                count = int(raw_count)
            except (TypeError, ValueError) as exc:
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

    for member in state.parallel:
        if member not in spec.states:
            raise NpaWorkflowError(f"state {state.name}: unknown parallel member {member!r}")
        if member == state.name:
            raise NpaWorkflowError(f"state {state.name}: parallel member cannot be itself")
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
                resolve_tokens(value, config=ctx.config, run=ctx.run)
            except TokenError as exc:
                raise NpaWorkflowError(f"state {state.name}: params.{key}: {exc}") from exc
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
                resolve_tokens(uri, config=state_scope, run=ctx.run)
            except TokenError as exc:
                raise NpaWorkflowError(f"state {state.name}: trigger.uri: {exc}") from exc
        for artifact in [*state.inputs, *state.outputs]:
            if not artifact.uri:
                continue
            try:
                resolve_tokens(
                    artifact.uri,
                    config=state_scope,
                    run=ctx.run,
                    state_outputs=ctx.state_outputs,
                )
            except TokenError as exc:
                if not str(exc).startswith("unknown state token:"):
                    raise NpaWorkflowError(f"state {state.name}: {exc}") from exc


def _validate_loop_max(state: StateSpec, config: dict[str, Any]) -> None:
    if state.loop is None or state.loop.max is None:
        return
    resolved = resolve_config_int(state.loop.max, config)
    if resolved < 1:
        raise NpaWorkflowError(f"state {state.name}: loop.max must be >= 1, got {resolved}")


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
        raise NpaWorkflowError("workflow must declare at least one terminal: true state")


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
                return int(config[attr])
            except (TypeError, ValueError) as exc:
                raise NpaWorkflowError(
                    f"config.{attr} must be an integer loop bound, got {config[attr]!r}"
                ) from exc
        if text.isdigit():
            return int(text)
    raise NpaWorkflowError(f"cannot resolve loop max from {value!r}")


def config_truthy(value: Any, config: dict[str, Any]) -> bool:
    if isinstance(value, str) and value.startswith("config."):
        attr = value[len("config.") :]
        return bool(config.get(attr))
    return bool(value)
