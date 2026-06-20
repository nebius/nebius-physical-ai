"""Load and validate ``apiVersion: npa.workflow/v0.0.1`` workflow specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.predicates import PREDICATES

API_VERSION = "npa.workflow/v0.0.1"


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
class StateSpec:
    name: str
    description: str = ""
    needs: list[str] = field(default_factory=list)
    run: RunSpec | None = None
    tool_ref: str = ""
    sequence: list[str] = field(default_factory=list)
    loop: LoopSpec | None = None
    transitions: list[TransitionSpec] = field(default_factory=list)
    next: str = ""
    inputs: list[ArtifactSpec] = field(default_factory=list)
    outputs: list[ArtifactSpec] = field(default_factory=list)
    resources: str = "default"
    terminal: bool = False


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
    data = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
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
        states[str(name)] = _parse_state(str(name), entry)

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


def _parse_state(name: str, entry: dict[str, Any]) -> StateSpec:
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
        loop=loop,
        transitions=transitions,
        next=str(entry.get("next") or ""),
        inputs=inputs,
        outputs=outputs,
        resources=str(entry.get("resources") or "default"),
        terminal=bool(entry.get("terminal")),
    )


def validate_spec(spec: NpaWorkflowSpec) -> None:
    if spec.api_version != API_VERSION:
        raise NpaWorkflowError(
            f"unsupported apiVersion {spec.api_version!r} (expected {API_VERSION})"
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
        if state.tool_ref:
            from npa.orchestration.npa_workflow.catalog import validate_tool_ref

            validate_tool_ref(state.tool_ref)
        if not state.terminal and not state.sequence and not state.run and not state.tool_ref:
            if not state.transitions and not state.next:
                raise NpaWorkflowError(
                    f"state {state.name}: must set run, toolRef, sequence, "
                    "transitions, next, or terminal"
                )
        if state.run and state.run.is_empty() and not state.tool_ref and not state.sequence:
            raise NpaWorkflowError(f"state {state.name}: empty run block")

        if state.loop:
            _validate_loop_max(state, spec.config)

    _assert_acyclic_needs(spec)
    _assert_terminal_exists(spec)
    _assert_bounded_control_flow_cycles(spec)


def _validate_loop_max(state: StateSpec, config: dict[str, Any]) -> None:
    if state.loop is None or state.loop.max is None:
        return
    try:
        resolved = resolve_config_int(state.loop.max, config)
    except NpaWorkflowError:
        return
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
            if not any(spec.states[item].loop for item in cycle):
                joined = " -> ".join(cycle)
                raise NpaWorkflowError(f"unbounded control-flow cycle detected: {joined}")
            return
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
        if text.startswith("config."):
            attr = text[len("config.") :]
            if attr not in config:
                raise NpaWorkflowError(f"config has no attribute {attr!r}")
            return int(config[attr])
        if text.isdigit():
            return int(text)
    raise NpaWorkflowError(f"cannot resolve loop max from {value!r}")


def config_truthy(value: Any, config: dict[str, Any]) -> bool:
    if isinstance(value, str) and value.startswith("config."):
        attr = value[len("config.") :]
        return bool(config.get(attr))
    return bool(value)
