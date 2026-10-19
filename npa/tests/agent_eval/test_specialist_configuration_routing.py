"""Exercise context admission, adaptive configuration selection and durable paid routing."""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from npa.agent_backend.specialists import configuration_router, token_router
from npa.agent_backend.specialists.capacity import _admission
from npa.agent_backend.specialists.config import (
    AgentConfiguration,
    ConfigurationRouting,
    ContextEndpoint,
    Operation,
    Profile,
    TeamConfig,
    fingerprint,
)
from npa.agent_backend.specialists.graph import initial_state
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.agent_backend.specialists.tools import WorkbenchTools


def _response(
    model, *, text="Done", tool=None, arguments=None, identifier="call-one", finish=None
):
    message = {"role": "assistant", "content": text}
    if tool:
        message["tool_calls"] = [
            {
                "id": identifier,
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(arguments or {})},
            }
        ]
    return {
        "model": model,
        "choices": [
            {
                "message": message,
                "finish_reason": finish or ("tool_calls" if tool else "stop"),
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5},
    }


class _Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat_completion(self, **arguments):
        self.calls.append(deepcopy(arguments))
        response = next(self.responses)
        return response(arguments) if callable(response) else deepcopy(response)


def _candidate(name, model="synthetic/fast", **changes):
    return AgentConfiguration(
        name=name,
        model=model,
        context_window=100000,
        description="Low cost for routine work; require diagnosis after failure",
        model_options={"max_tokens": 1000},
        tools_verified=True,
        input_price_per_million=0.1,
        output_price_per_million=0.3,
        **changes,
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src/value.txt").write_text("broken\n")
    fast = _candidate("fast-full")
    compact = _candidate("fast-compact", compact_context=True)
    deep = _candidate("deep-compact", "synthetic/deep", compact_context=True)
    policy = ConfigurationRouting(
        classifier=ContextEndpoint(
            model="synthetic/classifier",
            key_env="ROUTER_TEST_KEY",
            context_window=100000,
            model_options={"max_tokens": 1000},
        ),
        candidates=[fast, compact, deep],
    )
    profile = Profile(
        name="repair",
        description="Repair within scope",
        workspace=workspace,
        model=fast.model,
        configuration_routing=policy,
        read_paths=["src"],
        write_paths=["src"],
    )
    classifier = _Client(
        [_response("synthetic/classifier", text='{"selected_model":"fast-compact"}')]
    )
    worker = _Client([_response("synthetic/fast")])
    config = TeamConfig(
        state_directory=tmp_path / "state", profiles=[profile], default_profile="repair"
    )
    team = SpecialistTeam(config, clients={"repair": worker})
    monkeypatch.setenv("ROUTER_TEST_KEY", "synthetic-router-key")
    monkeypatch.setattr(
        configuration_router, "load_credentials", lambda: SimpleNamespace(tokens={})
    )
    monkeypatch.setattr(
        token_router, "TokenFactoryClient", lambda **_kwargs: classifier
    )
    team.submit("Inspect the sensor transform", specialist="repair", task_id="task")
    return SimpleNamespace(
        team=team, profile=profile, worker=worker, classifier=classifier, config=config
    )


def _tools(setup):
    return WorkbenchTools(setup.profile, setup.team.store, "task")


def _observations():
    messages = []
    for ordinal in range(2):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": str(ordinal),
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": str(ordinal),
                    "content": json.dumps(
                        {
                            "ok": True,
                            "path": "src/value.txt",
                            "sha256": "same",
                            "start_line": 1,
                            "end_line": 2,
                            "content": "x" * 12000,
                        }
                    ),
                },
            ]
        )
    return messages


def test_choice_drives_endpoint_options_and_reports_without_changing_grants(setup):
    before = setup.profile.model_dump()
    result = setup.team.work_once("repair")
    assert result["status"] == "completed"
    assert setup.worker.calls[0]["model"] == "synthetic/fast"
    assert setup.worker.calls[0]["extra"]["max_tokens"] == 1000
    report = setup.team.task_report("task")["configuration_routing"]
    assert report["current_configuration"] == "fast-compact"
    assert report["classifier_attempts"] == 1
    assert report["classifier_receipts"][0]["usage"] == {
        "input_tokens": 100,
        "output_tokens": 5,
    }
    assert setup.profile.model_dump() == before
    prompt = json.dumps(setup.classifier.calls)
    assert "history" in prompt and "criteria" in prompt
    assert "synthetic-router-key" not in prompt
    assert str(setup.profile.workspace) not in prompt


def test_capacity_selects_compact_view_and_preserves_full_history(setup):
    state = {"messages": _observations()}
    before = deepcopy(state)
    for candidate in setup.profile.configuration_routing.candidates:
        candidate.context_window = 22000
    setup.profile.configuration_routing.candidates.pop()
    candidate, view = configuration_router._select_configuration(
        state, setup.profile, _tools(setup)
    )
    assert candidate.name == "fast-compact"
    assert "content_omitted" in json.loads(view[1]["content"])
    assert len(json.loads(view[3]["content"])["content"]) == 12000
    assert state == before
    assert setup.classifier.calls == []
    receipt = setup.team.status("task")["events"][-1]
    assert not receipt["eligible"]["fast-full"]["admitted"]
    assert receipt["capacity"]["admitted"]


def test_every_call_rechecks_capacity_and_retains_current_route_without_classifier(
    setup,
):
    state = {"messages": _observations(), "configuration_route": "fast-full"}
    candidate, _ = configuration_router._select_configuration(
        state, setup.profile, _tools(setup)
    )
    assert candidate.name == "fast-full"
    assert setup.classifier.calls == []
    for candidate in setup.profile.configuration_routing.candidates:
        candidate.context_window = 22000
    setup.profile.configuration_routing.candidates.pop()
    selected, _ = configuration_router._select_configuration(
        state, setup.profile, _tools(setup)
    )
    assert selected.name == "fast-compact"
    assert setup.classifier.calls == []


def test_non_tool_endpoint_is_ineligible_even_with_capacity(setup):
    for candidate in setup.profile.configuration_routing.candidates:
        candidate.tools_verified = False
    setup.team.cancel("task")
    setup.team.submit("Inspect", specialist="repair", task_id="unsupported")
    result = setup.team.work_once("repair")
    assert result["status"] == "needs_attention"
    assert "no feasible configuration" in result["error"]
    assert not setup.worker.calls and not setup.classifier.calls


@pytest.mark.parametrize("target", ["worker", "classifier"])
def test_oversized_context_fails_before_paid_inference(setup, target):
    if target == "worker":
        for candidate in setup.profile.configuration_routing.candidates:
            candidate.context_window = 1100
    else:
        setup.profile.configuration_routing.classifier.context_window = 1100
    # Bind this task to the chosen policy before running it.
    setup.team.submit("Inspect", specialist="repair", task_id="overflow")
    result = setup.team.work_once("repair")  # Original policy is intentionally stale.
    assert result["status"] == "needs_attention"
    setup.team.cancel("task")
    result = setup.team.work_once("repair")
    assert result["status"] == "needs_attention"
    assert not setup.worker.calls and not setup.classifier.calls


@pytest.mark.parametrize(
    "answer,model,finish",
    [
        ('{"selected_model":"outside"}', "synthetic/classifier", "stop"),
        ('{"selected_model":"none"}', "synthetic/classifier", "stop"),
        ('{"selected_model":"fast-compact"}', "different", "stop"),
        ('{"selected_model":"fast-compact"}', "synthetic/classifier", "length"),
        (
            '{"selected_model":"fast-compact","extra":true}',
            "synthetic/classifier",
            "stop",
        ),
    ],
)
def test_invalid_or_abstaining_classifier_never_generates(setup, answer, model, finish):
    setup.classifier.responses = iter([_response(model, text=answer, finish=finish)])
    result = setup.team.work_once("repair")
    assert result["status"] == "needs_attention"
    assert not setup.worker.calls
    events = [
        item
        for item in setup.team.status("task")["events"]
        if item["type"] == "configuration_routing"
    ]
    assert events[-1]["usage"] == {"input_tokens": 100, "output_tokens": 5}


def test_lost_classifier_request_is_not_replayed_on_restart(setup):
    def crash(_arguments):
        raise KeyboardInterrupt

    setup.classifier.responses = iter([crash])
    with pytest.raises(KeyboardInterrupt):
        setup.team.work_once("repair")
    restarted = SpecialistTeam(setup.config, clients={"repair": setup.worker})
    result = restarted.work_once("repair")
    assert result["status"] == "needs_attention"
    assert len(setup.classifier.calls) == 1 and not setup.worker.calls
    assert "do not repeat" in result["error"]


def test_finalized_classification_reused_before_generation(setup):
    state = initial_state(setup.profile, setup.team.store._get("task")["goal"])
    configuration_router._select_configuration(state, setup.profile, _tools(setup))
    result = SpecialistTeam(setup.config, clients={"repair": setup.worker}).work_once(
        "repair"
    )
    assert result["status"] == "completed"
    assert len(setup.classifier.calls) == 1


@pytest.mark.parametrize("control", ["pause", "cancel", "pause-profile"])
def test_operator_control_during_classification_prevents_generation(setup, control):
    def control_during_request(_arguments):
        if control == "cancel":
            setup.team.cancel("task")
        elif control == "pause-profile":
            setup.team.pause(specialist="repair")
        else:
            setup.team.pause(task_id="task")
        return _response(
            "synthetic/classifier", text='{"selected_model":"fast-compact"}'
        )

    setup.classifier.responses = iter([control_during_request])
    setup.team.work_once("repair")
    assert not setup.worker.calls
    assert setup.team.status("task")["status"] != "completed"


def test_truncation_hands_off_with_history_and_without_executing_proposed_tools(setup):
    setup.classifier.responses = iter(
        [
            _response("synthetic/classifier", text='{"selected_model":"fast-compact"}'),
            _response("synthetic/classifier", text='{"selected_model":"deep-compact"}'),
        ]
    )
    setup.worker.responses = iter(
        [
            _response(
                "synthetic/fast",
                tool="run_operation",
                arguments={"name": "ungranted"},
                finish="length",
            ),
            _response("synthetic/deep"),
        ]
    )
    assert setup.team.work_once("repair")["status"] == "running"
    assert setup.team.work_once("repair")["status"] == "completed"
    assert len(setup.classifier.calls) == 2
    assert "truncated" in json.dumps(setup.classifier.calls[-1]["messages"])
    assert setup.team.status("task")["calls"] == []
    assert setup.team.task_report("task")["configuration_routing"]["handoff_count"] == 1


def test_failed_operation_receipt_reaches_next_model_without_replay(setup):
    operation = Operation(
        argv=["{python}", "-c", "raise SystemExit(1)"],
        description="Verify",
        handoff_on_failure=True,
    )
    setup.profile.operations["verify"] = operation
    setup.team.submit("Verify", specialist="repair", task_id="failure")
    # The older task cannot run under the new grants.
    assert setup.team.work_once("repair")["status"] == "needs_attention"
    setup.team.cancel("task")
    setup.worker.responses = iter(
        [
            _response(
                "synthetic/fast", tool="run_operation", arguments={"name": "verify"}
            ),
            _response("synthetic/deep", text="Verification failed; diagnosis required"),
        ]
    )
    setup.classifier.responses = iter(
        [
            _response("synthetic/classifier", text='{"selected_model":"fast-compact"}'),
            _response("synthetic/classifier", text='{"selected_model":"deep-compact"}'),
        ]
    )
    result = setup.team.work_once("repair")
    while result["status"] == "running":
        result = setup.team.work_once("repair")
    assert result["status"] == "completed"
    assert len(setup.team.status("failure")["calls"]) == 1
    assert '"returncode": 1' in json.dumps(setup.worker.calls[-1]["messages"]).replace(
        '\\"', '"'
    )
    assert "Operation failed" in json.dumps(setup.classifier.calls[-1]["messages"])


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"max_tokens": True},
        {"max_tokens": 0},
        {"max_tokens": 10000},
        {"max_tokens": "1000"},
    ],
)
def test_capacity_requires_real_explicit_output_reservation(options):
    with pytest.raises(ValueError, match="max_tokens"):
        ContextEndpoint(
            model="synthetic/model", context_window=10000, model_options=options
        )


def test_capacity_accounts_for_tools_unicode_and_framing():
    endpoint = ContextEndpoint(
        model="synthetic/model",
        context_window=10000,
        framing_tokens=500,
        model_options={"max_tokens": 1000},
    )
    messages = [{"role": "user", "content": "robot \u2603"}]
    receipt = _admission(endpoint, messages, tools=[{"description": "schema" * 20}])
    expected = (
        len(
            json.dumps(
                {"messages": messages, "tools": [{"description": "schema" * 20}]},
                ensure_ascii=True,
                sort_keys=True,
            ).encode("ascii")
        )
        + 500
        + 128
    )
    assert receipt["input_token_estimate"] == expected
    assert (
        receipt["output_tokens_reserved"] == 1000 and not receipt["exact_tokenization"]
    )
    with pytest.raises(ValueError, match="text and native tools"):
        _admission(endpoint, [{"role": "user", "content": [{"type": "image_url"}]}])


@pytest.mark.parametrize(
    "change",
    [
        {"model_router": "jev", "model_criteria": {"synthetic/fast": "routine"}},
        {"compact_context": True},
        {"fallback_models": [{"model": "synthetic/backup"}]},
    ],
)
def test_configuration_policy_rejects_ambiguous_legacy_routing(setup, change):
    with pytest.raises(ValueError, match="replaces legacy"):
        Profile.model_validate({**setup.profile.model_dump(), **change})


def test_configuration_contract_binds_durable_policy(setup):
    original = fingerprint(setup.profile)
    setup.profile.configuration_routing.candidates[0].context_window += 1
    assert fingerprint(setup.profile) != original
    assert setup.team.work_once("repair")["status"] == "needs_attention"
    assert not setup.worker.calls and not setup.classifier.calls


def test_native_compaction_preserves_failures_and_tool_ids(setup):
    state = {"messages": _observations()}
    state["messages"][1]["content"] = json.dumps(
        {"ok": False, "error": "sensor mismatch"}
    )
    original = hashlib.sha256(json.dumps(state).encode()).hexdigest()
    eligible, _ = configuration_router._candidates(
        state, setup.profile, _tools(setup).schemas()
    )
    view = eligible["fast-compact"][1]
    assert view[1] == state["messages"][1]
    assert [item.get("tool_call_id") for item in view] == [
        item.get("tool_call_id") for item in state["messages"]
    ]
    assert hashlib.sha256(json.dumps(state).encode()).hexdigest() == original


def test_configurations_of_same_model_have_distinct_output_and_reasoning_options(setup):
    candidate = setup.profile.configuration_routing.candidates[1]
    candidate.model_options = {"max_tokens": 3000, "reasoning_effort": "high"}
    setup.team.cancel("task")
    setup.team.submit("Inspect", specialist="repair", task_id="effort")
    assert setup.team.work_once("repair")["status"] == "completed"
    extra = setup.worker.calls[0]["extra"]
    assert extra["max_tokens"] == 3000 and extra["reasoning_effort"] == "high"


def test_hidden_reasoning_does_not_cross_routing_or_model_boundaries(setup):
    state = {
        "messages": [
            {"role": "user", "content": "Inspect"},
            {
                "role": "assistant",
                "content": "Observed",
                "reasoning_content": "opaque-hidden-state",
            },
        ],
        "configuration_route": "fast-full",
    }
    eligible, receipts = configuration_router._candidates(
        state, setup.profile, _tools(setup).schemas()
    )
    assert "reasoning_content" not in eligible["deep-compact"][1][-1]
    task, _criteria = configuration_router._routing_input(state, eligible, receipts)
    assert "opaque-hidden-state" not in json.dumps(task)
    assert state["messages"][-1]["reasoning_content"] == "opaque-hidden-state"


@pytest.mark.parametrize("price", [-1, float("inf"), float("nan")])
def test_invalid_prices_cannot_enter_routing_criteria(price):
    data = _candidate("fast").model_dump()
    with pytest.raises(ValueError):
        AgentConfiguration.model_validate({**data, "input_price_per_million": price})


def test_duplicate_configuration_ids_are_rejected(setup):
    data = setup.profile.configuration_routing.model_dump()
    data["candidates"][1]["name"] = data["candidates"][0]["name"]
    with pytest.raises(ValueError, match="unique"):
        ConfigurationRouting.model_validate(data)


def test_missing_classifier_key_blocks_generation(setup, monkeypatch):
    monkeypatch.delenv("ROUTER_TEST_KEY")
    result = setup.team.work_once("repair")
    assert result["status"] == "needs_attention"
    assert not setup.worker.calls and not setup.classifier.calls


def test_lost_worker_response_is_not_silently_repaid(setup):
    def lost(_arguments):
        raise KeyboardInterrupt

    setup.worker.responses = iter([lost])
    with pytest.raises(KeyboardInterrupt):
        setup.team.work_once("repair")
    result = SpecialistTeam(setup.config, clients={"repair": setup.worker}).work_once(
        "repair"
    )
    assert result["status"] == "needs_attention"
    assert len(setup.worker.calls) == len(setup.classifier.calls) == 1
    accounting = setup.team.task_report("task")["configuration_routing"][
        "inference_accounting"
    ]
    assert accounting["unknown_usage_requests"] == 1


def test_response_without_graph_checkpoint_is_not_repaid(setup, monkeypatch):
    from npa.agent_backend.specialists import graph

    record = graph._recorded_response

    def interrupted(*args, **kwargs):
        record(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(graph, "_recorded_response", interrupted)
    with pytest.raises(KeyboardInterrupt):
        setup.team.work_once("repair")
    monkeypatch.setattr(graph, "_recorded_response", record)
    result = SpecialistTeam(setup.config, clients={"repair": setup.worker}).work_once(
        "repair"
    )
    assert result["status"] == "needs_attention"
    assert len(setup.worker.calls) == 1
    accounting = setup.team.task_report("task")["configuration_routing"][
        "inference_accounting"
    ]
    assert (
        accounting["worker_requests"] == 1 and accounting["unknown_usage_requests"] == 0
    )


def test_repeated_failed_edit_hands_off_before_another_paid_generation(setup):
    from npa.agent_backend.specialists.configuration_execution import (
        _configuration_model,
    )

    messages = [{"role": "user", "content": "Repair"}]
    for identifier in ("one", "two"):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": identifier,
                            "function": {
                                "name": "edit_file",
                                "arguments": '{"path":"src/value.txt","expected_sha256":"same","old":"missing","new":"fixed"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": identifier,
                    "content": '{"ok":false,"error":"old text must match exactly once"}',
                },
            ]
        )
    result = _configuration_model(
        {"messages": messages, "configuration_route": "fast-compact"},
        setup.profile,
        _tools(setup),
        setup.worker,
    )
    assert result["configuration_excluded"] == ["fast-compact"]
    assert (
        result["recovering"] and not setup.worker.calls and not setup.classifier.calls
    )
