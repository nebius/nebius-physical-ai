"""Require real capture evidence without changing parameter or eager evaluation identity."""

from copy import deepcopy
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from npa.sdk.workbench.flex_pi import train
from npa.workbench.flex_pi import training_graphs as graphs
from npa.workbench.flex_pi.runtime import FlexPiError
from npa.workbench.flex_pi.service import create_app


@pytest.fixture
def compiled_model(monkeypatch):
    torch = pytest.importorskip("torch")
    from torch.utils._pytree import tree_unflatten

    model = torch.nn.Module()
    model.mot = torch.nn.Linear(2, 1, dtype=torch.float64)
    captures = []

    def capture(state, leaves, spec):
        positions = [
            i for i, value in enumerate(leaves) if isinstance(value, torch.Tensor)
        ]
        captures.append(True)

        def execute(*tensors):
            values = list(leaves)
            for i, tensor in zip(positions, tensors, strict=True):
                values[i] = tensor
            args, kwargs = tree_unflatten(values, spec)
            result = state.original(*args, **kwargs)
            state.replays["forward"] += 1

            def backward(gradient):
                state.replays["backward"] += 1
                return gradient

            result.register_hook(backward)
            return result

        return execute, positions

    monkeypatch.setattr(graphs, "_capture_native", capture)
    monkeypatch.setattr(graphs, "_check_runtime", lambda: None)
    original = deepcopy(model.state_dict())
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    graphs.configure_training_graphs(
        model, {"npa_cuda_graphs": "mot", "npa_activation_checkpointing": "off"}
    )
    assert [id(parameter) for parameter in model.parameters()] == parameter_ids
    assert set(original) == set(model.state_dict())
    return SimpleNamespace(
        torch=torch, model=model, original=original, captures=captures
    )


def test_capture_wrapper_preserves_state_gradients_and_eager_evaluation(compiled_model):
    fixture = compiled_model
    torch, model = fixture.torch, fixture.model
    reference = torch.nn.Linear(2, 1, dtype=torch.float64)
    reference.load_state_dict(
        {key.removeprefix("mot."): value for key, value in fixture.original.items()}
    )
    sample = torch.tensor([[0.2, -0.4]], dtype=torch.float64)
    for _ in range(3):
        model.mot(sample).square().sum().backward()
        reference(sample).square().sum().backward()
    for actual, expected in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        assert torch.equal(actual.grad, expected.grad)
    model.eval()
    assert torch.equal(model.mot(sample), reference(sample))
    model.train()
    with torch.no_grad():
        assert torch.equal(model.mot(sample), reference(sample))
    assert len(fixture.captures) == 1
    receipt = model._npa_training_graphs.receipt()
    assert receipt["captured_training_calls"] == 3
    assert receipt["native_replays"] == {"forward": 3, "backward": 3}
    assert receipt["eager_evaluation_calls"] == 2


def test_next_forward_and_receipt_require_completed_backward(compiled_model):
    fixture = compiled_model
    sample = fixture.torch.ones(1, 2, dtype=fixture.torch.float64)
    output = fixture.model.mot(sample)
    with pytest.raises(FlexPiError, match="backward completion"):
        fixture.model.mot(sample)
    with pytest.raises(FlexPiError, match="backward completion"):
        fixture.model._npa_training_graphs.receipt()
    output.sum().backward()
    graphs.check_training_graphs(SimpleNamespace(module=fixture.model))
    with pytest.raises(FlexPiError, match="replayed forward/backward"):
        fixture.model._npa_training_graphs.receipt()


def test_changed_static_input_contract_is_rejected(compiled_model):
    fixture = compiled_model
    fixture.model.mot(
        fixture.torch.ones(1, 2, dtype=fixture.torch.float64)
    ).sum().backward()
    with pytest.raises(FlexPiError, match="static CUDA graph contract"):
        fixture.model.mot(fixture.torch.ones(2, 2, dtype=fixture.torch.float64))
    with pytest.raises(FlexPiError, match="static CUDA graph contract"):
        fixture.model.mot(
            fixture.torch.ones(1, 2, dtype=fixture.torch.float64, requires_grad=True)
        )
    with pytest.raises(FlexPiError, match="unsupported static metadata"):
        graphs._input_signature([object()])


def test_native_capture_failure_propagates_without_eager_fallback(
    compiled_model, monkeypatch
):
    def fail(*args):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(graphs, "_capture_native", fail)
    with pytest.raises(RuntimeError, match="capture failed"):
        compiled_model.model.mot(
            compiled_model.torch.ones(1, 2, dtype=compiled_model.torch.float64)
        )
    assert compiled_model.model._npa_training_graphs.training_calls == 0


def test_eager_default_does_not_inspect_or_capture_cuda(monkeypatch):
    monkeypatch.setattr(
        graphs, "_check_runtime", lambda: pytest.fail("eager runtime inspected")
    )
    graphs.configure_training_graphs(object(), {})
    graphs.check_training_graphs(object())


@pytest.mark.parametrize(
    "version, available, suppress",
    [
        ("2.8.0+cu128", True, False),
        ("2.7.1+cpu", False, False),
        ("2.7.1+cu128", True, True),
    ],
)
def test_incompatible_runtime_or_error_suppression_is_rejected(
    monkeypatch, version, available, suppress
):
    torch = pytest.importorskip("torch")
    from torch import _dynamo

    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(_dynamo.config, "suppress_errors", suppress)
    with pytest.raises(FlexPiError, match="training CUDA graphs"):
        graphs._check_runtime()


def test_native_replay_observation_requires_pinned_closure_layout():
    with pytest.raises(FlexPiError, match="closure layout"):
        graphs._observe_native_replays(SimpleNamespace(forward=lambda: None), {})


@pytest.mark.parametrize("fails", [False, True])
def test_ddp_prepare_uses_ordered_side_stream_once_and_restores_on_error(
    monkeypatch, fails
):
    torch = pytest.importorskip("torch")
    from contextlib import contextmanager

    events = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            events.append((self.name, "wait", other.name))

    caller, side = Stream("caller"), Stream("side")
    active = [caller]

    @contextmanager
    def scope(stream):
        active.append(stream)
        try:
            yield
        finally:
            active.pop()

    def original(*args, **kwargs):
        assert args == ("model", "optimizer") and kwargs == {"device_placement": None}
        assert active[-1] is side
        assert events == [("side", "wait", "caller")]
        events.append("prepare")
        if fails:
            raise RuntimeError("prepare failed")
        return "wrapped model", "wrapped optimizer"

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: caller)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: side)
    monkeypatch.setattr(torch.cuda, "stream", scope)
    accelerator = SimpleNamespace(prepare=original)
    graphs.prepare_ddp_for_graph_capture(accelerator)
    if fails:
        with pytest.raises(RuntimeError, match="prepare failed"):
            accelerator.prepare("model", "optimizer", device_placement=None)
    else:
        assert accelerator.prepare("model", "optimizer", device_placement=None) == (
            "wrapped model",
            "wrapped optimizer",
        )
    assert accelerator.prepare is original
    assert active == [caller]
    assert events == [("side", "wait", "caller"), "prepare", ("caller", "wait", "side")]


def test_capture_detaches_samples_preserves_parameters_and_restores_rng(monkeypatch):
    torch = pytest.importorskip("torch")
    from torch.utils._pytree import tree_flatten

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(3.0))

        def forward(self, inputs, gain):
            return {"output": inputs * self.weight * gain}

    module = Model()
    source = torch.tensor([2.0], requires_grad=True)
    inputs = source * 2
    leaves, spec = tree_flatten(((inputs,), {"gain": 0.5}))
    state = SimpleNamespace(module=module, original=module.forward, replays={})
    original_fork = torch.random.fork_rng

    def fork(*, devices):
        assert devices == [0]
        return original_fork(devices=[])

    def capture(wrapper, samples, **options):
        assert options == {"num_warmup_iters": 11, "allow_unused_input": True}
        assert list(wrapper.parameters()) == list(module.parameters())
        assert samples[0].is_leaf and samples[0].requires_grad
        assert samples[0].data_ptr() != inputs.data_ptr()
        assert torch.equal(samples[0], inputs)
        torch.rand(5)
        torch.autograd.grad(
            wrapper(*samples)["output"].sum(), (*samples, *wrapper.parameters())
        )
        assert module.weight.grad is None
        return wrapper

    monkeypatch.setattr(torch.random, "fork_rng", fork)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "make_graphed_callables", capture)
    monkeypatch.setattr(graphs, "_observe_native_replays", lambda wrapper, counts: None)
    rng = torch.get_rng_state().clone()
    wrapper, positions = graphs._capture_native(state, leaves, spec)
    assert torch.equal(torch.get_rng_state(), rng)
    wrapper(*(leaves[i] for i in positions))["output"].sum().backward()
    assert source.grad.item() == 3.0
    assert module.weight.grad.item() == 2.0


def test_capture_autocasts_only_forward_and_preserves_eager_backward(monkeypatch):
    torch = pytest.importorskip("torch")
    from torch.utils._pytree import tree_flatten

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(2, 7, 5))

        def forward(self, inputs):
            assert torch.is_autocast_enabled("cpu")
            return torch.bmm(inputs * 10, self.weight * 10)

    torch.manual_seed(42)
    module = Model()
    inputs = torch.randn(2, 3, 7, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
        expected_output = module(inputs)
    expected_gradients = torch.autograd.grad(
        expected_output, (inputs, module.weight), torch.ones_like(expected_output)
    )
    leaves, spec = tree_flatten(((inputs,), {}))
    state = SimpleNamespace(module=module, original=module.forward, replays={})
    original_autocast = torch.autocast
    original_enabled = torch.is_autocast_enabled
    original_dtype = torch.get_autocast_dtype
    original_fork = torch.random.fork_rng
    monkeypatch.setattr(
        torch, "autocast", lambda device, **kw: original_autocast("cpu", **kw)
    )
    monkeypatch.setattr(
        torch, "is_autocast_enabled", lambda device: original_enabled("cpu")
    )
    monkeypatch.setattr(
        torch, "get_autocast_dtype", lambda device: original_dtype("cpu")
    )
    monkeypatch.setattr(
        torch.random, "fork_rng", lambda **kw: original_fork(devices=[])
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(graphs, "_observe_native_replays", lambda *args: None)

    def capture(wrapper, samples, **options):
        assert not original_enabled("cpu")
        output = wrapper(*samples)
        assert output.dtype == torch.bfloat16
        assert torch.equal(output, expected_output)
        assert not original_enabled("cpu")
        gradients = torch.autograd.grad(
            output, (*samples, *wrapper.parameters()), torch.ones_like(output)
        )
        for actual, expected in zip(gradients, expected_gradients, strict=True):
            assert torch.equal(actual, expected)
        return wrapper

    monkeypatch.setattr(torch.cuda, "make_graphed_callables", capture)
    with original_autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
        graphs._capture_native(state, leaves, spec)
        assert original_enabled("cpu")
    assert not original_enabled("cpu")


def test_native_observer_counts_only_successful_forward_and_backward_replays(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from functools import wraps

    class NativeGraph:
        def __init__(self):
            self.calls = 0
            self.fail = False

        def replay(self):
            if self.fail:
                raise RuntimeError("native replay failed")
            self.calls += 1

    monkeypatch.setattr(torch.cuda, "CUDAGraph", NativeGraph)
    fwd_graph, bwd_graph = NativeGraph(), NativeGraph()

    def decorated(function):
        @wraps(function)
        def call(*args):
            return function(*args)

        return call

    class Graphed:
        @staticmethod
        def forward():
            fwd_graph.replay()

        @staticmethod
        @decorated
        def backward():
            bwd_graph.replay()

    def graphed():
        return Graphed.forward()

    def new_fwd():
        return graphed()

    counts = {"forward": 0, "backward": 0}
    graphs._observe_native_replays(SimpleNamespace(forward=new_fwd), counts)
    new_fwd()
    Graphed.backward()
    assert counts == {"forward": 1, "backward": 1}
    assert fwd_graph.calls == bwd_graph.calls == 1
    bwd_graph.fail = True
    with pytest.raises(RuntimeError, match="native replay failed"):
        Graphed.backward()
    assert counts == {"forward": 1, "backward": 1}


@pytest.mark.parametrize(
    "mode, activation", [("auto", "off"), (False, "off"), ("mot", "on")]
)
def test_invalid_capture_policy_fails_before_launch(monkeypatch, mode, activation):
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **kw: pytest.fail("invalid policy launched")
    )
    with pytest.raises(FlexPiError, match="cuda-graphs"):
        train(
            output_path="s3://example-bucket/run",
            cuda_graphs=mode,
            activation_checkpointing=activation,
        )


def _receipt():
    return {
        "policy": "mot",
        "ranks": [
            {
                "rank": rank,
                "policy": "mot",
                "backend": "make_graphed_callables",
                "fullgraph": True,
                "dynamic": False,
                "evaluation": "eager",
                "captured_training_calls": 24,
                "eager_evaluation_calls": 0,
                "native_replays": {"forward": 24, "backward": 24},
                "skipped_cuda_graphs": 0,
                "recorded_cuda_graphs": {"forward": 1, "backward": 1, "inference": 0},
            }
            for rank in range(4)
        ],
    }


def test_every_gathered_rank_must_confirm_capture(compiled_model, monkeypatch):
    fixture = compiled_model
    state = fixture.model._npa_training_graphs
    state.training_calls = 24
    state.captured = object()
    state.replays.update(forward=24, backward=24)
    monkeypatch.setattr(fixture.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(fixture.torch.distributed, "get_world_size", lambda: 4)

    def gather(rows, local):
        rows[:] = [{**deepcopy(local), "rank": rank} for rank in range(4)]
        rows[3]["recorded_cuda_graphs"]["backward"] = 0

    monkeypatch.setattr(fixture.torch.distributed, "all_gather_object", gather)
    with pytest.raises(FlexPiError, match="training-only capture"):
        graphs.training_graphs_receipt(fixture.model, {"npa_cuda_graphs": "mot"})


@pytest.mark.parametrize(
    "key, value",
    [
        ("rank", 0),
        ("rank", True),
        ("policy", "off"),
        ("backend", "inductor"),
        ("fullgraph", 1),
        ("dynamic", True),
        ("evaluation", "compiled"),
        ("captured_training_calls", 1),
        ("native_replays", {"forward": 24, "backward": 0}),
        ("skipped_cuda_graphs", 1),
        ("recorded_cuda_graphs", {"forward": 2, "backward": 0, "inference": 0}),
        ("recorded_cuda_graphs", {"forward": 1, "backward": 1, "inference": 1}),
    ],
)
def test_one_rank_with_missing_or_wrong_capture_rejects_phase(key, value):
    receipt = _receipt()
    graphs.validate_graphs_receipt(receipt, "mot", 4)
    receipt["ranks"][2][key] = value
    with pytest.raises(FlexPiError, match="CUDA graph"):
        graphs.validate_graphs_receipt(receipt, "mot", 4)


@pytest.mark.parametrize("receipt", [None, {}, {"policy": "mot", "ranks": []}])
def test_missing_graph_receipt_is_rejected(receipt):
    with pytest.raises(FlexPiError, match="CUDA graph"):
        graphs.validate_graphs_receipt(receipt, "mot", 4)


def test_parent_rejects_successful_worker_without_capture_evidence(
    monkeypatch, tmp_path
):
    from npa.workbench.flex_pi import training

    monkeypatch.setattr(
        training, "_run_local_phase", lambda *a: {"reference_benchmark_beaten": False}
    )
    plan = training._plan(
        training.TrainingRequest("s3://example-bucket/run", cuda_graphs="mot")
    )
    with pytest.raises(FlexPiError, match="receipt"):
        training._run_phase(plan, tmp_path)


def test_cli_and_sdk_preserve_the_explicit_training_capture_policy():
    import json
    from typer.testing import CliRunner
    from npa.cli.main import app

    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "flex-pi",
            "train",
            "--output-path",
            "s3://example-bucket/run",
            "--activation-checkpointing",
            "off",
            "--cuda-graphs",
            "mot",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == train(
        output_path="s3://example-bucket/run",
        activation_checkpointing="off",
        cuda_graphs="mot",
        dry_run=True,
    )


@pytest.mark.parametrize("mode, status", [("off", 200), ("mot", 200), ("auto", 422)])
def test_authenticated_service_preserves_capture_policy(tmp_path, mode, status):
    manifest = tmp_path / "input.json"
    manifest.write_text("{}")
    service = create_app(
        token="test-token",
        output_root="s3://example-bucket/run",
        input_manifest=str(manifest),
    )
    with TestClient(service) as client:
        response = client.post(
            "/train",
            json={
                "output_path": "candidate",
                "cuda_graphs": mode,
                "activation_checkpointing": "off",
                "dry_run": True,
            },
            headers={"Authorization": "Bearer test-token"},
        )
    assert response.status_code == status
    if status == 200:
        assert response.json()["execution"]["cuda_graphs"] == mode
