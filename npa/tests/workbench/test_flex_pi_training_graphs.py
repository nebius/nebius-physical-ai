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
    model = torch.nn.Module()
    model.mot = torch.nn.Linear(2, 1, dtype=torch.float64)
    counters = {"skipped_cuda_graphs": 0, "compiled_graphs": 0}
    recorded = {"forward": 0, "backward": 0, "inference": 0}
    calls = []
    boundaries = []

    def compile_forward(function, **options):
        assert options == {"backend": "cudagraphs", "fullgraph": True, "dynamic": False}

        def execute(*args, **kwargs):
            assert torch._dynamo.config.optimize_ddp is False
            assert len(boundaries) == len(calls) + 1
            calls.append(True)
            counters["compiled_graphs"] = 1
            return function(*args, **kwargs)

        return execute

    monkeypatch.setattr(torch, "compile", compile_forward)
    monkeypatch.setattr(
        torch.compiler, "cudagraph_mark_step_begin", lambda: boundaries.append(True)
    )
    monkeypatch.setattr(graphs, "_check_runtime", lambda: None)
    monkeypatch.setattr(graphs, "_compiler_counters", lambda: counters.copy())
    monkeypatch.setattr(graphs, "_recorded_graphs", lambda: recorded.copy())
    original = deepcopy(model.state_dict())
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    graphs.configure_training_graphs(
        model, {"npa_cuda_graphs": "mot", "npa_activation_checkpointing": "off"}
    )
    assert [id(parameter) for parameter in model.parameters()] == parameter_ids
    assert set(original) == set(model.state_dict())
    return SimpleNamespace(
        torch=torch,
        model=model,
        original=original,
        counters=counters,
        recorded=recorded,
        calls=calls,
        boundaries=boundaries,
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
    assert len(fixture.calls) == 3
    assert len(fixture.boundaries) == 3
    fixture.recorded.update(forward=1, backward=1)
    receipt = model._npa_training_graphs.receipt()
    assert receipt["compiled_training_calls"] == 3
    assert receipt["eager_evaluation_calls"] == 2


def test_native_ddp_configuration_restored_after_capture_and_failure(compiled_model):
    fixture = compiled_model
    torch, state = fixture.torch, fixture.model._npa_training_graphs
    sample = torch.ones(1, 2, dtype=torch.float64)
    with torch._dynamo.config.patch(optimize_ddp=True):
        fixture.model.mot(sample).sum().backward()
        assert torch._dynamo.config.optimize_ddp is True

        def fail(*args, **kwargs):
            assert torch._dynamo.config.optimize_ddp is False
            raise RuntimeError("capture failed")

        state.compiled = fail
        with pytest.raises(RuntimeError, match="capture failed"):
            fixture.model.mot(sample)
        assert torch._dynamo.config.optimize_ddp is True


def test_compile_invocation_alone_is_not_capture_proof(compiled_model):
    fixture = compiled_model
    for _ in range(3):
        fixture.model.mot(fixture.torch.ones(1, 2, dtype=fixture.torch.float64))
    with pytest.raises(FlexPiError, match="recorded forward/backward"):
        fixture.model._npa_training_graphs.receipt()
    fixture.recorded["forward"] = 1
    with pytest.raises(FlexPiError, match="recorded forward/backward"):
        fixture.model._npa_training_graphs.receipt()


def test_forward_or_backward_fallback_cannot_be_published(compiled_model):
    fixture = compiled_model
    fixture.counters["skipped_cuda_graphs"] += 1
    with pytest.raises(FlexPiError, match="fell back"):
        graphs.check_training_graphs(SimpleNamespace(module=fixture.model))
    with pytest.raises(FlexPiError, match="fell back"):
        fixture.model.mot(fixture.torch.ones(1, 2, dtype=fixture.torch.float64))


def test_eager_default_does_not_inspect_or_compile_cuda(monkeypatch):
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


def test_recorded_graph_proof_traverses_backward_children_once(monkeypatch):
    torch = pytest.importorskip("torch")
    from torch._inductor import cudagraph_trees

    backward = SimpleNamespace(graph=object(), children={})
    forward = SimpleNamespace(graph=object(), children={2: [backward]})
    empty = SimpleNamespace(graph=None, children={})
    manager = SimpleNamespace(
        roots={1: [forward, forward], 3: [empty]},
        id_to_mode={
            key: SimpleNamespace(name=name)
            for key, name in ((1, "FORWARD"), (2, "BACKWARD"), (3, "INFERENCE"))
        },
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        cudagraph_trees, "get_manager", lambda device, create_if_none_exists: manager
    )
    assert graphs._recorded_graphs() == {"forward": 1, "backward": 1, "inference": 0}


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
                "backend": "cudagraphs",
                "fullgraph": True,
                "dynamic": False,
                "evaluation": "eager",
                "compiled_training_calls": 24,
                "eager_evaluation_calls": 0,
                "compiled_graphs": 1,
                "skipped_cuda_graphs": 0,
                "recorded_cuda_graphs": {"forward": 1, "backward": 1, "inference": 0},
            }
            for rank in range(4)
        ],
    }


def test_every_gathered_rank_must_confirm_capture(compiled_model, monkeypatch):
    fixture = compiled_model
    fixture.recorded.update(forward=1, backward=1)
    state = fixture.model._npa_training_graphs
    state.training_calls = 24
    fixture.counters["compiled_graphs"] = 1
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
        ("compiled_training_calls", 1),
        ("compiled_graphs", 0),
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
