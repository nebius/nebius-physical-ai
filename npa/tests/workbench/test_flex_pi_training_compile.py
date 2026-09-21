"""Reject failed compiler restoration and compilation inside steady windows."""

import copy

import pytest
import torch

from npa.workbench.flex_pi.training_compile import candidate_compilation
from npa.workbench.flex_pi.training_metrics import summarize_measurements


def test_candidate_restores_eager_call_after_failure_without_replacing_parameters(
    monkeypatch,
):
    model = torch.nn.Linear(2, 2)
    parameter_ids = [id(value) for value in model.parameters()]
    initial = copy.deepcopy(model.state_dict())
    monkeypatch.setattr(
        "npa.workbench.flex_pi.training_compile._normalizations", lambda _: [model]
    )

    def compile_fixture(**options):
        assert options == {"backend": "inductor", "mode": "default", "fullgraph": True}
        model._compiled_call_impl = object()

    monkeypatch.setattr(model, "compile", compile_fixture)
    with pytest.raises(ValueError), candidate_compilation(model, True):
        assert model._compiled_call_impl is not None
        raise ValueError("failed candidate")
    assert model._compiled_call_impl is None
    assert parameter_ids == [id(value) for value in model.parameters()]
    assert all(
        torch.equal(value, initial[name]) for name, value in model.state_dict().items()
    )


def _measured_rows():
    return [
        {
            "samples": 96,
            "seconds": 12.0,
            "compiler_by_rank": [
                {"unique_graphs": 6, "graph_breaks": 0, "compile_seconds": 4.0}
                for _ in range(4)
            ],
        }
        for _ in range(30)
    ]


@pytest.mark.parametrize(
    "change", ["recompile", "graph_break", "missing_rank", "disabled", "invalid_time"]
)
def test_any_rank_compiler_change_rejects_steady_throughput(change):
    rows = _measured_rows()
    assert summarize_measurements(rows)["median_samples_per_second"] == 8.0
    if change == "recompile":
        rows[15]["compiler_by_rank"][3]["unique_graphs"] += 1
    elif change == "graph_break":
        rows[5]["compiler_by_rank"][3]["graph_breaks"] = 1
    elif change == "missing_rank":
        rows[15]["compiler_by_rank"].pop()
    elif change == "disabled":
        rows[5]["compiler_by_rank"][3]["unique_graphs"] = 0
    else:
        rows[5]["compiler_by_rank"][3]["compile_seconds"] = float("nan")
    with pytest.raises(ValueError, match="compilation|graph breaks"):
        summarize_measurements(rows)
