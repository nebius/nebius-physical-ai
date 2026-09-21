"""Reject incomplete stochastic replay and hidden per-parameter parity failures."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from npa.workbench.flex_pi.training_parity import _preserve_model_attributes
from npa.workbench.flex_pi.training_parity_metrics import (
    ParityComparison,
    tensor_difference,
)
from npa.workbench.flex_pi.training_randomness import AnchorRandomness


@dataclass
class Flags:
    present_v: torch.Tensor
    cm_v: bool = True
    cm_d: bool = True
    cm_p: bool = True


def _sample_flags(configuration, batch_size, device):
    return Flags(torch.rand(batch_size, device=device) > 0.5)


def _draw(tape, anchors, *, dtype=torch.float32):
    tape.begin(anchors)
    with tape:
        flags = tape.flags(_sample_flags, None, len(anchors), torch.device("cpu"))
        events = []
        for _ in range(4):
            events.append(torch.randn_like(torch.zeros(len(anchors), 2, dtype=dtype)))
            events.append(
                torch.rand((len(anchors),), dtype=torch.float32, device="cpu")
            )
        tape.finish()
    return flags, events


def test_random_replay_preserves_each_anchor_and_combines_only_batch_dimension():
    entries = {}
    originals = [
        _draw(AnchorRandomness(entries, capture=True), [index]) for index in range(3)
    ]
    for index in range(3):
        flags, events = _draw(AnchorRandomness(entries, capture=False), [index])
        assert torch.equal(flags.present_v, originals[index][0].present_v)
        assert all(
            torch.equal(a, b) for a, b in zip(events, originals[index][1], strict=True)
        )
    flags, events = _draw(AnchorRandomness(entries, capture=False), [0, 1, 2])
    assert torch.equal(
        flags.present_v, torch.cat([row[0].present_v for row in originals])
    )
    for event, result in enumerate(events):
        assert torch.equal(result, torch.cat([row[1][event] for row in originals]))


def test_random_replay_rejects_missing_events_shape_and_dtype_changes():
    entries = {}
    tape = AnchorRandomness(entries, capture=True)
    tape.begin([0])
    with pytest.raises(RuntimeError, match="completely consumed"):
        tape.finish()
    _draw(tape, [0])
    with pytest.raises(RuntimeError, match="shape or dtype"):
        _draw(AnchorRandomness(entries, capture=False), [0], dtype=torch.float64)
    replay = AnchorRandomness(entries, capture=False)
    replay.begin([0])
    with replay, pytest.raises(RuntimeError, match="shape or dtype"):
        torch.randn_like(torch.zeros(1, 3))


@pytest.mark.parametrize("probability,training", [(0.0, True), (0.1, False)])
def test_deterministic_dropout_passes_through_without_consuming_randomness(
    probability, training
):
    tape = AnchorRandomness({}, capture=True)
    value = torch.ones(3)
    with tape:
        result = torch.nn.functional.dropout(value, p=probability, training=training)
    assert torch.equal(result, value)
    assert tape.event == 0


def test_unexpected_stochastic_operations_are_rejected():
    tape = AnchorRandomness({}, capture=True)
    with tape, pytest.raises(RuntimeError, match="unexpected"):
        torch.nn.functional.dropout(torch.ones(3), p=0.1, training=True)
    with tape, pytest.raises(RuntimeError, match="unexpected"):
        torch.randperm(3)


def test_one_changed_small_parameter_cannot_hide_in_aggregate_error():
    reference = {
        ("gradient", "large"): torch.ones(10000),
        ("gradient", "small"): torch.zeros(1),
    }
    comparison = ParityComparison(reference)
    comparison.observe("gradient", "large", torch.ones(10000))
    comparison.observe("gradient", "small", torch.tensor([0.0001]))
    report = comparison.finish()
    assert report["aggregates"][0]["passed"] is True
    assert report["passed"] is False
    assert report["parameters"][1]["relative_l2"] is None


def test_exact_replay_rejects_one_ulp_and_requires_complete_tensor_population():
    value = torch.ones(1)
    changed = torch.nextafter(value, torch.full_like(value, 2))
    comparison = ParityComparison({("update", "a"): value}, exact=True)
    comparison.observe("update", "a", changed)
    assert comparison.finish()["passed"] is False
    with pytest.raises(RuntimeError, match="population"):
        ParityComparison({("update", "a"): value}).finish()


def test_nonfinite_tensor_is_rejected_before_json_serialization():
    with pytest.raises(RuntimeError, match="nonfinite"):
        tensor_difference(torch.ones(1), torch.tensor([float("nan")]))


def test_mutable_upstream_attributes_restore_even_after_failure():
    cleared = []
    original = torch.ones(1)
    model = SimpleNamespace(
        _camera_intrinsics=original,
        vae=SimpleNamespace(
            model=SimpleNamespace(clear_cache=lambda: cleared.append(True))
        ),
    )
    with pytest.raises(ValueError), _preserve_model_attributes(model):
        model._camera_intrinsics = torch.zeros(3)
        model._batch_flex = object()
        raise ValueError("failed model call")
    assert model._camera_intrinsics is original
    assert not hasattr(model, "_batch_flex")
    assert cleared == [True]
