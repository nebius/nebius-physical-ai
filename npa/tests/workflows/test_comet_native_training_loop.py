from __future__ import annotations

import dataclasses

import pytest
from npa.workflows.behavior_challenge.native_training import (
    NativeTrainingPlan,
    run_native_training,
)


@dataclasses.dataclass
class State:
    step: int


class Runtime:
    def __init__(self, fail_publish=False):
        self.state = State(0)
        self.cursor = 0
        self.fail_publish = fail_publish
        self.events = []

    def next_delivered_batch(self):
        return {"x": self.cursor}, {"cursor": self.cursor}

    def train_step(self, state, batch):
        return State(state.step + 1), {
            "loss": 1,
            "grad_norm": 2,
            "param_norm": 3,
            "learning_rate": 4,
        }

    def synchronize(self, value):
        self.events.append("sync")

    def commit_cursor(self, delivery):
        self.cursor += 1
        self.events.append("commit")

    def state_step(self):
        return self.state.step

    def cursor_step(self):
        return self.cursor

    def durable_milestones(self):
        return {"0": self.released_parent_milestone()}

    def released_parent_milestone(self):
        return {"role": "released_parent_nonresumable"}

    def save_and_publish(self, root, step):
        self.events.append(f"publish:{step}")
        if self.fail_publish:
            raise RuntimeError("provider")
        return {"provider_manifest": {"provider_readback": True}}

    def retire_prior(self, step):
        self.events.append(f"retire:{step}")

    def runtime_record(self):
        return {"engine": "tiny_fixture"}


def test_loop_commits_after_sync_and_publishes_exact_milestones(tmp_path):
    runtime = Runtime()
    receipt = run_native_training(runtime, NativeTrainingPlan(4, (0, 2, 4)), tmp_path)
    assert receipt["logical_updates"] == 4
    assert runtime.events.index("sync") < runtime.events.index("commit")
    assert [value for value in runtime.events if value.startswith("publish")] == [
        "publish:2",
        "publish:4",
    ]


def test_publication_failure_never_retires(tmp_path):
    runtime = Runtime(fail_publish=True)
    with pytest.raises(RuntimeError, match="provider"):
        run_native_training(runtime, NativeTrainingPlan(2, (0, 2)), tmp_path)
    assert not any(value.startswith("retire") for value in runtime.events)


@pytest.mark.parametrize("milestones", [(1, 2), (0, 2, 2), (0, 1)])
def test_plan_rejects_noncanonical_schedule(milestones):
    with pytest.raises(ValueError):
        NativeTrainingPlan(2, milestones)
