"""Reject transient bootstrap success and recurrent memory crossing collection resets."""

import sys
from types import SimpleNamespace

import pytest

from npa.workflows.sim2real import isaac_franka_bootstrap as bootstrap
from npa.workflows.sim2real import isaac_recurrent_imitation as imitation


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def assessment_runtime(torch, monkeypatch):
    task = SimpleNamespace(
        num_envs=2,
        device="cpu",
        npa_scenario_rows=[{}, {}, {}, {}],
        npa_scenario_indices=torch.tensor([0, 1]),
        distance=torch.full((2,), 0.04),
        speed=torch.full((2,), 0.02),
    )
    scenarios = SimpleNamespace(
        STABLE_PLACEMENT_DISTANCE_M=0.05,
        STABLE_PLACEMENT_SPEED_MPS=0.03,
        STABLE_PLACEMENT_STEPS=3,
        _placement_state=lambda task, **kwargs: (task.distance, task.speed, None, None),
    )
    monkeypatch.setitem(sys.modules, "isaac_scenario_task", scenarios)
    return task, bootstrap._TerminalPlacementAssessment(task)


def _advance(torch, task, assessment, *, done=False):
    assessment.advance(task, torch.full((task.num_envs,), done, dtype=torch.bool))


def test_transient_hold_cannot_certify_terminal_actor_handoff(
    torch, assessment_runtime
):
    task, assessment = assessment_runtime
    for _ in range(3):
        _advance(torch, task, assessment)
    ever_successful = torch.tensor([True, True, False, False])
    assert int(ever_successful.sum()) * 2 >= len(ever_successful)
    task.speed.fill_(0.04)
    _advance(torch, task, assessment)
    _advance(torch, task, assessment, done=True)
    assert not assessment.successful_rows.any()
    assert assessment.evidence(ever_successful) == {
        "placement_criterion": "first_episode_last_pre_reset",
        "terminal_training_scenarios": 2,
        "ever_training_placements": 2,
    }
    with pytest.raises(RuntimeError, match="0/4 training scenarios"):
        bootstrap._require_successful_demonstrations(
            assessment.successful_rows, index="actor terminal assessment"
        )


@pytest.mark.parametrize("stable_samples,expected", [(2, False), (3, True)])
def test_terminal_outcome_uses_pre_reset_state_only(
    torch, assessment_runtime, stable_samples, expected
):
    task, assessment = assessment_runtime
    for _ in range(stable_samples):
        _advance(torch, task, assessment)
    # The newly reset state deliberately gives the opposite stability verdict.
    task.distance.fill_(0.2 if expected else 0.01)
    task.npa_scenario_indices = torch.tensor([2, 3])
    _advance(torch, task, assessment, done=True)
    assert assessment.successful_rows.tolist() == [expected, expected, False, False]
    assert assessment.completed_rows.tolist() == [True, True, False, False]
    assert assessment.stable_steps.tolist() == [0, 0]


def test_incomplete_episode_cannot_certify_handoff(torch, assessment_runtime):
    task, assessment = assessment_runtime
    for _ in range(5):
        _advance(torch, task, assessment)
    assert not assessment.completed_rows.any()
    assert not assessment.successful_rows.any()


def test_reset_dwell_and_later_retry_cannot_repair_first_episode(
    torch, assessment_runtime
):
    task, assessment = assessment_runtime
    for _ in range(2):
        _advance(torch, task, assessment)
    _advance(torch, task, assessment, done=True)
    for _ in range(3):
        _advance(torch, task, assessment)
    _advance(torch, task, assessment, done=True)
    assert assessment.completed_rows.tolist() == [True, True, False, False]
    assert not assessment.successful_rows.any()


def test_new_scenario_needs_its_own_complete_terminal_dwell(torch, assessment_runtime):
    task, assessment = assessment_runtime
    for _ in range(2):
        _advance(torch, task, assessment)
    task.npa_scenario_indices = torch.tensor([2, 3])
    _advance(torch, task, assessment, done=True)
    for _ in range(3):
        _advance(torch, task, assessment)
    task.npa_scenario_indices = torch.tensor([0, 1])
    _advance(torch, task, assessment, done=True)
    assert assessment.completed_rows.all()
    assert assessment.successful_rows.tolist() == [False, False, True, True]


@pytest.mark.parametrize("distance,speed", [(0.05, 0.02), (0.04, 0.03)])
def test_strict_thresholds_remain_exclusive(torch, assessment_runtime, distance, speed):
    task, assessment = assessment_runtime
    task.distance.fill_(distance)
    task.speed.fill_(speed)
    for _ in range(4):
        _advance(torch, task, assessment)
    _advance(torch, task, assessment, done=True)
    assert not assessment.successful_rows.any()


def test_simultaneous_duplicate_scenario_failure_is_not_overwritten(
    torch, assessment_runtime
):
    task, _ = assessment_runtime
    task.npa_scenario_indices.zero_()
    assessment = bootstrap._TerminalPlacementAssessment(task)
    task.speed[1] = 0.04
    for _ in range(3):
        _advance(torch, task, assessment)
    _advance(torch, task, assessment, done=True)
    assert assessment.completed_rows.tolist() == [True, False, False, False]
    assert not assessment.successful_rows.any()


class _CollectionEnvironment:
    def __init__(self, torch, task):
        self.torch, self.unwrapped = torch, task
        task.max_episode_length = 5
        self.steps = 0
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.unwrapped.npa_scenario_indices = self.torch.tensor([0, 1])
        self.unwrapped.speed.fill_(0.04)

    def get_observations(self):
        return {"policy": self.torch.zeros(2, 39)}

    def step(self, actions):
        self.steps += 1
        phase = self.steps % 5
        self.unwrapped.speed.fill_(0.02 if 1 <= phase <= 3 else 0.04)
        if phase == 0:
            offset = 2 * ((self.steps // 5) % 2)
            self.unwrapped.npa_scenario_indices = self.torch.tensor(
                [offset, offset + 1]
            )
        dones = self.torch.full((2,), phase == 0, dtype=self.torch.bool)
        return self.get_observations(), None, dones, {}


def _collection_teacher_type(torch):
    record_events = bootstrap._FrankaTeacher._record_stable_placements

    class Teacher:
        def __init__(self, task):
            self.task = task
            self.phase = torch.zeros(2, dtype=torch.long)
            self.elapsed = torch.zeros(2)
            self.stable_steps = torch.zeros(2, dtype=torch.long)
            self.successful_rows = torch.zeros(4, dtype=torch.bool)

        def actions(self):
            record_events(self)
            return torch.zeros(2, 8)

        def reset(self, dones):
            self.stable_steps[dones] = 0

    return Teacher


class _CollectionActor:
    obs_groups = ["policy"]

    def __init__(self, torch):
        self.torch = torch

    def __call__(self, observations):
        return self.torch.zeros(2, 8)

    def reset(self, dones=None):
        pass

    def eval(self):
        pass


def test_real_collection_returns_terminal_failures_and_keeps_ever_events_diagnostic(
    torch, monkeypatch, assessment_runtime
):
    task, _ = assessment_runtime
    environment = _CollectionEnvironment(torch, task)
    teacher_type = _collection_teacher_type(torch)
    monkeypatch.setattr(bootstrap, "_FrankaTeacher", teacher_type)
    observations, labels, success, diagnostics = bootstrap._training_round(
        environment, _CollectionActor(torch), learner_fraction=1.0
    )
    assert environment.resets == 1 and len(observations) == len(labels) == 30
    assert not success.any()
    assert diagnostics["terminal_training_scenarios"] == 4
    assert diagnostics["ever_training_placements"] == 4


def _batch(torch, values):
    tensor_dict = pytest.importorskip("tensordict").TensorDict
    observations = tensor_dict(
        {
            "policy": torch.tensor(values, dtype=torch.float32).reshape(-1, 1),
            "__done": torch.zeros(len(values), 1, dtype=torch.bool),
            "__phase": torch.zeros(len(values), 1, dtype=torch.long),
        },
        batch_size=[len(values)],
    )
    return observations, torch.zeros(len(values), 2)


def test_partial_collections_stay_separate_without_mutating_archived_resets(torch):
    first, first_labels = _batch(torch, [10, 20, 11, 21, 12, 22])
    second, second_labels = _batch(torch, [110, 120, 111, 121])
    replay = imitation.RecurrentDemonstrationReplay(2)
    replay._retain(first, first_labels)
    combined, labels = replay._retain(second, second_labels)
    assert not first["__done"].any() and not second["__done"].any()
    assert imitation._episode_lengths(combined, 2).tolist() == [3, 2, 3, 2]
    actor = SimpleNamespace(obs_groups=["policy"])
    sequences = imitation._episode_sequences(actor, combined, labels, 2)
    assert sequences["episodes"] == 4
    assert sequences["inputs"][:, :, 0].tolist() == [
        [10, 110, 20, 120],
        [11, 111, 21, 121],
        [12, 0, 22, 0],
    ]


def test_replay_gru_predictions_match_independently_reset_collections(torch):
    torch.manual_seed(7)
    first, first_labels = _batch(torch, [1, 2, 3, 4, 5, 6])
    second, second_labels = _batch(torch, [-1, -2, -3, -4])
    replay = imitation.RecurrentDemonstrationReplay(2)
    replay._retain(first, first_labels)
    combined, labels = replay._retain(second, second_labels)
    actor = SimpleNamespace(
        obs_groups=["policy"],
        rnn=SimpleNamespace(rnn=torch.nn.GRU(1, 4)),
        obs_normalizer=torch.nn.Identity(),
        mlp=torch.nn.Linear(4, 2),
        distribution=SimpleNamespace(deterministic_output=lambda value: value),
    )
    sequences = imitation._episode_sequences(actor, combined, labels, 2)
    actual = imitation._sequence_predictions(actor, sequences["inputs"])
    expected = imitation._sequence_predictions(actor, second["policy"].reshape(2, 2, 1))
    torch.testing.assert_close(actual[:2, [1, 3]], expected)
    joined = imitation._sequence_predictions(actor, combined["policy"].reshape(5, 2, 1))
    assert not torch.allclose(joined[3:], expected)


@pytest.mark.parametrize("sample_count", [0, 3])
def test_replay_refuses_incomplete_vector_steps(torch, sample_count):
    observations, labels = _batch(torch, list(range(sample_count)))
    replay = imitation.RecurrentDemonstrationReplay(2)
    with pytest.raises(RuntimeError, match="aligned samples|complete vector steps"):
        replay._retain(observations, labels)
