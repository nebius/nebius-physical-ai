"""Check navigation stage provenance, separation, and complete paired episodes."""

from __future__ import annotations

from npa.workflows.field_failure.artifacts import _read, _verify
from npa.workflows.field_failure.contracts import (
    _EpisodeEvidence,
    _Evaluation,
    _Reconstruction,
    _Training,
)
from npa.workflows.field_failure.durability import _check_claim


def _check_bundle(bundle):
    for artifact in [bundle.baseline.checkpoint, bundle.protocol]:
        _verify(artifact)
    for scenario in [*bundle.captures, *bundle.held_out]:
        _verify(scenario.asset)


def _check_reconstruction(payload, bundle, bundle_sha, root):
    record = _Reconstruction.model_validate(payload)
    if record.bundle_sha256 != bundle_sha:
        raise ValueError("reconstruction bundle provenance mismatch")
    prefix = _check_claim(
        "reconstruction", record, bundle_sha, bundle.adapters["reconstruct"], {}, root
    )
    expected = {(c.scenario_id, c.group_id, c.asset.sha256) for c in bundle.captures}
    actual = {(s.scenario_id, s.group_id, s.capture_sha256) for s in record.scenes}
    if actual != expected or len(record.scenes) != len(expected):
        raise ValueError("reconstruction must cover each failure capture exactly once")
    held_hashes = {s.asset.sha256 for s in bundle.held_out}
    if any(s.asset.sha256 in held_hashes for s in record.scenes):
        raise ValueError("reconstructed training scene overlaps held-out evidence")
    for scene in record.scenes:
        _verify(scene.asset, prefix)
    _verify(record.evidence, prefix)
    return record


def _check_training(
    payload, bundle, bundle_sha, reconstruction, reconstruction_sha, root
):
    record = _Training.model_validate(payload)
    if (
        record.bundle_sha256 != bundle_sha
        or record.reconstruction_sha256 != reconstruction_sha
    ):
        raise ValueError("training input provenance mismatch")
    prefix = _check_claim(
        "training",
        record,
        bundle_sha,
        bundle.adapters["train"],
        {"reconstruction": reconstruction_sha},
        root,
    )
    if record.initial_checkpoint_sha256 != bundle.baseline.checkpoint.sha256:
        raise ValueError("candidate was not fine-tuned from the existing baseline")
    if (
        record.candidate.checkpoint.sha256 == bundle.baseline.checkpoint.sha256
        or record.candidate.policy_id == bundle.baseline.policy_id
    ):
        raise ValueError("candidate must identify a changed checkpoint and policy")
    expected = [
        (record.training_scenario_ids, {s.scenario_id for s in reconstruction.scenes}),
        (record.training_group_ids, {s.group_id for s in reconstruction.scenes}),
        (record.training_scene_sha256, {s.asset.sha256 for s in reconstruction.scenes}),
    ]
    for actual, required in expected:
        if set(actual) != required or len(actual) != len(required):
            raise ValueError("training membership must exactly match reconstruction")
    _verify(record.candidate.checkpoint, prefix)
    _verify(record.evidence, prefix)
    return record


def _check_evaluation(payload, bundle, bundle_sha, policy, root, arm, inputs):
    record = _Evaluation.model_validate(payload)
    if (
        record.bundle_sha256 != bundle_sha
        or record.protocol_sha256 != bundle.protocol.sha256
    ):
        raise ValueError("evaluation bundle/protocol provenance mismatch")
    if record.adapter != bundle.adapters["evaluate"] or record.policy != policy:
        raise ValueError("evaluation adapter/checkpoint provenance mismatch")
    prefix = _check_claim(
        f"{arm}-evaluation",
        record,
        bundle_sha,
        bundle.adapters["evaluate"],
        inputs,
        root,
    )
    expected = {
        (s.scenario_id, seed): s.asset.sha256
        for s in bundle.held_out
        for seed in s.seeds
    }
    actual = {(e.scenario_id, e.seed): e.scene_sha256 for e in record.episodes}
    if actual != expected or len(record.episodes) != len(expected):
        raise ValueError(
            "evaluation must complete every held-out scenario/seed exactly once"
        )
    evidence = [e.evidence.sha256 for e in record.episodes]
    if len(evidence) != len(set(evidence)):
        raise ValueError("episodes must carry distinct rollout evidence")
    _verify_episodes(record, bundle.metrics, prefix)
    return record


def _verify_episodes(record, metrics, prefix):
    trajectories = set()
    for episode in record.episodes:
        _check_metrics(episode, metrics)
        _verify(episode.evidence, prefix)
        digest = _check_episode_evidence(episode, record, prefix)
        if digest in trajectories:
            raise ValueError("underlying trajectory bytes were reused across episodes")
        trajectories.add(digest)


def _check_episode_evidence(episode, evaluation, prefix):
    payload = _read(episode.evidence.uri, episode.evidence.sha256)[0]
    trace = _EpisodeEvidence.model_validate(payload)
    if (
        trace.bundle_sha256 != evaluation.bundle_sha256
        or trace.checkpoint_sha256 != evaluation.policy.checkpoint.sha256
        or trace.protocol_sha256 != evaluation.protocol_sha256
    ):
        raise ValueError("episode evidence has different policy/protocol provenance")
    measured = trace.model_dump(
        exclude={
            "schema_version",
            "bundle_sha256",
            "checkpoint_sha256",
            "protocol_sha256",
            "trajectory",
        }
    )
    if measured != episode.model_dump(exclude={"evidence"}):
        raise ValueError("evaluation summary differs from episode measurements")
    _verify(trace.trajectory, prefix)
    return trace.trajectory.sha256


def _check_metrics(episode, metrics):
    if set(episode.metrics) != {m.name for m in metrics}:
        raise ValueError("episode metric coverage differs from sealed protocol")
    for metric in metrics:
        value = episode.metrics[metric.name]
        if not metric.minimum <= value <= metric.maximum:
            raise ValueError("episode metric falls outside declared bounds")
