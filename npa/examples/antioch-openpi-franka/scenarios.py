"""Antioch wrapper around the shared NPA Isaac/OpenPI bridge implementation."""

from __future__ import annotations

import antioch

from reverse_policy_relay import ReversePolicyRelay


@antioch.scenario(tags=["npa-openpi-franka"])
def openpi_franka_camera_bridge(run: antioch.ScenarioRun) -> None:
    """Sustain camera-to-policy-to-position control while Antioch owns Kit."""

    from npa.workbench.antioch.openpi_isaac import run as run_bridge

    # Antioch's authenticated port tunnel is local -> assigned machine.  The
    # reverse relay lets a local connector carry the private Kubernetes policy
    # stream back through that tunnel without exposing the policy or copying a
    # Kubernetes credential into the hosted simulator.
    with ReversePolicyRelay(backend_port=18123, frontend_port=8000):
        report = run_bridge(launch_application=False)
    metrics = report["streaming_metrics"]
    run.add_result("policy_action_shape", report["policy_action_shape"])
    run.add_result("targets_executed", report["targets_executed"])
    run.add_result("observation_fps", metrics["observation_fps"])
    run.add_result("control_step_fps", metrics["control_step_fps"])
    run.add_result("inference_latency_ms_p95", metrics["inference_latency_ms_p95"])
    run.add_result("policy_round_trips", metrics["policy_round_trips"])
    run.add_result("camera_frame_sequence", metrics["latest_observation_sequence"])
    run.check(
        "OpenPI continuously returned exact finite 15x8 target chunks",
        report["policy_action_shape"] == [15, 8],
    )
    run.check(
        "camera frames, policy round trips, and safe targets advanced sustainably",
        metrics["ready"] is True
        and int(metrics["latest_observation_sequence"]) >= 3
        and int(metrics["policy_round_trips"]) >= 3
        and int(metrics["safely_applied_targets"]) >= 3,
    )
    run.check(
        "the soft-real-time boundary remained fail-closed without a hard-RT claim",
        report["fail_closed"] is True
        and report["soft_real_time"] is True
        and report["hard_real_time"] is False,
    )
