"""Apply NPA's context-bound evidence fixes to pinned Isaac Lab-Arena.

The upstream replay policy loads a recorded initial state but its standalone
policy runner never applies that state.  The same runner leaves Arena's metric
recorder in ``/tmp`` and starts viewport recording through an episode reset,
which makes an exact-state replay awkward to capture.  NPA fixes those three
integration gaps while keeping the real upstream environment, policy, task,
metric, and renderer implementations in authority. The viewport is captured by
an Isaac Lab post-step recorder before automatic termination reset; Gym consumes
that exact frame instead of rendering the next episode's reset state. Initial
and terminal PNGs retain the same real renderer output and action-step binding.
Scalar phase journals identify policy, controller, environment, and capture
progress without recording their arguments or changing solver decisions.

The final patch still separates viewport rendering from embodiment observation
cameras.  Each replacement is anchored to the exact 0.3.0 source context so a
future upstream change fails the image build instead of silently drifting.
"""

from __future__ import annotations

import argparse
from pathlib import Path


POLICY_RUNNER_RESET = """\
    try:
        obs, _ = env.reset()
        policy.reset()
"""

POLICY_RUNNER_RESET_PATCHED = """\
    try:
        # NPA integration: apply the replay's actual recorded simulator state.
        replay_initial_state = getattr(policy, "get_initial_state", None)
        if callable(replay_initial_state):
            initial_state = replay_initial_state()
            if initial_state is None:
                raise RuntimeError("replay policy did not provide an initial state")
            obs, _ = env.unwrapped.reset_to(initial_state, None, is_relative=True)
            print("Applied replay initial state with Isaac Lab reset_to(is_relative=True)")
        else:
            obs, _ = env.reset()
        policy.reset()
"""

POLICY_RUNNER_ENVIRONMENT = """\
        output_dir = timestamped_run_dir(args_cli.output_base_dir)
        video_cfg = VideoRecordingCfg(
            record_viewport_video=args_cli.record_viewport_video,
            record_camera_video=args_cli.record_camera_video,
            video_base_dir=output_dir,
        )
        env = arena_builder.make_registered(render_mode=video_cfg.render_mode)
"""

POLICY_RUNNER_ENVIRONMENT_PATCHED = """\
        output_dir = timestamped_run_dir(args_cli.output_base_dir)
        os.makedirs(output_dir, exist_ok=True)
        video_cfg = VideoRecordingCfg(
            record_viewport_video=args_cli.record_viewport_video,
            record_camera_video=args_cli.record_camera_video,
            video_base_dir=output_dir,
        )
        env_cfg, env_kwargs = arena_builder.compose_manager_cfg()
        if env_cfg.recorders is not None:
            env_cfg.recorders.dataset_export_dir_path = output_dir
            env_cfg.recorders.dataset_filename = f"simulator_ground_truth_rank{local_rank}"
        if args_cli.record_viewport_video:
            from npa.workbench.isaac_arena.simulator_video import configure_video_capture
            configure_video_capture(env_cfg)
        # Replay uses the simulator's reset_to API. Gym's OrderEnforcing wrapper
        # only recognizes reset(), so use the base env before adding RecordVideo.
        env = arena_builder.make_registered(env_cfg, env_kwargs, render_mode=video_cfg.render_mode).unwrapped
        from npa.workbench.isaac_arena.simulator_phases import configure_phase_journal
        configure_phase_journal(env, output_dir, local_rank)
"""

POLICY_RUNNER_STEP = """\
                actions = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(actions)
"""

POLICY_RUNNER_STEP_PATCHED = """\
                from npa.workbench.isaac_arena.simulator_phases import phase_scope
                with phase_scope(env, "policy_action", num_steps_completed + 1):
                    actions = policy.get_action(env, obs)
                with phase_scope(env, "env_step", num_steps_completed + 1):
                    obs, _, terminated, truncated, _ = env.step(actions)
                from npa.workbench.isaac_arena.action_evidence import record_executed_action
                record_executed_action(env, actions, num_steps_completed + 1)
"""

POLICY_RUNNER_POLICY = """\
        policy = build_policy_from_cli(policy_cls, args_cli)

        # Simulation length.
"""

POLICY_RUNNER_POLICY_PATCHED = """\
        policy = build_policy_from_cli(policy_cls, args_cli)
        from npa.workbench.isaac_arena.action_evidence import configure_action_evidence
        configure_action_evidence(env, output_dir, args_cli.policy_type, local_rank)

        # Simulation length.
"""

POLICY_RUNNER_LENGTH = """\
        # Simulation length.
        if policy.has_length():
            num_steps = policy.length()
            num_episodes = None
        else:
            if args_cli.num_steps is not None:
                num_steps = args_cli.num_steps
                num_episodes = None
                print(f"[Rank {local_rank}/{world_size}] Simulation length: {num_steps} steps")
            elif args_cli.num_episodes is not None:
                num_steps = None
                num_episodes = args_cli.num_episodes
                print(f"[Rank {local_rank}/{world_size}] Simulation length: {num_episodes} episodes")
            else:
                raise ValueError(f"[Rank {local_rank}/{world_size}] Either num_steps or num_episodes must be provided")
"""

POLICY_RUNNER_LENGTH_PATCHED = """\
        # Video evidence is episode-bound even when a replay policy advertises
        # a longer source-action horizon. Stop at the requested native terminal
        # rather than auto-resetting and recording actions from another episode.
        if video_cfg.enabled and args_cli.num_episodes is not None:
            num_steps = None
            num_episodes = args_cli.num_episodes
            print(f"[Rank {local_rank}/{world_size}] Video simulation length: {num_episodes} episodes")
        elif policy.has_length():
            num_steps = policy.length()
            num_episodes = None
        else:
            if args_cli.num_steps is not None:
                num_steps = args_cli.num_steps
                num_episodes = None
                print(f"[Rank {local_rank}/{world_size}] Simulation length: {num_steps} steps")
            elif args_cli.num_episodes is not None:
                num_steps = None
                num_episodes = args_cli.num_episodes
                print(f"[Rank {local_rank}/{world_size}] Simulation length: {num_episodes} episodes")
            else:
                raise ValueError(f"[Rank {local_rank}/{world_size}] Either num_steps or num_episodes must be provided")
"""

_SIMULATION_LENGTH_MARKER = "        # Simulation length.\n"
POLICY_RUNNER_POLICY_AND_LENGTH = (
    POLICY_RUNNER_POLICY.removesuffix(_SIMULATION_LENGTH_MARKER) + POLICY_RUNNER_LENGTH
)
POLICY_RUNNER_POLICY_AND_LENGTH_PATCHED = (
    POLICY_RUNNER_POLICY_PATCHED.removesuffix(_SIMULATION_LENGTH_MARKER)
    + POLICY_RUNNER_LENGTH_PATCHED
)

POLICY_RUNNER_CAMERA_CONTEXT = """\
        # Re-apply enable_cameras: the full parse resets it to default False.
        if args_cli.record_camera_video or args_cli.record_viewport_video:
            args_cli.enable_cameras = True
"""

POLICY_RUNNER_ROLLOUT_END = """\
        if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
            return env.unwrapped.compute_metrics()
        return None


def list_variations(args_parser: argparse.ArgumentParser) -> None:
"""

POLICY_RUNNER_ROLLOUT_END_PATCHED = """\
        if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
            return env.unwrapped.compute_metrics()
        return None
    finally:
        # Retain unfinished measurements before env.close stops physics and
        # deletes the scene/recorder managers. This never scores an episode.
        from npa.workbench.isaac_arena.simulator_video import finalize_video_capture
        from npa.workbench.isaac_arena.action_evidence import finalize_action_evidence
        from npa.workbench.isaac_arena.simulator_phases import finalize_phase_journal
        try:
            finalize_video_capture(env)
        finally:
            try:
                finalize_action_evidence(env)
            finally:
                finalize_phase_journal(env)


def list_variations(args_parser: argparse.ArgumentParser) -> None:
"""

VIDEO_TRIGGER = """\
        env = RecordVideo(
            env,
            video_folder=video_cfg.video_base_dir,
            episode_trigger=lambda episode: episode == 0,
            video_length=video_length,
            name_prefix=video_cfg.viewport_name_prefix,
            disable_logger=True,
        )
"""

VIDEO_TRIGGER_PATCHED = """\
        # Start after the first action rather than forcing an episode reset.  A
        # replay can therefore apply its recorded initial state directly while
        # retaining every action-bearing frame in the evidence interval.
        # The simulator captures before automatic reset; Gym consumes that frame.
        from npa.workbench.isaac_arena.simulator_video import cached_frame_env
        env = RecordVideo(
            cached_frame_env(env),
            video_folder=video_cfg.video_base_dir,
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            name_prefix=video_cfg.viewport_name_prefix,
            disable_logger=True,
        )
"""

REVOLUTE_POST_STEP = """\
    def record_post_step(self):
        openness = self.object.get_openness(self._env)
        return self.name, openness
"""

REVOLUTE_POST_STEP_PATCHED = """\
    def record_post_reset(self, env_ids):
        # NPA integration: bind the trace to simulator state before the first action.
        openness = self.object.get_openness(self._env)
        if env_ids is not None:
            openness = openness[env_ids]
        return self.name, openness

    def record_post_step(self):
        openness = self.object.get_openness(self._env)
        return self.name, openness
"""

EMBODIMENT_IMPORT = """\
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
"""

EMBODIMENT_IMPORT_PATCHED = """\
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Any
"""

EMBODIMENT_ASSIGNMENT = """\
        self.enable_cameras = enable_cameras
"""

EMBODIMENT_ASSIGNMENT_PATCHED = """\
        # NPA's viewport recorder still needs the upstream render flag, but it
        # does not consume robot-mounted sensor observations.  Mask only those
        # sensors so ``env.render()`` continues to advance the Kit viewport.
        viewport_only = os.environ.get("NPA_ISAAC_ARENA_VIEWPORT_ONLY") == "1"
        self.enable_cameras = enable_cameras and not viewport_only
"""


def _replace_once(path: Path, source: str, replacement: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(source) != 1:
        raise RuntimeError(f"pinned Arena {label} patch context changed")
    path.write_text(text.replace(source, replacement), encoding="utf-8")


def _patch_runner(policy_runner: Path) -> None:
    _replace_once(
        policy_runner,
        POLICY_RUNNER_RESET,
        POLICY_RUNNER_RESET_PATCHED,
        "replay-state",
    )
    _replace_once(
        policy_runner,
        POLICY_RUNNER_ENVIRONMENT,
        POLICY_RUNNER_ENVIRONMENT_PATCHED,
        "ground-truth-output",
    )
    _replace_once(
        policy_runner,
        POLICY_RUNNER_ROLLOUT_END,
        POLICY_RUNNER_ROLLOUT_END_PATCHED,
        "live-diagnostic-finalization",
    )
    _replace_once(
        policy_runner,
        POLICY_RUNNER_STEP,
        POLICY_RUNNER_STEP_PATCHED,
        "scalar-phase-diagnostics",
    )
    _replace_once(
        policy_runner,
        POLICY_RUNNER_POLICY_AND_LENGTH,
        POLICY_RUNNER_POLICY_AND_LENGTH_PATCHED,
        "action-evidence-and-episode-bound-video",
    )
    policy_text = policy_runner.read_text(encoding="utf-8")
    if policy_text.count(POLICY_RUNNER_CAMERA_CONTEXT) != 1:
        raise RuntimeError("pinned Arena viewport-camera patch context changed")


def _patch_recorders(video_recording: Path, revolute_metric: Path) -> None:
    _replace_once(
        video_recording,
        VIDEO_TRIGGER,
        VIDEO_TRIGGER_PATCHED,
        "video-trigger",
    )
    _replace_once(
        revolute_metric,
        REVOLUTE_POST_STEP,
        REVOLUTE_POST_STEP_PATCHED,
        "revolute-ground-truth",
    )


def _patch_embodiment(embodiment_base: Path) -> None:
    _replace_once(
        embodiment_base,
        EMBODIMENT_IMPORT,
        EMBODIMENT_IMPORT_PATCHED,
        "embodiment-import",
    )
    _replace_once(
        embodiment_base,
        EMBODIMENT_ASSIGNMENT,
        EMBODIMENT_ASSIGNMENT_PATCHED,
        "embodiment-camera",
    )


def patch_sources(
    policy_runner: Path,
    embodiment_base: Path,
    video_recording: Path,
    revolute_metric: Path,
) -> None:
    """Apply the complete exact-source integration patch.

    Args:
        policy_runner: Pinned upstream evaluation entrypoint.
        embodiment_base: Pinned upstream camera-bearing embodiment class.
        video_recording: Pinned upstream viewport wrapper factory.
        revolute_metric: Pinned upstream door-openness recorder.

    Returns:
        None.

    Raises:
        RuntimeError: Any expected source context changed.
    """

    _patch_runner(policy_runner)
    _patch_recorders(video_recording, revolute_metric)
    _patch_embodiment(embodiment_base)


def main() -> None:
    """Patch the four pinned upstream source files selected by the build.

    Args:
        None.

    Returns:
        None.

    Raises:
        RuntimeError: A source does not match the pinned patch context.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_runner", type=Path)
    parser.add_argument("embodiment_base", type=Path)
    parser.add_argument("video_recording", type=Path)
    parser.add_argument("revolute_metric", type=Path)
    args = parser.parse_args()
    patch_sources(
        args.policy_runner,
        args.embodiment_base,
        args.video_recording,
        args.revolute_metric,
    )


if __name__ == "__main__":
    main()
