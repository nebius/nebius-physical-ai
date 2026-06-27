"""Sim2Real workflow orchestration — the single Python control plane."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from npa.workflows.sim2real.config import build_config_from_env
from npa.workflows.sim2real.models import Sim2RealLoopConfig, Sim2RealLoopError
from npa.workflows.sim2real.state import WorkflowState


class Sim2RealWorkflow:
    """Run the 13-stage VLM→RL loop with explicit stage boundaries.

    This is the canonical orchestrator. SkyPilot runbooks, SDK helpers, and
    legacy CLIs should delegate here instead of re-implementing bash loops or
    ad-hoc state polling.
    """

    def __init__(self, config: Sim2RealLoopConfig) -> None:
        config.validate()
        self.config = config
        self._local_dir = self._resolve_output_dir(config)

    @classmethod
    def from_env(cls, **overrides: Any) -> Sim2RealWorkflow:
        return cls(build_config_from_env(**overrides))

    @property
    def local_dir(self) -> Path:
        return self._local_dir

    def run_preamble(self) -> WorkflowState:
        from npa.workflows.sim2real.engine import run_preamble

        payload = run_preamble(self.config)
        return WorkflowState.from_payload(self._local_dir, payload)

    def run_outer_iteration(
        self,
        *,
        outer_iteration: int,
        initial_quality: float | None = None,
    ) -> WorkflowState:
        from npa.workflows.sim2real.engine import run_single_outer_iteration

        state = WorkflowState.load(self._local_dir)
        quality = (
            float(initial_quality)
            if initial_quality is not None
            else state.current_quality
        )
        iteration = run_single_outer_iteration(
            self.config,
            local_dir=self._local_dir,
            outer_iteration=outer_iteration,
            initial_quality=quality,
            # Resume the same policy from the prior outer iteration's checkpoint so
            # "send back for more RL" (stage 11B) compounds instead of restarting.
            resume_checkpoint_uri=state.last_checkpoint_uri,
        )
        state.final_inner = iteration["inner"]
        state.final_eval = iteration["heldout_report"]
        state.final_decision = iteration["decision"]
        state.outer_history.append(iteration["history_entry"])
        state.current_quality = float(iteration["next_quality"])
        produced_checkpoint = str(iteration.get("checkpoint_uri") or "").strip()
        if produced_checkpoint:
            state.last_checkpoint_uri = produced_checkpoint
        state.next_outer_iteration = outer_iteration + 1
        state.status = "outer_iteration_completed"
        state.save()
        from npa.workflows.sim2real.engine import sync_workflow_state_to_s3

        sync_workflow_state_to_s3(self.config, self._local_dir)
        return state

    def run_finalize(self, *, upload: bool | None = None) -> dict[str, Any]:
        from npa.workflows.sim2real.engine import run_finalize

        state = WorkflowState.load(self._local_dir)
        if not state.final_decision or not state.final_inner or not state.final_eval:
            raise Sim2RealLoopError(
                "finalize requires at least one completed outer iteration"
            )
        report = run_finalize(
            self.config,
            local_dir=self._local_dir,
            stage_records=list(state.stage_records),
            components=list(state.components),
            outer_history=list(state.outer_history),
            final_inner=dict(state.final_inner),
            final_eval=dict(state.final_eval),
            final_decision=dict(state.final_decision),
            upload=upload,
        )
        state.status = "completed"
        state.report_path = str(self._local_dir / "reports" / "sim2real-report.json")
        state.save()
        from npa.workflows.sim2real.engine import (
            _read_workflow_state,
            _write_workflow_state,
            sync_workflow_state_to_s3,
        )

        finalize_state = _read_workflow_state(self._local_dir)
        finalize_state["status"] = "completed"
        finalize_state["report_path"] = state.report_path
        _write_workflow_state(self._local_dir, finalize_state, config=self.config)
        sync_workflow_state_to_s3(self.config, self._local_dir)
        return report

    def run(self, *, upload: bool | None = None) -> dict[str, Any]:
        """Execute preamble → outer loop → finalize in one process."""

        from npa.workflows.sim2real.engine import _config_from_workflow_state

        state = self.run_preamble()
        self.config = _config_from_workflow_state(self.config, state.to_payload())
        for outer_iteration in range(1, self.config.outer_iterations + 1):
            state = self.run_outer_iteration(outer_iteration=outer_iteration)
            if state.should_promote():
                break
        return self.run_finalize(upload=upload)

    def run_staged(
        self,
        *,
        upload: bool | None = None,
        initial_quality: float | None = None,
    ) -> dict[str, Any]:
        """Resume or execute the full staged path from persisted state.

        If ``workflow_state.json`` is missing, runs preamble first. Then runs
        remaining outer iterations and finalize. Intended for runbook parity
        without bash control flow.
        """

        state_path = WorkflowState.path_for(self._local_dir)
        if not state_path.exists():
            state = self.run_preamble()
            from npa.workflows.sim2real.engine import _config_from_workflow_state

            self.config = _config_from_workflow_state(self.config, state.to_payload())
        else:
            state = WorkflowState.load(self._local_dir)

        if initial_quality is not None:
            state.current_quality = float(initial_quality)
            state.save()

        start = state.next_outer_iteration
        for outer_iteration in range(start, self.config.outer_iterations + 1):
            state = self.run_outer_iteration(outer_iteration=outer_iteration)
            if state.should_promote():
                break

        if state.status != "completed":
            return self.run_finalize(upload=upload)
        from npa.workflows.sim2real.engine import run_finalize

        return run_finalize(
            self.config,
            local_dir=self._local_dir,
            stage_records=list(state.stage_records),
            components=list(state.components),
            outer_history=list(state.outer_history),
            final_inner=dict(state.final_inner or {}),
            final_eval=dict(state.final_eval or {}),
            final_decision=dict(state.final_decision or {}),
            upload=upload,
        )

    @staticmethod
    def _resolve_output_dir(config: Sim2RealLoopConfig) -> Path:
        if config.output_dir is not None:
            path = Path(config.output_dir)
        else:
            import tempfile

            path = Path(tempfile.mkdtemp(prefix=f"npa-{config.run_id}-"))
        path.mkdir(parents=True, exist_ok=True)
        return path


def run_full_loop(config: Sim2RealLoopConfig, *, upload: bool | None = None) -> dict[str, Any]:
    """Backward-compatible entrypoint used by SDK and tests."""

    return Sim2RealWorkflow(config).run(upload=upload)
