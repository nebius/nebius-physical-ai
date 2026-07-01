"""Persistent workflow state for staged Sim2Real execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.utils import _utc_now, _write_json_artifact


WORKFLOW_STATE_SCHEMA = "npa.sim2real.workflow_state.v1"
PROMOTE_DECISION = "promote_checkpoint"


@dataclass
class WorkflowState:
    """Typed view over ``state/workflow_state.json`` for stage hand-offs."""

    run_id: str
    local_artifact_dir: Path
    current_quality: float
    status: str = "initialized"
    stage_records: list[dict[str, Any]] = field(default_factory=list)
    components: list[dict[str, Any]] = field(default_factory=list)
    train_envs_uri: str = ""
    heldout_envs_uri: str = ""
    scene_spec_uri: str = ""
    robot_spec_uri: str = ""
    env_count: int = 0
    train_env_count: int = 0
    heldout_env_count: int = 0
    outer_history: list[dict[str, Any]] = field(default_factory=list)
    final_inner: dict[str, Any] | None = None
    final_eval: dict[str, Any] | None = None
    final_decision: dict[str, Any] | None = None
    next_outer_iteration: int = 1
    # Latest real-policy checkpoint URI produced by the outer loop; the next outer
    # iteration resumes the SAME policy from it (stage 11B "more RL" compounds).
    last_checkpoint_uri: str = ""
    report_path: str | None = None
    updated_at: str = field(default_factory=_utc_now)

    @classmethod
    def path_for(cls, local_dir: Path) -> Path:
        return local_dir / "state" / "workflow_state.json"

    @classmethod
    def load(cls, local_dir: Path) -> WorkflowState:
        path = cls.path_for(local_dir)
        if not path.exists():
            raise FileNotFoundError(f"missing workflow state: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_payload(local_dir, payload)

    @classmethod
    def from_payload(cls, local_dir: Path, payload: dict[str, Any]) -> WorkflowState:
        artifact_dir = Path(payload.get("local_artifact_dir") or local_dir)
        return cls(
            run_id=str(payload["run_id"]),
            local_artifact_dir=artifact_dir,
            current_quality=float(payload.get("current_quality", 0.0)),
            status=str(payload.get("status", "unknown")),
            stage_records=list(payload.get("stage_records") or []),
            components=list(payload.get("components") or []),
            train_envs_uri=str(payload.get("train_envs_uri") or ""),
            heldout_envs_uri=str(payload.get("heldout_envs_uri") or ""),
            scene_spec_uri=str(payload.get("scene_spec_uri") or ""),
            robot_spec_uri=str(payload.get("robot_spec_uri") or ""),
            env_count=int(payload.get("env_count") or 0),
            train_env_count=int(payload.get("train_env_count") or 0),
            heldout_env_count=int(payload.get("heldout_env_count") or 0),
            outer_history=list(payload.get("outer_history") or []),
            final_inner=payload.get("final_inner"),
            final_eval=payload.get("final_eval"),
            final_decision=payload.get("final_decision"),
            next_outer_iteration=int(payload.get("next_outer_iteration") or 1),
            last_checkpoint_uri=str(payload.get("last_checkpoint_uri") or ""),
            report_path=payload.get("report_path"),
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": WORKFLOW_STATE_SCHEMA,
            "run_id": self.run_id,
            "status": self.status,
            "local_artifact_dir": str(self.local_artifact_dir),
            "stage_records": self.stage_records,
            "components": self.components,
            "train_envs_uri": self.train_envs_uri,
            "heldout_envs_uri": self.heldout_envs_uri,
            "scene_spec_uri": self.scene_spec_uri,
            "robot_spec_uri": self.robot_spec_uri,
            "env_count": self.env_count,
            "train_env_count": self.train_env_count,
            "heldout_env_count": self.heldout_env_count,
            "outer_history": self.outer_history,
            "final_inner": self.final_inner,
            "final_eval": self.final_eval,
            "final_decision": self.final_decision,
            "current_quality": self.current_quality,
            "next_outer_iteration": self.next_outer_iteration,
            "last_checkpoint_uri": self.last_checkpoint_uri,
            "report_path": self.report_path,
            "updated_at": self.updated_at,
        }

    def save(self) -> dict[str, Any]:
        self.updated_at = _utc_now()
        record = _write_json_artifact(self.path_for(self.local_artifact_dir), self.to_payload())
        return record["payload"]

    @property
    def decision(self) -> str:
        if not self.final_decision:
            return ""
        return str(self.final_decision.get("decision") or "")

    def should_promote(self) -> bool:
        return self.decision == PROMOTE_DECISION
