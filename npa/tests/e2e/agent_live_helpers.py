from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from npa.cli.agent import (
    AGENT_UI_VERSION,
    DEFAULT_AGENT_NAME,
    DEFAULT_PROJECT_ALIAS,
    _agent_record,
    _load_auth_secret,
    _record_tls_verify,
)


@dataclass(frozen=True)
class AgentLiveContext:
    project: str
    name: str
    auth_user: str
    auth_password: str
    agent_url: str
    rerun_url: str
    sim_viz_url: str
    sim_assets_url: str
    tls_verify: bool

    @property
    def api_base(self) -> str:
        return self.agent_url.rstrip("/")

    def auth(self) -> tuple[str, str]:
        return (self.auth_user, self.auth_password)

    def get(self, path: str, **kwargs: object) -> httpx.Response:
        url = path if path.startswith("http") else f"{self.api_base}{path}"
        kwargs.setdefault("verify", self.tls_verify)
        kwargs.setdefault("timeout", 10.0)
        return httpx.get(url, auth=self.auth(), **kwargs)

    def post(self, path: str, **kwargs: object) -> httpx.Response:
        url = path if path.startswith("http") else f"{self.api_base}{path}"
        kwargs.setdefault("verify", self.tls_verify)
        kwargs.setdefault("timeout", 30.0)
        return httpx.post(url, auth=self.auth(), **kwargs)

    def put(self, path: str, **kwargs: object) -> httpx.Response:
        url = path if path.startswith("http") else f"{self.api_base}{path}"
        kwargs.setdefault("verify", self.tls_verify)
        kwargs.setdefault("timeout", 10.0)
        return httpx.put(url, auth=self.auth(), **kwargs)


def load_agent_live_context() -> AgentLiveContext:
    project = os.environ.get("NPA_AGENT_PROJECT", DEFAULT_PROJECT_ALIAS)
    name = os.environ.get("NPA_AGENT_NAME", DEFAULT_AGENT_NAME)
    record = _agent_record(project, name)
    if not record:
        raise RuntimeError(f"missing agent config for {project}/{name}")
    auth_user, auth_password = _load_auth_secret(str(record.get("auth_secret_path", "")))
    agent_url = str(record.get("agent_url", ""))
    rerun_url = str(record.get("rerun_url", ""))
    sim_viz_url = str(record.get("sim_viz_url", rerun_url))
    sim_assets_url = str(record.get("sim_assets_url", agent_url))
    return AgentLiveContext(
        project=project,
        name=name,
        auth_user=auth_user,
        auth_password=auth_password,
        agent_url=agent_url,
        rerun_url=rerun_url,
        sim_viz_url=sim_viz_url,
        sim_assets_url=sim_assets_url,
        tls_verify=_record_tls_verify(record),
    )


STOCK_FRANKA_SELECTION = {
    "scene_spec_uri": "stock://scene/default",
    "robot_spec_uri": "stock://robot/franka",
    "cameras_uri": "stock://cameras/default",
    "robot_preset": "franka",
    "sim_backend": "isaac",
    "props": ["cube"],
}

UI_BUTTON_IDS = (
    "chatActionS3",
    "chatActionCosmos",
    "chatActionWatch",
    "chatActionWorkflow",
    "newChatSession",
    "workflowUpload",
    "workflowValidate",
    "workflowPlan",
    "workflowSubmitYaml",
    "loadFrankaRerun",
    "loadRerunViewer",
    "openRerun",
    "applySelection",
    "submitWorkflow",
    "workflowStatus",
    "loadRunData",
    "artifactRefreshRuns",
    "artifactLoadRunArtifacts",
)

UI_WIRING_MARKERS = (
    "function bindClick(",
    "function wireUi(",
    "function showToast(",
    "initNpaAgentUi",
    "DOMContentLoaded",
    f'name="npa-ui-version" content="{AGENT_UI_VERSION}"',
)

RERUN_STATIC_CANDIDATES = (
    "/rerun/index.js",
    "/rerun/re_viewer.js",
    "/rerun/favicon.ico",
    "/rerun/version",
)

ONBOARD_SOLUTION_PROMPT = (
    "add an open source repo, containerize, push to registry, and run a GPU smoke on live infra"
)

ONBOARD_OSS_REPO_PROMPT = (
    "onboard https://github.com/githubtraining/hellogitworld.git on Ubuntu, "
    "build the container, push to registry, and run a deploy smoke on live infra"
)
CREATE_BYOF_WORKFLOW_PROMPT = (
    "create a BYOF Isaac Lab workflow for live infra with placeholder repo and task"
)


def assert_grounded_onboard_solution_reply(payload: dict[str, object]) -> str:
    assert payload.get("ok") is True
    assert payload.get("grounded") is True
    reply = str(payload.get("reply") or "")
    assert reply
    assert "npa workbench byof run" in reply or "run_byof_repo.py" in reply
    assert "--base-profile" in reply or "--base-image" in reply
    assert "byof-onboard" in reply or "skills/workflows/byof-onboard" in reply
    assert "oss-solution-registry-onboard" in reply
    assert "upstream docs" in reply
    assert "live Nebius" in reply
    assert "<repo-url>" in reply
    assert "container-verify" in reply or "<task>" in reply
    assert "registry" in reply.lower()
    assert not reply.strip().startswith("GET /api"), "raw GET path instead of onboarding guidance"
    apis_used = payload.get("apis_used")
    assert isinstance(apis_used, list) and apis_used
    assert "tools" in apis_used
    return reply
