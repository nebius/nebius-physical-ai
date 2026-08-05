"""Hermetic PAIDF orchestration contract across restart boundaries."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from npa.clients.config import persist_workflow_src_s3_uri, write_config
from npa.orchestration.npa_workflow.artifact_load import load_final_artifact_into_agent
from npa.orchestration.npa_workflow.run_state import RunManifest, build_actionable_run_status
from npa.orchestration.npa_workflow.src_staging import stage_npa_source
from npa.orchestration.npa_workflow.submission_state import (
    load_submission_state,
    update_submission_state,
)
from npa.orchestration.skypilot.k8s_gpu_catalog import (
    KubernetesGpuCatalog,
    wait_for_kubernetes_accelerators,
)


class HermeticS3:
    def __init__(self) -> None:
        self.s3 = self
        self.objects: dict[tuple[str, str], bytes] = {}
        self.upload_count = 0

    def upload_file(self, local_file: str, uri: str) -> str:
        bucket, key = uri.removeprefix("s3://").split("/", 1)
        self.objects[(bucket, key)] = Path(local_file).read_bytes()
        self.upload_count += 1
        return uri

    def get_object(self, *, Bucket: str, Key: str):  # noqa: ANN201, N803
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"Body": BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, *, Bucket: str, Key: str):  # noqa: ANN201, N803
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def get_paginator(self, _name: str):  # noqa: ANN201
        return self

    def paginate(self, *, Bucket: str, Prefix: str):  # noqa: ANN201, N803
        return [
            {
                "Contents": [
                    {"Key": key}
                    for bucket, key in self.objects
                    if bucket == Bucket and key.startswith(Prefix)
                ]
            }
        ]


class Response:
    def __init__(self, status: int, payload: dict) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _source_tree(root: Path) -> Path:
    (root / "src/npa").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='npa'\n", encoding="utf-8")
    (root / "src/npa/__init__.py").write_text("", encoding="utf-8")
    return root


def _paidf_manifest(status: str = "running") -> RunManifest:
    names = ["annotate", "augment", "evaluate", "relabel", "curator", "gate", "publish"]
    steps = [{"state": name, "status": "SUCCEEDED"} for name in names]
    steps.extend(
        [
            {"state": "fiftyone-curate", "status": "SUBMITTED"},
            {"state": "finalize", "status": "SUBMITTED"},
        ]
    )
    return RunManifest(
        workflow="physical-ai-data-factory",
        run_id="paidf-hermetic",
        api_version="npa.workflow/v0.0.1",
        status=status,
        sky_job_id="77",
        run_prefix_uri="s3://unit/physical-ai-data-factory/paidf-hermetic",
        steps=steps,
        updated_at="2026-08-05T11:00:00Z",
    )


def test_configure_stage_restart_gpu_delay_and_startup_failure(
    tmp_path: Path, monkeypatch
) -> None:
    write_config(
        {
            "default_project": "demo",
            "projects": {
                "demo": {
                    "project_id": "project-placeholder",
                    "region": "eu-north1",
                    "agents": {},
                }
            },
        }
    )
    s3 = HermeticS3()
    source = _source_tree(tmp_path / "source")
    uri = stage_npa_source(bucket="unit", source_root=source, client=s3, max_workers=1)
    persist_workflow_src_s3_uri(uri, "demo")
    update_submission_state(
        "demo",
        "paidf-hermetic",
        {"source": {"uri": uri, "status": "verified"}},
    )
    uploads_before_restart = s3.upload_count

    # Process restarts after staging: the commit manifest prevents every source
    # object from being uploaded again.
    assert stage_npa_source(
        bucket="unit", source_root=source, client=s3, max_workers=1
    ) == uri
    assert s3.upload_count == uploads_before_restart

    catalogs = iter(
        [
            KubernetesGpuCatalog(quantities_by_accelerator={}),
            KubernetesGpuCatalog(
                quantities_by_accelerator={"RTXPRO-6000": frozenset({1})}
            ),
        ]
    )
    clock = iter([0.0, 0.0, 1.0])
    readiness = wait_for_kubernetes_accelerators(
        ["RTXPRO6000:1"],
        context="hermetic",
        timeout=10,
        poll_interval=1,
        discover=lambda: next(catalogs),
        allocatable=lambda: 1,
        monotonic=lambda: next(clock),
        sleeper=lambda _seconds: None,
    )
    assert readiness["RTXPRO6000:1"].resolved == "RTXPRO-6000:1"

    controller = "\n".join(
        [
            'container not found ("ray-node")',
            "cannot exec in a deleted state",
            'container not found ("ray-node")',
        ]
    )
    status = build_actionable_run_status(
        _paidf_manifest(),
        live_status="RUNNING",
        task_rows=[{"task_id": 7, "status": "PENDING", "retry_count": 2}],
        controller_output=controller,
        project="demo",
        now=datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc),
    )
    assert status["status"] == "FAILED_STARTUP"
    assert status["active_stage_index"] == 8
    assert status["stages"]["fiftyone-curate"]["scheduler_state"] == "PENDING"
    assert "workflow logs paidf-hermetic --stage fiftyone-curate" in (
        status["stages"]["fiftyone-curate"]["log_command"]
    )
    ledger = str(load_submission_state("demo", "paidf-hermetic"))
    assert "credential" not in ledger.lower()
    assert "secret" not in ledger.lower()


def test_success_restart_finishes_only_missing_agent_load(tmp_path: Path, monkeypatch) -> None:
    import npa.cli.agent as agent

    secret = tmp_path / "agent.env"
    secret.write_text("AGENT_USER=user\nAGENT_PASSWORD=never-serialize\n", encoding="utf-8")
    monkeypatch.setattr(
        agent,
        "resolve_project_agents",
        lambda _project: {"agent": {"agent_url": "https://agent.invalid"}},
    )
    monkeypatch.setattr(
        agent,
        "_agent_record",
        lambda _project, _name: {
            "agent_url": "https://agent.invalid",
            "auth_secret_path": str(secret),
        },
    )
    monkeypatch.setattr(agent, "_record_tls_verify", lambda _record: True)
    s3 = HermeticS3()
    key = "physical-ai-data-factory/paidf-hermetic/reports/sim2real.rrd"
    s3.objects[("unit", key)] = b"rrd"
    update_submission_state(
        "demo",
        "paidf-hermetic",
        {
            "launch": {"status": "completed", "sky_job_id": "77"},
            "workflow": {"status": "succeeded"},
        },
    )
    artifact_uri = f"s3://unit/{key}"
    methods: list[str] = []

    def request(method: str, _url: str, **_kwargs):  # noqa: ANN202
        methods.append(method)
        if method == "POST":
            return Response(200, {"ok": True})
        if len(methods) == 1:
            return Response(200, {"rerun_ready": False})
        return Response(
            200,
            {
                "artifact_uri": artifact_uri,
                "artifact_render": "rerun",
                "rerun_ready": True,
            },
        )

    result = load_final_artifact_into_agent(
        project="demo",
        run_id="paidf-hermetic",
        run_prefix_uri="s3://unit/physical-ai-data-factory/paidf-hermetic",
        storage_client=s3,
        http_request=request,
    )
    resumed = load_final_artifact_into_agent(
        project="demo",
        run_id="paidf-hermetic",
        run_prefix_uri="s3://unit/physical-ai-data-factory/paidf-hermetic",
        storage_client=s3,
        http_request=request,
    )

    assert result.verified and resumed.verified
    assert methods == ["GET", "POST", "GET", "GET"]
    state = load_submission_state("demo", "paidf-hermetic")
    assert state["launch"]["sky_job_id"] == "77"  # no duplicate launch
    assert state["artifact_load"]["artifact_uri"] == artifact_uri
    assert state["artifact_load"]["verified"] is True
    assert "never-serialize" not in str(state)
