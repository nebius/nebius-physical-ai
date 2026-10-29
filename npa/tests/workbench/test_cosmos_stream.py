"""Hosted adapter contracts with synthetic transports, not live GPU evidence."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from npa.workbench.cosmos.stream_augmentation import StreamAugmentation
from npa.workbench.cosmos.stream_client import StreamClient, StreamError, read_api_key


class Storage:
    def __init__(self):
        self.objects = {}
        self.s3 = SimpleNamespace(
            meta=SimpleNamespace(
                endpoint_url="https://storage.test-region.nebius.cloud"
            )
        )

    def upload_file(self, path, uri):
        self.objects[uri] = Path(path).read_bytes()
        return uri

    def download_path(self, uri, path):
        if uri not in self.objects:
            raise FileNotFoundError()
        Path(path).write_bytes(self.objects[uri])
        return path


class Service:
    def __init__(self, model="nvidia/Cosmos3-Nano"):
        self.model = model
        self.admissions = []
        self.responses = []
        self.fail_poll = False
        self.corrupt_source = False
        self.manifest_override = {}
        self.connection = {
            "bucket": "example-bucket",
            "read_prefix": "run/input",
            "write_prefix": "run/output",
            "purpose": "input_output",
            "provider": "nebius",
            "output_eligible": True,
            "endpoint_url": "https://storage.test-region.nebius.cloud",
            "region": "test-region",
            "verification": {
                "status": "verified",
                "endpoint_region": "test-region",
                "bucket_region": "test-region",
            },
        }

    def handle(self, request):
        assert request.headers["Authorization"] == "Bearer synthetic-secret"
        path = request.url.path
        if path == "/v1/auth/me":
            value = {
                "owner_id": "synthetic-owner",
                "allowed_models": [self.model],
            }
        elif path == "/v1/models":
            value = {
                "data": [
                    {
                        "id": self.model,
                        "capability_matrix": [
                            {
                                "task": "full_source_v2v_augmentation",
                                "runtime_implemented": True,
                                "endpoint_exposed": True,
                                "guardrails_required": True,
                                "evidence_status": "qualified",
                            }
                        ],
                    }
                ]
            }
        elif path.startswith("/v1/storage/connections/"):
            value = {"connection": self.connection}
        elif path == "/v1/augmentations":
            evaluation = json.loads(request.content)["evaluation"]
            assert evaluation == {
                "mode": "technical_report",
                "rubric_id": "paidf-source-timeline",
                "criteria": ["complete_decode", "timeline_coverage"],
            }
            self.admissions.append(
                (request.headers["Idempotency-Key"], json.loads(request.content))
            )
            value = {"job_id": "job-synthetic", "status": "queued"}
        elif path == "/v1/jobs/job-synthetic":
            if self.fail_poll:
                raise httpx.ConnectError("synthetic interruption")
            value = {"job_id": "job-synthetic", "status": "succeeded"}
        elif path.endswith("/artifacts"):
            value = {
                "manifest": {
                    "job_id": "job-synthetic",
                    "prompt_sha256": hashlib.sha256(b"appearance").hexdigest(),
                    "model": self.model,
                    "effective": {
                        "guardrails": True,
                        "structural_control": "edge",
                        "seed": 17,
                        "num_inference_steps": 24,
                        "guidance_scale": 5.0,
                        "control_guidance": 1.5,
                    },
                    "artifacts": [
                        {"name": n, "sha256": hashlib.sha256(self.bytes(n)).hexdigest()}
                        for n in ("source.mp4", "generated.mp4")
                    ],
                    **self.manifest_override,
                }
            }
        else:
            return httpx.Response(200, content=self.bytes(path.rsplit("/", 1)[-1]))
        self.responses.append(copy.deepcopy(value))
        return httpx.Response(200, json=value)

    def bytes(self, name):
        return (
            b"different-source"
            if name == "source.mp4" and self.corrupt_source
            else name.encode()
        )


def adapter(service, storage):
    api = StreamClient(
        "https://stream.example.invalid",
        "synthetic-secret",
        transport=httpx.MockTransport(service.handle),
    )
    return StreamAugmentation(
        origin=api.origin,
        key_file="",
        input_connection="input",
        output_connection="output",
        input_uri="s3://example-bucket/run/input/source.mp4",
        output_uri="s3://example-bucket/run/output",
        checkpoint=service.model.removeprefix("nvidia/"),
        storage=storage,
        environ={},
        api=api,
    )


def generate(client, tmp_path, directory="download"):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source.mp4")
    return client.generate(
        index=0,
        attempt=0,
        run_id="stable-run",
        source=source,
        directory=tmp_path / directory,
        prompt="appearance",
        seed=17,
        steps=24,
        guidance=5.0,
        control_guidance=1.5,
    )


def test_interrupted_poll_resumes_same_job_without_new_admission(tmp_path):
    service, storage = Service(), Storage()
    service.fail_poll = True
    with pytest.raises(StreamError, match="transport"):
        generate(adapter(service, storage), tmp_path)
    assert len(service.admissions) == 1
    service.fail_poll = False
    result = generate(adapter(service, storage), tmp_path)
    assert len(service.admissions) == 1
    assert result["endpoint_execution"]["status"] == "verified"
    assert Path(result["output_path"]).read_bytes() == b"generated.mp4"
    assert b"synthetic-secret" not in b"".join(storage.objects.values())


@pytest.mark.parametrize("model", ["nvidia/Cosmos3-Nano", "nvidia/Cosmos3-Super"])
def test_explicit_model_selection_preserves_each_receipt_shape(tmp_path, model):
    service = Service(model)
    if model.endswith("Super"):
        service.manifest_override["effective"] = {
            "guardrails": True,
            "structural_control": "edge",
            "effective_generation": {
                "seed": 17,
                "num_inference_steps": 24,
                "guidance_scale": 5.0,
                "control_guidance": 1.5,
            },
        }
    result = generate(adapter(service, Storage()), tmp_path)
    assert result["endpoint_execution"]["model"] == model
    assert service.admissions[0][1]["model"] == model


@pytest.mark.parametrize(
    "mutation", ["provider", "eligible", "verification", "endpoint", "scope"]
)
def test_storage_fails_before_admission(mutation):
    service, storage = Service(), Storage()
    if mutation == "provider":
        service.connection["provider"] = "aws"
    elif mutation == "eligible":
        service.connection["output_eligible"] = False
    elif mutation == "verification":
        service.connection["verification"]["bucket_region"] = "other"
    elif mutation == "endpoint":
        storage.s3.meta.endpoint_url = "https://s3.example.invalid"
    else:
        service.connection["write_prefix"] = "another-owner"
    with pytest.raises(StreamError):
        adapter(service, storage)
    assert service.admissions == []


def test_changed_source_is_rejected_even_with_valid_download_hash(tmp_path):
    service = Service()
    service.corrupt_source = True
    with pytest.raises(StreamError, match="different source"):
        generate(adapter(service, Storage()), tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [("model", "nvidia/Cosmos3-Super"), ("job_id", "job-other"), ("effective", {})],
)
def test_unproven_effective_settings_are_rejected(tmp_path, field, value):
    service = Service()
    service.manifest_override[field] = value
    with pytest.raises(StreamError, match="receipt"):
        generate(adapter(service, Storage()), tmp_path)


def test_private_key_permissions_and_symlink(tmp_path):
    key = tmp_path / "key"
    key.write_text("synthetic-secret")
    key.chmod(0o644)
    with pytest.raises(StreamError):
        read_api_key(str(key), {})
    key.chmod(0o600)
    assert read_api_key(str(key), {}) == "synthetic-secret"
    link = tmp_path / "link"
    link.symlink_to(key)
    with pytest.raises(StreamError):
        read_api_key(str(link), {})


@pytest.mark.parametrize("prefix", ["", "WMH_API_KEY="])
def test_private_key_file_accepts_raw_and_operator_delivery_format(tmp_path, prefix):
    key = tmp_path / "key"
    key.write_text(prefix + "synthetic-secret\n")
    key.chmod(0o600)
    assert read_api_key(str(key), {}) == "synthetic-secret"


@pytest.mark.parametrize(
    "value", ["WMH_API_KEY=", "WMH_API_KEY=first\nWMH_API_KEY=second"]
)
def test_operator_key_file_rejects_empty_or_multiple_credentials(tmp_path, value):
    key = tmp_path / "key"
    key.write_text(value)
    key.chmod(0o600)
    with pytest.raises(StreamError, match="required"):
        read_api_key(str(key), {})


def test_rate_limited_reads_wait_but_submissions_are_not_automatically_replayed(
    monkeypatch,
):
    calls = []
    pauses = []

    def handler(request):
        calls.append(request.method)
        if len(calls) == 1 or request.method == "POST":
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json={"status": "running"})

    monkeypatch.setattr("npa.workbench.cosmos.stream_client.time.sleep", pauses.append)
    api = StreamClient(
        "https://stream.example.invalid",
        "synthetic-secret",
        transport=httpx.MockTransport(handler),
    )
    assert api.request("GET", "/v1/jobs/job-synthetic") == {"status": "running"}
    assert pauses == [3.0] and calls == ["GET", "GET"]
    with pytest.raises(StreamError, match="429"):
        api.request("POST", "/v1/augmentations", body={}, identity="retained-identity")
    assert calls == ["GET", "GET", "POST"]


@pytest.mark.parametrize(
    "origin",
    [
        "http://example.invalid",
        "https://user:secret@example.invalid",
        "https://example.invalid?q=secret",
        "https://example.invalid/path",
        "https://example.invalid\\other",
        "https://example.invalid\n",
        "https://example.invalid:bad",
    ],
)
def test_ambiguous_or_insecure_origin_is_rejected(origin):
    with pytest.raises(StreamError):
        StreamClient(origin, "synthetic-secret")


def test_degraded_evaluation_cannot_spend_on_another_refinement(tmp_path):
    from npa.workbench.cosmos.stream_augmentation import require_completed_evaluation

    report = tmp_path / "cosmos_evaluator.json"
    report.write_text(json.dumps({"status": "degraded", "score": 0.0}))
    with pytest.raises(StreamError, match="repair the evaluator"):
        require_completed_evaluation(str(tmp_path))
    # A real measured failure may refine; an infrastructure failure may not.
    report.write_text(
        json.dumps({"status": "completed", "score": 0.0, "passed": False})
    )
    require_completed_evaluation(str(tmp_path))


def test_redirect_is_not_followed_and_hash_mismatch_is_rejected(tmp_path):
    calls = []

    def transport(request):
        calls.append(str(request.url))
        return httpx.Response(
            302, headers={"Location": "https://other.example.invalid/key"}
        )

    api = StreamClient(
        "https://stream.example.invalid",
        "synthetic-secret",
        transport=httpx.MockTransport(transport),
    )
    with pytest.raises(StreamError, match="302"):
        api.request("GET", "/v1/auth/me")
    assert len(calls) == 1
    api = StreamClient(
        "https://stream.example.invalid",
        "synthetic-secret",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=b"tampered")
        ),
    )
    with pytest.raises(StreamError, match="checksum"):
        api.download(
            "job-synthetic", {"name": "video.mp4", "sha256": "a" * 64}, tmp_path
        )
    assert not (tmp_path / "video.mp4").exists()


def test_hosted_workflow_selects_cpu_and_has_no_native_model_access_requirement():
    from npa.cli.workbench.workflow import _workflow_access_requirements
    from npa.orchestration.npa_workflow.interpreter import build_reachability_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    path = Path(__file__).resolve().parents[3] / "workflows/main/paidf-cosmos3.yaml"
    spec = load_spec(path)
    assert spec.config["use_hosted_endpoint"] is False
    native = {step.state for step in build_reachability_plan(spec).steps}
    assert "generate-variants" in native and "generate-variants-endpoint" not in native
    assert _workflow_access_requirements(spec)
    spec.config["use_hosted_endpoint"] = True
    hosted = {step.state for step in build_reachability_plan(spec).steps}
    assert "generate-variants-endpoint" in hosted and "generate-variants" not in hosted
    assert spec.states["generate-variants-endpoint"].resources == "adapter"
    assert not _workflow_access_requirements(spec)


@pytest.mark.parametrize("model", ["Cosmos3-Nano", "Cosmos3-Super"])
@pytest.mark.parametrize("decision", ["promote_checkpoint", "loop_back"])
@pytest.mark.parametrize("count", ["2", "12"])
def test_hosted_switch_preserves_surrounding_workflow(model, decision, count):
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    path = Path(__file__).resolve().parents[3] / "workflows/main/paidf-cosmos3.yaml"
    spec = load_spec(path)
    spec.config.update(
        cosmos3_checkpoint=model, variant_count=count, variant_parallelism=count
    )
    native = build_plan(spec, run_id="backend-contract", assume_decision=decision)
    spec.config["use_hosted_endpoint"] = True
    hosted = build_plan(spec, run_id="backend-contract", assume_decision=decision)
    generation = {"generate-variants", "generate-variants-endpoint"}

    def surrounding(plan):
        fields = ("state", "argv", "shell", "resources", "inputs", "outputs")
        return [
            tuple(getattr(step, field) for field in fields)
            for step in plan.steps
            if step.state not in generation | {"verify-endpoint-evaluation"}
        ]

    assert surrounding(native) == surrounding(hosted)
    for plan, expected in (
        (native, "generate-variants"),
        (hosted, "generate-variants-endpoint"),
    ):
        steps = [step for step in plan.steps if step.state in generation]
        assert steps and all(
            step.state == expected and model in step.argv for step in steps
        )
    assert all(step.state != "verify-endpoint-evaluation" for step in native.steps)
    assert any(step.state == "verify-endpoint-evaluation" for step in hosted.steps)


def test_static_config_transition_executes_without_a_decision_artifact(tmp_path):
    from npa.orchestration.npa_workflow.interpreter import run_workflow
    from npa.orchestration.npa_workflow.spec import load_spec

    path = tmp_path / "workflow.yaml"
    path.write_text("""apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata: {name: config-branch}
config: {enabled: true, decision_uri: 's3://example-bucket/not-yet-written.json'}
initial: select
states:
  select:
    run: {argv: ['true']}
    transitions:
      - {if: config.enabled, goto: hosted}
      - {goto: native}
  hosted:
    run: {argv: ['true']}
    terminal: true
  native:
    run: {argv: ['false']}
    terminal: true
""")

    def unexpected(*args):
        raise AssertionError("a static config branch must not read a decision")

    result = run_workflow(
        load_spec(path),
        run_id="static-branch",
        execute=True,
        decision_reader=unexpected,
    )
    assert result["status"] == "completed"
