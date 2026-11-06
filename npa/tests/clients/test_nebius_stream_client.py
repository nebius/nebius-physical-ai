"""Verify the portable Stream client without a private server or credentials."""

import copy
import hashlib
import importlib.util
import io
import json
import ssl
from pathlib import Path
from urllib.request import Request

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "public_cosmos_client",
    ROOT / "skills/tools/nebius-stream-beta/scripts/cosmos_client.py",
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def private_file(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    path.chmod(0o600)
    return path


def contract():
    return {
        "auth": {"owner_id": "alpha"},
        "openapi": {"paths": {"/v1/videos": {"post": {}}}},
        "models": {
            "data": [
                {
                    "id": "nvidia/Cosmos3-Nano",
                    "capability_matrix": [
                        {
                            "profiles": ["nano-480p-20"],
                            "runtime_implemented": True,
                            "endpoint_exposed": True,
                            "evidence_status": "qualified",
                        }
                    ],
                }
            ]
        },
    }


def request():
    return {
        "model": "nvidia/Cosmos3-Nano",
        "prompt": "A robot moves a block.",
        "output": {
            "connection_id": "customer-output",
            "prefix": "s3://example-bucket/generated/run-001",
        },
        "generation": {"profile": "nano-480p-20", "seed": 101},
    }


class FakeClient:
    origin = "https://api.example.test"

    def __init__(self):
        self.attempts = []
        self.fail = False

    def contract(self):
        return contract()

    def request(self, method, route, payload=None, identity=None):
        if route.startswith("/v1/storage/connections/"):
            return {
                "connection": {
                    "purpose": "output",
                    "provider": "nebius",
                    "output_eligible": True,
                    "verification": {"status": "verified"},
                }
            }
        if method == "POST":
            self.attempts.append(identity)
            if self.fail:
                raise helper.ClientError("HTTPS request failed")
            return {"job_id": "job-001"}
        if route == "/v1/auth/me":
            return {"owner_id": "alpha"}
        if route.endswith("/artifacts"):
            return {
                "manifest": {
                    "artifacts": [
                        {
                            "name": "generated.mp4",
                            "sha256": hashlib.sha256(b"verified-output").hexdigest(),
                        }
                    ]
                }
            }
        return {"status": "succeeded"}

    def open(self, *args):
        return io.BytesIO(b"verified-output")


def test_submission_disconnect_retains_identity_and_recovery_does_not_duplicate(
    tmp_path,
):
    tmp_path.chmod(0o700)
    client = FakeClient()
    state = tmp_path / "run.json"
    client.fail = True
    with pytest.raises(helper.ClientError):
        helper.submit(client, request(), state, "/v1/videos")
    stored = helper.read_json(state)
    assert stored["idempotency_key"] and "job_id" not in stored
    client.fail = False
    assert helper.submit(client, request(), state, "/v1/videos")["job_id"] == "job-001"
    assert client.attempts[0] == client.attempts[1]
    helper.submit(client, request(), state, "/v1/videos")
    assert len(client.attempts) == 2
    changed = {**request(), "prompt": "Different intended request"}
    with pytest.raises(helper.ClientError, match="different request"):
        helper.submit(client, changed, state, "/v1/videos")
    client.origin = "https://other.example.test"
    with pytest.raises(helper.ClientError, match="different request"):
        helper.submit(client, request(), state, "/v1/videos")


def test_download_integrity_and_resume(tmp_path):
    tmp_path.chmod(0o700)
    client = FakeClient()
    state = tmp_path / "run.json"
    helper.submit(client, request(), state, "/v1/videos")
    result = helper.resume(client, state, tmp_path / "results")
    assert result["verified_artifacts"] == 1
    assert (tmp_path / "results/generated.mp4").read_bytes() == b"verified-output"
    client.open = lambda *args: io.BytesIO(b"corrupt")
    with pytest.raises(helper.ClientError, match="SHA-256 mismatch"):
        helper.resume(client, state, tmp_path / "results")
    assert (tmp_path / "results/generated.mp4").read_bytes() == b"verified-output"
    assert not list((tmp_path / "results").glob(".download-*"))


def test_file_permissions_and_symlinks(tmp_path):
    key = private_file(tmp_path, "key", "synthetic-key")
    assert helper.read_private(key) == b"synthetic-key"
    key.chmod(0o644)
    with pytest.raises(helper.ClientError, match="private"):
        helper.read_private(key)
    key.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(key)
    with pytest.raises(helper.ClientError):
        helper.read_private(link)


def test_client_verifies_tls_and_refuses_redirect(monkeypatch, tmp_path):
    key = private_file(tmp_path, "key", "synthetic-key")
    context = ssl.create_default_context()
    builder = []
    monkeypatch.setattr(helper.ssl, "create_default_context", lambda **kwargs: context)
    monkeypatch.setattr(
        helper, "build_opener", lambda *handlers: builder.extend(handlers)
    )
    helper.Client("https://api.example.test", key)
    handler = next(h for h in builder if isinstance(h, helper.HTTPSHandler))
    assert handler._context.check_hostname is True
    assert handler._context.verify_mode == ssl.CERT_REQUIRED
    assert next(h for h in builder if isinstance(h, helper.ProxyHandler)).proxies == {}
    redirect = next(h for h in builder if isinstance(h, helper.NoRedirect))
    for status in [301, 302, 303, 307, 308]:
        with pytest.raises(helper.ClientError, match="redirect refused"):
            redirect.redirect_request(
                Request(
                    "https://api.example.test/v1/models",
                    headers={"Authorization": "Bearer synthetic-key"},
                ),
                None,
                status,
                "redirect",
                {},
                "https://foreign.example.test/",
            )
    with pytest.raises(helper.ClientError, match="HTTPS API origin"):
        helper.Client("http://api.example.test", key)


def test_unsupported_or_unqualified_profile_fails_before_submit():
    value = request()
    value["generation"]["profile"] = "imagined-profile"
    with pytest.raises(helper.ClientError, match="qualified"):
        helper.qualify(contract(), value, "/v1/videos")
    value = request()
    source = contract()
    source["models"]["data"][0]["capability_matrix"][0]["profile_evidence"] = {
        "nano-480p-20": "failed"
    }
    with pytest.raises(helper.ClientError, match="qualified"):
        helper.qualify(source, value, "/v1/videos")


def reasoning_contract(model, modalities=("text", "image", "video")):
    return {
        "auth": {"owner_id": "alpha"},
        "openapi": {"paths": {"/v1/reasoning": {"post": {}}}},
        "models": {
            "data": [
                {
                    "id": model,
                    "capability_matrix": [
                        {
                            "task": "reasoning_understanding",
                            "profiles": ["cosmos3-reasoner-chat-v1"],
                            "profile_evidence": {
                                "cosmos3-reasoner-chat-v1": "qualified"
                            },
                            "input_modalities": list(modalities),
                            "output_modalities": ["text"],
                            "guardrails_required": True,
                            "runtime_implemented": True,
                            "endpoint_exposed": True,
                            "evidence_status": "qualified",
                        }
                    ],
                }
            ]
        },
    }


def reasoning_request(name):
    path = ROOT / "skills/tools/nebius-stream-beta/examples" / name
    return json.loads(path.read_text())


class ReasoningClient(FakeClient):
    def __init__(self, model, modalities=("text", "image", "video")):
        super().__init__()
        self.discovery = reasoning_contract(model, modalities)

    def contract(self):
        return self.discovery


@pytest.mark.parametrize(
    ("example", "modality"),
    [
        ("reasoning-text.json", "text"),
        ("reasoning-image.json", "image"),
        ("reasoning-video.json", "video"),
    ],
)
@pytest.mark.parametrize("model", ["nvidia/Cosmos3-Nano", "nvidia/Cosmos3-Super"])
def test_reasoning_examples_admit_once_and_recover_with_same_identity(
    tmp_path, example, modality, model
):
    tmp_path.chmod(0o700)
    client = ReasoningClient(model)
    value = reasoning_request(example)
    value["model"] = model
    state_path = tmp_path / f"{modality}.json"
    client.fail = True
    with pytest.raises(helper.ClientError, match="HTTPS request failed"):
        helper.submit(client, value, state_path, "/v1/reasoning", modality)
    retained = helper.read_json(state_path)
    assert retained["route"] == "/v1/reasoning" and "job_id" not in retained
    client.fail = False
    state = helper.submit(client, value, state_path, "/v1/reasoning", modality)
    assert state["job_id"] == "job-001"
    assert client.attempts == [retained["idempotency_key"]] * 2
    assert (
        helper.resume(client, state_path, tmp_path / "results")["verified_artifacts"]
        == 1
    )
    assert value["generation"]["seed"] == 0
    assert value["generation"]["temperature"] == 0.0


@pytest.mark.parametrize(
    "failure",
    [
        "route",
        "model",
        "task",
        "profile",
        "input_modality",
        "output_modality",
        "guardrails",
        "runtime",
        "exposure",
        "evidence",
    ],
)
def test_reasoning_rejects_unqualified_contract_before_admission(tmp_path, failure):
    tmp_path.chmod(0o700)
    client = ReasoningClient("nvidia/Cosmos3-Nano")
    value = reasoning_request("reasoning-video.json")
    row = client.discovery["models"]["data"][0]["capability_matrix"][0]
    if failure == "route":
        client.discovery["openapi"]["paths"] = {}
    elif failure == "model":
        value["model"] = "nvidia/Cosmos-Reason"
    elif failure == "task":
        row["task"] = "text_to_video"
    elif failure == "profile":
        row["profiles"] = []
    elif failure == "input_modality":
        row["input_modalities"] = ["text"]
    elif failure == "output_modality":
        row["output_modalities"] = ["video"]
    elif failure == "guardrails":
        row["guardrails_required"] = False
    elif failure == "runtime":
        row["runtime_implemented"] = False
    elif failure == "exposure":
        row["endpoint_exposed"] = False
    else:
        row["profile_evidence"] = {"cosmos3-reasoner-chat-v1": "failed"}
    with pytest.raises(helper.ClientError):
        helper.submit(
            client, value, tmp_path / "reasoning.json", "/v1/reasoning", "video"
        )
    assert not client.attempts and not (tmp_path / "reasoning.json").exists()


@pytest.mark.parametrize(
    ("modality", "change", "message"),
    [
        (None, None, "requires --input-modality"),
        ("text", "keep_input", "must not include"),
        ("image", "drop_input", "requires an input"),
        ("image", "video_sampler", "video sampling"),
        ("video", "two_samplers", "only one video sampler"),
        ("text", "zero_tokens", "max_tokens"),
        ("text", "zero_top_p", "top_p"),
    ],
)
def test_reasoning_request_validation_happens_before_admission(
    tmp_path, modality, change, message
):
    tmp_path.chmod(0o700)
    client = ReasoningClient("nvidia/Cosmos3-Nano")
    value = reasoning_request("reasoning-text.json")
    if change == "keep_input":
        value["input"] = {"connection_id": "input", "uri": "s3://BUCKET/image.png"}
    elif change == "drop_input":
        value = reasoning_request("reasoning-image.json")
        value["model"] = "nvidia/Cosmos3-Nano"
        value.pop("input")
    elif change == "video_sampler":
        value = reasoning_request("reasoning-image.json")
        value["model"] = "nvidia/Cosmos3-Nano"
        value["generation"]["video_num_frames"] = 24
    elif change == "two_samplers":
        value = reasoning_request("reasoning-video.json")
        value["generation"]["video_fps"] = 2
    elif change == "zero_tokens":
        value["generation"]["max_tokens"] = 0
    elif change == "zero_top_p":
        value["generation"]["top_p"] = 0
    with pytest.raises(helper.ClientError, match=message):
        helper.submit(
            client, value, tmp_path / "reasoning.json", "/v1/reasoning", modality
        )
    assert not client.attempts and not (tmp_path / "reasoning.json").exists()


def test_reasoning_cli_keeps_zero_values_and_retains_private_state(
    tmp_path, monkeypatch, capsys
):
    tmp_path.chmod(0o700)
    key = private_file(tmp_path, "key", "synthetic-key")
    request_path = private_file(
        tmp_path, "request.json", json.dumps(reasoning_request("reasoning-text.json"))
    )
    client = ReasoningClient("nvidia/Cosmos3-Nano")
    monkeypatch.setenv("WMH_BASE_URL", client.origin)
    monkeypatch.setenv("WMH_API_KEY_FILE", str(key))
    monkeypatch.setattr(helper, "Client", lambda *args: client)
    monkeypatch.setattr(
        helper.sys,
        "argv",
        [
            "cosmos_client.py",
            "submit",
            "--route",
            "/v1/reasoning",
            "--input-modality",
            "text",
            "--request",
            str(request_path),
            "--state",
            str(tmp_path / "state.json"),
        ],
    )
    helper.main()
    assert json.loads(capsys.readouterr().out)["job_id"] == "job-001"
    assert helper.read_json(tmp_path / "state.json")["route"] == "/v1/reasoning"


def transfer_contract(model):
    value = contract()
    value["openapi"]["paths"] = {"/v1/transfers": {"post": {}}}
    value["models"]["data"][0].update(
        id=model,
        capability_matrix=[
            {
                "task": "structural_transfer",
                "profiles": ["structural-transfer-832x480-121-v1"],
                "controls": ["edge", "depth", "seg", "blur"],
                "control_combinations": ["blur+depth+edge+seg"],
                "control_weights_supported": True,
                "negative_prompt_supported": True,
                "guardrails_required": True,
                "runtime_implemented": True,
                "endpoint_exposed": True,
                "evidence_status": "qualified",
            }
        ],
    )
    return value


def transfer_request(model, frames):
    path = ROOT / "skills/tools/nebius-stream-beta/examples/structural-transfer.json"
    value = json.loads(path.read_text())
    value["model"] = model
    value["input"]["uri"] = f"s3://example-bucket/inputs/source-{frames}.mp4"
    return value


class TransferClient(FakeClient):
    def __init__(self, model):
        super().__init__()
        self.discovery = transfer_contract(model)
        self.payloads = []

    def contract(self):
        return self.discovery

    def request(self, method, route, payload=None, identity=None):
        if method == "POST":
            self.payloads.append((route, copy.deepcopy(payload), identity))
        return super().request(method, route, payload, identity)


@pytest.mark.parametrize("model", ["nvidia/Cosmos3-Super", "nvidia/Cosmos3-Nano"])
@pytest.mark.parametrize("frames", [227, 302])
def test_transfer_keeps_full_source_controls_and_prompts_through_resume(
    tmp_path, model, frames
):
    tmp_path.chmod(0o700)
    client = TransferClient(model)
    value = transfer_request(model, frames)
    state_path = tmp_path / "transfer.json"
    client.fail = True
    with pytest.raises(helper.ClientError, match="HTTPS request failed"):
        helper.submit(client, value, state_path, "/v1/transfers")
    retained = helper.read_json(state_path)
    assert "job_id" not in retained
    client.fail = False
    state = helper.submit(client, value, state_path, "/v1/transfers")
    assert helper.status_route(state) == "/v1/jobs/job-001"
    assert client.attempts == [retained["idempotency_key"]] * 2
    assert all(
        route == "/v1/transfers" and body == value for route, body, _ in client.payloads
    )
    helper.submit(client, value, state_path, "/v1/transfers")
    result = helper.resume(client, state_path, tmp_path / "results")
    assert result["verified_artifacts"] == 1 and len(client.attempts) == 2
    assert (tmp_path / "results/generated.mp4").read_bytes() == b"verified-output"
    for change in ("weight", "negative_prompt"):
        changed = copy.deepcopy(value)
        if change == "weight":
            changed["controls"]["depth"]["weight"] = 2
        else:
            changed["negative_prompt"] = "Different negative prompt"
        with pytest.raises(helper.ClientError, match="different request"):
            helper.submit(client, changed, state_path, "/v1/transfers")
    assert len(client.attempts) == 2


@pytest.mark.parametrize(
    "failure",
    [
        "route",
        "profile",
        "wrong-task",
        "controls",
        "combination",
        "weights",
        "negative",
        "guardrails",
        "runtime",
        "exposure",
        "evidence",
    ],
)
def test_transfer_requires_its_own_qualified_capabilities_before_post(
    tmp_path, failure
):
    tmp_path.chmod(0o700)
    client = TransferClient("nvidia/Cosmos3-Nano")
    value = transfer_request("nvidia/Cosmos3-Nano", 302)
    row = client.discovery["models"]["data"][0]["capability_matrix"][0]
    if failure == "route":
        client.discovery["openapi"]["paths"] = {}
    elif failure == "profile":
        row["profiles"] = ["another-profile"]
    elif failure == "wrong-task":
        row["task"] = "full_source_v2v_augmentation"
    elif failure == "controls":
        row["controls"] = ["edge"]
    elif failure == "combination":
        row["control_combinations"] = []
    elif failure == "weights":
        row["control_weights_supported"] = False
    elif failure == "negative":
        row["negative_prompt_supported"] = False
    elif failure == "guardrails":
        row["guardrails_required"] = False
    elif failure == "runtime":
        row["runtime_implemented"] = False
    elif failure == "exposure":
        row["endpoint_exposed"] = False
    else:
        row["profile_evidence"] = {value["generation"]["profile"]: "failed"}
    with pytest.raises(helper.ClientError):
        helper.submit(client, value, tmp_path / "transfer.json", "/v1/transfers")
    assert not client.attempts and not (tmp_path / "transfer.json").exists()


@pytest.mark.parametrize("weight", [True, "1", -1, float("inf"), 10**400, 0])
def test_transfer_invalid_weights_never_reach_admission(tmp_path, weight):
    tmp_path.chmod(0o700)
    client = TransferClient("nvidia/Cosmos3-Super")
    value = transfer_request("nvidia/Cosmos3-Super", 227)
    for control in value["controls"].values():
        control["weight"] = weight
    with pytest.raises(helper.ClientError, match="weights"):
        helper.submit(client, value, tmp_path / "transfer.json", "/v1/transfers")
    assert not client.attempts


def test_transfer_accepts_unit_weight_without_relative_weight_or_negative_support():
    model = "nvidia/Cosmos3-Nano"
    source = transfer_contract(model)
    row = source["models"]["data"][0]["capability_matrix"][0]
    row["control_weights_supported"] = row["negative_prompt_supported"] = False
    value = transfer_request(model, 302)
    value["controls"] = {"edge": {"weight": 3}}
    value["negative_prompt"] = ""
    helper.qualify(source, value, "/v1/transfers")


@pytest.mark.parametrize(
    "connection_change",
    [
        {"purpose": "input"},
        {"provider": "other"},
        {"output_eligible": False},
        {"verification": {"status": "pending"}},
        {"verification": {}},
    ],
)
def test_submit_refuses_unverified_output_before_admission(tmp_path, connection_change):
    tmp_path.chmod(0o700)
    client = FakeClient()
    original = client.request

    def request_with_unverified_output(method, route, *args):
        response = original(method, route, *args)
        if route.startswith("/v1/storage/connections/"):
            response["connection"].update(connection_change)
        return response

    client.request = request_with_unverified_output
    state = tmp_path / "run.json"
    with pytest.raises(helper.ClientError, match="Nebius verification"):
        helper.submit(client, request(), state, "/v1/videos")
    assert client.attempts == []
    assert not state.exists()


def test_get_backpressure_honors_retry_after_and_never_retries_post(
    tmp_path, monkeypatch
):
    from email.message import Message
    from urllib.error import HTTPError

    key = private_file(tmp_path, "key", "synthetic-key")
    client = helper.Client("https://api.example.test", key)
    sleeps = []
    monkeypatch.setattr(helper.time, "sleep", sleeps.append)
    monkeypatch.setattr(helper.random, "uniform", lambda a, b: 0.125)
    headers = Message()
    headers["Retry-After"] = "3"
    attempts = []

    def opening(request, **kwargs):
        attempts.append(request.get_method())
        if len(attempts) == 1:
            raise HTTPError(
                request.full_url, 429, "busy", headers, io.BytesIO(b"private response")
            )
        return io.BytesIO(b"verified-output")

    monkeypatch.setattr(client.opener, "open", opening)
    with client.open("GET", "/v1/jobs/job-001/artifacts/generated.mp4") as stream:
        assert stream.read() == b"verified-output"
    assert attempts == ["GET", "GET"]
    assert sleeps == [3.125]
    attempts.clear()
    sleeps.clear()
    with pytest.raises(helper.ClientError, match="HTTP 429"):
        client.open("POST", "/v1/videos", request(), "original-idempotency-key")
    assert attempts == ["POST"] and sleeps == []


def test_get_backpressure_accepts_http_date(tmp_path, monkeypatch):
    from email.message import Message
    from urllib.error import HTTPError

    key = private_file(tmp_path, "key", "synthetic-key")
    client = helper.Client("https://api.example.test", key)
    sleeps = []
    calls = []
    monkeypatch.setattr(helper.time, "time", lambda: 0.0)
    monkeypatch.setattr(helper.time, "sleep", sleeps.append)
    monkeypatch.setattr(helper.random, "uniform", lambda a, b: 0)
    headers = Message()
    headers["Retry-After"] = "Thu, 01 Jan 1970 00:00:04 GMT"

    def opening(request, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise HTTPError(request.full_url, 429, "busy", headers, io.BytesIO())
        return io.BytesIO(b"ok")

    monkeypatch.setattr(client.opener, "open", opening)
    with client.open("GET", "/v1/jobs/job-001/artifacts") as stream:
        assert stream.read() == b"ok"
    assert sleeps == [4.0]


def test_query_encoding_keeps_origin_and_rejects_raw_routes(tmp_path, monkeypatch):
    from urllib.parse import parse_qs, urlsplit

    client = helper.Client(
        "https://api.example.test", private_file(tmp_path, "key", "synthetic-key")
    )
    calls = []
    monkeypatch.setattr(
        client.opener,
        "open",
        lambda request, **kwargs: calls.append(request) or io.BytesIO(b"{}"),
    )
    client.request(
        "GET",
        "/v1/usage/events",
        query={"owner_id": "owner&key=foreign", "include_synthetic": True, "cursor": 2},
    )
    parsed = urlsplit(calls[0].full_url)
    assert parsed.netloc == "api.example.test" and parsed.path == "/v1/usage/events"
    assert parse_qs(parsed.query) == {
        "owner_id": ["owner&key=foreign"],
        "include_synthetic": ["true"],
        "cursor": ["2"],
    }
    assert "synthetic-key" not in calls[0].full_url
    with pytest.raises(helper.ClientError, match="invalid API route"):
        client.request("GET", "/v1/usage/events?owner_id=other")
    with pytest.raises(helper.ClientError, match="invalid API query"):
        client.request("POST", "/v1/videos", query={"owner_id": "other"})


class UsageClient:
    origin = "https://api.example.test"

    def __init__(self):
        self.calls = []
        self.rows = [
            {"sequence": number, "owner_id": "alpha", "event_id": f"event-{number}"}
            for number in (1, 4, 7)
        ]

    def request(self, method, route, *, query=None):
        if route == "/v1/auth/me":
            return {"owner_id": "alpha"}
        self.calls.append(query.copy())
        high = query.get("watermark", 9)
        rows = [row for row in self.rows if query["cursor"] < row["sequence"] <= high]
        page, more = rows[: query["limit"]], len(rows) > query["limit"]
        next_cursor = page[-1]["sequence"] if more else None
        return {
            "schema_version": "world-model-hosting.usage.v1",
            "events": page,
            "watermark": high,
            "next_cursor": next_cursor,
            "resume_cursor": next_cursor if more else high,
        }


def test_usage_export_crash_after_page_reuses_watermark_and_never_duplicates(
    tmp_path, monkeypatch
):
    tmp_path.chmod(0o700)
    client = UsageClient()
    state_path, destination = tmp_path / "export-state.json", tmp_path / "events"
    original = helper.save_private
    crashed = []

    def save(path, value):
        if Path(path) == state_path and value.get("cursor") == 1 and not crashed:
            crashed.append(True)
            raise OSError("simulated crash before cursor commit")
        original(path, value)

    monkeypatch.setattr(helper, "save_private", save)
    with pytest.raises(OSError):
        helper.usage_export(
            client, state_path, destination, {"include_synthetic": True}, limit=1
        )
    assert helper.read_json(state_path)["cursor"] == 0
    assert helper.read_json(state_path)["watermark"] == 9
    assert len(list(destination.glob("*.json"))) == 1
    client.rows.append({"sequence": 10, "owner_id": "alpha", "event_id": "future"})
    result = helper.usage_export(
        client, state_path, destination, {"include_synthetic": True}, limit=1
    )
    assert result == {
        "watermark": 9,
        "cursor": 9,
        "complete": True,
        "pages": 3,
        "events": 3,
    }
    rows = [
        row
        for path in sorted(destination.glob("*.json"))
        for row in json.loads(path.read_text())["events"]
    ]
    assert [row["sequence"] for row in rows] == [1, 4, 7]
    assert all(
        query["owner_id"] == "alpha" and query["include_synthetic"]
        for query in client.calls
    )
    assert all(query.get("watermark") == 9 for query in client.calls[1:])
    calls = len(client.calls)
    helper.usage_export(
        client, state_path, destination, {"include_synthetic": True}, limit=1
    )
    assert len(client.calls) == calls
    with pytest.raises(helper.ClientError, match="different options"):
        helper.usage_export(
            client, state_path, destination, {"include_synthetic": False}, limit=1
        )
    newer = helper.usage_export(
        client,
        tmp_path / "next.json",
        tmp_path / "next",
        {"include_synthetic": True},
        after=9,
    )
    assert newer["events"] == 0  # This fake's initial high watermark remains 9.


def test_usage_export_rejects_cross_owner_and_cursor_cycles(tmp_path):
    tmp_path.chmod(0o700)
    client = UsageClient()
    client.rows[0]["owner_id"] = "other"
    with pytest.raises(helper.ClientError, match="cross-owner"):
        helper.usage_export(
            client, tmp_path / "bad.json", tmp_path / "bad", {}, limit=1
        )
    assert not (tmp_path / "bad").exists()
    client.rows[0]["owner_id"] = "alpha"
    original = client.request

    def request(method, route, *, query=None):
        value = original(method, route, query=query)
        if route == "/v1/usage/events":
            value["next_cursor"] = 0
        return value

    client.request = request
    with pytest.raises(helper.ClientError, match="non-advancing"):
        helper.usage_export(
            client, tmp_path / "cycle.json", tmp_path / "cycle", {}, limit=1
        )


def test_completed_batch_downloads_successful_children_and_preserves_failure(tmp_path):
    tmp_path.chmod(0o700)
    state_path = tmp_path / "batch-state.json"
    helper.save_private(
        state_path,
        {
            "origin": FakeClient.origin,
            "owner_id": "alpha",
            "route": "/v1/batches",
            "batch_id": "batch-one",
        },
    )
    client = FakeClient()
    original = client.request

    def request(method, route, *args):
        if route == "/v1/batches/batch-one":
            return {
                "status": "completed",
                "jobs": [
                    {"job_id": "job-good", "status": "succeeded"},
                    {"job_id": "job-bad", "status": "failed"},
                ],
            }
        return original(method, route, *args)

    client.request = request
    with pytest.raises(helper.ClientError, match="unsuccessful children"):
        helper.resume(client, state_path, tmp_path / "results")
    assert (
        tmp_path / "results/job-good/generated.mp4"
    ).read_bytes() == b"verified-output"
    assert (
        json.loads((tmp_path / "results/batch.json").read_text())["jobs"][1]["status"]
        == "failed"
    )
    assert not (tmp_path / "results/job-bad").exists()


def test_large_private_export_page_can_be_atomically_replaced_on_retry(tmp_path):
    tmp_path.chmod(0o700)
    path = tmp_path / "page.json"
    helper.save_private(path, b"x" * 2_000_000)
    helper.save_private(path, b"y" * 2_000_000)
    assert path.read_bytes() == b"y" * 2_000_000
    assert path.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("value", ["2026-10-27T12:30:00Z", "2026-10-27T08:30:00-04:00"])
def test_usage_time_filters_require_timezone_and_normalize_to_utc(value):
    import argparse

    assert helper.utc_epoch(value) == 1793104200
    with pytest.raises(argparse.ArgumentTypeError):
        helper.utc_epoch("2026-10-27T12:30:00")


def test_storage_only_secret_init_preserves_existing_api_key(
    tmp_path, monkeypatch, capsys
):
    tmp_path.chmod(0o700)
    key = private_file(tmp_path, "api-key.txt", "existing-api-value")
    destination = tmp_path / "separate-input-s3.json"
    monkeypatch.setenv("WMH_API_KEY_FILE", str(key))
    monkeypatch.setenv("WMH_S3_CREDENTIAL_FILE", str(destination))
    monkeypatch.setattr(helper.sys, "argv", ["cosmos_client.py", "init-storage-secret"])
    monkeypatch.setattr(helper.sys.stdin, "isatty", lambda: True)
    values = iter(["input-access-value", "input-secret-value", ""])
    monkeypatch.setattr(helper.getpass, "getpass", lambda prompt: next(values))
    helper.main()
    assert key.read_text() == "existing-api-value"
    assert (
        helper.storage_secrets(destination)["aws_secret_access_key"]
        == "input-secret-value"
    )
    output = capsys.readouterr()
    assert "input-secret-value" not in output.out + output.err
    assert "existing-api-value" not in output.out + output.err
    with pytest.raises(helper.ClientError, match="unused"):
        helper.main()
