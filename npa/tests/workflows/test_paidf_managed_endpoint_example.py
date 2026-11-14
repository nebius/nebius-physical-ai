"""Exercise the managed-endpoint PAIDF client: logical tasks, receipts, offline HTML."""

import base64
import copy
import hashlib
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import subprocess
import threading

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "npa/examples/paidf-cosmos3-api-only"
SPEC = importlib.util.spec_from_file_location(
    "paidf_managed_example", EXAMPLE / "run.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def campaign():
    return runner._campaign(EXAMPLE / "paidf-managed-endpoint.yaml")


@pytest.fixture
def private_work(tmp_path):
    tmp_path.chmod(0o700)
    return tmp_path


@pytest.fixture(autouse=True)
def no_client_media_processing(monkeypatch):
    """The successful client path must never start FFmpeg/ffprobe."""
    original = subprocess.run

    def guarded(arguments, *args, **kwargs):
        if Path(str(arguments[0])).name in {"ffmpeg", "ffprobe"}:
            raise AssertionError("client attempted local media processing")
        return original(arguments, *args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", guarded)


def _bindings():
    return {
        "wmh_base_url": "https://api.example.test",
        "wmh_api_key_file": "/private/key.txt",
        "input_connection_id": "input",
        "output_connection_id": "output",
        "source_uri": "s3://example-bucket/source.mp4",
        "output_prefix": "s3://example-bucket/output",
    }


def _contract(status="available"):
    return {
        "auth": {"owner_id": "test-owner"},
        "openapi": {"paths": {runner.ROUTE: {"post": {}}}},
        "models": {
            "data": [
                {
                    "id": runner.MODEL,
                    "managed_finishing": {
                        "schema_version": "managed-finishing-v1",
                        "status": status,
                        "request_field": "output_policy",
                        "customer_compute_controls": "none",
                    },
                    "capability_matrix": [
                        {
                            "task": "full_source_v2v_augmentation",
                            "endpoint_exposed": True,
                            "runtime_implemented": True,
                            "evidence_status": "qualified",
                            "guardrails_required": True,
                        }
                    ],
                }
            ]
        },
    }


class _Service:
    origin = "https://api.example.test"

    def __init__(self, module, status="available"):
        self.module = module
        self.status = status
        self.jobs = {}
        self.lock = threading.Lock()
        self.fail_once = True
        self.posts = []
        self.bodies = []

    def contract(self):
        return _contract(self.status)

    def request(self, method, route, payload=None, identity=None):
        with self.lock:
            if route.startswith("/v1/storage/connections/"):
                return {
                    "connection": {
                        "purpose": "output",
                        "provider": "nebius",
                        "output_eligible": True,
                        "verification": {"status": "verified"},
                    }
                }
            if route == "/v1/auth/me":
                return {"owner_id": "test-owner"}
            if method == "POST":
                self.posts.append(identity)
                self.bodies.append(payload)
                self.jobs.setdefault(identity, "job-" + str(len(self.jobs) + 1))
                if self.fail_once:
                    self.fail_once = False
                    raise self.module.ClientError("lost successful response")
                return {"job_id": self.jobs[identity]}
            return {"status": "succeeded"}


def test_requests_are_logical_tasks_with_output_policy_and_no_compute(campaign):
    body = runner._request(campaign, _bindings(), campaign["variants"][0])
    assert body["output_policy"] == {
        "border_preservation": "restore_detected_source_borders",
        "browser_preview": "lossless_rgb_webm",
    }
    text = json.dumps(body).lower()
    for forbidden in ("gpu", "replica", "node", "pod", "container", "concurrency"):
        assert forbidden not in text


@pytest.mark.parametrize(
    "mutation",
    [
        "gpu_count",
        "replicas",
        "customer_compute",
        "media_processing",
        "policy_field",
        "policy_value",
    ],
)
def test_campaign_rejects_customer_compute_or_unsupported_policy(
    campaign, private_work, mutation
):
    changed = copy.deepcopy(campaign)
    if mutation == "gpu_count":
        changed["generation"]["gpu_count"] = 4
    elif mutation == "replicas":
        changed["output_policy"]["replicas"] = 2
    elif mutation == "customer_compute":
        changed["execution"]["customer_compute"] = "local-cpu"
    elif mutation == "media_processing":
        changed["execution"]["customer_media_processing"] = "ffmpeg"
    elif mutation == "policy_field":
        del changed["output_policy"]["browser_preview"]
    else:
        changed["output_policy"]["border_preservation"] = "prompt"
    path = private_work / "campaign.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError):
        runner._campaign(path)


def test_service_without_managed_finishing_never_admits_or_falls_back(
    campaign, private_work, monkeypatch
):
    module = runner._client_module()
    service = _Service(module, status="not_available")
    monkeypatch.setattr(runner, "_api_client", lambda *args: service)
    with pytest.raises(ValueError, match="managed finishing"):
        runner._admit(module, private_work, campaign, _bindings())
    assert not service.posts


def test_lost_admission_reuses_identity_and_retains_twelve_jobs(
    campaign, private_work, monkeypatch
):
    module = runner._client_module()
    service = _Service(module)
    monkeypatch.setattr(runner, "_api_client", lambda *args: service)
    with pytest.raises(module.ClientError, match="lost successful response"):
        runner._admit(module, private_work, campaign, _bindings())
    runner._admit(module, private_work, campaign, _bindings())
    runner._admit(module, private_work, campaign, _bindings())
    states = [module.read_json(path) for path in (private_work / "jobs").glob("*.json")]
    assert (
        len(service.jobs) == len(states) == len({row["job_id"] for row in states}) == 12
    )
    assert len(service.posts) == 13 and len(set(service.posts)) == 12
    assert all(
        body["output_policy"]["browser_preview"] == "lossless_rgb_webm"
        for body in service.bodies
    )
    assert all(
        path.stat().st_mode & 0o077 == 0 for path in private_work.rglob("*.json")
    )


def test_resume_recovers_retained_jobs_without_generation(
    campaign, private_work, monkeypatch
):
    module = runner._client_module()
    service = _Service(module)
    service.fail_once = False
    monkeypatch.setattr(runner, "_api_client", lambda *args: service)
    runner._admit(module, private_work, campaign, _bindings())
    posts = len(service.posts)
    recovered = []

    def download(client, job_id, destination):
        recovered.append(job_id)
        return {"job_id": job_id}

    monkeypatch.setattr(module, "download_job", download)
    runner._resume(module, private_work, campaign, _bindings())
    assert len(recovered) == len(set(recovered)) == 12
    assert len(service.posts) == posts


def test_campaign_and_binding_drift_refuse_before_admission(campaign, private_work):
    module = runner._client_module()
    runner._state(module, private_work, campaign, _bindings())
    changed = copy.deepcopy(campaign)
    changed["variants"][0]["prompt"] += " Changed."
    with pytest.raises(ValueError, match="workflow changed"):
        runner._state(module, private_work, changed, _bindings())
    with pytest.raises(ValueError, match="bindings changed"):
        runner._state(
            module,
            private_work,
            campaign,
            {**_bindings(), "output_prefix": "s3://example-bucket/changed"},
        )


def test_private_config_rejects_symlink_and_public_mode(private_work):
    module = runner._client_module()
    path = private_work / "config.json"
    module.save_private(path, _bindings())
    linked = private_work / "linked.json"
    linked.symlink_to(path)
    with pytest.raises(module.ClientError):
        runner._config(module, linked)
    path.chmod(0o644)
    with pytest.raises(module.ClientError):
        runner._config(module, path)


def test_work_directory_rejects_git_checkout_and_symlink(private_work):
    module = runner._client_module()
    checkout = private_work / "checkout"
    checkout.mkdir(mode=0o700)
    subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
    with pytest.raises(ValueError, match="outside Git"):
        runner._workspace(module, checkout)
    link = private_work / "link"
    link.symlink_to(checkout, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        runner._workspace(module, link)


# ----------------------------------------------------------------- server receipts

SAMPLES = [
    {"x": 256, "y": 120, "region": "scene", "rgb": [1, 2, 3]},
    {"x": 48, "y": 240, "region": "border", "rgb": [0, 0, 0]},
]


def _write(path, payload):
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _published_variant(module, work, campaign, variant, index):
    directory = work / "downloads" / variant["id"]
    # Create each level privately, independent of the caller's umask.
    module.private_directory(work / "downloads")
    module.private_directory(directory)
    job_id = f"job-{index}"
    module.save_private(work / "jobs" / (variant["id"] + ".json"), {"job_id": job_id})
    hashes = {
        "source.mp4": _write(directory / "source.mp4", b"same-source"),
        "generated.mp4": _write(
            directory / "generated.mp4", f"finished-{index}".encode()
        ),
        "model-raw.mp4": _write(directory / "model-raw.mp4", f"raw-{index}".encode()),
        "generated-preview.webm": _write(
            directory / "generated-preview.webm", f"preview-{index}".encode()
        ),
        "source-preview.webm": _write(
            directory / "source-preview.webm", b"source-preview"
        ),
    }
    timeline = {
        "decoded_frames": 336,
        "frame_rate": "24/1",
        "width": 832,
        "height": 480,
    }
    report = {
        "policy": {
            "version": "managed-finishing-v1",
            **campaign["output_policy"],
            "customer_fields": sorted(campaign["output_policy"]),
        },
        "source": timeline,
        "finished": timeline,
        "timeline_preserved": True,
        "padding_detection": {"status": "detected", "bounds": [96, 0, 736, 480]},
        "border_preservation": {
            "status": "verified",
            "scene_pixels_unchanged": True,
            "padding_matches_source": True,
        },
        "browser_preview": {
            "status": "verified",
            "rgb_matches_reference_every_frame": True,
            "browser_reference_frame0_samples": SAMPLES,
        },
        "source_browser_preview": {
            "status": "verified",
            "browser_reference_frame0_samples": SAMPLES,
        },
        "semantic_evaluation": {"status": "not_performed"},
        "lineage": {
            "source_sha256": hashes["source.mp4"],
            "model_raw_sha256": hashes["model-raw.mp4"],
            "finished_sha256": hashes["generated.mp4"],
            "browser_preview_sha256": hashes["generated-preview.webm"],
            "source_browser_preview_sha256": hashes["source-preview.webm"],
        },
    }
    hashes["finishing-report.json"] = _write(
        directory / "finishing-report.json", json.dumps(report).encode()
    )
    sampling = campaign["generation"]["sampling"]
    manifest = {
        "job_id": job_id,
        "manifest": {
            "job_id": job_id,
            "model": runner.MODEL,
            "prompt_sha256": hashlib.sha256(variant["prompt"].encode()).hexdigest(),
            "effective": {
                "guardrails": True,
                "structural_control": "edge",
                "effective_generation": {"seed": variant["seed"], **sampling},
            },
            "technical_validation": {
                "complete_decode": True,
                "timeline_coverage": "all source frames via edge control",
            },
            "finishing": {
                "policy": report["policy"],
                "report_sha256": hashes["finishing-report.json"],
                "generation_dispatches": 1,
            },
            "timing": {"finishing_cpu_seconds": 45.0},
            "artifacts": [
                {"name": name, "sha256": digest} for name, digest in hashes.items()
            ],
        },
    }
    module.save_private(directory / "manifest.json", manifest)
    module.save_private(
        directory / "metrics.json",
        {
            "runner_stage_bounds": {
                "super_request": {
                    "started_at_unix_seconds": 100.0 + 200 * index,
                    "completed_at_unix_seconds": 290.0 + 200 * index,
                }
            }
        },
    )
    return directory


@pytest.fixture
def published(campaign, private_work):
    module = runner._client_module()
    for index, variant in enumerate(campaign["variants"]):
        _published_variant(module, private_work, campaign, variant, index)
    return module


def test_collect_verifies_server_receipts_and_reports_concurrency_separately(
    campaign, private_work, published
):
    summary = runner._collect(published, private_work, campaign)
    assert len(summary["variants"]) == 12
    assert summary["customer_media_processing"] == "none"
    assert summary["client_admission_parallelism"] == 12
    assert summary["observed_service_concurrency"] == {
        "max_overlapping_model_requests": 1,
        "measured_jobs": 12,
    }
    assert all(row["border"]["status"] == "verified" for row in summary["variants"])


@pytest.mark.parametrize(
    "mutation",
    ["hash", "policy", "lineage", "border", "preview", "dispatches", "timeline"],
)
def test_collect_rejects_unverified_or_mismatched_service_evidence(
    campaign, private_work, published, mutation
):
    directory = private_work / "downloads" / campaign["variants"][3]["id"]
    manifest = published.read_json(directory / "manifest.json")
    report = json.loads((directory / "finishing-report.json").read_text())
    if mutation == "hash":
        (directory / "generated.mp4").write_bytes(b"tampered")
    elif mutation == "policy":
        manifest["manifest"]["finishing"]["policy"]["browser_preview"] = "none"
    elif mutation == "dispatches":
        manifest["manifest"]["finishing"]["generation_dispatches"] = 2
    else:
        if mutation == "lineage":
            report["lineage"]["model_raw_sha256"] = "0" * 64
        elif mutation == "border":
            report["border_preservation"]["scene_pixels_unchanged"] = False
        elif mutation == "preview":
            report["browser_preview"]["rgb_matches_reference_every_frame"] = False
        else:
            report["timeline_preserved"] = False
        digest = _write(
            directory / "finishing-report.json", json.dumps(report).encode()
        )
        manifest["manifest"]["finishing"]["report_sha256"] = digest
        for item in manifest["manifest"]["artifacts"]:
            if item["name"] == "finishing-report.json":
                item["sha256"] = digest
    published.save_private(directory / "manifest.json", manifest)
    with pytest.raises(ValueError):
        runner._collect(published, private_work, campaign)


def _review(campaign, rejected="variant-11-soft-green-accent"):
    rows = []
    for variant in campaign["variants"]:
        ok = variant["id"] != rejected
        rows.append(
            {
                "variant_id": variant["id"],
                "verdict": "accepted" if ok else "rejected",
                "criteria": {name: ok for name in runner.CRITERIA},
                "notes": "<b>sampled</b> frames" if ok else "spurious object",
            }
        )
    return {"variants": rows}


class _Videos(HTMLParser):
    def __init__(self):
        super().__init__()
        self.videos = []

    def handle_starttag(self, tag, attrs):
        if tag == "video":
            self.videos.append(dict(attrs))


def test_report_embeds_thirteen_server_previews_with_color_references(
    campaign, private_work, published
):
    review = private_work / "review.json"
    published.save_private(review, _review(campaign))
    runner._report(published, private_work, campaign, review)
    document = (private_work / "paidf-managed-endpoint-proof.html").read_text()
    parser = _Videos()
    parser.feed(document)
    assert len(parser.videos) == 13
    assert all(
        video["src"].startswith("data:video/webm;base64,") for video in parser.videos
    )
    assert base64.b64decode(parser.videos[1]["src"].split(",", 1)[1]) == b"preview-0"
    assert json.loads(parser.videos[1]["data-reference-frame0"]) == SAMPLES
    assert (
        "default-src &#x27;none&#x27;" not in document
        and "default-src 'none'" in document
    )
    assert "<b>sampled</b>" not in document and "&lt;b&gt;sampled&lt;/b&gt;" in document
    assert "11 of 12 accepted" in document
    assert "Observed service model-request concurrency" in document


def test_review_does_not_accept_false_criteria(campaign, private_work):
    module = runner._client_module()
    rows = _review(campaign)
    rows["variants"][0]["criteria"]["contacts_physically_plausible"] = False
    path = private_work / "review.json"
    module.save_private(path, rows)
    with pytest.raises(ValueError):
        runner._review(module, path, campaign)
