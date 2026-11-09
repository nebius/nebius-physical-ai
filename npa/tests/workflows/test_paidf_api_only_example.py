"""Exercise durable fanout, media integrity and offline HTML for the hosted example."""

import copy
import hashlib
from html.parser import HTMLParser
import importlib.util
from pathlib import Path
import shutil
import subprocess
import threading

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "npa/examples/paidf-cosmos3-api-only"
SPEC = importlib.util.spec_from_file_location(
    "paidf_api_only_example", EXAMPLE / "run.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def campaign():
    return runner._campaign(EXAMPLE / "campaign.yaml")


@pytest.fixture
def private_work(tmp_path):
    tmp_path.chmod(0o700)
    return tmp_path


def _bindings():
    return {
        "wmh_base_url": "https://api.example.test",
        "wmh_api_key_file": "/private/key.txt",
        "input_connection_id": "input",
        "output_connection_id": "output",
        "source_uri": "s3://example-bucket/source.mp4",
        "output_prefix": "s3://example-bucket/output",
    }


def _contract():
    return {
        "auth": {"owner_id": "test-owner"},
        "openapi": {"paths": {runner.ROUTE: {"post": {}}}},
        "models": {
            "data": [
                {
                    "id": runner.MODEL,
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

    def __init__(self, module):
        self.module = module
        self.jobs = {}
        self.lock = threading.Lock()
        self.fail_once = True
        self.posts = []
        self.calls = []

    def contract(self):
        return _contract()

    def request(self, method, route, payload=None, identity=None):
        with self.lock:
            self.calls.append((method, route))
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
                self.jobs.setdefault(identity, "job-" + str(len(self.jobs) + 1))
                if self.fail_once:
                    self.fail_once = False
                    raise self.module.ClientError("lost successful response")
                return {"job_id": self.jobs[identity]}
            return {"status": "succeeded"}


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
    assert len(service.posts) == 13
    assert len(set(service.posts)) == 12
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


def test_unqualified_service_never_admits_paid_work(
    campaign, private_work, monkeypatch
):
    module = runner._client_module()
    service = _Service(module)
    contract = _contract()
    contract["models"]["data"][0]["capability_matrix"][0]["runtime_implemented"] = False
    monkeypatch.setattr(service, "contract", lambda: contract)
    monkeypatch.setattr(runner, "_api_client", lambda *args: service)
    with pytest.raises(module.ClientError, match="qualified"):
        runner._admit(module, private_work, campaign, _bindings())
    assert not service.posts


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


@pytest.mark.parametrize("mutation", ["duplicate", "unsafe", "nan"])
def test_invalid_campaign_refuses(campaign, private_work, mutation):
    if mutation == "duplicate":
        campaign["variants"][1]["seed"] = campaign["variants"][0]["seed"]
    elif mutation == "unsafe":
        campaign["variants"][0]["id"] = "../../escape"
    else:
        campaign["generation"]["sampling"]["guidance_scale"] = float("nan")
    path = private_work / "campaign.yaml"
    path.write_text(yaml.safe_dump(campaign))
    with pytest.raises(ValueError):
        runner._campaign(path)


@pytest.fixture
def media_campaign(campaign, private_work):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for real media verification")
    module = runner._client_module()
    video = private_work / "fixture.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=32x32:r=24:d=0.25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    reviews = []
    for index, variant in enumerate(campaign["variants"]):
        directory = private_work / "downloads" / variant["id"]
        module.private_directory(directory)
        for name in ("source.mp4", "generated.mp4"):
            module.save_private(directory / name, video.read_bytes())
        job = "job-" + str(index)
        module.save_private(
            private_work / "jobs" / (variant["id"] + ".json"), {"job_id": job}
        )
        published = {
            "job_id": job,
            "model": runner.MODEL,
            "prompt_sha256": hashlib.sha256(variant["prompt"].encode()).hexdigest(),
            "effective": {
                "guardrails": True,
                "structural_control": "edge",
                "effective_generation": {
                    "seed": variant["seed"],
                    **campaign["generation"]["sampling"],
                },
            },
            "technical_validation": {
                "complete_decode": True,
                "timeline_coverage": True,
            },
            "artifacts": [
                {"name": name, "sha256": runner._sha256(directory / name)}
                for name in ("source.mp4", "generated.mp4")
            ],
        }
        module.save_private(
            directory / "manifest.json", {"job_id": job, "manifest": published}
        )
        reviews.append(
            {
                "variant_id": variant["id"],
                "verdict": "accepted",
                "criteria": dict.fromkeys(runner.CRITERIA, True),
                "notes": "<script>private note</script>",
            }
        )
    review_path = private_work / "review.json"
    module.save_private(review_path, {"variants": reviews})
    return module, review_path


class _Videos(HTMLParser):
    def __init__(self):
        super().__init__()
        self.videos = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        if tag == "video":
            self.videos.append(dict(attrs)["src"])
        if tag == "script":
            self.scripts.append(attrs)


def test_report_embeds_thirteen_verified_videos_and_escapes_notes(
    campaign, private_work, media_campaign
):
    module, review_path = media_campaign
    runner._report(module, private_work, campaign, review_path)
    output = private_work / "paidf-cosmos3-proof.html"
    parser = _Videos()
    parser.feed(output.read_text())
    assert len(parser.videos) == 13
    assert all(url.startswith("data:video/mp4;base64,") for url in parser.videos)
    assert not parser.scripts
    assert "&lt;script&gt;private note&lt;/script&gt;" in output.read_text()
    assert output.stat().st_mode & 0o077 == 0
    assert len(module.read_json(private_work / "technical.json")["variants"]) == 12


def test_report_rechecks_media_instead_of_trusting_old_technical_evidence(
    campaign, private_work, media_campaign
):
    module, review_path = media_campaign
    runner._collect(module, private_work, campaign)
    output = (
        private_work / "downloads" / campaign["variants"][0]["id"] / "generated.mp4"
    )
    output.write_bytes(b"corrupted after successful collection")
    with pytest.raises(ValueError, match="manifest"):
        runner._report(module, private_work, campaign, review_path)
    assert not (private_work / "paidf-cosmos3-proof.html").exists()


def test_review_does_not_accept_false_criteria(campaign, private_work):
    module = runner._client_module()
    records = [
        {
            "variant_id": row["id"],
            "verdict": "accepted",
            "notes": "Inspected sampled frames.",
            "criteria": dict.fromkeys(runner.CRITERIA, True),
        }
        for row in campaign["variants"]
    ]
    records[0]["criteria"]["contacts_physically_plausible"] = False
    path = private_work / "review.json"
    module.save_private(path, {"variants": records})
    with pytest.raises(ValueError, match="accepted review"):
        runner._review(module, path, campaign)
    records[0]["verdict"] = "rejected"
    module.save_private(path, {"variants": records})
    assert (
        runner._review(module, path, campaign)[records[0]["variant_id"]]["verdict"]
        == "rejected"
    )
