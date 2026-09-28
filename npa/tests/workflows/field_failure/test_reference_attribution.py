"""Verify public scan credit against sealed capture bytes and frozen scene lineage."""

import hashlib
import io
import json
from types import SimpleNamespace

import pytest

from npa.workbench.nurec.navigation_sample import ARCHIVE_SHA256, ARCHIVE_URL
from npa.workflows.field_failure import artifacts
from npa.workflows.field_failure.reference_demo_attribution import sample_credit

ROOT = "s3://example/reference-demo"


@pytest.fixture
def sealed_sample(monkeypatch):
    capture = {
        "source": {
            "dataset": "TUM RGB-D benchmark, fr3/long_office_household",
            "archive_sha256": ARCHIVE_SHA256,
            "url": ARCHIVE_URL,
            "license": "CC-BY-4.0",
            "attribution": "untrusted custom text must not be copied",
            "private_metadata": "custom capture metadata must not be copied",
        },
    }
    objects = {}
    plan = {"cohorts": {"geometry": {}}}

    def seal(value):
        capture_bytes = artifacts._encode(value)
        digest = hashlib.sha256(capture_bytes).hexdigest()
        files = {"capture.json": digest}
        record = {
            "schema": "npa.navigation.publication.v1",
            "attempt": "attempts/" + "a" * 32,
            "files": files,
            "manifest_sha256": artifacts._digest(artifacts._encode(files)),
        }
        for name in ("claim", "completion"):
            objects[ROOT + "/sample/" + name + ".json"] = artifacts._encode(record)
        objects[ROOT + "/sample/" + record["attempt"] + "/capture.json"] = capture_bytes
        plan["cohorts"]["geometry"]["capture_manifest_sha256"] = digest

    seal(capture)
    client = SimpleNamespace(
        get_object=lambda Bucket, Key: {
            "Body": io.BytesIO(objects["s3://" + Bucket + "/" + Key])
        }
    )
    monkeypatch.setattr(artifacts, "_client", lambda: client)
    return SimpleNamespace(objects=objects, plan=plan, capture=capture, seal=seal)


def test_credit_retains_public_license_modifications_and_frozen_identity(sealed_sample):
    credit = sample_credit(ROOT, sealed_sample.plan)
    assert credit["archive_sha256"] == ARCHIVE_SHA256
    assert (
        credit["capture_manifest_sha256"]
        == (sealed_sample.plan["cohorts"]["geometry"]["capture_manifest_sha256"])
    )
    assert "J. Sturm" in credit["attribution"] and "IROS 2012" in credit["attribution"]
    assert credit["license_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert "TSDF" in credit["modifications"] and "JPEG" in credit["modifications"]
    assert "must not be copied" not in json.dumps(credit)


@pytest.mark.parametrize("field", ["dataset", "url", "archive_sha256", "license"])
def test_different_verified_capture_is_not_credited_as_tum(sealed_sample, field):
    sealed_sample.capture["source"][field] = "custom-source"
    sealed_sample.seal(sealed_sample.capture)
    assert sample_credit(ROOT, sealed_sample.plan) is None


@pytest.mark.parametrize("broken", ["capture", "claim", "frozen_geometry", "manifest"])
def test_credit_requires_intact_publication_and_frozen_lineage(sealed_sample, broken):
    if broken == "capture":
        uri = next(
            uri for uri in sealed_sample.objects if uri.endswith("/capture.json")
        )
        sealed_sample.objects[uri] += b" "
    elif broken == "claim":
        sealed_sample.objects[ROOT + "/sample/claim.json"] = b"{}"
    elif broken == "frozen_geometry":
        sealed_sample.plan["cohorts"]["geometry"]["capture_manifest_sha256"] = "b" * 64
    else:
        uri = ROOT + "/sample/completion.json"
        record = json.loads(sealed_sample.objects[uri])
        record["files"]["capture.json"] = "b" * 64
        sealed_sample.objects[uri] = artifacts._encode(record)
    with pytest.raises(ValueError, match="SHA-256|claim|lineage|digest"):
        sample_credit(ROOT, sealed_sample.plan)


def _selection():
    return {
        "eligible": False,
        "baseline_success_rate": 0.0,
        "candidate_success_rate": 0.0,
        "success_rate_gain": 0.0,
        "reasons": ["Synthetic test comparison did not improve"],
        "final_cohort_consumed": False,
        "regional": None,
        "paired": {"passed": True, "paired_cases": 2, "violations": []},
    }


def _preview():
    from PIL import Image
    from npa.workflows.preview_html import image_preview

    return [
        {
            "title": "Synthetic rendering fixture",
            "frames": [
                {
                    "label": "Unit frame",
                    "images": [
                        {
                            "label": "Unit image",
                            "data": image_preview(Image.new("RGB", (2, 2))),
                        }
                    ],
                }
            ],
        }
    ]


def test_published_standalone_html_retains_verified_public_credit(
    sealed_sample, monkeypatch
):
    from npa.workflows.field_failure import reference_demo_report as report
    from npa.workflows.field_failure import reference_demo_media as media

    sealed_sample.plan.update(num_envs=2, baseline_iterations=1, candidate_iterations=1)
    sealed_sample.objects[ROOT + "/reference-plan.json"] = artifacts._encode(
        sealed_sample.plan
    )
    published = {}

    def put(data, uri, **kwargs):
        assert kwargs["if_none_match"] and uri not in published
        published[uri] = data

    storage = SimpleNamespace(put_bytes_conditional=put)
    for module in (report, artifacts):
        monkeypatch.setattr(module, "_storage", lambda: storage)
    monkeypatch.setattr(media, "preview_groups", lambda *_: _preview())
    result = report.publish_report(
        SimpleNamespace(output_root=ROOT, run_id="unit"), selection=_selection()
    )
    rendered = published[ROOT + "/reports/index.html"].decode()
    assert "J. Sturm" in rendered and "creativecommons.org/licenses/by/4.0/" in rendered
    assert "TSDF" in rendered and "JPEG" in rendered
    assert "must not be copied" not in rendered and "s3://" not in rendered
    assert (
        json.loads(published[ROOT + "/reports/result.json"])["public_sample"]
        == result["public_sample"]
    )
