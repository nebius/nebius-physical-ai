"""Reject evidence that misattributes a workflow or breaks the offline viewer."""

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[3] / "workflows/cosmos-data-factory"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "cdf_proof", DIRECTORY / "build-proof.py"
)
PACKER = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(PACKER)


@pytest.fixture
def evidence():
    """Return minimal author metadata without fabricated run results."""
    return {
        "assets": [{"id": "seed", "mime_type": "image/jpeg"}],
        "robot_assets": ["seed"],
        "workflows": [
            {
                "id": id,
                "spec_filename": filename,
                "evidence": {
                    "media_ids": ["seed"],
                    "source": {
                        "url": "https://example.org/source",
                        "media_ids": ["seed"],
                    },
                    "comparisons": [{"media_ids": ["seed"]}],
                },
            }
            for id, filename in PACKER.WORKFLOW_SPECS.items()
        ],
    }


@pytest.mark.parametrize("id,states", [("iaa", 9), ("evg", 12)])
def test_workflow_metadata_comes_from_matching_spec(id, states):
    row = {"id": id, "spec_filename": PACKER.WORKFLOW_SPECS[id]}
    assert len(PACKER._workflow_details(row, DIRECTORY)["stages"]) == states
    row["spec_filename"] = PACKER.WORKFLOW_SPECS["evg" if id == "iaa" else "iaa"]
    with pytest.raises(ValueError, match="ID must match"):
        PACKER._workflow_details(row, DIRECTORY)


@pytest.mark.parametrize("location", ["robot", "workflow", "source", "comparison"])
def test_unknown_media_rejected_before_embedding(evidence, tmp_path, location):
    details = evidence["workflows"][0]["evidence"]
    references = {
        "robot": evidence["robot_assets"],
        "workflow": details["media_ids"],
        "source": details["source"]["media_ids"],
        "comparison": details["comparisons"][0]["media_ids"],
    }
    references[location].append("missing")
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="Unknown embedded media IDs"):
        PACKER._report_data(path, DIRECTORY)


def test_duplicate_asset_ids_rejected(evidence):
    evidence["assets"].append(dict(evidence["assets"][0]))
    with pytest.raises(ValueError, match="must be unique"):
        PACKER._validate_media_references(evidence)


@pytest.mark.parametrize(
    "url", ["javascript:alert(1)", "data:text/html,test", "//example.org", "relative"]
)
def test_non_http_source_links_rejected(evidence, url):
    evidence["workflows"][0]["evidence"]["source"]["url"] = url
    with pytest.raises(ValueError, match="absolute HTTP or HTTPS"):
        PACKER._validate_media_references(evidence)


def test_image_only_watchpoints_rejected(evidence):
    evidence["workflows"][0]["evidence"]["explanation"] = {
        "watchpoints": [{"seconds": 0}],
    }
    with pytest.raises(ValueError, match="require an embedded workflow video"):
        PACKER._validate_media_references(evidence)


def test_reviewed_media_and_video_watchpoints_accepted(evidence):
    evidence["assets"].append({"id": "clip", "mime_type": "video/mp4"})
    details = evidence["workflows"][1]["evidence"]
    details["media_ids"].append("clip")
    details["explanation"] = {"watchpoints": [{"seconds": 0}]}
    PACKER._validate_media_references(evidence)


@pytest.mark.parametrize("failure", ["hash", "mime", "traversal", "symlink"])
def test_unqualified_media_rejected(tmp_path, failure):
    evidence_directory = tmp_path / "evidence"
    evidence_directory.mkdir()
    media = evidence_directory / "seed.jpg"
    media.write_bytes(b"fixture-media")
    asset = {
        "id": "seed",
        "path": "seed.jpg",
        "sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
        "mime_type": "image/jpeg",
        "alt": "fixture",
        "caption": "fixture",
    }
    messages = {
        "hash": "Asset hash mismatch",
        "mime": "Unsupported embedded media",
        "traversal": "within the evidence directory",
        "symlink": "within the evidence directory",
    }
    if failure == "hash":
        media.write_bytes(b"modified-media")
    elif failure == "mime":
        asset["mime_type"] = "text/plain"
    else:
        outside = tmp_path / "outside.jpg"
        outside.write_bytes(media.read_bytes())
        if failure == "traversal":
            asset["path"] = "../outside.jpg"
        else:
            media.unlink()
            media.symlink_to(outside)
    with pytest.raises(ValueError, match=messages[failure]):
        PACKER._embedded_asset(asset, evidence_directory)
