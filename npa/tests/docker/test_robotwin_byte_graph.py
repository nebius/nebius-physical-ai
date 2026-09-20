"""Verify RoboTwin's Docker compatibility manifest cannot escape OCI coverage."""

import json

import pytest

from test_image_byte_scan import CHECKOUT, file, js, run, tar_data, write
from test_image_byte_scan_oci import authorize, oci_fixture
from image_byte_scan import core as W, robotwin_verification as R


@pytest.mark.parametrize("mutation", [None, "config", "layer", "extra-image"])
def test_robotwin_docker_manifest_is_bound_to_complete_oci_graph(tmp_path, mutation):
    tmp_path.chmod(0o700)
    files, expected, _ = oci_fixture()
    manifest = json.loads(files["blobs/sha256/" + expected["image_manifest_digest"][7:]])
    compatibility = [{
        "Config": "blobs/sha256/" + expected["image_config_digest"][7:],
        "Layers": ["blobs/sha256/" + row["digest"][7:] for row in manifest["layers"]],
        "RepoTags": ["synthetic/robotwin:fixture"],
    }]
    if mutation == "config":
        compatibility[0]["Config"] = "blobs/sha256/" + "0" * 64
    elif mutation == "layer":
        compatibility[0]["Layers"].reverse()
    elif mutation == "extra-image":
        compatibility *= 2
    files["manifest.json"] = js(compatibility)
    with W.authorized_roots(tmp_path, CHECKOUT):
        binding = write(tmp_path / "candidate.tar", tar_data([file(n, b) for n, b in files.items()]))
        if mutation:
            with pytest.raises(W.ScanError, match="oci_docker_"):
                R.verify(binding, expected["expected_image_id"])
            return
        verified = R.verify(binding, expected["expected_image_id"])
        assert verified["regular_files_read"] == 3
        report, records = run(tmp_path, authorize(tmp_path, files, verified))
    assert report["valid"] and report["complete"]
    assert any(row.get("sha256") == W.sha(files["manifest.json"]) for row in records)
