"""Exercise the live test's real encoder and reporting without network access."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

pytest_plugins = ["pytester"]

_LIVE_TEST = Path(__file__).parent / "e2e" / "test_convert_rrd_s3_live_e2e.py"
_DESTINATIONS = {
    "dataset": "s3://synthetic-private-bucket/private-dataset.rrd",
    "predictions": "s3://synthetic-private-bucket/private-predictions.rrd",
}
_CONFIG_ERRORS = {
    "missing-file": "Cannot read the private destination file",
    "malformed-json": "Cannot read the private destination file",
    "invalid-uri": "Each destination must be an exact S3 RRD object URI",
    "permissions": "Destination file must be owner-only",
    "duplicate": "Each conversion mode requires its own destination object",
}


@pytest.mark.parametrize("scenario", ["success", "existing", "read-error", "upload-error", "corrupt", *_CONFIG_ERRORS])
def test_private_live_reports_and_storage_guards(pytester, monkeypatch, scenario: str) -> None:
    pytester.makepyfile(test_private_live=_LIVE_TEST.read_text())
    pytester.makeini("[pytest]\nmarkers = e2e: opted-in integration test\njunit_family = legacy\n")
    destination_file = pytester.path / "destinations.json"
    destination_file.write_text(json.dumps(_DESTINATIONS))
    destination_file.chmod(0o600)
    if scenario == "missing-file":
        destination_file.unlink()
    elif scenario == "malformed-json":
        destination_file.write_text('{"dataset": "' + _DESTINATIONS["dataset"])
    elif scenario == "invalid-uri":
        destination_file.write_text(json.dumps({**_DESTINATIONS, "dataset": _DESTINATIONS["dataset"] + "?private"}))
    elif scenario == "permissions":
        destination_file.chmod(0o644)
    elif scenario == "duplicate":
        destination_file.write_text(json.dumps(dict.fromkeys(_DESTINATIONS, _DESTINATIONS["dataset"])))
    monkeypatch.setenv("NPA_INTEGRATION_E2E", "1")
    monkeypatch.setenv("NPA_E2E_CONVERT_RRD_S3_DESTINATIONS_FILE", str(destination_file))
    monkeypatch.setenv("SYNTHETIC_RRD_SCENARIO", scenario)
    pytester.makeconftest(
        """
import importlib
import io
import logging
import os
from pathlib import Path
import warnings

import pytest
from botocore.exceptions import ClientError
from botocore.response import StreamingBody
from npa.clients.storage import StorageClient

@pytest.fixture(autouse=True)
def synthetic_storage(monkeypatch, request):
    scenario = os.environ["SYNTHETIC_RRD_SCENARIO"]
    calls = Path(__file__).with_name("upload-calls")
    objects = {}

    def noisy_failure(bucket, key):
        uri = f"s3://{bucket}/{key}"
        print(uri)
        os.write(2, uri.encode())
        logging.error(uri)
        warnings.warn(uri)
        raise OSError(uri)

    class Transport:
        def get_object(self, *, Bucket, Key):
            if scenario == "existing":
                return {"Body": io.BytesIO(b"existing"), "ETag": '"existing"'}
            if scenario == "read-error":
                noisy_failure(Bucket, Key)
            if Key not in objects:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            payload = b"invalid RRD bytes" if scenario == "corrupt" else objects[Key]
            return {"Body": StreamingBody(io.BytesIO(payload), len(payload)), "ETag": '"synthetic"'}

        def upload_file(self, local_file, bucket, key):
            with calls.open("a") as stream:
                stream.write("upload\\n")
            if scenario == "upload-error":
                noisy_failure(bucket, key)
            objects[key] = Path(local_file).read_bytes()

    storage = StorageClient.__new__(StorageClient)
    storage._s3 = Transport()
    monkeypatch.setattr(request.module, "_storage_client", lambda *_args: storage)
    for name in ("lerobot_to_rerun", "groot_predictions_to_rerun"):
        adapter = importlib.import_module(f"npa.viz.adapters.{name}")
        monkeypatch.setattr(adapter, "_storage_client", lambda *_args: storage)
"""
    )
    result = pytester.runpytest_subprocess(
        "-q", "--showlocals", "--junitxml=report.xml", "-o", "junit_logging=all",
        "-o", "log_cli=true", "-o", "log_cli_level=DEBUG",
    )
    if scenario == "success":
        result.assert_outcomes(passed=2)
    elif scenario in _CONFIG_ERRORS:
        result.assert_outcomes(errors=2)
        result.stdout.fnmatch_lines([f"*{_CONFIG_ERRORS[scenario]}*"])
    else:
        result.assert_outcomes(failed=2)
    if scenario == "existing":
        result.stdout.fnmatch_lines(["*Refusing to overwrite an existing destination object*"])
    elif scenario in {"read-error", "upload-error", "corrupt"}:
        error_type = "AssertionError" if scenario == "corrupt" else "OSError"
        result.stdout.fnmatch_lines([f"*Private S3 upload/readback failed ({error_type})*"])
    calls = pytester.path / "upload-calls"
    assert (len(calls.read_text().splitlines()) if calls.exists() else 0) == (
        0 if scenario in {"existing", "read-error", *_CONFIG_ERRORS} else 2
    )
    report = (pytester.path / "report.xml").read_text()
    output = result.stdout.str() + result.stderr.str() + report
    for private_value in (*_DESTINATIONS.values(), "synthetic-private-bucket", "private-dataset.rrd", "private-predictions.rrd"):
        assert private_value not in output
    if scenario == "success":
        cases = ElementTree.fromstring(report).findall(".//testcase")
        assert len(cases) == 2
        for case in cases:
            properties = {entry.attrib["name"]: entry.attrib["value"] for entry in case.findall("./properties/property")}
            assert len(properties["rrd_sha256"]) == 64
            assert int(properties["rrd_bytes"]) > 0
            assert properties["decoded_frames_per_entity"] == "4"
            assert properties["decoded_joint_entities"] == ("2" if "predictions" in case.attrib["name"] else "1")
