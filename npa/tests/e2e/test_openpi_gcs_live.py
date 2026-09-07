"""Actual public DROID staging through the OpenPI storage preparation path."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import sys

import pytest

from npa.workflows.byof import openpi_full_droid as droid
from npa.workflows.byof import openpi_gcs as gcs


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.public_inputs,
    pytest.mark.skipif(os.environ.get("NPA_INTEGRATION_E2E") != "1",
                       reason="requires explicit public GCS live validation"),
]


def test_real_pinned_filter_dictionary_staging_and_cache_reuse(tmp_path):
    """Download the real pinned 28 MB mapping and validate its complete identity."""
    executable = tmp_path / "npa-openpi-gcs"
    source = Path(droid.__file__).with_name("openpi_gcs.py")
    executable.write_text(
        "#!/bin/sh\nexec " + shlex.join([sys.executable, str(source)]) + ' "$@"\n'
    )
    executable.chmod(0o700)
    cache = droid._configure_openpi_cache(tmp_path / "work")
    first = droid._stage_filter_dictionary(str(executable), cache)
    second = droid._stage_filter_dictionary(str(executable), cache)
    assert first["size_bytes"] == droid.FILTER_DICTIONARY_BYTES
    assert first["sha256"] == droid.FILTER_DICTIONARY_SHA256
    assert first["entry_count"] > 0
    assert first["cache_reused"] is False
    assert second["cache_reused"] is True
    evidence = {
        "status": "completed", "component": "OpenPI public DROID filter staging",
        "size_bytes": first["size_bytes"], "sha256": first["sha256"],
        "entry_count": first["entry_count"], "verified_cache_reuse": True,
    }
    (tmp_path / "validation.json").write_text(json.dumps(evidence, indent=2))


def test_real_public_prefix_inventory_sync_and_corruption_repair(tmp_path, capsys):
    """Exercise provider pagination, checksum synchronization and actual repair."""
    source = droid.FILTER_DICTIONARY_URI.rsplit("/", 1)[0]
    gcs.run(["ls", "-l", "-r", source + "/**"])
    listing = capsys.readouterr().out.splitlines()
    expected_count = len(listing)
    expected_bytes = sum(int(line.split()[0]) for line in listing)
    assert expected_count >= 1
    assert expected_bytes >= droid.FILTER_DICTIONARY_BYTES
    destination = tmp_path / "dataset"
    gcs.run(["-m", "rsync", "-r", "-c", source, str(destination)])
    files = [path for path in destination.rglob("*") if path.is_file()]
    assert len(files) == expected_count
    assert sum(path.stat().st_size for path in files) == expected_bytes
    target = destination / droid.FILTER_DICTIONARY_URI.rsplit("/", 1)[1]
    assert droid._validate_filter_dictionary(target)["sha256"] == droid.FILTER_DICTIONARY_SHA256
    with target.open("r+b") as handle:
        handle.write(b"corrupt")
    gcs.run(["-m", "rsync", "-r", "-c", source, str(destination)])
    assert droid._validate_filter_dictionary(target)["sha256"] == droid.FILTER_DICTIONARY_SHA256
    evidence = {
        "status": "completed", "component": "OpenPI public GCS checksum synchronization",
        "object_count": expected_count, "size_bytes": expected_bytes,
        "verified_corruption_repair": True,
    }
    (tmp_path / "validation.json").write_text(json.dumps(evidence, indent=2))
