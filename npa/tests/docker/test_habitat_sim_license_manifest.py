"""Validate the exact Habitat source, wheel, and runtime licensing closure."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "npa/docker/workbench/habitat-sim"


def _wheel_lock(name: str) -> dict[str, str]:
    result = {}
    for line in (PACKAGE / name).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        package, digest = line.split(" --hash=sha256:")
        result[package.lower().replace("_", "-")] = digest
    return result


def test_every_locked_wheel_has_exact_license_bytes_and_role() -> None:
    licenses = json.loads((PACKAGE / "licenses.json").read_text())
    rows = licenses["python_wheels"]
    observed = {
        f"{row['name'].lower().replace('_', '-')}=={row['version']}": row
        for row in rows
    }
    expected = {
        **_wheel_lock("requirements-build.lock"),
        **_wheel_lock("requirements-runtime.lock"),
    }
    assert set(observed) == set(expected)
    for package, digest in expected.items():
        row = observed[package]
        assert row["sha256"] == digest
        assert row["license"] and row["role"]
        assert row["license_sha256"]
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", value) for value in row["license_sha256"]
        )


def test_source_license_manifest_matches_exact_selected_dependencies() -> None:
    source = json.loads((PACKAGE / "source-manifest.json").read_text())
    licenses = json.loads((PACKAGE / "licenses.json").read_text())
    expected = {
        (source["source"]["repository"], source["source"]["revision"]),
        *((item["repository"], item["revision"]) for item in source["dependencies"]),
    }
    assert len(expected) == 15
    license_names = {row["name"] for row in licenses["source_licenses"]}
    assert {item["name"] for item in source["dependencies"]} <= license_names
    assert "habitat-sim" in license_names
    assert next(
        row for row in licenses["source_licenses"] if row["name"] == "imath"
    ) == {
        "name": "imath",
        "license": "BSD-3-Clause",
        "sha256": "c20236d3b39fd20eba8e3d1fb3b892a5483df2e7d8d61bf43f165d3fac22f601",
    }


def test_runtime_asset_is_not_misclassified_as_baked() -> None:
    licenses = json.loads((PACKAGE / "licenses.json").read_text())
    text = json.dumps(licenses)
    assert licenses["classification"] == "public-eligible-unbuilt"
    assert "Skokloster scene archive or members" in licenses["not_baked"]
    assert "credentials" in text and "generated outputs" in text


def test_pbr_projection_keeps_mit_and_cc0_dispositions_separate() -> None:
    licenses = json.loads((PACKAGE / "licenses.json").read_text())
    source = json.loads((PACKAGE / "source-manifest.json").read_text())
    pbr = licenses["pbr_source_projection"]
    configuration = set(pbr["habitat_configuration"]["files"])
    lut = set(pbr["brdf_lut"]["files"])
    environment_maps = set(pbr["environment_maps"]["files"])
    required = {row["path"] for row in source["source"]["required_projection_files"]}
    assert pbr["habitat_configuration"]["license"] == "MIT"
    assert pbr["brdf_lut"]["license"] == "MIT"
    assert pbr["environment_maps"]["license"] == "CC0-1.0"
    assert configuration | lut | environment_maps | {"data/pbr/license.txt"} == required
    assert not configuration & lut
    assert not configuration & environment_maps
    assert not lut & environment_maps
    assert pbr["upstream_notice"] == {
        "path": "data/pbr/license.txt",
        "bytes": 1416,
        "sha256": "63c9792282c792c05c4b39704b52329b59cace1957b06db4c731b5bd58428238",
    }
    notices = (PACKAGE / "THIRD_PARTY_NOTICES.md").read_text()
    assert "BRDF lookup texture — MIT" in notices
    assert "Poly Haven environment maps — CC0 1.0" in notices
    assert "/usr/src/habitat-sim/data/pbr/license.txt" in notices


def test_rsync_reciprocal_source_is_accompanied_not_merely_offered() -> None:
    licenses = json.loads((PACKAGE / "licenses.json").read_text())
    rsync = next(row for row in licenses["ubuntu_sources"] if row["name"] == "rsync")
    assert rsync == {
        "name": "rsync",
        "license": "GPL-3.0+",
        "role": "runtime-bootstrap",
        "source_delivery": "accompanying-exact-source",
        "source_lock": "apt-runtime.lock#corresponding_sources[rsync]",
        "image_path": "/usr/share/doc/npa-habitat-sim/ubuntu-sources/rsync",
    }
    assert "source_offer_required" not in json.dumps(licenses)


def test_runtime_lock_contains_matplotlibs_pinned_packaging_dependency() -> None:
    build = _wheel_lock("requirements-build.lock")
    runtime = _wheel_lock("requirements-runtime.lock")
    assert runtime["packaging==26.3"] == build["packaging==26.3"]
    licenses = json.loads((PACKAGE / "licenses.json").read_text())
    packaging = next(
        row for row in licenses["python_wheels"] if row["name"] == "packaging"
    )
    assert packaging["role"] == "build-and-runtime"


def test_pillow_runtime_wheel_is_fixed_and_carries_exact_bundled_notices() -> None:
    licenses = json.loads((PACKAGE / "licenses.json").read_text())
    pillow = next(row for row in licenses["python_wheels"] if row["name"] == "pillow")
    assert pillow == {
        "name": "pillow",
        "version": "12.3.0",
        "sha256": "f0606c8bf2cdefea14a43530f7657cbbb7ecf1c4222512492ef4a4434a9501ec",
        "license": "MIT-CMU AND bundled notices",
        "license_sha256": [
            "dda12a98c1979cf3d94df1cff45d27a4cb3f04a60c76f76902ac54cac03ec0ce"
        ],
        "role": "runtime",
    }
