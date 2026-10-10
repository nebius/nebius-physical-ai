"""Exact causal package update cannot silently resolve different bytes."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "docker/workbench/curobo"
SPEC = importlib.util.spec_from_file_location(
    "curobo_security_apt", ROOT / "install_security_apt.py"
)
assert SPEC and SPEC.loader
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


@pytest.fixture
def manifest():
    return json.loads((ROOT / "security-apt.lock.json").read_text())


def test_manifest_retains_exact_binary_and_corresponding_source_inputs(manifest):
    assert [row["name"] for row in installer.verify_manifest(manifest)] == [
        "libssl3t64",
        "openssl",
    ]
    assert manifest["source"]["version"] == "3.0.13-0ubuntu3.16"
    assert len(manifest["source"]["files"]) == 3
    assert (
        manifest["binary_index_sha256"]
        == "9736348a03098fc12dd70f3eec09501a08717a81d232fc87e2bc0ac22ef1425b"
    )
    assert (
        manifest["source_index_sha256"]
        == "e732731f8822786bd5605666694da905c5ecd9d535775f8de9f78368c797683f"
    )


@pytest.mark.parametrize(
    "mutation",
    ["schema", "extra", "missing", "version", "architecture", "url", "digest", "size"],
)
def test_unexpected_manifest_rejected_before_network_or_process(
    manifest, monkeypatch, mutation
):
    monkeypatch.setattr(installer, "_download_https", lambda *_: pytest.fail("network"))
    monkeypatch.setattr(
        installer.subprocess, "run", lambda *_a, **_k: pytest.fail("process")
    )
    if mutation == "schema":
        manifest["schema"] = "different"
    elif mutation == "extra":
        manifest["packages"].append(copy.deepcopy(manifest["packages"][0]))
    elif mutation == "missing":
        manifest["packages"].pop()
    else:
        key = {"digest": "sha256", "size": "bytes"}.get(mutation, mutation)
        manifest["packages"][0][key] = -1 if key == "bytes" else "wrong"
    with pytest.raises(ValueError):
        installer.install(manifest)


@pytest.mark.parametrize("failure", [None, "digest", "control", "installed"])
def test_installer_verifies_bytes_metadata_and_final_state(
    manifest, monkeypatch, failure
):
    data = {
        row["url"]: f"synthetic {row['name']} package".encode()
        for row in manifest["packages"]
    }
    for row in manifest["packages"]:
        row["sha256"] = hashlib.sha256(data[row["url"]]).hexdigest()
        row["bytes"] = len(data[row["url"]])
    calls = []
    monkeypatch.setattr(
        installer,
        "_download_https",
        lambda url: data[url] + (b"corrupt" if failure == "digest" else b""),
    )

    def check_output(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "dpkg-deb":
            name = Path(argv[2]).name.split("_")[0]
            assert Path(argv[2]).read_bytes() == next(
                data[row["url"]] for row in manifest["packages"] if row["name"] == name
            )
            return f"Package: {name}\nVersion: {'wrong' if failure == 'control' else '3.0.13-0ubuntu3.16'}\nArchitecture: amd64\n"
        assert argv[:2] == ["dpkg-query", "--show"]
        return (
            "wrong"
            if failure == "installed"
            else "libssl3t64\t3.0.13-0ubuntu3.16\tamd64\nopenssl\t3.0.13-0ubuntu3.16\tamd64\n"
        )

    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[:2] == ["dpkg", "--install"] and len(argv) == 4
        assert kwargs == {"check": True}
        assert all(Path(p).is_file() for p in argv[2:])

    monkeypatch.setattr(installer.subprocess, "check_output", check_output)
    monkeypatch.setattr(installer.subprocess, "run", run)
    if failure:
        with pytest.raises(ValueError):
            installer.install(manifest)
        if failure in ("digest", "control"):
            assert not any(call[0] == "dpkg" for call in calls)
    else:
        assert installer.install(manifest)["installed_verified"] is True
        assert len(calls) == 4
