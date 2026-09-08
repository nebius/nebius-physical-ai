"""Verify the public Cosmos bootstrap preserves runtime and publication boundaries."""

from __future__ import annotations

import re
import importlib.util
import subprocess
import hashlib
from pathlib import Path

import yaml
import pytest

ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = ROOT / "npa/docker/workbench/cosmos3-super-benchmark"
DOCKERFILE = IMAGE_DIR / "Dockerfile"
BUILD = IMAGE_DIR / "build.sh"
CONTRACT = ROOT / "npa/docker/workbench/packaging-contract.yaml"
PUBLIC_PARENT_DIGEST = (
    "sha256:3342bbe44bd1c00ebf05ab4c9d7286058a94bb5ce90b49b164b23604d3acf180"
)


def test_wrapper_inherits_public_bootstrap_and_defers_serving_payload() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert f"npa-cosmos3-serving@{PUBLIC_PARENT_DIGEST}" in text
    assert "FROM ${BASE_IMAGE}" in text
    assert "vllm/vllm-omni:" not in text
    assert "-r /opt/npa-cosmos3-serving/packaging-requirements.txt" in text
    assert not re.search(r"pip install[^\n]*(torch|vllm|cuda)", text)
    assert 'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"' in text
    assert "openssh-server rsync sudo" in text
    assert "dpkg-query -W openssh-server rsync sudo" in text
    assert "sudo service ssh start; sudo service ssh status" in text
    assert "sudo service ssh stop; sudo rm -f /etc/ssh/ssh_host_*" in text
    assert 'org.opencontainers.image.revision="${NPA_SOURCE_SHA}"' in text
    assert not re.search(r"(?i)(HF_TOKEN|NGC_API_KEY|ACCEPT.*=YES)", text)
    assert "USER ubuntu" in text
    assert "UV_CACHE_DIR=/home/ubuntu/.cache/uv" in text
    assert 'ENTRYPOINT ["/usr/local/bin/npa-cosmos3-super-benchmark-entrypoint"]' in text


def test_wrapper_is_public_runtime_fetch_with_notices() -> None:
    from npa.deploy.images import (
        RESTRICTED_PUBLICATION_TOOLS,
        is_publicly_redistributable,
    )

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    entry = contract["images"]["cosmos3-super-benchmark"]
    assert entry["redistribution"] == "public"
    assert "cosmos3-super-benchmark" not in RESTRICTED_PUBLICATION_TOOLS
    assert is_publicly_redistributable("cosmos3-super-benchmark") is True
    assert (IMAGE_DIR / "REDISTRIBUTION.md").is_file()
    assert (IMAGE_DIR / "THIRD_PARTY_NOTICES.md").is_file()
    assert (IMAGE_DIR / "LICENSE-APACHE-2.0").read_bytes() == (ROOT / "LICENSE").read_bytes()


def _guardrail_policy():
    path = IMAGE_DIR / "prepare_guardrail_runtime.py"
    spec = importlib.util.spec_from_file_location("cosmos_guardrail_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_disabled_guardrails_do_not_fetch_assets(monkeypatch):
    policy = _guardrail_policy()
    monkeypatch.delenv("NPA_COSMOS3_SERVE_GUARDRAILS", raising=False)
    monkeypatch.setattr(policy.subprocess, "run", lambda *a, **kw: pytest.fail("download"))
    assert policy.main() == 0


def test_enabled_guardrails_preserve_materializer_failure(monkeypatch):
    policy = _guardrail_policy()
    monkeypatch.setenv("NPA_COSMOS3_SERVE_GUARDRAILS", "on")

    def fail(command, **options):
        assert command[1].endswith("/prepare_guardrail_payload.py")
        assert options == {"check": True}
        raise subprocess.CalledProcessError(3, command)

    monkeypatch.setattr(policy.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        policy.main()


def test_invalid_guardrail_mode_fails_closed(monkeypatch):
    policy = _guardrail_policy()
    monkeypatch.setenv("NPA_COSMOS3_SERVE_GUARDRAILS", "yes")
    monkeypatch.setattr(policy.subprocess, "run", lambda *a, **kw: pytest.fail("download"))
    with pytest.raises(ValueError, match="on or off"):
        policy.main()


def test_local_builder_refuses_publication_before_invoking_docker() -> None:
    text = BUILD.read_text(encoding="utf-8")
    result = subprocess.run(["bash", str(BUILD), "--push"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "trusted" in result.stderr
    assert "^dev-[0-9a-f]{40}$" in text
    assert "env -u HF_TOKEN -u NGC_API_KEY -u NEBIUS_IAM_TOKEN" in text


def test_bootstrap_shell_commands_forward_without_model_or_license_access():
    result = subprocess.run(
        ["bash", str(IMAGE_DIR / "entrypoint.sh"), "/bin/sh", "-c",
         'printf "%s\\n" "$@"', "sentinel", "spaced argument", "--flag"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == ["spaced argument", "--flag"]


def test_runtime_mode_requires_a_command_before_fetching():
    result = subprocess.run(
        ["bash", str(IMAGE_DIR / "entrypoint.sh"), "--runtime"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "requires a workload command" in result.stderr


@pytest.mark.parametrize("selection", ["NO", "", "invalid"])
@pytest.mark.parametrize("tool", ["cosmos3-super-benchmark", "cosmos3-nano-video"])
def test_runtime_entrypoints_reject_opt_out_before_touching_cache(monkeypatch, tool, selection):
    monkeypatch.setenv("NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE", selection)
    if tool == "cosmos3-super-benchmark":
        command = ["bash", str(IMAGE_DIR / "entrypoint.sh"), "--runtime", "true"]
    else:
        command = ["bash", str(IMAGE_DIR.parent / tool / "runtime_bootstrap.sh"), "true"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 78
    assert "declined" in result.stderr or "must be YES or NO" in result.stderr


def test_effective_runtime_bootstrap_hashes_match_reviewed_source():
    serving = IMAGE_DIR.parent / "cosmos3-serving"
    sources = {name: serving / name for name in [
        "requirements.lock", "runtime_bootstrap.sh", "entrypoint.sh", "smoke_serving.sh",
        "access_preflight.py", "hf_snapshot_pin.py", "backport_setuptools_manifest.py", "verify_env.py",
    ]}
    sources["prepare_guardrail_payload.py"] = serving / "prepare_guardrail_runtime.py"
    sources["prepare_guardrail_runtime.py"] = IMAGE_DIR / "prepare_guardrail_runtime.py"
    sources["packaging-requirements.txt"] = IMAGE_DIR.parent / "sonic/packaging-requirements.txt"
    expected = {f"/opt/npa-cosmos3-serving/{name}": hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in sources.items()}
    rows = [line.split() for line in (IMAGE_DIR / "bootstrap-source-sha256s.txt").read_text().splitlines()]
    assert {path: digest for digest, path in rows} == expected


@pytest.mark.parametrize("tool", ["cosmos3-super-benchmark", "cosmos3-nano-video"])
def test_replacement_overlay_updates_parent_pins_and_checks_runtime_as_nonroot(tool):
    docker = (IMAGE_DIR.parent / tool / "Dockerfile").read_text()
    serving = (IMAGE_DIR.parent / "cosmos3-serving/Dockerfile").read_text()
    names = ("VLLM_OMNI_REVISION", "VLLM_OMNI_SOURCE_SHA256", "COSMOS3_CLOSURE_SHA256", "DEBIAN_SNAPSHOT")
    for name in names:
        expected = re.search(rf"(?m)^ARG {name}=(.+)$", serving).group(1)
        assert f"ARG {name}={expected}" in docker
    assert "apt-get upgrade -y" in docker
    assert "snapshot.debian.org/archive/debian(-security)?/" in docker
    assert 'npa.vllm_omni.revision="${VLLM_OMNI_REVISION}"' in docker
    runtime_user = "USER ubuntu" if tool == "cosmos3-super-benchmark" else "USER 10001:10001"
    after_user = docker.split(runtime_user)[-1]
    assert "test -w /opt/npa-cosmos3-serving/runtime" in after_user
    assert "sha256sum -c /opt/npa-cosmos3-serving/bootstrap-source-sha256s.txt" in after_user
    assert "python /opt/npa-cosmos3-serving/verify_env.py" in after_user


@pytest.mark.parametrize("tool", ["cosmos3-super-benchmark", "cosmos3-nano-video"])
def test_overlay_snapshot_substitution_updates_both_inherited_repositories(tool):
    docker = (IMAGE_DIR.parent / tool / "Dockerfile").read_text()
    snapshot = re.search(r"(?m)^ARG DEBIAN_SNAPSHOT=(.+)$", docker).group(1)
    expression = re.search(r'&& sed -ri ("[^"\n]+")', docker).group(1)
    inherited = (
        "Types: deb\n"
        "URIs: https://snapshot.debian.org/archive/debian/20260817T000000Z\n"
        "Suites: bookworm bookworm-updates\n\n"
        "Types: deb\n"
        "URIs: http://snapshot.debian.org/archive/debian-security/20260817T000000Z\n"
        "Suites: bookworm-security\n"
    )
    result = subprocess.run(
        ["bash", "-c", f"sed -r {expression}"],
        env={"DEBIAN_SNAPSHOT": snapshot}, input=inherited,
        capture_output=True, text=True, check=True,
    )
    assert result.stdout == inherited.replace("20260817T000000Z", snapshot)
