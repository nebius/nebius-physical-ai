"""Guard the sim2real no-registry image-pin fallbacks against drift.

``sim2real/constants.py`` carries no-registry fallback tags that must stay in
sync with the canonical ``[tool.npa.supported-tools]`` pins in ``pyproject.toml``
(mirrored by ``deploy/images.py``). The tag-audit script only matches
fully-qualified ``npa-<tool>:<tag>`` references, so it cannot catch drift in
these bare constants — this test closes that gap.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
import yaml

from npa.deploy.images import supported_tool_version
from npa.workflows.sim2real import constants


_EXACT_SOURCE_DOCKERFILES = (
    "cosmos2-transfer/Dockerfile",
    "cosmos3-reason/Dockerfile",
    "isaac-lab/Dockerfile",
    "lerobot-vlm-rl/Dockerfile",
    "rerun-viewer/Dockerfile",
    "sim2real-envgen/Dockerfile",
    "sim2real-eval/Dockerfile",
    "sim2real-control/Dockerfile",
)

_STANDARD_WORKFLOW_PASSTHROUGH_DOCKERFILES = (
    "cosmos3-reason/Dockerfile",
    "isaac-lab/Dockerfile",
    "sim2real-control/Dockerfile",
    "sim2real-envgen/Dockerfile",
    "sim2real-eval/Dockerfile",
)


@pytest.mark.parametrize(
    ("constant_name", "tool"),
    [
        ("DEFAULT_ENVGEN_TAG", "envgen"),
        ("DEFAULT_REFERENCE_POLICY_TAG", "reference-policy"),
        ("DEFAULT_TRAINER_TAG", "lerobot-vlm-rl"),
        ("DEFAULT_EVAL_TAG", "loop-eval"),
    ],
)
def test_sim2real_constant_matches_supported_tool_version(
    constant_name: str, tool: str
) -> None:
    constant_value = getattr(constants, constant_name)
    assert constant_value == supported_tool_version(tool), (
        f"{constant_name}={constant_value!r} drifted from canonical "
        f"{tool}={supported_tool_version(tool)!r} (pyproject supported-tools)"
    )


def test_canonical_sim2real_workflow_requires_operator_pinned_images() -> None:
    """The standard workflow must not supply mutable image-tag fallbacks."""

    path = (
        Path(__file__).resolve().parents[2]
        / "workflows"
        / "workbench"
        / "npa-workflows"
        / "sim2real.yaml"
    )
    runbook = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = runbook["config"]
    image_inputs = (
        "controller_image",
        "transfer_image",
        "envgen_image",
        "isaac_image",
        "viewer_image",
    )
    assert config["require_baked_npa"] == "1"
    assert config["baked_npa_import"] == "npa.workflows.sim2real.workflow_stage"
    assert all(config[name] == "" for name in image_inputs)
    resources = runbook["resources"]
    assert all(
        str(resource["image"]).startswith("{{config.")
        for resource in resources.values()
    )


@pytest.mark.parametrize("relative_path", _EXACT_SOURCE_DOCKERFILES)
def test_exact_source_images_copy_forced_workflow_package_data(
    relative_path: str,
) -> None:
    """An exact-source image must be installable from its minimal build context."""

    dockerfile = (
        Path(__file__).resolve().parents[2] / "docker" / "workbench" / relative_path
    ).read_text(encoding="utf-8")
    assert "COPY" in dockerfile
    assert "workflows /opt/npa/workflows" in dockerfile
    assert "pyproject.toml /opt/npa/pyproject.toml" in dockerfile


def test_isaac_exact_source_image_uses_light_package_imports() -> None:
    """Isaac's purpose-built venv must not require unrelated NPA SDK extras."""

    dockerfile = (
        Path(__file__).resolve().parents[2]
        / "docker"
        / "workbench"
        / "isaac-lab"
        / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert "NPA_SKIP_EAGER_IMPORTS=1" in dockerfile
    assert "NPA_BAKED_PYTHON=/opt/npa/sim/venv/bin/python" in dockerfile
    assert "npa-exact-source.pth" in dockerfile
    assert "env -u PYTHONPATH /opt/npa/sim/venv/bin/python -c" in dockerfile
    assert (
        "import boto3, kubernetes, mcap, npa.workflows.sim2real.runtime_attestation"
        in (dockerfile)
    )


def test_cpu_controller_is_small_pinned_and_resolver_closed() -> None:
    root = Path(__file__).resolve().parents[2] / "docker" / "workbench"
    dockerfile = (root / "sim2real-control" / "Dockerfile").read_text()
    assert "python:3.11-slim-trixie@sha256:" in dockerfile
    assert "npa-sim2real-control" not in dockerfile
    assert "Genesis" not in dockerfile
    assert "CUDA" not in dockerfile
    assert "--no-deps" in dockerfile
    assert "pip check" in dockerfile
    assert "NPA_SOURCE_SHA" in dockerfile
    assert "NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}" in dockerfile
    assert "NPA_SKIP_EAGER_IMPORTS=1" in dockerfile
    assert "ARG DEBIAN_SNAPSHOT=20260801T000000Z" in dockerfile
    assert "snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}" in dockerfile
    assert "npa-exact-source.pth" in dockerfile
    assert "env -u PYTHONPATH python -c" in dockerfile
    for prerequisite in (
        "openssh-server",
        "rsync",
        "sudo",
        "netcat-openbsd",
        "NOPASSWD",
    ):
        assert prerequisite in dockerfile

    repair_dockerfile = (
        root / "sim2real-control" / "Dockerfile.k8s-prereqs"
    ).read_text()
    assert "npa-exact-source.pth" in repair_dockerfile
    assert "env -u PYTHONPATH python3 -c" in repair_dockerfile

    requirements = (
        root / "common" / "sim2real-controller-requirements.txt"
    ).read_text()
    lines = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines
    assert all(line.count("==") == 1 for line in lines)
    assert "httpx==0.28.1" in lines
    assert "PyYAML==6.0.3" in lines
    assert "from npa.clients.token_factory import TokenFactoryClient" in dockerfile
    assert "NEBIUS_TOKEN_FACTORY_KEY" in dockerfile
    assert "build-smoke" in dockerfile
    assert "TokenFactoryClient()" in dockerfile


def test_isaac_cache_warmer_is_nonroot_and_uses_fs_group() -> None:
    root = Path(__file__).resolve().parents[2] / "docker" / "workbench"
    documents = list(
        yaml.safe_load_all((root / "common" / "warm-isaac-cache.yaml").read_text())
    )
    job = next(item for item in documents if item and item.get("kind") == "Job")
    pod = job["spec"]["template"]["spec"]
    security = pod["securityContext"]
    assert security["runAsNonRoot"] is True
    assert (
        security["runAsUser"] == security["runAsGroup"] == security["fsGroup"] == 1000
    )
    assert security["seccompProfile"] == {"type": "RuntimeDefault"}
    container = pod["containers"][0]
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]


def test_standard_workflow_entrypoint_execs_orchestrator_argv() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "docker/workbench/common/workflow_runtime_entrypoint.sh"
    )
    result = subprocess.run(
        ["bash", str(script), "/bin/sh", "-c", "printf standard-workflow-ready"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == "standard-workflow-ready"


def test_rerun_viewer_is_exact_source_stage14_runtime() -> None:
    root = Path(__file__).resolve().parents[2] / "docker" / "workbench"
    dockerfile = (root / "rerun-viewer" / "Dockerfile").read_text(encoding="utf-8")
    requirements = (root / "common" / "sim2real-viewer-requirements.txt").read_text(
        encoding="utf-8"
    )
    assert 'org.opencontainers.image.revision="${NPA_SOURCE_SHA}"' in dockerfile
    assert "NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}" in dockerfile
    assert "NPA_BAKED_PYTHON=/opt/rerun/venv/bin/python" in dockerfile
    assert "npa-exact-source.pth" in dockerfile
    assert "sim2real-viewer-requirements.txt" in dockerfile
    assert "--no-deps" in dockerfile
    assert "pip check" in dockerfile
    assert (
        'ENTRYPOINT ["/opt/npa/docker/workbench/rerun-viewer/entrypoint.sh"]'
        in dockerfile
    )
    for prerequisite in ("openssh-server", "rsync", "sudo", "netcat-openbsd"):
        assert prerequisite in dockerfile
    lines = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert all(line.count("==") == 1 for line in lines)
    for dependency in ("boto3==1.43.62", "mcap==1.4.0", "rerun-sdk==0.31.4"):
        assert dependency in lines


@pytest.mark.parametrize("relative_path", _STANDARD_WORKFLOW_PASSTHROUGH_DOCKERFILES)
def test_standard_workflow_images_use_passthrough_entrypoint(
    relative_path: str,
) -> None:
    dockerfile = (
        Path(__file__).resolve().parents[2] / "docker/workbench" / relative_path
    ).read_text(encoding="utf-8")
    assert "workflow_runtime_entrypoint.sh" in dockerfile
    assert 'ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]' in dockerfile
    assert 'CMD ["--help"]' not in dockerfile


def test_cosmos2_exact_source_image_uses_light_package_imports() -> None:
    """Transfer's minimal environment must not require unrelated NPA SDK extras."""

    dockerfile = (
        Path(__file__).resolve().parents[2]
        / "docker"
        / "workbench"
        / "cosmos2-transfer"
        / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert "NPA_SKIP_EAGER_IMPORTS=1" in dockerfile
    assert "import npa.workflows.sim2real.runtime_attestation" in dockerfile


@pytest.mark.parametrize(
    "relative_path",
    (
        "cosmos3-reason/Dockerfile",
        "lerobot-vlm-rl/Dockerfile",
        "sim2real-envgen/Dockerfile",
        "sim2real-eval/Dockerfile",
        "isaac-lab/Dockerfile",
    ),
)
def test_exact_source_runtime_installs_are_resolver_closed(relative_path: str) -> None:
    """Specialized images may add only exact packages without transitive drift."""

    dockerfile = (
        Path(__file__).resolve().parents[2] / "docker" / "workbench" / relative_path
    ).read_text(encoding="utf-8")
    assert "--no-deps" in dockerfile
    assert "pip check" in dockerfile
    assert (
        "find /opt/npa/src /opt/npa/workflows -type d -exec chmod a+rx {} +"
        in dockerfile
    )
    assert (
        "find /opt/npa/src /opt/npa/workflows -type f -exec chmod a+r {} +"
        in dockerfile
    )
    assert "chmod -R" not in dockerfile
    assert "pip install tomli " not in dockerfile
    assert ">=" not in dockerfile


def test_cosmos3_system_packages_use_an_immutable_snapshot() -> None:
    dockerfile = (
        Path(__file__).resolve().parents[2]
        / "docker"
        / "workbench"
        / "cosmos3-reason"
        / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert "ARG UBUNTU_SNAPSHOT=20260801T053000Z" in dockerfile
    assert "https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" in dockerfile
    assert "rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list" in dockerfile
    assert "pip install --no-deps -e /opt/npa" not in dockerfile
    assert "PYTHONPATH=/opt/npa/src" in dockerfile
    for requirement in (
        "tomli==2.4.1",
        "accelerate==1.14.0",
        "huggingface_hub==0.36.0",
        "qwen-vl-utils==0.0.14",
        "safetensors==0.8.0",
        "transformers==4.57.6",
    ):
        assert requirement in dockerfile


def test_cosmos3_baked_runtime_survives_skypilot_pythonpath_scrubbing() -> None:
    """The immutable Reason image must import its stage after Sky removes PYTHONPATH.

    SkyPilot deliberately launches managed tasks with ``env -u PYTHONPATH``.  A source
    tree copied into the image is therefore not enough: the interpreter selected by
    ``require_baked_npa`` needs an image-local path record too.  Compositional live
    attempt 16 reached both real Stage 8 siblings before exposing this boundary.
    """

    dockerfile = (
        Path(__file__).resolve().parents[2]
        / "docker"
        / "workbench"
        / "cosmos3-reason"
        / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert "NPA_BAKED_PYTHON=/opt/npa/venv/bin/python" in dockerfile
    assert "python -m pip uninstall -y npa" in dockerfile
    assert "npa-exact-source.pth" in dockerfile
    assert "env -u PYTHONPATH /opt/npa/venv/bin/python -c" in dockerfile
    assert "from npa.workflows.sim2real.workflow_stage import main" in dockerfile


def test_envgen_baked_runtime_survives_skypilot_pythonpath_scrubbing() -> None:
    """EnvGen's exact source remains importable when Sky clears PYTHONPATH."""

    dockerfile = (
        Path(__file__).resolve().parents[2]
        / "docker"
        / "workbench"
        / "sim2real-envgen"
        / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert "NPA_BAKED_PYTHON=/opt/npa/venv/bin/python" in dockerfile
    assert "npa-exact-source.pth" in dockerfile
    assert "env -u PYTHONPATH /opt/npa/venv/bin/python -c" in dockerfile
    assert "from npa.workflows.sim2real.workflow_stage import main" in dockerfile


def test_cosmos3_reason_has_complete_skypilot_bootstrap_runtime() -> None:
    dockerfile = (
        Path(__file__).resolve().parents[2]
        / "docker"
        / "workbench"
        / "cosmos3-reason"
        / "Dockerfile"
    ).read_text(encoding="utf-8")
    for package in ("openssh-server", "procps", "rsync", "sudo"):
        assert package in dockerfile
    assert "rm -f /etc/ssh/ssh_host_*" in dockerfile


def test_sim2real_control_plane_requirement_closure_is_exact() -> None:
    requirements = (
        Path(__file__).resolve().parents[2]
        / "docker"
        / "workbench"
        / "common"
        / "sim2real-control-requirements.txt"
    ).read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines
    assert all(line.count("==") == 1 for line in lines)
    for expected in (
        "kubernetes==33.1.0",
        "mcap==1.4.0",
        "durationpy==0.10",
        "google-auth==2.56.3",
        "cryptography==50.0.0",
        "cffi==2.1.1",
        "pycparser==3.0",
        "lz4==4.4.5",
        "zstandard==0.25.0",
    ):
        assert expected in lines

    root = Path(__file__).resolve().parents[2] / "docker" / "workbench"
    for relative_path in ("lerobot-vlm-rl/Dockerfile", "isaac-lab/Dockerfile"):
        dockerfile = (root / relative_path).read_text(encoding="utf-8")
        assert "sim2real-control-requirements.txt" in dockerfile
        assert "--no-deps" in dockerfile

    isaac_dockerfile = (root / "isaac-lab" / "Dockerfile").read_text(encoding="utf-8")
    assert (
        "find /opt/npa/src /opt/npa/workflows -type d -exec chmod a+rx {} +"
        in isaac_dockerfile
    )
    assert (
        "find /opt/npa/src /opt/npa/workflows -type f -exec chmod a+r {} +"
        in isaac_dockerfile
    )
    assert "chmod -R" not in isaac_dockerfile
    assert "PYTHONPATH=/opt/npa/src" in isaac_dockerfile

    genesis_requirements = (
        root / "common" / "sim2real-genesis-requirements.txt"
    ).read_text(encoding="utf-8")
    genesis_lines = [
        line.strip()
        for line in genesis_requirements.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert genesis_lines == [
        "huggingface-hub==0.35.3",
        "tomli==2.4.1",
    ]
    for relative_path in (
        "sim2real-envgen/Dockerfile",
        "sim2real-eval/Dockerfile",
        "lerobot-vlm-rl/Dockerfile",
    ):
        dockerfile = (root / relative_path).read_text(encoding="utf-8")
        assert "sim2real-control-requirements.txt" in dockerfile
        assert "sim2real-genesis-requirements.txt" in dockerfile
        assert "python -m pip uninstall -y transformers" in dockerfile
        assert 'd.metadata["Name"].lower() == "transformers"' in dockerfile
        assert "pip check" in dockerfile
