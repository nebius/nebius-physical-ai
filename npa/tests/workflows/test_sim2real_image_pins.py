"""Guard the sim2real no-registry image-pin fallbacks against drift.

``sim2real/constants.py`` carries no-registry fallback tags that must stay in
sync with the canonical ``[tool.npa.supported-tools]`` pins in ``pyproject.toml``
(mirrored by ``deploy/images.py``). The tag-audit script only matches
fully-qualified ``npa-<tool>:<tag>`` references, so it cannot catch drift in
these bare constants — this test closes that gap.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

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
    "rerun-viewer/Dockerfile",
)

_STANDARD_WORKFLOW_PASSTHROUGH_DOCKERFILES = (
    "cosmos3-reason/Dockerfile",
    "isaac-lab/Dockerfile",
    "sim2real-control/Dockerfile",
    "sim2real-envgen/Dockerfile",
    "sim2real-eval/Dockerfile",
)

_ENVGEN_PARENT = (
    "ghcr.io/nebius/nebius-physical-ai/npa-envgen@"
    "sha256:55f1541c1a86e9d865963753cfa38fa1dbf88928c8d28d9ee30e2851a5604b88"
)


def test_rerun_viewer_bakes_exact_stage14_runtime() -> None:
    dockerfile = (
        Path(__file__).resolve().parents[2]
        / "docker"
        / "workbench"
        / "rerun-viewer"
        / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert "NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}" in dockerfile
    assert "NPA_BAKED_PYTHON=/opt/rerun/venv/bin/python" in dockerfile
    assert "sim2real-controller-requirements.txt" in dockerfile
    assert "sim2real-control-requirements.txt" in dockerfile
    assert "env -u PYTHONPATH python -c" in dockerfile


@pytest.mark.parametrize(
    ("constant_name", "tool"),
    [
        ("DEFAULT_ENVGEN_TAG", "envgen"),
        ("DEFAULT_REFERENCE_POLICY_TAG", "reference-policy"),
        ("DEFAULT_TRAINER_TAG", "lerobot-vlm-rl"),
        ("DEFAULT_EVAL_TAG", "loop-eval"),
        ("DEFAULT_ISAAC_TAG", "isaac-lab"),
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

    path = Path(__file__).resolve().parents[3] / "workflows" / "main" / "sim2real.yaml"
    runbook = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = runbook["config"]
    image_inputs = (
        "controller_image",
        "cosmos3_image",
        "envgen_image",
        "isaac_image",
        "viewer_image",
    )
    assert config["require_baked_npa"] == "1"
    assert config["baked_npa_import"] == "npa.workflows.sim2real.workflow_stage"
    assert all(config[name] == "" for name in image_inputs)
    assert config["required_immutable_images"] == list(image_inputs)
    resources = runbook["resources"]
    assert all(
        str(resource["image"]).startswith("{{config.")
        for resource in resources.values()
    )


def test_envgen_requires_an_explicit_source_attested_parent() -> None:
    root = Path(__file__).resolve().parents[2] / "docker" / "workbench"
    text = (root / "sim2real-envgen/Dockerfile").read_text(encoding="utf-8")
    assert "ARG BASE_IMAGE\nFROM ${BASE_IMAGE} AS sanitized" in text
    assert "ARG BASE_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-genesis@" not in text
    assert "npa-envgen@sha256:" not in text


def test_envgen_bootstrap_tools_use_the_current_secure_floors() -> None:
    """Fail before image publication if a scanned bootstrap tool is stale."""

    root = Path(__file__).resolve().parents[2] / "docker" / "workbench"
    text = (root / "sim2real-envgen/Dockerfile").read_text(encoding="utf-8")
    assert "'pip==26.2.2' 'setuptools==84.1.0'" in text
    assert "pip==25.3" not in text
    assert "setuptools==79.0.1" not in text
    assert "setuptools==83.0.0" not in text
    # uv venvs have no pip by default.  The secure bootstrap above cannot run
    # in a clean image unless the venv is explicitly seeded first.
    assert 'uv venv --seed --python "${ENVGEN_PYTHON_VERSION}"' in text


def test_cosmos3_reason_repairs_inherited_scanner_critical_python_packages() -> None:
    """The Stage 3 final image must exclude vulnerable parent-image history."""

    root = Path(__file__).resolve().parents[2] / "docker" / "workbench"
    text = (root / "cosmos3-reason/Dockerfile").read_text(encoding="utf-8")

    # pip 26.2 inherited a scanner-visible vendored certifi 2026.06.17 even
    # though the application package below was current.  Pinning only the
    # application package leaves that complete-byte vulnerability in the
    # published image, so the bootstrap itself must be upgraded too.
    assert '"pip==26.2.2"' in text
    assert '"certifi==2026.10.19"' in text
    assert '"Pillow==13.0.0"' in text
    assert '"cryptography==50.0.2"' in text
    assert '"cffi==2.1.1"' in text
    assert '"pycparser==3.0"' in text
    assert '"Pillow==12.2.0"' not in text
    assert "FROM ${BASE_IMAGE} AS sanitized" in text
    final_stage = text.split("FROM scratch AS runtime", maxsplit=1)[1]
    assert "COPY --from=sanitized / /" in final_stage
    for contract in (
        "NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}",
        "NPA_BAKED_PYTHON=/opt/npa/venv/bin/python",
        "NVIDIA_VISIBLE_DEVICES=all",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        'NVIDIA_REQUIRE_CUDA="cuda>=13.0',
        'ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]',
    ):
        assert contract in final_stage


def test_sim2real_manual_uses_canonical_cosmos_and_cache_warm_roles() -> None:
    """A fresh operator must not be directed to a mismatched image or raw Job."""

    guide = (
        Path(__file__).resolve().parents[3]
        / "docs/workbench/guides/sim2real-workflow.md"
    ).read_text(encoding="utf-8")

    assert "| `COSMOS3_IMAGE` | `npa-cosmos3`" in guide
    assert "The cache-warming Job below is created directly by" not in guide
    assert "Do not add a direct `kubectl` cache-warming\nJob." in guide
    assert "`stage-00-isaac-cache-warm`" in guide


@pytest.mark.parametrize(
    ("relative_path", "child_image"),
    (
        ("lerobot-vlm-rl/Dockerfile", "npa-lerobot-vlm-rl"),
        ("sim2real-eval/Dockerfile", "npa-loop-eval"),
    ),
)
def test_component_images_pin_the_rebuilt_envgen_parent(
    relative_path: str, child_image: str
) -> None:
    root = Path(__file__).resolve().parents[2] / "docker" / "workbench"
    text = (root / relative_path).read_text(encoding="utf-8")
    assert f"ARG BASE_IMAGE={_ENVGEN_PARENT}" in text
    assert 'npa.base.image="npa-envgen"' in text
    assert f"{child_image}@sha256:" not in text
    assert "pip install --no-deps -e /opt/npa" not in text
    assert "env -u PYTHONPATH python -c" in text


def test_sim2real_build_selects_a_source_attested_envgen_parent() -> None:
    script = (
        Path(__file__).resolve().parents[2] / "docker/workbench/sim2real-build.sh"
    ).read_text(encoding="utf-8")
    assert 'SIM2REAL_ENVGEN_BASE_IMAGE="${SIM2REAL_ENVGEN_BASE_IMAGE:-}"' in script
    assert (
        "SKIP_COSMOS3_REASON requires an explicit SIM2REAL_ENVGEN_BASE_IMAGE" in script
    )
    assert 'SIM2REAL_ENVGEN_BASE_IMAGE="npa-cosmos3-reason:${VLM_TAG}"' in script
    assert (
        "SIM2REAL_ENVGEN_BASE_IMAGE must carry org.opencontainers.image.revision="
        in script
    )
    assert "require_current_secure_genesis_closure" not in script
    assert "SIM2REAL_GPU_BASE_IMAGE" not in script
    assert "GENESIS_IMAGE" not in script
    assert "npa-loop-eval@sha256:" not in script
    assert (
        'SIM2REAL_COMPONENT_BASE_IMAGE="${SIM2REAL_COMPONENT_BASE_IMAGE:-npa-envgen:${ENVGEN_TAG}}"'
        in script
    )
    assert script.count('"BASE_IMAGE=${SIM2REAL_COMPONENT_BASE_IMAGE}"') == 2
    assert "BUILD_LEGACY_GENESIS_EVAL:-0" in script
    assert "--envgen-only" in script


def test_sim2real_build_help_does_not_execute_its_parent_example() -> None:
    script = Path(__file__).resolve().parents[2] / "docker/workbench/sim2real-build.sh"
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=script.parents[3]
    ).strip()
    result = subprocess.run(
        ["bash", str(script), "--help"],
        cwd=script.parents[3],
        env={**os.environ, "NPA_SOURCE_SHA": source_sha},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    assert "npa-cosmos3-reason" in result.stdout
    assert "Genesis" in result.stdout


def test_sim2real_build_requires_an_explicit_parent_when_cosmos_is_skipped() -> None:
    """Skipping the producer must not silently select a stale EnvGen parent."""

    script = Path(__file__).resolve().parents[2] / "docker/workbench/sim2real-build.sh"
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=script.parents[3]
    ).strip()
    env = {
        **os.environ,
        "NPA_SOURCE_SHA": source_sha,
        "SIM2REAL_ENVGEN_BASE_IMAGE": "",
        "SKIP_COSMOS3_REASON": "test",
    }
    result = subprocess.run(
        ["bash", str(script)],
        cwd=script.parents[3],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert (
        "SKIP_COSMOS3_REASON requires an explicit SIM2REAL_ENVGEN_BASE_IMAGE"
        in result.stderr
    )
    assert source_sha in result.stderr


def test_sim2real_build_rejects_a_stale_explicit_envgen_parent_before_any_build(
    tmp_path,
) -> None:
    """A supplied local tag is not evidence unless its source label matches."""

    script = Path(__file__).resolve().parents[2] / "docker/workbench/sim2real-build.sh"
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=script.parents[3]
    ).strip()
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nprintf '%s\\n' 'e'$(printf 'e%.0s' $(seq 1 39))\n")
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script)],
        cwd=script.parents[3],
        env={
            **os.environ,
            "NPA_SOURCE_SHA": source_sha,
            "SIM2REAL_ENVGEN_BASE_IMAGE": "local-envgen-parent",
            "SKIP_COSMOS3_REASON": "test",
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert (
        "SIM2REAL_ENVGEN_BASE_IMAGE must carry org.opencontainers.image.revision="
        in result.stderr
    )
    assert "found e" in result.stderr


def test_sim2real_build_does_not_apply_the_genesis_closure_to_envgen(
    tmp_path: Path,
) -> None:
    """A source-attested EnvGen parent may reach the normal build path."""

    script = Path(__file__).resolve().parents[2] / "docker/workbench/sim2real-build.sh"
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=script.parents[3]
    ).strip()
    invoked = tmp_path / "docker-invoked"
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        f"if [ \"$1\" = image ]; then printf '%s\\n' '{source_sha}'; exit 0; fi\n"
        f"touch {invoked}\nexit 99\n"
    )
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script)],
        cwd=script.parents[3],
        env={
            **os.environ,
            "NPA_SOURCE_SHA": source_sha,
            "SIM2REAL_ENVGEN_BASE_IMAGE": "local-envgen-parent",
            "SKIP_COSMOS3_REASON": "test",
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 99
    assert invoked.exists()


def test_sim2real_build_envgen_only_builds_its_current_source_parent_and_no_siblings(
    tmp_path: Path,
) -> None:
    """The trusted EnvGen route needs the producer, but no unrelated roles."""

    script = Path(__file__).resolve().parents[2] / "docker/workbench/sim2real-build.sh"
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=script.parents[3]
    ).strip()
    docker_log = tmp_path / "docker-argv"
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        f"if [ \"$1\" = image ]; then printf '%s\\n' '{source_sha}'; exit 0; fi\n"
        f"printf '%s\\n' \"$@\" >> '{docker_log}'\n"
    )
    docker.chmod(0o755)
    tag = f"dev-{source_sha}"
    result = subprocess.run(
        ["bash", str(script), "--registry", "registry.example", "--envgen-only"],
        cwd=script.parents[3],
        env={
            **os.environ,
            "NPA_SOURCE_SHA": source_sha,
            "VLM_TAG": tag,
            "ENVGEN_TAG": tag,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    argv = docker_log.read_text(encoding="utf-8")
    assert argv.count("build\n") == 2
    assert "cosmos3-reason/Dockerfile" in argv
    assert "sim2real-envgen/Dockerfile" in argv
    assert f"registry.example/npa-envgen:{tag}" in argv
    assert "npa-sim2real-control" not in argv
    assert "npa-reference-policy" not in argv
    assert "npa-lerobot-vlm-rl" not in argv
    assert "npa-loop-eval" not in argv
    assert "npa-rerun-viewer" not in argv


def test_trusted_public_envgen_build_uses_the_source_attested_parent_route() -> None:
    """The publisher must not call EnvGen's mandatory-parent Dockerfile alone."""

    workflow = (
        Path(__file__).resolve().parents[3]
        / ".github/workflows/publish-public-images.yml"
    ).read_text(encoding="utf-8")
    route = 'if [[ "$TOOL" == envgen ]]; then'
    assert route in workflow
    assert "npa/docker/workbench/sim2real-build.sh" in workflow
    assert '--registry "$NPA_PUBLIC_REGISTRY" --envgen-only' in workflow
    assert 'VLM_TAG="dev-${DEVELOPMENT_SHA}"' in workflow
    assert 'ENVGEN_TAG="dev-${DEVELOPMENT_SHA}"' in workflow
    assert workflow.index(route) < workflow.index("docker buildx build")


def test_trusted_public_envgen_build_step_executes_the_parent_route(
    tmp_path: Path,
) -> None:
    """The publisher's EnvGen branch creates both local source-attested roles."""

    root = Path(__file__).resolve().parents[3]
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=root
    ).strip()
    workflow = yaml.safe_load(
        (root / ".github/workflows/publish-public-images.yml").read_text(
            encoding="utf-8"
        )
    )
    build_step = next(
        step
        for step in workflow["jobs"]["build-development"]["steps"]
        if step.get("name") == "Build immutable development image locally"
    )
    docker_log = tmp_path / "docker-argv"
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        f"if [ \"$1\" = image ]; then printf '%s\\n' '{source_sha}'; exit 0; fi\n"
        f"printf '%s\\n' \"$@\" >> '{docker_log}'\n"
    )
    docker.chmod(0o755)
    image = f"registry.example/npa-envgen:dev-{source_sha}"
    result = subprocess.run(
        ["bash", "-c", build_step["run"]],
        cwd=root,
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "TOOL": "envgen",
            "DEVELOPMENT_SHA": source_sha,
            "NPA_PUBLIC_REGISTRY": "registry.example",
            "IMAGE": image,
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    argv = docker_log.read_text(encoding="utf-8")
    assert argv.count("build\n") == 2
    assert "cosmos3-reason/Dockerfile" in argv
    assert "sim2real-envgen/Dockerfile" in argv
    assert "buildx" not in argv
    assert image in argv


def test_sonic_keeps_an_explicit_amd64_build_platform() -> None:
    dockerfile = (
        Path(__file__).resolve().parents[2] / "docker/workbench/sonic/Dockerfile"
    ).read_text(encoding="utf-8")
    assert "ARG NPA_BUILD_PLATFORM=linux/amd64" in dockerfile
    assert "FROM --platform=${NPA_BUILD_PLATFORM} ${BASE_IMAGE}" in dockerfile


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


def test_isaac_exact_source_image_imports_the_runtime_stage() -> None:
    """Isaac's purpose-built venv must satisfy the exact SkyPilot stage probe."""

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
        "import boto3, kubernetes, mcap, "
        "npa.workflows.sim2real.cache_warm, "
        "npa.workflows.sim2real.runtime_attestation, "
        "npa.workflows.sim2real.workflow_stage" in (dockerfile)
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
    assert "sim2real-controller-requirements.txt" in dockerfile
    assert "sim2real-control-requirements.txt" in dockerfile
    assert "from npa.clients.token_factory import TokenFactoryClient" in dockerfile
    assert (
        "from npa.workbench.cosmos.reason import run_token_factory_rollout_vlm"
        in dockerfile
    )
    assert "from npa.workflows.sim2real import robot_contract" in dockerfile
    assert "from npa.workflows.sim2real.stage8_cosmos3 import run" in dockerfile
    assert "NPA_SOURCE_SHA" in dockerfile
    assert "NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}" in dockerfile
    assert "NPA_SKIP_EAGER_IMPORTS=1" in dockerfile
    assert "ARG DEBIAN_SNAPSHOT=20261002T000000Z" in dockerfile
    assert "ARG MIN_PERL_BASE_VERSION=5.40.1-6+deb13u1" in dockerfile
    assert "apt-get upgrade -y --no-install-recommends" in dockerfile
    assert 'dpkg --compare-versions "${installed}" ge "$2"' in dockerfile
    assert "snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}" in dockerfile
    assert "npa-exact-source.pth" in dockerfile
    assert "env -u PYTHONPATH python -c" in dockerfile
    for prerequisite in (
        "openssh-server",
        "rsync",
        "sudo",
        "netcat-openbsd",
        "NOPASSWD",
        "rm -f /etc/ssh/ssh_host_*",
        "PasswordAuthentication no",
        "PermitRootLogin no",
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
    assert "defusedxml==0.7.1" in lines
    assert "from npa.clients.token_factory import TokenFactoryClient" in dockerfile
    assert "NEBIUS_TOKEN_FACTORY_KEY" in dockerfile
    assert "build-smoke" in dockerfile
    assert "TokenFactoryClient()" in dockerfile

    control_requirements = (
        root / "common" / "sim2real-control-requirements.txt"
    ).read_text()
    control_lines = {
        line.strip()
        for line in control_requirements.splitlines()
        if line.strip() and not line.startswith("#")
    }
    for dependency in (
        "anyio==4.14.2",
        "h11==0.16.0",
        "httpcore==1.0.9",
        "httpx==0.28.1",
        "typing-extensions==4.16.0",
    ):
        assert dependency in control_lines


def test_isaac_cache_warmer_is_a_nonroot_canonical_workflow_state() -> None:
    root = Path(__file__).resolve().parents[3]
    runbook = yaml.safe_load((root / "workflows/main/sim2real.yaml").read_text())
    resource = runbook["resources"]["isaac-cache-warm"]
    pod = resource["kubernetes"]["pod_config"]["spec"]
    security = pod["securityContext"]
    assert security["runAsNonRoot"] is True
    assert (
        security["runAsUser"] == security["runAsGroup"] == security["fsGroup"] == 1000
    )
    assert security["seccompProfile"] == {"type": "RuntimeDefault"}
    container = pod["containers"][0]
    # The standard SkyPilot first-party-image bootstrap uses passwordless sudo
    # before it executes this non-root state. A container-level no-new-privs
    # override would block that documented bootstrap contract.
    assert "securityContext" not in container
    assert runbook["states"]["stage-00-isaac-cache-warm"]["resources"] == (
        "isaac-cache-warm"
    )


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
    for dependency in (
        "boto3==1.43.62",
        "mcap==1.4.0",
        "pillow==13.0.0",
        "rerun-sdk==0.38.1",
    ):
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


def test_rerun_viewer_exact_source_image_selects_one_light_cli() -> None:
    """The PAIDF visualization toolRef must be registered in the viewer image."""

    dockerfile = (
        Path(__file__).resolve().parents[2]
        / "docker"
        / "workbench"
        / "rerun-viewer"
        / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert "NPA_SKIP_EAGER_IMPORTS=1" in dockerfile
    selectors = [
        token
        for token in dockerfile.split()
        if token.startswith("NPA_LIGHT_WORKBENCH_TOOL=")
    ]
    assert selectors == ["NPA_LIGHT_WORKBENCH_TOOL=rerun-viewer"]


@pytest.mark.parametrize(
    "relative_path",
    (
        "cosmos3-reason/Dockerfile",
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
    runtime_installs = re.sub(r'NVIDIA_REQUIRE_CUDA="[^"]+"', "", dockerfile)
    assert ">=" not in runtime_installs


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
    assert "cryptography.exceptions import InvalidSignature" in dockerfile
    assert (
        "npa.workflows.sim2real.cosmos3_augment import run_cosmos3_structural_augmentation"
        in dockerfile
    )


def test_envgen_baked_runtime_survives_skypilot_pythonpath_scrubbing() -> None:
    """EnvGen's exact source remains importable when Sky clears PYTHONPATH."""

    dockerfile = (
        Path(__file__).resolve().parents[2]
        / "docker"
        / "workbench"
        / "sim2real-envgen"
        / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert "NPA_BAKED_PYTHON=/opt/npa/envgen-venv/bin/python" in dockerfile
    assert "npa-exact-source.pth" in dockerfile
    assert 'env -u PYTHONPATH "${ENVGEN_VENV}/bin/python" -c' in dockerfile
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
    lerobot_vlm = (root / "lerobot-vlm-rl/Dockerfile").read_text(encoding="utf-8")
    assert "sim2real-genesis-requirements.txt" not in lerobot_vlm
    assert 'datasets", "lerobot", "genesis-world", "torch' in lerobot_vlm
    isaac_dockerfile = (root / "isaac-lab/Dockerfile").read_text(encoding="utf-8")
    assert "sim2real-control-requirements.txt" in isaac_dockerfile
    assert "--no-deps" in isaac_dockerfile

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
    assert "npa.workflows.sim2real.cache_warm" in isaac_dockerfile
    assert "npa.workflows.sim2real.workflow_stage" in isaac_dockerfile

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
        "GitPython==3.2.1",
        "pillow==13.0.0",
    ]
    for relative_path in ("sim2real-eval/Dockerfile",):
        dockerfile = (root / relative_path).read_text(encoding="utf-8")
        assert "sim2real-control-requirements.txt" in dockerfile
        assert "sim2real-genesis-requirements.txt" in dockerfile
        assert "python -m pip uninstall -y transformers" in dockerfile
        assert 'd.metadata["Name"].lower() == "transformers"' in dockerfile
        assert "pip check" in dockerfile

    envgen_dockerfile = (root / "sim2real-envgen/Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "sim2real-envgen-requirements.txt" in envgen_dockerfile
    assert "sim2real-genesis-requirements.txt" not in envgen_dockerfile
