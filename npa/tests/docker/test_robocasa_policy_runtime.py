"""Static contract for the combined RoboCasa + LeRobot ACT evaluation runtime."""

import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = ROOT / "npa" / "docker" / "workbench" / "robocasa"
DOCKERFILE = IMAGE_DIR / "Dockerfile"
BUILD_SCRIPT = DOCKERFILE.with_name("build.sh")
SMOKE_SCRIPT = ROOT / "npa" / "src" / "npa" / "smoke" / "test_robocasa_functional.py"
GENERATOR = IMAGE_DIR / "generate-locks.sh"
RUNTIME_INPUT = IMAGE_DIR / "requirements.in"
RUNTIME_LOCK = IMAGE_DIR / "requirements.lock"
BUILD_LOCK = IMAGE_DIR / "build-requirements.lock"
LEROBOT_LOCK = IMAGE_DIR / "lerobot-requirements.lock"
BASE_INVENTORY = IMAGE_DIR.parent / "base-image-security.json"
ROBOCASA_COMMIT = "8f3c96ec8d1bfcd8126cad2bca887da98d30e997"
ROBOSUITE_COMMIT = "85abee228d1c43ab1939bce33028099945d453b4"
PINNED_CUDA_BASE_NAME = "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"
PINNED_CUDA_BASE_DIGEST = (
    "sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4"
)
PINNED_CUDA_BASE = f"{PINNED_CUDA_BASE_NAME}@{PINNED_CUDA_BASE_DIGEST}"
OPENCV_PROVIDERS = {
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
}


def _locked_names(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {
        match.group(1).lower().replace("_", "-")
        for match in re.finditer(r"(?m)^([A-Za-z0-9_.-]+)(?:==| @ )", text)
    }


def _committed_builder_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    repo = tmp_path / "repo"
    image_dir = repo / "npa/docker/workbench/robocasa"
    image_dir.mkdir(parents=True)
    script = image_dir / "build.sh"
    script.write_bytes(BUILD_SCRIPT.read_bytes())
    script.chmod(0o755)
    (image_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (repo / "npa/tracked.txt").write_text("committed context\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "robocasa-build-test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "RoboCasa Build Test"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "add", "npa"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    return repo, script, source_sha


def _requirement_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    starts = [match.start() for match in re.finditer(r"(?m)^[A-Za-z0-9_.-]+", text)]
    return [
        text[start : starts[index + 1] if index + 1 < len(starts) else len(text)]
        for index, start in enumerate(starts)
    ]


def _from_refs(text: str) -> list[str]:
    return [
        match.group(1) for match in re.finditer(r"(?m)^FROM\s+(?:--\S+\s+)*(\S+)", text)
    ]


def test_robocasa_keeps_known_good_gymnasium_and_policy_only_lerobot() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    runtime_input = RUNTIME_INPUT.read_text(encoding="utf-8")
    runtime_lock = RUNTIME_LOCK.read_text(encoding="utf-8")
    lerobot_lock = LEROBOT_LOCK.read_text(encoding="utf-8")

    assert "gymnasium==0.29.1" in runtime_lock
    assert "lerobot==0.5.1" in lerobot_lock
    assert "--require-hashes --only-binary=:all: --no-deps" in dockerfile
    assert "-r /opt/robocasa/locks/lerobot-requirements.lock" in dockerfile
    for requirement in (
        "av>=15.0.0,<16.0.0",
        "diffusers>=0.27.2,<0.36.0",
        "pyserial>=3.5,<4.0",
        "draccus==0.10.0",
        "einops>=0.8.0,<0.9.0",
        "opencv-python>=4.9,<4.14",
    ):
        assert requirement in runtime_input
    assert "opencv-python-headless" not in runtime_input
    assert "from lerobot.policies.act.modeling_act import ACTPolicy" in dockerfile
    assert "from lerobot.policies.factory import make_pre_post_processors" in dockerfile
    assert "opencv == ['opencv-python']" in dockerfile
    assert "--no-build-isolation --no-deps -e /opt/robocasa/source" in dockerfile


def test_robocasa_act_runtime_stays_within_lerobot_dependency_bounds() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    runtime_lock = RUNTIME_LOCK.read_text(encoding="utf-8")

    assert "torch==2.9.0+cu128" in runtime_lock
    assert "torchvision==0.24.0+cu128" in runtime_lock
    assert "torch==2.12.1" not in runtime_lock
    assert "torchvision==0.27.1" not in runtime_lock
    assert "req.specifier.contains(version(req.name), prereleases=True)" in dockerfile
    assert "torchvision.__version__ == '0.24.0+cu128'" in dockerfile


def test_robocasa_policy_runtime_has_one_cuda_wheel_family() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    runtime_lock = RUNTIME_LOCK.read_text(encoding="utf-8")
    locked_names = _locked_names(RUNTIME_LOCK)

    assert runtime_lock.count("torch==2.9.0+cu128") == 1
    assert runtime_lock.count("torchvision==0.24.0+cu128") == 1
    assert any(
        name.startswith("nvidia-") and name.endswith("-cu12") for name in locked_names
    )
    assert not any(name.endswith("-cu13") for name in locked_names)
    assert not {"cuda-bindings", "cuda-pathfinder", "cuda-toolkit"} & locked_names
    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in dockerfile
    assert "forbidden_names = {" in dockerfile
    assert "name.endswith('-cu13')" in dockerfile
    assert "torch.__version__ == '2.9.0+cu128'" in dockerfile
    assert "torch.version.cuda == '12.8'" in dockerfile


def test_robocasa_public_runtime_excludes_restricted_optional_payloads() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    dependency_install = text.split("# Install RoboCasa source", maxsplit=1)[0]
    runtime_lock = RUNTIME_LOCK.read_text(encoding="utf-8")

    assert "IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg" in text
    assert "        ffmpeg fuse netcat-openbsd" in text
    assert "imageio_ffmpeg-0.6.0.tar.gz#sha256=" in runtime_lock
    assert "imageio_ffmpeg-0.6.0-py3-none" not in runtime_lock
    assert "imageio[ffmpeg]" not in runtime_lock
    assert "*/imageio_ffmpeg/binaries/ffmpeg*" in dependency_install
    assert "imageio_ffmpeg.get_ffmpeg_exe() == '/usr/bin/ffmpeg'" in dependency_install
    assert "imageio.mimsave(video," in dependency_install
    assert "itertools.islice(reader, 3)" in dependency_install
    assert "len(frames) == 2" in dependency_install
    assert dependency_install.count("render_dataset_with_omniverse.py") >= 2
    assert (
        "*/robosuite/scripts/__pycache__/render_dataset_with_omniverse*.pyc"
        in dependency_install
    )


def test_robocasa_system_install_layer_removes_builder_resolver_state() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    start = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("RUN apt-get update")
    )
    end = start
    while lines[end].rstrip().endswith("\\"):
        end += 1
    install_run = "".join(lines[start : end + 1]).replace("\\\n", " ")
    commands = [command.strip() for command in install_run.split("&&")]

    install = next(
        index for index, command in enumerate(commands) if "apt-get install" in command
    )
    venv = commands.index("python3.12 -m venv /opt/robocasa/venv")
    cleanup = next(
        index
        for index, command in enumerate(commands)
        if (tokens := shlex.split(command))
        and tokens[0] == "rm"
        and "/run/systemd/resolve" in tokens
    )
    absent_path = commands.index("test ! -e /run/systemd/resolve")
    absent_symlink = commands.index("test ! -L /run/systemd/resolve")

    cleanup_tokens = shlex.split(commands[cleanup])
    assert "/var/lib/apt/lists/*" in cleanup_tokens
    assert install < venv < cleanup < absent_path < absent_symlink
    assert absent_symlink == len(commands) - 1


def test_robocasa_runtime_is_non_root_without_passwordless_sudo() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    smoke = SMOKE_SCRIPT.read_text(encoding="utf-8")

    user_lines = [
        line.strip() for line in text.splitlines() if line.startswith("USER ")
    ]
    assert user_lines[-1] == "USER ubuntu"
    assert "NOPASSWD" not in text
    assert "openssh-server" not in text
    assert "rsync sudo" not in text
    assert "chown -R ubuntu:ubuntu /opt/robocasa/source/robocasa/models/assets" in text
    assert 'RUN test "$(id -u)" != "0"' in text
    assert "python /app/smoke_functional.py" in text
    assert "os.geteuid() == 0" in smoke
    assert 'NamedTemporaryFile(prefix=".npa-write-"' in smoke
    assert "kitchen_task_registration(download_assets=False)" in smoke
    assert "_execute_capability_in_worker(" in smoke
    assert "registered_env_count" in smoke
    assert "chown -R ubuntu:ubuntu /app " not in text
    assert "chown -R ubuntu:ubuntu /opt/robocasa/source\n" not in text
    assert "chown -R ubuntu:ubuntu /app /opt/robocasa\n" not in text
    copy_end = text.index(
        "COPY src/npa/smoke/test_robocasa_functional.py /app/smoke_functional.py"
    )
    normalize_permissions = text.index(
        "RUN chmod -R u=rwX,go=rX /app/npa /app/smoke_functional.py"
    )
    runtime_import = text.index(
        'RUN python -c "from npa.workbench.robocasa.service import app;'
    )
    assert copy_end < normalize_permissions < runtime_import


def test_robocasa_image_binds_committed_source_revision() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    build_script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert _from_refs(dockerfile) == [PINNED_CUDA_BASE]
    assert "ARG BASE_IMAGE" not in dockerfile
    assert "${BASE_IMAGE}" not in dockerfile
    assert "--base-image" not in build_script
    assert "ROBOCASA_BASE_IMAGE" not in build_script
    assert "--build-arg BASE_IMAGE" not in build_script
    assert "ARG NPA_SOURCE_SHA" in dockerfile
    assert 'org.opencontainers.image.revision="${NPA_SOURCE_SHA}"' in dockerfile
    assert f'org.opencontainers.image.base.name="{PINNED_CUDA_BASE_NAME}"' in dockerfile
    assert (
        f'org.opencontainers.image.base.digest="{PINNED_CUDA_BASE_DIGEST}"'
        in dockerfile
    )
    assert f'npa.base_image="{PINNED_CUDA_BASE}"' in dockerfile
    assert "NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}" in dockerfile
    assert "ROBOCASA_REQUIRE_IMAGE_SOURCE_SHA=1" in dockerfile
    assert "FROM --platform=" not in dockerfile
    assert (
        'REPO_ROOT="$(git -C "${NPA_ROOT}" rev-parse --show-toplevel)"' in build_script
    )
    assert (
        'git -C "${REPO_ROOT}" archive --format=tar "${NPA_SOURCE_SHA}:npa"'
        in build_script
    )
    assert "| docker build \\\n      --platform linux/amd64 \\" in build_script
    assert "\n      --provenance=false \\" in build_script
    assert "-f docker/workbench/robocasa/Dockerfile \\" in build_script
    assert build_script.rstrip().endswith('echo "Built ${IMAGE}"')
    assert '"${NPA_ROOT}"\n\necho "Built ${IMAGE}"' not in build_script
    assert "grep -Eq '^[0-9a-f]{40}$'" in dockerfile
    assert "COPY src/npa/clients/storage.py /app/npa/clients/storage.py" in dockerfile
    assert (
        "COPY src/npa/cli/path_contract.py /app/npa/cli/path_contract.py" in dockerfile
    )
    assert "from npa.workbench.robocasa.service import app" in dockerfile
    assert 'rev-parse HEAD)" != "${NPA_SOURCE_SHA}"' in build_script
    assert "status --porcelain=v1 --untracked-files=all -- ." in build_script


def test_robocasa_builder_streams_the_committed_npa_subtree(
    tmp_path: Path,
) -> None:
    repo, script, source_sha = _committed_builder_fixture(tmp_path)
    # Untracked data outside npa must never enter the committed subtree archive.
    (repo / "private-input").write_text("not build input\n", encoding="utf-8")

    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['DOCKER_STDIN']).write_bytes(sys.stdin.buffer.read())\n"
        "pathlib.Path(os.environ['DOCKER_ARGV']).write_text(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    archive = tmp_path / "context.tar"
    argv = tmp_path / "docker-argv.json"
    result = subprocess.run(
        [
            str(script),
            "--registry",
            "registry.example.invalid/npa",
            "--tag",
            f"dev-{source_sha}",
        ],
        cwd=repo / "npa",
        env={
            **os.environ,
            "PATH": f"{binary}{os.pathsep}{os.environ['PATH']}",
            "NPA_SOURCE_SHA": source_sha,
            "DOCKER_STDIN": str(archive),
            "DOCKER_ARGV": str(argv),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    with tarfile.open(fileobj=io.BytesIO(archive.read_bytes())) as context:
        names = set(context.getnames())
    assert "docker/workbench/robocasa/Dockerfile" in names
    assert "tracked.txt" in names
    assert "private-input" not in names
    docker_argv = json.loads(argv.read_text(encoding="utf-8"))
    assert docker_argv[0] == "build"
    assert docker_argv[-1] == "-"
    assert docker_argv[docker_argv.index("-f") + 1] == (
        "docker/workbench/robocasa/Dockerfile"
    )


def test_robocasa_builder_fails_closed_when_git_status_fails(
    tmp_path: Path,
) -> None:
    repo, script, source_sha = _committed_builder_fixture(tmp_path)
    binary = tmp_path / "bin"
    binary.mkdir()
    real_git = shutil.which("git")
    assert real_git is not None
    git = binary / "git"
    git.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "if 'status' in sys.argv[1:]:\n"
        "    raise SystemExit(17)\n"
        "real = os.environ['REAL_GIT']\n"
        "os.execv(real, [real, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    git.chmod(0o755)
    marker = tmp_path / "docker-ran"
    docker = binary / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib\n"
        "pathlib.Path(os.environ['DOCKER_MARKER']).touch()\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)

    result = subprocess.run(
        [
            str(script),
            "--registry",
            "registry.example.invalid/npa",
            "--tag",
            f"dev-{source_sha}",
        ],
        cwd=repo / "npa",
        env={
            **os.environ,
            "PATH": f"{binary}{os.pathsep}{os.environ['PATH']}",
            "REAL_GIT": real_git,
            "NPA_SOURCE_SHA": source_sha,
            "DOCKER_MARKER": str(marker),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "cannot verify the build-context checkout is clean" in result.stderr
    assert not marker.exists()


def test_robocasa_upstreams_use_verified_immutable_commits() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert f"ROBOCASA_SOURCE_COMMIT={ROBOCASA_COMMIT}" in text
    assert f'npa.robocasa.revision="{ROBOCASA_COMMIT}"' in text
    assert f"ROBOSUITE_SOURCE_COMMIT={ROBOSUITE_COMMIT}" in text
    assert f'npa.robosuite.revision="{ROBOSUITE_COMMIT}"' in text
    assert "git -C /opt/robocasa/source fetch --depth 1 origin" in text
    assert '"${ROBOCASA_SOURCE_COMMIT}"' in text
    assert 'rev-parse HEAD)" = \\\n      "${ROBOCASA_SOURCE_COMMIT}"' in text
    assert "ROBOCASA_REPO_REF" not in text and "v1.0" not in text
    robosuite_install = text.split('"robosuite @ git+', 1)[0].rsplit(
        "PIP_CONFIG_FILE=/dev/null", 1
    )[1]
    assert "--no-build-isolation --no-deps" in robosuite_install
    assert "distribution('robosuite').read_text('direct_url.json')" in text
    assert "vcs.get('commit_id') == '${ROBOSUITE_SOURCE_COMMIT}'" in text


def test_every_robocasa_from_base_has_one_security_inventory_entry() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    inventory = json.loads(BASE_INVENTORY.read_text(encoding="utf-8"))
    from_refs = _from_refs(dockerfile)

    assert from_refs == [PINNED_CUDA_BASE]
    for image in from_refs:
        entries = [entry for entry in inventory if entry["image"] == image]
        assert len(entries) == 1, image
        assert entries[0] == {
            "name": "nvidia-cuda-12-4-1-cudnn-devel-ubuntu22-04",
            "image": PINNED_CUDA_BASE,
            "purge_linux_libc_dev": False,
            "upgrade_os": False,
        }


def test_robocasa_python_locks_are_hash_complete_and_target_specific() -> None:
    for lock in (BUILD_LOCK, RUNTIME_LOCK, LEROBOT_LOCK):
        text = lock.read_text(encoding="utf-8")
        assert "npa/docker/workbench/robocasa/generate-locks.sh" in text
        blocks = _requirement_blocks(lock)
        assert blocks and all("--hash=sha256:" in block for block in blocks)

    runtime_names = _locked_names(RUNTIME_LOCK)
    assert runtime_names & OPENCV_PROVIDERS == {"opencv-python"}
    assert {"gymnasium", "torch", "torchvision"} <= runtime_names
    assert _locked_names(LEROBOT_LOCK) == {"lerobot"}


def test_robocasa_lock_generation_uses_only_anonymous_indexes() -> None:
    generator = GENERATOR.read_text(encoding="utf-8")

    assert "uv 0.12.5 (x86_64-unknown-linux-gnu)" in generator
    assert "--python-version 3.12" in generator
    assert "--python-platform x86_64-manylinux_2_28" in generator
    assert "--torch-backend cu128" in generator
    assert "--config-file /dev/null" in generator
    assert "--default-index https://pypi.org/simple" in generator
    assert "--keyring-provider disabled" in generator
    assert "unset UV_INDEX" in generator and "PIP_INDEX_URL" in generator
    assert not re.search(r"https?://[^/\s]+@", generator)


def test_robocasa_local_builder_has_no_publication_path() -> None:
    build_script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "--push" not in build_script
    assert "docker push" not in build_script
    for gate in ("payload", "complete-byte", "Trivy", "SBOM", "GPU"):
        assert gate in build_script


def test_robocasa_candidate_version_is_consistent() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    build_script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "ARG ROBOCASA_VERSION=0.1.1" in dockerfile
    assert 'ROBOCASA_VERSION="${ROBOCASA_VERSION:-0.1.1}"' in build_script
    assert "npa.cuda_architectures" not in dockerfile
