"""SSH-based application deployment to the VM."""

from __future__ import annotations

import json
import os
import shlex
import time
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from jinja2 import Environment, FileSystemLoader

from npa.clients.env import render_docker_env_file, render_shell_env_file
from npa.clients.ssh import SSHClient
from npa.cli.agent_source_archive import create_agent_source_archive
from npa.workbench.model_cache import (
    RUNTIME_DOCKER,
    docker_model_cache_volumes,
    model_cache_env,
    resolve_model_cache_root,
)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_DEPLOY_DIR = Path(__file__).parent.parent.parent.parent / "deploy"
_NPA_PACKAGE_ROOT = Path(__file__).parent.parent.parent.parent


class ConfiguratorError(Exception):
    pass


class HealthCheckMode(str, Enum):
    public = "public"
    ssh = "ssh"
    auto = "auto"


def _step(n: int, total: int, msg: str) -> None:
    print(f"  [{n}/{total}] {msg}", flush=True)


def _step_ok(n: int, total: int, msg: str) -> None:
    print(f"  [{n}/{total}] {msg} done", flush=True)


def install_lerobot(ssh: SSHClient) -> bool:
    """Check if LeRobot is installed; return True if already present."""
    code, out, _ = ssh.run(
        "/opt/lerobot/venv/bin/python -c 'import lerobot; print(lerobot.__version__)' 2>/dev/null"
    )
    if code == 0 and out.strip():
        return True
    return False


def install_container_runtime(
    ssh: SSHClient,
    *,
    ssh_user: str = "ubuntu",
    gpu: bool = True,
) -> None:
    """Install Docker and, for GPU workbenches, NVIDIA Container Toolkit."""
    gpu_install = ""
    if gpu:
        gpu_install = """
if ! dpkg-query -W nvidia-container-toolkit >/dev/null 2>&1; then
  sudo rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
fi

sudo nvidia-ctk runtime configure --runtime=docker
"""

    install_cmd = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
      | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  fi
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $VERSION_CODENAME stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

{gpu_install}
sudo systemctl restart docker
sudo usermod -aG docker {shlex.quote(ssh_user)} || true
"""
    ssh.run_or_raise(
        f"bash -lc {shlex.quote(install_cmd)}", label="Container runtime install"
    )


def write_remote_env_file(
    ssh: SSHClient,
    remote_path: str,
    env: dict[str, Any],
    *,
    owner: str = "ubuntu",
) -> None:
    """Atomically install a private shell env file on the VM."""
    write_remote_text_file(
        ssh, remote_path, render_shell_env_file(env), owner=owner, mode="0600"
    )


def write_remote_docker_env_file(
    ssh: SSHClient,
    remote_path: str,
    env: dict[str, Any],
    *,
    owner: str = "ubuntu",
) -> None:
    """Atomically install a private Docker env file without shell quoting."""
    write_remote_text_file(
        ssh, remote_path, render_docker_env_file(env), owner=owner, mode="0600"
    )


def write_remote_text_file(
    ssh: SSHClient,
    remote_path: str,
    content: str,
    *,
    owner: str = "ubuntu",
    mode: str = "0644",
) -> None:
    """Stage privately, then atomically replace a root-managed destination.

    Both staging directories are created with mode 0700 before file creation.
    The final temporary file resides on the destination filesystem, so mv -T
    publishes one complete file without opening an existing destination.
    """
    parent = shlex.quote(str(Path(remote_path).parent))
    target = shlex.quote(remote_path)
    template = shlex.quote(str(Path(remote_path).parent / ".npa-install.XXXXXXXXXX"))
    with ssh.temporary_directory() as directory:
        source = f"{directory}/payload"
        ssh.upload_private_text(content, source)
        script = (
            "set -eu; "
            f"sudo mkdir -p -- {parent}; "
            f"npa_install_dir=$(sudo mktemp -d {template}); "
            "trap 'sudo rm -rf -- \"$npa_install_dir\"' EXIT HUP INT TERM; "
            f"sudo install -m {shlex.quote(mode)} -o {shlex.quote(owner)} "
            f"-g {shlex.quote(owner)} -- {shlex.quote(source)} \"$npa_install_dir/payload\"; "
            f"sudo mv -fT -- \"$npa_install_dir/payload\" {target}"
        )
        ssh.run_or_raise(f"bash -lc {shlex.quote(script)}", label="install remote file")


def docker_exec_cmd(container_name: str, command: str) -> str:
    """Wrap a shell command for execution inside a Workbench container."""
    return f"sudo docker exec {shlex.quote(container_name)} bash -lc {shlex.quote(command)}"


def deploy_workbench_container(
    ssh: SSHClient,
    *,
    image_ref: str,
    container_name: str,
    env_file: str | None = None,
    volumes: Sequence[str] = (),
    work_dirs: Sequence[str] = (),
    group_add: Sequence[str] = (),
    devices: Sequence[str] = (),
    command: str = "-lc 'tail -f /dev/null'",
    ssh_user: str = "ubuntu",
    gpu: bool = True,
) -> None:
    """Install Docker and run a Workbench image as a long-lived container."""
    install_container_runtime(ssh, ssh_user=ssh_user, gpu=gpu)

    # Weights a workbench image is not allowed to bake must survive `docker rm`,
    # which this deploy runs every time. A host-backed cache turns the second
    # deploy of an image into a local read instead of another gated download.
    cache_root = resolve_model_cache_root(runtime=RUNTIME_DOCKER)
    cache_volumes = docker_model_cache_volumes(root=cache_root)
    cache_env = model_cache_env(cache_root)
    volumes = (*volumes, *cache_volumes)
    cache_host_dirs = tuple(volume.split(":", 1)[0] for volume in cache_volumes)

    if work_dirs:
        dirs = " ".join(shlex.quote(path) for path in work_dirs)
        ssh.run_or_raise(
            f"sudo mkdir -p {dirs} && sudo chown -R "
            f"{shlex.quote(ssh_user)}:{shlex.quote(ssh_user)} {dirs}"
        )
    if cache_host_dirs:
        # Deliberately not part of work_dirs, which are recursively chowned on every
        # deploy: this tree is a growing cache of downloaded weights, and walking it
        # is wasted work at best. On a root-squashed network mount `chown` also
        # returns non-zero, which would abort the deploy over a directory that is
        # already owned correctly. Own the root only, and let the container's own
        # user create what it needs beneath it.
        #
        # Mode 3777 rather than the ssh user's ownership alone: the container's
        # runtime uid is the image's business, not this host's, and the two agree
        # only by convention (both are 1000 today). If they ever diverge, an
        # unwritable cache root is not a slow deploy but a broken one -- the tool's
        # first mkdir fails. Sticky keeps one tool from deleting another's blobs and
        # setgid keeps the group stable, matching the Kubernetes init Job.
        dirs = " ".join(shlex.quote(path) for path in cache_host_dirs)
        ssh.run_or_raise(
            f"sudo install -d -o {shlex.quote(ssh_user)} -g {shlex.quote(ssh_user)} "
            f"-m 3777 {dirs}"
        )

    ssh.run_or_raise(
        f"sudo docker pull {shlex.quote(image_ref)}", label="Docker image pull"
    )

    gpu_flag = "--gpus all " if gpu else ""
    env_flag = f"--env-file {shlex.quote(env_file)} " if env_file else ""
    resolved_group_add: list[str] = []
    for group in group_add:
        if str(group).isdigit():
            resolved_group_add.append(str(group))
            continue
        code, out, _ = ssh.run(
            f"getent group {shlex.quote(str(group))} | cut -d: -f3"
        )
        resolved_group_add.append(out.strip() if code == 0 and out.strip() else str(group))
    group_flags = " ".join(
        f"--group-add {shlex.quote(group)}" for group in resolved_group_add
    )
    group_flags = f"{group_flags} " if group_flags else ""
    device_flags = " ".join(f"--device {shlex.quote(device)}" for device in devices)
    device_flags = f"{device_flags} " if device_flags else ""
    volume_flags = " ".join(f"-v {shlex.quote(volume)}" for volume in volumes)
    # `-e` wins over `--env-file`, so a configured shared cache supersedes the
    # per-tool cache path an image bakes into its own env file.
    cache_env_flags = " ".join(
        f"-e {shlex.quote(f'{key}={value}')}" for key, value in sorted(cache_env.items())
    )
    cache_env_flags = f"{cache_env_flags} " if cache_env_flags else ""
    run_cmd = (
        f"sudo docker rm -f {shlex.quote(container_name)} >/dev/null 2>&1 || true\n"
        f"sudo docker run -d {gpu_flag}--ipc=host --network host "
        f"--name {shlex.quote(container_name)} --restart unless-stopped "
        f"{group_flags}{device_flags}{env_flag}{cache_env_flags}{volume_flags} "
        f"{shlex.quote(image_ref)} {command}"
    )
    ssh.run_or_raise(run_cmd)


def deploy_server(
    ssh: SSHClient,
    server_config: dict[str, Any],
) -> None:
    """Copy the npa package to the VM, render server config, install systemd unit."""
    with ssh.temporary_directory() as directory:
        # 1. Package and upload the npa source
        archive_path = create_agent_source_archive(_NPA_PACKAGE_ROOT.parent)
        try:
            # Upload via SSH (paramiko sftp)
            _sftp_upload(ssh, archive_path, f"{directory}/npa.tgz")
        finally:
            os.unlink(archive_path)

        # 2. Extract and install on the VM
        ssh.run_or_raise(
            f"mkdir {shlex.quote(directory + '/source')} && "
            f"tar -xzf {shlex.quote(directory + '/npa.tgz')} -C {shlex.quote(directory + '/source')} && "
            f"/opt/lerobot/venv/bin/pip install -q {shlex.quote(directory + '/source/npa[server]')}"
        )

        # 3. Render and upload server.yaml
        env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)))
        template = env.get_template("server.yaml.j2")
        rendered = template.render(**server_config)

        write_remote_text_file(ssh, "/etc/npa/server.yaml", rendered, owner="root")

        # 4. Write env file for systemd from the server config
        env_vars: dict[str, Any] = {
            "NPA_SERVER_HOST": server_config.get("server_host", "0.0.0.0"),
            "NPA_SERVER_PORT": server_config.get("server_port", 8080),
            "NPA_CHECKPOINT_DIR": server_config.get("checkpoint_dir", "/opt/lerobot/checkpoints"),
            "NPA_CHECKPOINT_BUCKET": server_config.get("checkpoint_bucket", ""),
            "NPA_JOB_STATUS_DIR": server_config.get("job_status_dir", "/opt/lerobot/job_status"),
            "NPA_LOG_DIR": server_config.get("log_dir", "/var/log/npa-lerobot"),
            "AWS_ENDPOINT_URL": server_config.get("storage_endpoint", ""),
        }
        shared_env = server_config.get("shared_env", {})
        if isinstance(shared_env, dict):
            env_vars.update(shared_env)
        write_remote_env_file(ssh, "/etc/npa-lerobot-server/env", env_vars, owner="root")

        # 5. Upload and enable systemd unit
        service_src = _DEPLOY_DIR / "npa-lerobot-server.service"
        if service_src.exists():
            write_remote_text_file(
                ssh, "/etc/systemd/system/npa-lerobot-server.service",
                service_src.read_text(), owner="root",
            )
            ssh.run_or_raise(
                "sudo systemctl daemon-reload && "
                "sudo systemctl enable npa-lerobot-server"
            )

        # 6. Create required directories
        ssh.run_or_raise(
            "sudo mkdir -p /var/log/npa-lerobot /opt/lerobot/checkpoints /opt/lerobot/job_status && "
            "sudo chown ubuntu:ubuntu /var/log/npa-lerobot /opt/lerobot/checkpoints /opt/lerobot/job_status"
        )

        # 7. Restart service
        ssh.run_or_raise("sudo systemctl restart npa-lerobot-server")


def deploy_lerobot_container(
    ssh: SSHClient,
    *,
    image_ref: str,
    server_config: dict[str, Any],
    ssh_user: str = "ubuntu",
    container_name: str = "npa-lerobot",
) -> None:
    """Install Docker/NVIDIA runtime and run the LeRobot server container."""
    hf_cache_dir = str(server_config.get("hf_cache_dir") or "/opt/lerobot/hf_cache")
    # This deploy predates the shared cache and has its own `docker run`, so it has
    # to opt in explicitly or it would be the one tool on the box still discarding
    # everything except its LeRobot datasets: HF_LEROBOT_HOME covers those, but the
    # transformers and torch downloads a policy pulls in have nowhere to go.
    cache_root = resolve_model_cache_root(runtime=RUNTIME_DOCKER)
    cache_volumes = docker_model_cache_volumes(root=cache_root)
    cache_env = model_cache_env(cache_root)
    # The per-deploy directory stays mounted and stays authoritative for LeRobot's
    # own datasets: it may already hold them, and this deploy is not the place to
    # silently move an operator's data to a new path.
    cache_env.pop("HF_LEROBOT_HOME", None)
    cache_env.pop("LEROBOT_HF_HOME", None)
    install_cmd = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

sudo install -d -m 0755 -o {shlex.quote(ssh_user)} -g {shlex.quote(ssh_user)} \
  /opt/lerobot \
  /opt/lerobot/checkpoints \
  /opt/lerobot/job_status \
  /opt/lerobot/dataset_cache \
  /opt/lerobot/checkpoint_cache \
  {shlex.quote(hf_cache_dir)} \
  /opt/lerobot/benchmarks \
  {" ".join(shlex.quote(v.split(":", 1)[0]) for v in cache_volumes)} \
  /var/log/npa-lerobot
sudo touch /opt/lerobot/.env
sudo chown {shlex.quote(ssh_user)}:{shlex.quote(ssh_user)} /opt/lerobot/.env
sudo chmod 600 /opt/lerobot/.env

if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
      | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  fi
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $VERSION_CODENAME stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

if ! dpkg-query -W nvidia-container-toolkit >/dev/null 2>&1; then
  sudo rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
fi

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo usermod -aG docker {shlex.quote(ssh_user)} || true
"""
    ssh.run_or_raise(
        f"bash -lc {shlex.quote(install_cmd)}", label="LeRobot runtime install"
    )

    ssh.run_or_raise(
        f"sudo docker pull {shlex.quote(image_ref)}", label="Docker image pull"
    )

    env_args = {
        "NPA_SERVER_HOST": server_config.get("server_host", "0.0.0.0"),
        "NPA_SERVER_PORT": "8080",
        "NPA_CHECKPOINT_DIR": server_config.get("checkpoint_dir", "/opt/lerobot/checkpoints"),
        "NPA_CHECKPOINT_BUCKET": server_config.get("checkpoint_bucket", ""),
        "NPA_JOB_STATUS_DIR": server_config.get("job_status_dir", "/opt/lerobot/job_status"),
        "NPA_LOG_DIR": server_config.get("log_dir", "/var/log/npa-lerobot"),
        "AWS_ENDPOINT_URL": server_config.get("storage_endpoint", ""),
        "HF_LEROBOT_HOME": hf_cache_dir,
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "PYTHONUNBUFFERED": "1",
    }
    shared_env = server_config.get("shared_env", {})
    if isinstance(shared_env, dict):
        env_args.update(shared_env)
    if server_config.get("cuda_visible_devices"):
        env_args["CUDA_VISIBLE_DEVICES"] = server_config["cuda_visible_devices"]
    if server_config.get("gpu_count"):
        env_args["NPA_GPU_COUNT"] = str(server_config["gpu_count"])
    env_args.update(cache_env)
    # A later env file retains the former --env override precedence without
    # exposing shared credentials in either the SSH shell or Docker argv.
    write_remote_docker_env_file(
        ssh, "/opt/lerobot/container.env", env_args, owner=ssh_user
    )
    volume_flags = " ".join(
        [
            "-v /opt/lerobot/.env:/opt/lerobot/.env:ro",
            "-v /opt/lerobot/checkpoints:/opt/lerobot/checkpoints",
            "-v /opt/lerobot/job_status:/opt/lerobot/job_status",
            "-v /opt/lerobot/dataset_cache:/opt/lerobot/dataset_cache",
            "-v /opt/lerobot/checkpoint_cache:/opt/lerobot/checkpoint_cache",
            # HF_LEROBOT_HOME points here, and this deploy recreates the container
            # on every run: without the bind mount the datasets and policy weights
            # LeRobot pulls from Hugging Face are discarded with the old container
            # and downloaded again.
            f"-v {shlex.quote(hf_cache_dir)}:{shlex.quote(hf_cache_dir)}",
            "-v /opt/lerobot/benchmarks:/opt/lerobot/benchmarks",
            "-v /var/log/npa-lerobot:/var/log/npa-lerobot",
            *(f"-v {shlex.quote(volume)}" for volume in cache_volumes),
        ]
    )
    run_cmd = (
        "sudo systemctl stop npa-lerobot-server >/dev/null 2>&1 || true\n"
        f"sudo docker rm -f {shlex.quote(container_name)} >/dev/null 2>&1 || true\n"
        f"sudo docker run -d --gpus all --ipc=host --network host "
        f"--name {shlex.quote(container_name)} --restart unless-stopped "
        f"--env-file /opt/lerobot/.env --env-file /opt/lerobot/container.env {volume_flags} "
        f"{shlex.quote(image_ref)}"
    )
    ssh.run_or_raise(run_cmd)


def health_check(endpoint: str, *, retries: int = 10, backoff: float = 3.0) -> bool:
    """Poll the /health endpoint until success or timeout."""
    import httpx

    url = f"{endpoint}/health"
    for attempt in range(retries):
        try:
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(backoff)
    return False


def health_check_ssh(
    ssh: SSHClient,
    port: int,
    *,
    path: str = "/health",
    retries: int = 10,
    backoff: float = 3.0,
) -> bool:
    """Poll a VM-local HTTP health endpoint through SSH."""
    url = f"http://127.0.0.1:{port}{path}"
    for _attempt in range(retries):
        code, _out, _err = ssh.run(f"curl -fsS {shlex.quote(url)} >/dev/null")
        if code == 0:
            return True
        time.sleep(backoff)
    return False


def health_check_auto(
    endpoint: str,
    *,
    mode: HealthCheckMode | str = HealthCheckMode.auto,
    ssh: SSHClient | None = None,
    port: int | None = None,
    host: str = "",
    retries: int = 10,
    backoff: float = 3.0,
    auto_public_retries: int = 3,
) -> tuple[bool, str]:
    """Run public/SSH/auto health checks and return (healthy, note)."""
    selected = mode if isinstance(mode, HealthCheckMode) else HealthCheckMode(str(mode))
    if selected == HealthCheckMode.public:
        return health_check(endpoint, retries=retries, backoff=backoff), ""
    if selected == HealthCheckMode.ssh:
        if ssh is None or port is None:
            return False, ""
        return health_check_ssh(ssh, port, retries=retries, backoff=backoff), ""

    public_retries = min(retries, max(1, auto_public_retries))
    if health_check(endpoint, retries=public_retries, backoff=backoff):
        return True, ""
    if ssh is not None and port is not None and health_check_ssh(
        ssh,
        port,
        retries=retries,
        backoff=backoff,
    ):
        return True, f"Public port {port} unreachable; service healthy via SSH on {host}."
    return False, ""


def _decode_env_file_value(value: str) -> str:
    if len(value) >= 2 and value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("'\\''", "'")
    return value


def parse_env_file_content(content: str, keys: Sequence[str]) -> dict[str, str]:
    selected = set(keys)
    values: dict[str, str] = {}
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export "):]
        key, value = stripped.split("=", 1)
        if key in selected:
            values[key] = _decode_env_file_value(value)
    return values


def read_remote_env_keys(
    ssh: SSHClient,
    remote_path: str,
    keys: Sequence[str],
) -> dict[str, str]:
    """Read selected env keys without shell expansion."""
    _code, out, _err = ssh.run_or_raise(f"sudo cat {shlex.quote(remote_path)}")
    return parse_env_file_content(out, keys)


def audit_remote_env(
    ssh: SSHClient,
    remote_path: str,
    expected: dict[str, str],
) -> list[str]:
    """Return shared credential keys missing or mismatched in a remote env file."""
    keys = [key for key, value in expected.items() if value]
    if not keys:
        return []
    actual = read_remote_env_keys(ssh, remote_path, keys)
    return [key for key in keys if actual.get(key, "") != expected[key]]


def write_manifest(
    ssh: SSHClient,
    tool: str,
    version: str,
    deployed_by: str,
) -> None:
    """Write /etc/npa/manifest.json on the VM."""
    manifest = json.dumps(
        {
            "tool": tool,
            "version": version,
            "deployed_by": deployed_by,
            "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        indent=2,
    )
    ssh.run_or_raise(
        f"sudo mkdir -p /etc/npa && "
        f"echo '{manifest}' | sudo tee /etc/npa/manifest.json >/dev/null"
    )


def _sftp_upload(ssh: SSHClient, local_path: str, remote_path: str) -> None:
    """Upload through the shared private, atomic SFTP implementation."""
    ssh.upload_file(local_path, remote_path)
