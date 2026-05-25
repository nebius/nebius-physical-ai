"""Two-VM expert distillation: L40S (Genesis sim) + H100 (LeRobot training).

Provisions both VMs via Terraform, installs the correct runtime on each,
runs the 5-stage pipeline with S3 artifact handoff, and optionally tears
down infrastructure afterward.  Teardown runs in a finally block that
covers provisioning too, so non-preemptible GPU instances are never leaked.

VM layout:
    L40S (sim)   — Stages 1, 2, 3, 5  (Genesis + adapter)
    H100 (train) — Stage 4             (LeRobot ACT/Diffusion/SmolVLA)

Artifact flow (S3):
    s3://{bucket}/distill/{run_id}/teacher/   <- Stage 1 checkpoint
    s3://{bucket}/distill/{run_id}/dataset/   <- Stage 3 output  (sim -> train)
    s3://{bucket}/distill/{run_id}/student/   <- Stage 4 output  (train -> sim)
    s3://{bucket}/distill/{run_id}/eval/      <- Stage 5 metrics

Demos (Stage 2) stay on the sim VM — only the converted dataset is uploaded.

Usage:
    npa workbench workflow distill [--teardown] [--skip-infra]
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from npa.clients.config import SSHConfig, write_config
from npa.clients.nebius import NebiusError, bootstrap_environment
from npa.clients.ssh import SSHClient, SSHError
from npa.deploy.provisioner import ProvisionerError
from npa.workflows.distill import generate_run_id

logger = logging.getLogger(__name__)

# ── Environment constants ──────────────────────────────────────────────────


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


TENANT_ID = os.environ.get("NPA_TENANT_ID") or os.environ.get("NEBIUS_ACCOUNT_ID", "")
PROJECT_ID = _required_env("NPA_PROJECT_ID")
DEFAULT_S3_BUCKET = _required_env("NPA_S3_BUCKET")
REGION = os.environ.get("NPA_REGION", "eu-north1")
PROJECT_ALIAS = os.environ.get("NPA_PROJECT_ALIAS", REGION)

# Path to the npa package root (the directory containing pyproject.toml).
_NPA_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Bundled setup scripts inside the package.
_SETUP_DIR = Path(__file__).resolve().parent.parent / "setup"

# Miniforge install prefix on the VM.  Every conda invocation in this
# module uses absolute paths so it works in paramiko's non-login,
# non-interactive shell where ~/.bashrc is NOT sourced.
_CONDA_PREFIX = "/opt/conda"
_CONDA_BIN = f"{_CONDA_PREFIX}/bin/conda"

# Tar exclusion patterns to keep the uploaded archive small (<1 MB).
# Without these, .venv/ alone adds ~250 MB of irrelevant data.
_TAR_EXCLUDES = [
    "--exclude=.venv",
    "--exclude=__pycache__",
    "--exclude=*.pyc",
    "--exclude=.git",
    "--exclude=.mypy_cache",
    "--exclude=.pytest_cache",
    "--exclude=*.egg-info",
]


@dataclass
class VMSpec:
    """Terraform variables that differ per VM."""

    name: str
    gpu_platform: str
    gpu_preset: str
    conda_env: str        # conda env name created by the setup script
    setup_script: str     # filename inside npa/src/npa/setup/
    npa_extra: str        # pip extra for ``npa[<extra>]``


SIM_VM = VMSpec(
    name="l40s-distill-genesis",
    gpu_platform="gpu-l40s-a",
    gpu_preset="1gpu-40vcpu-160gb",
    conda_env="genesis",
    setup_script="install_genesis.sh",
    npa_extra="genesis",
)

TRAIN_VM = VMSpec(
    name="h100-distill-lerobot",
    gpu_platform="gpu-h100-sxm",
    gpu_preset="1gpu-16vcpu-200gb",
    conda_env="lerobot",
    setup_script="install_lerobot.sh",
    npa_extra="adapter",
)


class TwoVMDistillError(Exception):
    pass


def _conda_activate(conda_env: str) -> str:
    """Return a shell prefix that activates a conda env and loads S3 creds.

    Uses the absolute path to the conda binary only for the initial
    ``shell.bash hook`` eval — that injects a ``conda()`` shell function
    into the current shell.  The subsequent ``conda activate`` must call
    that shell function (not the binary) because only the function can
    modify the current shell's PATH and environment variables.

    Also sources ``/opt/lerobot/.env`` (written by cloud-init) so that
    S3 credentials (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
    ``NEBIUS_S3_ENDPOINT``) are available to boto3 and npa commands
    running in the conda env.
    """
    return (
        f'set -a && test -f /opt/lerobot/.env && . /opt/lerobot/.env; set +a && '
        f'eval "$({_CONDA_BIN} shell.bash hook)" && '
        f'conda activate {conda_env} && '
    )


# ── Infrastructure helpers ─────────────────────────────────────────────────


def _provision_vm(
    spec: VMSpec,
    nebius_creds: dict[str, str],
    *,
    ssh_public_key_path: str = "~/.ssh/id_ed25519.pub",
) -> dict[str, Any]:
    """Provision a single VM via Terraform. Returns Terraform outputs dict."""
    from npa.deploy import provisioner

    s3_bucket = nebius_creds.get("s3_bucket") or DEFAULT_S3_BUCKET
    s3_endpoint = nebius_creds["s3_endpoint"]

    logger.info("Preparing Terraform working dir for %s ...", spec.name)
    tf_dir = str(provisioner.prepare_working_dir(
        PROJECT_ALIAS,
        spec.name,
        bucket=s3_bucket,
        region=REGION,
        endpoint=s3_endpoint,
    ))

    logger.info("Running terraform init for %s ...", spec.name)
    try:
        provisioner.init(
            tf_dir=tf_dir,
            backend_config={
                "access_key": nebius_creds["nebius_api_key"],
                "secret_key": nebius_creds["nebius_secret_key"],
            },
        )
    except ProvisionerError as exc:
        raise TwoVMDistillError(f"terraform init failed for {spec.name}: {exc}") from exc

    tf_vars: dict[str, str] = {
        "nebius_project_id": nebius_creds["nebius_project_id"],
        "iam_token": nebius_creds["iam_token"],
        "service_account_id": nebius_creds["service_account_id"],
        "nebius_region": REGION,
        "instance_name": spec.name,
        "gpu_platform": spec.gpu_platform,
        "gpu_preset": spec.gpu_preset,
        "enable_preemptible": "false",
        "ssh_public_key_path": ssh_public_key_path,
        "nebius_api_key": nebius_creds["nebius_api_key"],
        "nebius_secret_key": nebius_creds["nebius_secret_key"],
        "s3_bucket": s3_bucket,
        "s3_endpoint": s3_endpoint,
    }

    logger.info("Running terraform apply for %s (%s) ...", spec.name, spec.gpu_platform)
    try:
        outputs = provisioner.apply(tf_dir=tf_dir, tf_vars=tf_vars)
    except ProvisionerError as exc:
        raise TwoVMDistillError(f"terraform apply failed for {spec.name}: {exc}") from exc

    vm_ip = outputs.get("vm_ip", "")
    if not vm_ip:
        raise TwoVMDistillError(f"No VM IP returned for {spec.name}")

    logger.info("%s provisioned — IP %s", spec.name, vm_ip)
    return outputs


def _destroy_vm(spec: VMSpec, nebius_creds: dict[str, str]) -> None:
    """Tear down a VM via Terraform.

    If the local Terraform working directory does not exist (e.g.
    ``--skip-infra`` was used), it is re-created via
    ``prepare_working_dir`` and re-initialized from the S3 remote
    backend so that ``terraform destroy`` can pull existing state.
    """
    from npa.deploy import provisioner

    s3_bucket = nebius_creds.get("s3_bucket") or DEFAULT_S3_BUCKET
    tf_dir = str(provisioner.working_dir_path(PROJECT_ALIAS, spec.name))
    if not Path(tf_dir).exists():
        # Re-create the working dir and pull state from S3 backend.
        logger.info(
            "Local Terraform dir missing for %s — re-creating from S3 state.",
            spec.name,
        )
        tf_dir = str(provisioner.prepare_working_dir(
            PROJECT_ALIAS,
            spec.name,
            bucket=s3_bucket,
            region=REGION,
            endpoint=nebius_creds["s3_endpoint"],
        ))

    try:
        provisioner.init(
            tf_dir=tf_dir,
            backend_config={
                "access_key": nebius_creds["nebius_api_key"],
                "secret_key": nebius_creds["nebius_secret_key"],
            },
        )
    except ProvisionerError as exc:
        raise TwoVMDistillError(f"terraform init (destroy) failed for {spec.name}: {exc}") from exc

    tf_vars: dict[str, str] = {
        "nebius_project_id": nebius_creds["nebius_project_id"],
        "iam_token": nebius_creds["iam_token"],
        "service_account_id": nebius_creds["service_account_id"],
        "nebius_region": REGION,
        "instance_name": spec.name,
        "gpu_platform": spec.gpu_platform,
        "gpu_preset": spec.gpu_preset,
        "enable_preemptible": "false",
        "nebius_api_key": nebius_creds["nebius_api_key"],
        "nebius_secret_key": nebius_creds["nebius_secret_key"],
        "s3_bucket": s3_bucket,
        "s3_endpoint": nebius_creds["s3_endpoint"],
    }

    logger.info("Destroying %s ...", spec.name)
    try:
        provisioner.destroy(tf_dir=tf_dir, tf_vars=tf_vars)
    except ProvisionerError as exc:
        raise TwoVMDistillError(f"terraform destroy failed for {spec.name}: {exc}") from exc

    logger.info("%s destroyed.", spec.name)


def _wait_for_ssh(
    ssh: SSHClient,
    label: str,
    *,
    retries: int = 60,
    interval: float = 5.0,
) -> None:
    """Block until the VM accepts SSH and cloud-init has finished."""
    logger.info("Waiting for SSH on %s ...", label)
    for attempt in range(1, retries + 1):
        try:
            code, _, _ = ssh.run("true")
            if code == 0:
                break
        except SSHError:
            pass
        if attempt == retries:
            raise TwoVMDistillError(f"SSH to {label} not ready after {retries} attempts")
        time.sleep(interval)

    logger.info("SSH to %s connected. Waiting for cloud-init ...", label)
    for attempt in range(1, 120 + 1):
        try:
            code, _, _ = ssh.run("test -f /var/lib/cloud/instance/boot-finished")
            if code == 0:
                logger.info("cloud-init finished on %s.", label)
                return
        except SSHError:
            pass
        if attempt == 120:
            raise TwoVMDistillError(f"cloud-init on {label} not finished after 20 min")
        time.sleep(10)


# ── VM runtime setup ──────────────────────────────────────────────────────


def _sftp_upload(ssh: SSHClient, local_path: str, remote_path: str) -> None:
    """Upload a file via SFTP using paramiko."""
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key_path = os.path.expanduser(ssh._config.key_path)
    try:
        client.connect(
            hostname=ssh._config.host,
            username=ssh._config.user,
            key_filename=key_path,
            timeout=15,
            look_for_keys=False,
        )
        sftp = client.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()
    finally:
        client.close()


def _setup_vm(ssh: SSHClient, spec: VMSpec, label: str) -> None:
    """Install conda env, framework, and npa CLI on a freshly provisioned VM.

    1. Tar only the pip-installable parts of the npa package (src/,
       pyproject.toml, deploy/, setup/ — excludes .venv, __pycache__,
       .git), upload via SFTP.
    2. Extract to ``/opt/npa/repo/npa/`` on the VM so that the setup
       scripts' ``pip install -e /opt/npa/repo/npa[...]`` path works.
    3. Install Miniforge to ``/opt/conda`` via sudo if conda is not
       present, using absolute paths so it works in non-login shells.
    4. Run the bundled setup script (install_genesis.sh or
       install_lerobot.sh) which creates the conda env, installs the
       framework, and ``pip install -e npa[extra]``.
    5. Verify the ``npa`` entrypoint is on PATH inside the conda env.
    """
    user = ssh._config.user

    logger.info("[%s] Uploading npa package ...", label)

    # Package only the pip-installable parts of the npa source tree.
    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
        archive_path = tmp.name
    try:
        subprocess.run(
            ["tar", "-czf", archive_path]
            + _TAR_EXCLUDES
            + ["-C", str(_NPA_PACKAGE_ROOT), "."],
            check=True,
            capture_output=True,
        )
        try:
            ssh.run_or_raise(
                f"sudo mkdir -p /opt/npa && sudo chown {user}:{user} /opt/npa"
            )
        except SSHError as exc:
            raise TwoVMDistillError(f"[{label}] Failed to create /opt/npa: {exc}") from exc

        _sftp_upload(ssh, archive_path, "/tmp/npa-src.tgz")
    finally:
        os.unlink(archive_path)

    # Extract to /opt/npa/repo/npa/ so the setup scripts' install path
    # (``pip install -e /opt/npa/repo/npa[...]``) resolves correctly.
    # The repo root is /opt/npa/repo/ and the npa package is a
    # subdirectory — matching the git layout.
    logger.info("[%s] Extracting npa package on VM ...", label)
    try:
        ssh.run_or_raise(
            "rm -rf /opt/npa/repo/npa && mkdir -p /opt/npa/repo/npa && "
            "tar -xzf /tmp/npa-src.tgz -C /opt/npa/repo/npa 2>/dev/null && "
            "rm -f /tmp/npa-src.tgz"
        )
    except SSHError as exc:
        raise TwoVMDistillError(f"[{label}] Failed to extract npa package: {exc}") from exc

    # Verify pyproject.toml landed in the right place.
    try:
        ssh.run_or_raise("test -f /opt/npa/repo/npa/pyproject.toml")
    except SSHError as exc:
        raise TwoVMDistillError(
            f"[{label}] pyproject.toml not found at /opt/npa/repo/npa/pyproject.toml "
            f"after extraction — tar archive may be malformed: {exc}"
        ) from exc

    # Install Miniforge if conda is not present.  Uses sudo to write
    # to /opt/conda, then chowns to the SSH user.  The entire install
    # uses absolute paths (no ~/.bashrc dependency) so later paramiko
    # exec_command() calls can find it.
    logger.info("[%s] Ensuring conda is available ...", label)
    try:
        code, _, _ = ssh.run(f"test -x {_CONDA_BIN}")
    except SSHError:
        code = 1
    if code != 0:
        logger.info("[%s] Installing Miniforge to %s ...", label, _CONDA_PREFIX)
        try:
            ssh.run_or_raise(
                "curl -fsSL -o /tmp/miniforge.sh "
                "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && "
                f"sudo bash /tmp/miniforge.sh -b -p {_CONDA_PREFIX} && "
                "rm -f /tmp/miniforge.sh && "
                f"sudo chown -R {user}:{user} {_CONDA_PREFIX}"
            )
        except SSHError as exc:
            raise TwoVMDistillError(f"[{label}] Miniforge install failed: {exc}") from exc

    # Double-check conda is executable.
    try:
        ssh.run_or_raise(f"{_CONDA_BIN} --version")
    except SSHError as exc:
        raise TwoVMDistillError(
            f"[{label}] conda not executable at {_CONDA_BIN} after install: {exc}"
        ) from exc

    # Run the setup script.  We prefix PATH with /opt/conda/bin so the
    # script's bare ``conda`` calls work even without .bashrc.
    setup_script = _SETUP_DIR / spec.setup_script
    if not setup_script.exists():
        raise TwoVMDistillError(
            f"Setup script not found: {setup_script}. "
            f"Expected at npa/src/npa/setup/{spec.setup_script}"
        )

    logger.info("[%s] Uploading setup script %s ...", label, spec.setup_script)
    _sftp_upload(ssh, str(setup_script), f"/tmp/{spec.setup_script}")

    logger.info("[%s] Running setup script (this may take several minutes) ...", label)
    try:
        code, stdout, stderr = ssh.run(
            f'export PATH="{_CONDA_PREFIX}/bin:$PATH" && '
            f'bash /tmp/{spec.setup_script}',
            stream=True,
        )
    except SSHError as exc:
        raise TwoVMDistillError(f"[{label}] Setup script SSH error: {exc}") from exc

    if code != 0:
        raise TwoVMDistillError(
            f"[{label}] Setup script failed (exit {code}): "
            f"{stderr.strip()[-500:] if stderr else '(no stderr)'}"
        )

    # Verify npa entrypoint inside the conda env, using absolute conda path.
    logger.info("[%s] Verifying npa CLI in conda env '%s' ...", label, spec.conda_env)
    verify_cmd = (
        f'{_conda_activate(spec.conda_env)}'
        f"npa --help >/dev/null 2>&1 && echo NPA_CLI_OK"
    )
    try:
        code, stdout, _ = ssh.run(verify_cmd)
    except SSHError as exc:
        raise TwoVMDistillError(f"[{label}] npa CLI verification SSH error: {exc}") from exc

    if code != 0 or "NPA_CLI_OK" not in stdout:
        raise TwoVMDistillError(
            f"[{label}] npa CLI not available in conda env '{spec.conda_env}'. "
            f"Setup script may have failed to install it."
        )

    logger.info("[%s] VM setup complete.", label)


def _write_s3_env(
    ssh: SSHClient,
    nebius_creds: dict[str, str],
    label: str,
) -> None:
    """Update S3 credentials in /opt/lerobot/.env, preserving other vars.

    Cloud-init seeds /opt/lerobot/.env with S3 credentials *and* runtime
    settings (MUJOCO_GL, HF_LEROBOT_HOME, PYOPENGL_PLATFORM, etc.).
    On reused VMs the S3 keys may be stale while the runtime vars are
    still correct, so we read-merge-write instead of overwriting wholesale.
    """
    s3_endpoint = nebius_creds.get("s3_endpoint", "")
    s3_bucket = nebius_creds.get("s3_bucket", "")
    s3_vars = {
        "AWS_ACCESS_KEY_ID": nebius_creds.get("nebius_api_key", ""),
        "AWS_SECRET_ACCESS_KEY": nebius_creds.get("nebius_secret_key", ""),
        "NEBIUS_S3_ENDPOINT": s3_endpoint,
        "AWS_ENDPOINT_URL": s3_endpoint,
        "NEBIUS_S3_BUCKET": s3_bucket,
    }

    logger.info("[%s] Updating S3 credentials in /opt/lerobot/.env", label)
    try:
        # Read existing env file (may not exist on a fresh VM).
        code, existing_content, _ = ssh.run("cat /opt/lerobot/.env 2>/dev/null")
        existing_vars: dict[str, str] = {}
        if code == 0 and existing_content.strip():
            for line in existing_content.splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    existing_vars[key.strip()] = value.strip()

        # Merge: S3 vars override, everything else is preserved.
        merged = {**existing_vars, **s3_vars}
        env_content = "".join(f"{k}={v}\n" for k, v in merged.items())

        ssh.run_or_raise(
            f"sudo mkdir -p /opt/lerobot && "
            f"cat > /tmp/lerobot-env << 'ENVEOF'\n{env_content}ENVEOF\n"
            f"sudo mv /tmp/lerobot-env /opt/lerobot/.env && "
            f"sudo chown {ssh._config.user}:{ssh._config.user} /opt/lerobot/.env && "
            f"sudo chmod 600 /opt/lerobot/.env"
        )
    except SSHError as exc:
        raise TwoVMDistillError(
            f"[{label}] Failed to update /opt/lerobot/.env: {exc}"
        ) from exc


def _deploy_http_server(
    ssh: SSHClient,
    spec: VMSpec,
    nebius_creds: dict[str, str],
    label: str,
    *,
    server_port: int = 8080,
) -> None:
    """Install and start npa-lerobot-server on a VM using the conda env.

    This mirrors what ``configurator.deploy_server`` does for the venv-based
    deploy path, but uses the conda env's python so the distill training VM
    is manageable via ``npa workbench lerobot status/serve/...``.
    """
    user = ssh._config.user
    conda_python = f"{_CONDA_PREFIX}/envs/{spec.conda_env}/bin/python"

    # 1. Install server extras (fastapi, uvicorn, numpy) into the conda env.
    logger.info("[%s] Installing npa[server] in conda env '%s' ...", label, spec.conda_env)
    try:
        ssh.run_or_raise(
            f'{_conda_activate(spec.conda_env)}'
            f'pip install -e "/opt/npa/repo/npa[server]"'
        )
    except SSHError as exc:
        raise TwoVMDistillError(
            f"[{label}] Failed to install npa[server]: {exc}"
        ) from exc

    # 2. Create required directories.
    logger.info("[%s] Creating server directories ...", label)
    try:
        ssh.run_or_raise(
            f"sudo mkdir -p /var/log/npa-lerobot /opt/lerobot/checkpoints "
            f"/opt/lerobot/job_status /etc/npa-lerobot-server && "
            f"sudo chown {user}:{user} /var/log/npa-lerobot "
            f"/opt/lerobot/checkpoints /opt/lerobot/job_status"
        )
    except SSHError as exc:
        raise TwoVMDistillError(
            f"[{label}] Failed to create server directories: {exc}"
        ) from exc

    # 3. Write env file for the systemd service.
    s3_endpoint = nebius_creds.get("s3_endpoint", "")
    s3_bucket = nebius_creds.get("s3_bucket", "")
    env_content = (
        f"NPA_SERVER_HOST=0.0.0.0\n"
        f"NPA_SERVER_PORT={server_port}\n"
        f"NPA_CHECKPOINT_DIR=/opt/lerobot/checkpoints\n"
        f"NPA_CHECKPOINT_BUCKET=s3://{s3_bucket}/checkpoints/\n"
        f"NPA_JOB_STATUS_DIR=/opt/lerobot/job_status\n"
        f"NPA_LOG_DIR=/var/log/npa-lerobot\n"
        f"AWS_ENDPOINT_URL={s3_endpoint}\n"
        f"AWS_ACCESS_KEY_ID={nebius_creds.get('nebius_api_key', '')}\n"
        f"AWS_SECRET_ACCESS_KEY={nebius_creds.get('nebius_secret_key', '')}\n"
    )
    try:
        ssh.run_or_raise(
            f"cat > /tmp/npa-server.env << 'ENVEOF'\n{env_content}ENVEOF\n"
            f"sudo mv /tmp/npa-server.env /etc/npa-lerobot-server/env && "
            f"sudo chmod 600 /etc/npa-lerobot-server/env"
        )
    except SSHError as exc:
        raise TwoVMDistillError(
            f"[{label}] Failed to write server env file: {exc}"
        ) from exc

    # 4. Write systemd unit that uses the conda env's python.
    unit = (
        "[Unit]\n"
        "Description=NPA LeRobot Server\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"User={user}\n"
        f"Group={user}\n"
        "EnvironmentFile=/etc/npa-lerobot-server/env\n"
        f"ExecStart={conda_python} -m npa.server.app\n"
        "WorkingDirectory=/opt/lerobot\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        "StandardOutput=append:/var/log/npa-lerobot/server.log\n"
        "StandardError=append:/var/log/npa-lerobot/server.log\n"
        "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )
    try:
        ssh.run_or_raise(
            f"cat > /tmp/npa-lerobot-server.service << 'UNITEOF'\n{unit}UNITEOF\n"
            f"sudo mv /tmp/npa-lerobot-server.service /etc/systemd/system/ && "
            f"sudo systemctl daemon-reload && "
            f"sudo systemctl enable npa-lerobot-server && "
            f"sudo systemctl restart npa-lerobot-server"
        )
    except SSHError as exc:
        raise TwoVMDistillError(
            f"[{label}] Failed to start npa-lerobot-server: {exc}"
        ) from exc

    # 5. Health check — poll until the server responds.
    logger.info("[%s] Waiting for server health check on port %d ...", label, server_port)
    vm_ip = ssh._config.host
    for attempt in range(1, 16):
        try:
            code, stdout, _ = ssh.run(
                f"curl -sf http://127.0.0.1:{server_port}/health"
            )
            if code == 0:
                logger.info("[%s] Server healthy.", label)
                return
        except SSHError:
            pass
        time.sleep(3)

    raise TwoVMDistillError(
        f"[{label}] npa-lerobot-server not healthy after 45s on port {server_port}"
    )


# ── S3 sync helpers ────────────────────────────────────────────────────────


def _s3_upload(
    ssh: SSHClient,
    conda_env: str,
    local_path: str,
    s3_bucket: str,
    s3_prefix: str,
) -> None:
    """Upload a directory from a remote VM to S3.

    Raises TwoVMDistillError if the source directory is missing or empty.
    """
    activate = _conda_activate(conda_env)
    script = (
        f"import boto3, os, pathlib, sys; "
        f"s3 = boto3.client('s3', "
        f"endpoint_url=os.environ.get('NEBIUS_S3_ENDPOINT', os.environ.get('AWS_ENDPOINT_URL', '')), "
        f"aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID', ''), "
        f"aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', '')); "
        f"base = pathlib.Path('{local_path}'); "
        f"files = [f for f in base.rglob('*') if f.is_file()]; "
        f"n = len(files); "
        f"print(f's3_upload_count={{n}}'); "
        f"[s3.upload_file(str(f), '{s3_bucket}', '{s3_prefix}' + str(f.relative_to(base))) "
        f"for f in files]; "
        f"print('s3_upload_done')"
    )
    cmd = f'{activate}python3 -c "{script}"'

    try:
        code, stdout, stderr = ssh.run(cmd, stream=True)
    except SSHError as exc:
        raise TwoVMDistillError(f"S3 upload failed: {exc}") from exc

    if code != 0:
        raise TwoVMDistillError(f"S3 upload failed (exit {code}): {stderr.strip()[-500:]}")
    if "s3_upload_done" not in stdout:
        raise TwoVMDistillError("S3 upload: completion marker not found in output")

    # Verify at least one file was transferred.
    for line in stdout.splitlines():
        if line.startswith("s3_upload_count="):
            count = int(line.split("=", 1)[1])
            if count == 0:
                raise TwoVMDistillError(
                    f"S3 upload transferred 0 files from {local_path}. "
                    f"The source directory may be missing or empty."
                )
            logger.info("  S3 upload: %d files transferred.", count)
            break


def _s3_download(
    ssh: SSHClient,
    conda_env: str,
    s3_bucket: str,
    s3_prefix: str,
    local_path: str,
) -> None:
    """Download from S3 to a directory on a remote VM.

    Raises TwoVMDistillError if no objects are found under the S3 prefix.
    """
    activate = _conda_activate(conda_env)
    script = (
        f"import boto3, os, pathlib; "
        f"s3 = boto3.client('s3', "
        f"endpoint_url=os.environ.get('NEBIUS_S3_ENDPOINT', os.environ.get('AWS_ENDPOINT_URL', '')), "
        f"aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID', ''), "
        f"aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', '')); "
        f"pag = s3.get_paginator('list_objects_v2'); "
        f"dest = pathlib.Path('{local_path}'); "
        f"keys = ["
        f"o['Key'] "
        f"for page in pag.paginate(Bucket='{s3_bucket}', Prefix='{s3_prefix}') "
        f"for o in page.get('Contents', []) "
        f"if o['Key'][len('{s3_prefix}'):]"
        f"]; "
        f"[("
        f"os.makedirs(str((dest / k[len('{s3_prefix}'):]).parent), exist_ok=True), "
        f"s3.download_file('{s3_bucket}', k, str(dest / k[len('{s3_prefix}'):]))) "
        f"for k in keys"
        f"]; "
        f"print(f's3_download_count={{len(keys)}}'); "
        f"print('s3_download_done')"
    )
    cmd = f'mkdir -p {local_path} && {activate}python3 -c "{script}"'

    try:
        code, stdout, stderr = ssh.run(cmd, stream=True)
    except SSHError as exc:
        raise TwoVMDistillError(f"S3 download failed: {exc}") from exc

    if code != 0:
        raise TwoVMDistillError(f"S3 download failed (exit {code}): {stderr.strip()[-500:]}")
    if "s3_download_done" not in stdout:
        raise TwoVMDistillError("S3 download: completion marker not found in output")

    # Verify at least one file was transferred.
    for line in stdout.splitlines():
        if line.startswith("s3_download_count="):
            count = int(line.split("=", 1)[1])
            if count == 0:
                raise TwoVMDistillError(
                    f"S3 download transferred 0 files from "
                    f"s3://{s3_bucket}/{s3_prefix}. "
                    f"The upstream stage may not have produced output."
                )
            logger.info("  S3 download: %d files transferred.", count)
            break


# ── Stage runner ───────────────────────────────────────────────────────────


def _run_stage(
    ssh: SSHClient,
    conda_env: str,
    stage_name: str,
    command: str,
    remote_base: str,
) -> dict[str, Any]:
    """Execute a stage command on a remote VM via SSH."""
    activate = _conda_activate(conda_env)
    full_cmd = f"mkdir -p {remote_base} && {activate}{command}"

    logger.info("[%s] Running: %s", stage_name, command)

    try:
        code, stdout, stderr = ssh.run(full_cmd, stream=True)
    except SSHError as exc:
        return {
            "status": "failed",
            "error": f"SSH error: {exc}",
        }

    if code != 0:
        return {
            "status": "failed",
            "exit_code": code,
            "stderr": stderr.strip()[-500:] if stderr else "",
        }

    # Attempt to extract structured JSON output from stdout.  CLI commands
    # invoked with ``--output-format json`` print a pretty-printed JSON
    # object as their last output block.  We find it by scanning backwards
    # for the closing ``}`` and matching opening ``{``.
    output = {}
    if stdout:
        stripped = stdout.strip()
        # Find the last top-level JSON object in stdout.
        end = stripped.rfind("}")
        if end != -1:
            # Walk backwards to find the matching opening brace.
            depth = 0
            start = end
            for i in range(end, -1, -1):
                if stripped[i] == "}":
                    depth += 1
                elif stripped[i] == "{":
                    depth -= 1
                if depth == 0:
                    start = i
                    break
            if depth == 0:
                try:
                    output = json.loads(stripped[start:end + 1])
                except (json.JSONDecodeError, ValueError):
                    pass

    return {
        "status": "success",
        "exit_code": code,
        "output": output,
    }


# ── Workbench config persistence ──────────────────────────────────────────


def _save_workbench_config(
    spec: VMSpec,
    tf_outputs: dict[str, Any],
    nebius_creds: dict[str, str],
    *,
    include_endpoint: bool = True,
) -> None:
    """Persist VM details into ~/.npa/config.yaml so later stages can
    resolve them via ``resolve_config(project=..., name=...)``.

    Parameters
    ----------
    include_endpoint:
        Write the ``endpoint`` key only when the VM runs the HTTP server.
        The sim VM does not run one, so pointing at port 8080 there would
        advertise a dead endpoint.
    """
    vm_ip = tf_outputs["vm_ip"]
    wb_entry: dict[str, Any] = {
        "gpu_platform": spec.gpu_platform,
        "gpu_preset": spec.gpu_preset,
        "tf_instance_name": spec.name,
        "workbench_type": "genesis" if not include_endpoint else "lerobot",
        "ssh": {
            "host": vm_ip,
            "user": tf_outputs.get("ssh_user", "ubuntu"),
            "key_path": tf_outputs.get("ssh_key_path", "~/.ssh/id_ed25519"),
        },
        "storage": {
            "checkpoint_bucket": f"s3://{nebius_creds['s3_bucket']}/checkpoints/",
            "endpoint_url": nebius_creds["s3_endpoint"],
            "aws_access_key_id": nebius_creds["nebius_api_key"],
            "aws_secret_access_key": nebius_creds["nebius_secret_key"],
        },
    }
    if include_endpoint:
        wb_entry["endpoint"] = f"http://{vm_ip}:8080"

    write_config({
        "projects": {
            PROJECT_ALIAS: {
                "project_id": PROJECT_ID,
                "tenant_id": TENANT_ID,
                "region": REGION,
                "workbenches": {
                    spec.name: wb_entry,
                },
            },
        },
    })

    # Deep-merge does not delete keys — scrub any stale endpoint that
    # survived from a previous run's config so resolve_config() won't
    # find a dead HTTP endpoint for a VM that never runs a server.
    if not include_endpoint:
        from npa.clients.config import _load_yaml, CONFIG_PATH
        import yaml as _yaml

        cfg = _load_yaml()
        wb = (cfg.get("projects", {})
              .get(PROJECT_ALIAS, {})
              .get("workbenches", {})
              .get(spec.name, {}))
        if "endpoint" in wb:
            del wb["endpoint"]
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with CONFIG_PATH.open("w") as _f:
                _yaml.dump(cfg, _f, default_flow_style=False, sort_keys=False)


# ── Credential resolution ─────────────────────────────────────────────────


def _resolve_creds_from_config() -> dict[str, str]:
    """Build the nebius_creds dict from saved config (for --skip-infra).

    Uses the stored workbench storage keys + environment fields so that
    ``bootstrap_environment()`` does not need to run.
    """
    from npa.clients.config import resolve_environment, resolve_ssh_config

    env = resolve_environment(PROJECT_ALIAS)
    if env is None:
        raise TwoVMDistillError(
            f"No environment config found for project '{PROJECT_ALIAS}' in "
            f"~/.npa/config.yaml. Run without --skip-infra first."
        )

    # Pull S3 credentials from either workbench's saved storage config.
    # The sim VM has no HTTP endpoint, so use resolve_ssh_config which
    # does not require the endpoint field.
    wb_cfg = resolve_ssh_config(project=PROJECT_ALIAS, name=SIM_VM.name)

    s3_endpoint = wb_cfg.storage.endpoint_url
    if not s3_endpoint:
        s3_endpoint = f"https://storage.{env.region}.nebius.cloud"

    bucket = wb_cfg.storage.checkpoint_bucket
    # checkpoint_bucket is stored as "s3://bucket/checkpoints/" — extract just the bucket name.
    if bucket.startswith("s3://"):
        bucket = bucket[len("s3://"):].split("/")[0]
    elif not bucket:
        raise TwoVMDistillError(
            "No S3 bucket found in saved config. Run without --skip-infra first."
        )

    return {
        "nebius_project_id": env.project_id,
        "nebius_region": env.region,
        "s3_bucket": bucket,
        "s3_endpoint": s3_endpoint,
        "nebius_api_key": wb_cfg.storage.aws_access_key_id,
        "nebius_secret_key": wb_cfg.storage.aws_secret_access_key,
        # These are only needed for provisioning, not for S3 ops.
        "iam_token": "",
        "service_account_id": "",
    }


# ── Result persistence ─────────────────────────────────────────────────────


def _save_result(base_dir: Path, result: dict[str, Any]) -> None:
    path = base_dir / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(result, f, indent=2)


# ── Main workflow ──────────────────────────────────────────────────────────


def distill(
    *,
    teardown: bool = False,
    skip_infra: bool = False,
    skip_setup: bool = False,
    n_envs: int = 4096,
    teacher_max_iterations: int = 500,
    demo_domain_randomize: bool = True,
    demo_fps: int = 20,
    demo_seed: int = 42,
    allow_failure_demos: bool = False,
    student_policy: str = "act",
    student_epochs: int = 100,
    student_batch_size: int = 64,
    eval_n_episodes: int = 1024,
    eval_seed: int = 7777,
    action_space: str = "cartesian",
) -> dict[str, Any]:
    """Provision two VMs and run the full distillation pipeline.

    Args:
        teardown: Destroy both VMs after the workflow completes (including
            on failure — covers partial provisioning too).
        skip_infra: Skip VM provisioning; assume VMs are already running
            and their config is in ~/.npa/config.yaml.  Also skips the
            Nebius bootstrap — S3 credentials are read from saved config.
        skip_setup: Skip the runtime setup phase (conda env + npa install).
            Use when VMs already have the correct environment from a
            previous run.
        n_envs: Parallel envs for Genesis simulation stages.
        teacher_max_iterations: PPO training iterations.
        demo_domain_randomize: Apply domain randomization in demo generation.
        demo_fps: Camera frame rate for demo recording.
        student_policy: Policy type for student training (act, diffusion, smolvla).
        student_epochs: Training epochs for student.
        student_batch_size: Batch size for student training.
        eval_n_episodes: Number of eval episodes for the student.
        action_space: "cartesian" (4D: delta xyz + gripper) or
            "joint" (8D: delta joint positions + gripper).

    Returns:
        Result dict with run_id, per-stage status, and infrastructure details.
    """
    run_id = generate_run_id()
    base_dir = Path(f"./runs/{run_id}")
    base_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "run_id": run_id,
        "config": {
            "sim_vm": SIM_VM.name,
            "train_vm": TRAIN_VM.name,
            "region": REGION,
            "n_envs": n_envs,
            "student_policy": student_policy,
        },
        "infrastructure": {},
        "stages": {},
    }

    # ── Step 1: Resolve credentials ────────────────────────────────
    if skip_infra:
        logger.info("=== Resolving credentials from saved config (--skip-infra) ===")
        nebius_creds = _resolve_creds_from_config()
    else:
        logger.info("=== Bootstrapping Nebius environment ===")
        try:
            nebius_creds = bootstrap_environment(
                PROJECT_ID,
                TENANT_ID,
                REGION,
                on_status=lambda msg: logger.info("  %s", msg),
            )
        except NebiusError as exc:
            raise TwoVMDistillError(f"Nebius bootstrap failed: {exc}") from exc

    s3_bucket = nebius_creds.get("s3_bucket") or DEFAULT_S3_BUCKET
    s3_prefix = f"distill/{run_id}/"

    # Track which VMs exist and should be torn down.  Populated during
    # provisioning (append on success) or from saved config when
    # --skip-infra is used, so --teardown works in both modes.
    teardown_specs: list[VMSpec] = []

    try:
        # ── Step 2: Provision VMs ──────────────────────────────────
        if not skip_infra:
            logger.info("=== Provisioning sim VM: %s ===", SIM_VM.name)
            sim_outputs = _provision_vm(SIM_VM, nebius_creds)
            teardown_specs.append(SIM_VM)
            _save_workbench_config(SIM_VM, sim_outputs, nebius_creds, include_endpoint=False)
            result["infrastructure"]["sim"] = {
                "name": SIM_VM.name,
                "ip": sim_outputs["vm_ip"],
                "gpu": SIM_VM.gpu_platform,
            }

            logger.info("=== Provisioning train VM: %s ===", TRAIN_VM.name)
            train_outputs = _provision_vm(TRAIN_VM, nebius_creds)
            teardown_specs.append(TRAIN_VM)
            _save_workbench_config(TRAIN_VM, train_outputs, nebius_creds)

            # Set default workbench to the train VM since it runs the
            # HTTP server — bare `npa workbench lerobot status` should
            # resolve to the VM that actually has a reachable endpoint.
            write_config({"default_workbench": TRAIN_VM.name})
            result["infrastructure"]["train"] = {
                "name": TRAIN_VM.name,
                "ip": train_outputs["vm_ip"],
                "gpu": TRAIN_VM.gpu_platform,
            }

            sim_ip = sim_outputs["vm_ip"]
            sim_user = sim_outputs.get("ssh_user", "ubuntu")
            sim_key = sim_outputs.get("ssh_key_path", "~/.ssh/id_ed25519")
            train_ip = train_outputs["vm_ip"]
            train_user = train_outputs.get("ssh_user", "ubuntu")
            train_key = train_outputs.get("ssh_key_path", "~/.ssh/id_ed25519")
        else:
            # Read VM details from existing config.  The sim VM has no
            # HTTP endpoint so use resolve_ssh_config for both.
            from npa.clients.config import resolve_ssh_config

            sim_cfg = resolve_ssh_config(project=PROJECT_ALIAS, name=SIM_VM.name)
            train_cfg = resolve_ssh_config(project=PROJECT_ALIAS, name=TRAIN_VM.name)

            sim_ip = sim_cfg.ssh.host
            sim_user = sim_cfg.ssh.user
            sim_key = sim_cfg.ssh.key_path
            train_ip = train_cfg.ssh.host
            train_user = train_cfg.ssh.user
            train_key = train_cfg.ssh.key_path

            result["infrastructure"]["sim"] = {"name": SIM_VM.name, "ip": sim_ip}
            result["infrastructure"]["train"] = {"name": TRAIN_VM.name, "ip": train_ip}

            # VMs exist from a prior run — mark both for teardown.
            teardown_specs = [SIM_VM, TRAIN_VM]

        _save_result(base_dir, result)

        sim_ssh = SSHClient(SSHConfig(host=sim_ip, user=sim_user, key_path=sim_key))
        train_ssh = SSHClient(SSHConfig(host=train_ip, user=train_user, key_path=train_key))

        # ── Steps 3-7: setup + pipeline ────────────────────────────
        _run_pipeline(
            sim_ssh=sim_ssh,
            train_ssh=train_ssh,
            nebius_creds=nebius_creds,
            skip_infra=skip_infra,
            skip_setup=skip_setup,
            run_id=run_id,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            n_envs=n_envs,
            teacher_max_iterations=teacher_max_iterations,
            demo_domain_randomize=demo_domain_randomize,
            demo_fps=demo_fps,
            demo_seed=demo_seed,
            allow_failure_demos=allow_failure_demos,
            student_policy=student_policy,
            student_epochs=student_epochs,
            student_batch_size=student_batch_size,
            eval_n_episodes=eval_n_episodes,
            eval_seed=eval_seed,
            action_space=action_space,
            result=result,
            base_dir=base_dir,
        )

    except TwoVMDistillError:
        result.setdefault("status", "failed")
        _save_result(base_dir, result)
        raise

    finally:
        # Teardown always runs — even on failure, even on partial
        # provisioning — so we never leak non-preemptible GPU instances.
        # Works with both --skip-infra and fresh provisioning.
        if teardown and teardown_specs:
            logger.info("=== Tearing down infrastructure (%d VMs) ===", len(teardown_specs))
            # _destroy_vm needs a valid IAM token + service_account_id
            # for Terraform.  When --skip-infra these are empty, so
            # bootstrap just enough to get them.
            if not nebius_creds.get("iam_token") or not nebius_creds.get("service_account_id"):
                try:
                    fresh = bootstrap_environment(
                        PROJECT_ID, TENANT_ID, REGION,
                        on_status=lambda msg: logger.info("  %s", msg),
                    )
                    nebius_creds.update(fresh)
                except NebiusError as exc:
                    logger.warning(
                        "Nebius bootstrap for teardown failed: %s. "
                        "VMs may need manual cleanup.", exc,
                    )
            for spec in teardown_specs:
                try:
                    _destroy_vm(spec, nebius_creds)
                except TwoVMDistillError as exc:
                    logger.warning("Destroy failed for %s: %s", spec.name, exc)

    result.setdefault("status", "success")
    result["s3_base"] = f"s3://{s3_bucket}/{s3_prefix}"
    _save_result(base_dir, result)

    logger.info(
        "=== Workflow %s — run_id=%s, s3=%s ===",
        result["status"].upper(),
        run_id,
        result["s3_base"],
    )
    return result


def _run_pipeline(
    *,
    sim_ssh: SSHClient,
    train_ssh: SSHClient,
    nebius_creds: dict[str, str],
    skip_infra: bool,
    skip_setup: bool,
    run_id: str,
    s3_bucket: str,
    s3_prefix: str,
    n_envs: int,
    teacher_max_iterations: int,
    demo_domain_randomize: bool,
    demo_fps: int,
    demo_seed: int,
    allow_failure_demos: bool,
    student_policy: str,
    student_epochs: int,
    student_batch_size: int,
    eval_n_episodes: int,
    eval_seed: int,
    action_space: str,
    result: dict[str, Any],
    base_dir: Path,
) -> None:
    """Run the wait -> setup -> stages pipeline.

    Extracted so that ``distill`` can wrap it in try/finally
    for teardown.
    """
    # ── Step 3: Wait for VMs ────────────────────────────────────────
    if not skip_infra:
        _wait_for_ssh(sim_ssh, f"sim ({SIM_VM.name})")
        _wait_for_ssh(train_ssh, f"train ({TRAIN_VM.name})")

    # ── Step 4: Install runtime on each VM ──────────────────────────
    if not skip_setup:
        logger.info("=== Setting up sim VM runtime ===")
        _setup_vm(sim_ssh, SIM_VM, f"sim ({SIM_VM.name})")

        logger.info("=== Setting up train VM runtime ===")
        _setup_vm(train_ssh, TRAIN_VM, f"train ({TRAIN_VM.name})")

    # Update S3 credentials in /opt/lerobot/.env on both VMs so that
    # _conda_activate() sources them and S3 uploads/downloads use the
    # correct keys (on reused VMs the cloud-init-seeded keys may be
    # stale while other runtime vars in the file are still correct).
    _write_s3_env(sim_ssh, nebius_creds, f"sim ({SIM_VM.name})")
    _write_s3_env(train_ssh, nebius_creds, f"train ({TRAIN_VM.name})")

    # Deploy the HTTP management server on the training VM so it is
    # reachable via ``npa workbench lerobot status/serve/...``.
    logger.info("=== Deploying HTTP server on train VM ===")
    _deploy_http_server(
        train_ssh, TRAIN_VM, nebius_creds,
        f"train ({TRAIN_VM.name})",
    )

    remote_base = f"/opt/npa/runs/{run_id}"

    # ── Stage 1: Train teacher (sim VM — Genesis) ──────────────────
    logger.info("=== [1/5] Training teacher on %s ===", SIM_VM.name)
    stage_result = _run_stage(
        sim_ssh,
        SIM_VM.conda_env,
        "train_teacher",
        f"npa workbench genesis train-teacher "
        f"--n-envs {n_envs} "
        f"--max-iterations {teacher_max_iterations} "
        f"--action-space {action_space} "
        f"--output {remote_base}/teacher/",
        remote_base,
    )
    result["stages"]["train_teacher"] = stage_result
    _save_result(base_dir, result)

    if stage_result["status"] != "success":
        result["status"] = "failed"
        _save_result(base_dir, result)
        raise TwoVMDistillError("Stage train_teacher failed")

    # Upload teacher checkpoint to S3.
    logger.info("  Uploading teacher checkpoint to S3 ...")
    _s3_upload(
        sim_ssh, SIM_VM.conda_env,
        f"{remote_base}/teacher/",
        s3_bucket, f"{s3_prefix}teacher/",
    )

    # ── Stage 2: Generate demos (sim VM — Genesis) ─────────────────
    # Demo generation renders cameras per-env, so use fewer parallel
    # envs than teacher training (which is physics-only).  64 envs
    # generates 64 episodes per batch — enough for a training dataset.
    demo_n_envs = min(n_envs, 64)
    logger.info("=== [2/5] Generating demos on %s (n_envs=%d) ===", SIM_VM.name, demo_n_envs)
    stage_result = _run_stage(
        sim_ssh,
        SIM_VM.conda_env,
        "generate_demos",
        f"npa workbench genesis generate-demos "
        f"--checkpoint {remote_base}/teacher/model.pt "
        f"--n-envs {demo_n_envs} "
        f"{'--domain-randomize' if demo_domain_randomize else '--no-domain-randomize'} "
        f"--fps {demo_fps} "
        f"--seed {demo_seed} "
        f"{'--allow-failure-demos' if allow_failure_demos else '--no-failure-demos'} "
        f"--action-space {action_space} "
        f"--output {remote_base}/demos/ "
        f"--output-format json",
        remote_base,
    )
    result["stages"]["generate_demos"] = stage_result
    _save_result(base_dir, result)

    if stage_result["status"] != "success":
        result["status"] = "failed"
        _save_result(base_dir, result)
        raise TwoVMDistillError("Stage generate_demos failed")

    demo_output = stage_result.get("output", {})
    if demo_output.get("includes_failures"):
        logger.warning(
            "Demo dataset includes non-successful rollouts "
            "(teacher_success_rate=%.2f%%). Student will train on failure "
            "trajectories — this may degrade distillation quality.",
            demo_output.get("teacher_success_rate", 0) * 100,
        )

    # ── Teacher eval (held-out baseline for distillation gap) ────────
    # Run the teacher on the eval seed (no cameras, privileged state
    # only) to get a valid held-out baseline.  This is fast — physics
    # only, no rendering.
    logger.info("=== Evaluating teacher under held-out conditions ===")
    teacher_eval_result = _run_stage(
        sim_ssh,
        SIM_VM.conda_env,
        "eval_teacher",
        f"npa workbench genesis eval-teacher "
        f"--checkpoint {remote_base}/teacher/model.pt "
        f"--n-envs {min(n_envs, eval_n_episodes)} "
        f"--seed {eval_seed} "
        f"--action-space {action_space} "
        f"--output-format json",
        remote_base,
    )
    teacher_success_rate = None
    if teacher_eval_result["status"] == "success":
        teval_output = teacher_eval_result.get("output", {})
        teacher_success_rate = teval_output.get("teacher_success_rate")
        if teacher_success_rate is not None:
            logger.info("  Teacher held-out success rate: %.1f%%", teacher_success_rate * 100)
    else:
        logger.warning("Teacher eval failed — distillation gap will not be computed.")
    result["stages"]["eval_teacher"] = teacher_eval_result
    _save_result(base_dir, result)

    # ── Stage 3: Convert to LeRobotDataset (sim VM — CPU) ──────────
    logger.info("=== [3/5] Converting demos to LeRobotDataset on %s ===", SIM_VM.name)
    stage_result = _run_stage(
        sim_ssh,
        SIM_VM.conda_env,
        "convert",
        f"npa adapter convert "
        f"--input {remote_base}/demos/ "
        f"--output {remote_base}/dataset/ "
        f"--fps {demo_fps} "
        f"--robot franka_panda",
        remote_base,
    )
    result["stages"]["convert"] = stage_result
    _save_result(base_dir, result)

    if stage_result["status"] != "success":
        result["status"] = "failed"
        _save_result(base_dir, result)
        raise TwoVMDistillError("Stage convert failed")

    # Upload dataset to S3 for cross-VM handoff.
    logger.info("  Uploading dataset to S3 (sim -> S3 -> train) ...")
    _s3_upload(
        sim_ssh, SIM_VM.conda_env,
        f"{remote_base}/dataset/",
        s3_bucket, f"{s3_prefix}dataset/",
    )

    # ── Stage 4: Train student (train VM — LeRobot) ────────────────
    # Download dataset from S3 to the training VM.
    logger.info("  Downloading dataset from S3 to train VM ...")
    _s3_download(
        train_ssh, TRAIN_VM.conda_env,
        s3_bucket, f"{s3_prefix}dataset/",
        f"{remote_base}/dataset/",
    )

    logger.info("=== [4/5] Training student (%s) on %s ===", student_policy, TRAIN_VM.name)
    stage_result = _run_stage(
        train_ssh,
        TRAIN_VM.conda_env,
        "train_student",
        f"npa workbench lerobot train-student "
        f"--dataset {remote_base}/dataset/ "
        f"--policy {student_policy} "
        f"--epochs {student_epochs} "
        f"--batch-size {student_batch_size} "
        f"--output-dir {remote_base}/student/",
        remote_base,
    )
    result["stages"]["train_student"] = stage_result
    _save_result(base_dir, result)

    if stage_result["status"] != "success":
        result["status"] = "failed"
        _save_result(base_dir, result)
        raise TwoVMDistillError("Stage train_student failed")

    # Upload student checkpoint to S3 for cross-VM handoff.
    logger.info("  Uploading student checkpoint to S3 (train -> S3 -> sim) ...")
    _s3_upload(
        train_ssh, TRAIN_VM.conda_env,
        f"{remote_base}/student/",
        s3_bucket, f"{s3_prefix}student/",
    )

    # ── Stage 5: Eval student (sim VM — Genesis) ───────────────────
    # Download student checkpoint from S3 to sim VM.
    logger.info("  Downloading student checkpoint from S3 to sim VM ...")
    _s3_download(
        sim_ssh, SIM_VM.conda_env,
        s3_bucket, f"{s3_prefix}student/",
        f"{remote_base}/student/",
    )

    # Pass the held-out teacher success rate (computed earlier) so
    # eval_student can report the distillation gap.
    tsr_flag = f"--teacher-success-rate {teacher_success_rate} " if teacher_success_rate is not None else ""
    logger.info("=== [5/5] Evaluating student on %s ===", SIM_VM.name)
    stage_result = _run_stage(
        sim_ssh,
        SIM_VM.conda_env,
        "eval_student",
        f"npa workbench genesis eval-student "
        f"--checkpoint {remote_base}/student/ "
        f"--n-envs {min(n_envs, eval_n_episodes)} "
        f"--n-episodes {eval_n_episodes} "
        f"--seed {eval_seed} "
        f"--action-space {action_space} "
        f"{tsr_flag}"
        f"--output {remote_base}/eval/",
        remote_base,
    )
    result["stages"]["eval_student"] = stage_result
    _save_result(base_dir, result)

    if stage_result["status"] != "success":
        result["status"] = "failed"
        _save_result(base_dir, result)
        raise TwoVMDistillError("Stage eval_student failed")

    # Upload eval results to S3 (best-effort).
    try:
        _s3_upload(
            sim_ssh, SIM_VM.conda_env,
            f"{remote_base}/eval/",
            s3_bucket, f"{s3_prefix}eval/",
        )
    except TwoVMDistillError:
        logger.warning("Failed to upload eval artifacts — non-fatal.")


# ── CLI entry point ────────────────────────────────────────────────────────

def main() -> None:
    """Parse args and run the expert distillation workflow."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Expert distillation: L40S (Genesis) + H100 (LeRobot).",
    )
    parser.add_argument(
        "--teardown", action="store_true",
        help="Destroy both VMs after the workflow completes (even on failure).",
    )
    parser.add_argument(
        "--skip-infra", action="store_true",
        help="Skip provisioning and Nebius bootstrap; resolve VMs and S3 "
             "credentials from ~/.npa/config.yaml.",
    )
    parser.add_argument(
        "--skip-setup", action="store_true",
        help="Skip runtime setup (conda env + npa install). Use when VMs "
             "already have the correct environment.",
    )
    parser.add_argument(
        "--n-envs", type=int, default=4096,
        help="Parallel environments for simulation (default: 4096).",
    )
    parser.add_argument(
        "--teacher-max-iterations", type=int, default=500,
        help="PPO training iterations for teacher (default: 500).",
    )
    parser.add_argument(
        "--student-policy", default="act",
        choices=["act", "diffusion", "smolvla"],
        help="Student policy type (default: act).",
    )
    parser.add_argument(
        "--student-epochs", type=int, default=100,
        help="Training epochs for student (default: 100).",
    )
    parser.add_argument(
        "--student-batch-size", type=int, default=64,
        help="Batch size for student training (default: 64).",
    )
    parser.add_argument(
        "--eval-n-episodes", type=int, default=1024,
        help="Number of eval episodes (default: 1024).",
    )

    args = parser.parse_args()

    # Validate inputs before doing real work.
    if args.n_envs <= 0:
        parser.error(f"--n-envs must be positive, got {args.n_envs}")
    if args.teacher_max_iterations <= 0:
        parser.error(f"--teacher-max-iterations must be positive, got {args.teacher_max_iterations}")
    if args.student_epochs <= 0:
        parser.error(f"--student-epochs must be positive, got {args.student_epochs}")
    if args.student_batch_size <= 0:
        parser.error(f"--student-batch-size must be positive, got {args.student_batch_size}")
    if args.eval_n_episodes <= 0:
        parser.error(f"--eval-n-episodes must be positive, got {args.eval_n_episodes}")

    try:
        result = distill(
            teardown=args.teardown,
            skip_infra=args.skip_infra,
            skip_setup=args.skip_setup,
            n_envs=args.n_envs,
            teacher_max_iterations=args.teacher_max_iterations,
            student_policy=args.student_policy,
            student_epochs=args.student_epochs,
            student_batch_size=args.student_batch_size,
            eval_n_episodes=args.eval_n_episodes,
        )
    except TwoVMDistillError as exc:
        logger.error("Workflow failed: %s", exc)
        sys.exit(1)

    if result.get("status") != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
