"""Run short Slurm client commands locally or through the Soperator login pod."""

from __future__ import annotations

import subprocess

from .diagnostics import _cleanup, _failure, _record


def _execute(settings, command, evidence_uri):
    try:
        if settings["transport"] == "local":
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
        elif settings["transport"] == "soperator":
            result = _pod_execute(
                settings, ["chroot", "/mnt/jail", *command], evidence_uri
            )
        else:
            raise ValueError("production transport must be local or soperator")
    except Exception as error:
        _failure(evidence_uri, error)
        raise
    if result.returncode or result.stderr:
        _record(
            evidence_uri,
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
    if result.returncode:
        raise RuntimeError(f"Slurm client failed with exit code {result.returncode}")
    return result.stdout.strip()


def _pod_execute(settings, command, evidence_uri):
    from kubernetes import client, config
    from kubernetes.stream import stream

    configuration = client.Configuration()
    if settings.get("context"):
        config.load_kube_config(
            context=settings["context"], client_configuration=configuration
        )
    else:
        config.load_incluster_config(client_configuration=configuration)
    with client.ApiClient(configuration) as api_client:
        core = client.CoreV1Api(api_client)
        connection = stream(
            core.connect_get_namespaced_pod_exec,
            settings["login_pod"],
            settings["namespace"],
            container=settings["login_container"],
            command=command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
        try:
            return _read_connection(connection, command, evidence_uri)
        finally:
            _cleanup(connection.close, evidence_uri)


def _read_connection(connection, command, evidence_uri):
    stdout, stderr = [], []
    while connection.is_open():
        connection.update(timeout=1)
        stdout.append(connection.read_stdout())
        stderr.append(connection.read_stderr())
    stdout.append(connection.read_stdout())
    stderr.append(connection.read_stderr())
    if connection.returncode is None:
        _record(
            evidence_uri,
            {
                "stdout": "".join(stdout),
                "stderr": "".join(stderr),
                "exit_status_missing": True,
            },
        )
        raise RuntimeError("Slurm client connection closed without an exit status")
    return subprocess.CompletedProcess(
        command, connection.returncode, "".join(stdout), "".join(stderr)
    )
