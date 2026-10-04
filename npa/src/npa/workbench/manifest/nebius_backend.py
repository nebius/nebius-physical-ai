"""Nebius backend: translates a validated invocation into an ephemeral
Kubernetes Job via kubectl, then tears it down.

Reuse, don't rebuild: the cluster, GPU operator, and image pulls are
existing machinery. This backend only renders the job spec from
(image, argv, resources, outputs, payload) and parses the results.

Small (json/text) artifacts travel inline as base64 log markers.
Binary artifacts go to the descriptor's artifact_store (S3); the runner
uploads them and reports URIs. S3 inputs are downloaded in-pod by the
runner. The Job carries the repo-standard labels plus a TTL so orphans
are visible to teardown tooling and eventually reaped.
"""

from __future__ import annotations

import base64
import json
import subprocess
import time
import uuid
from pathlib import Path

from .runtime import BackendResult

RUNNER_PY = r"""
import base64, json, os, subprocess, sys

argv = json.loads(os.environ["MANIFEST_ARGV"])
outputs = json.loads(os.environ["MANIFEST_OUTPUTS"])  # [[name, path, format]]
s3_inputs = json.loads(os.environ.get("MANIFEST_S3_INPUTS", "[]"))
s3_outputs = set(json.loads(os.environ.get("MANIFEST_S3_OUTPUTS", "[]")))
s3_prefix = os.environ.get("MANIFEST_S3_PREFIX", "")

s3 = None
if s3_inputs or s3_outputs:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "boto3==1.43.108"], check=False
    )
    import boto3

    s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])

# Declared system dependencies (descriptor `environment.apt`).
apt_pkgs = json.loads(os.environ.get("MANIFEST_APT", "[]"))
if apt_pkgs:
    print(f"@@APT:{','.join(apt_pkgs)}@@", flush=True)
    r = subprocess.run(["apt-get", "update", "-qq"])
    if r.returncode != 0:
        print("@@APTFAIL:update@@", flush=True)
        sys.exit(3)
    r = subprocess.run(["apt-get", "install", "-y", "-qq", *apt_pkgs])
    if r.returncode != 0:
        print("@@APTFAIL@@", flush=True)
        sys.exit(3)

# Declared workload dependencies (descriptor `environment.pip`).
for pkg in json.loads(os.environ.get("MANIFEST_PIP", "[]")):
    print(f"@@PIP:{pkg}@@", flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])
    if r.returncode != 0:
        print(f"@@PIPFAIL:{pkg}@@", flush=True)
        sys.exit(3)


def _split_s3_uri(uri):
    assert uri.startswith("s3://"), f"bad s3 uri {uri!r}"
    bucket, _, key = uri[5:].partition("/")
    return bucket, key


for s3_uri, cpath in s3_inputs:
    bucket, key = _split_s3_uri(s3_uri)
    d = os.path.dirname(cpath)
    if d:
        os.makedirs(d, exist_ok=True)
    s3.download_file(bucket, key, cpath)
    print(f"@@S3IN:{cpath}@@", flush=True)

for _oname, cpath, _fmt in outputs:
    d = os.path.dirname(cpath)
    if d:
        os.makedirs(d, exist_ok=True)

proc = subprocess.run(argv)
exit_code = proc.returncode
print(f"@@EXIT:{exit_code}@@", flush=True)

for oname, cpath, fmt in outputs:
    if not os.path.exists(cpath):
        print(f"@@ARTIFACT_MISSING:{oname}@@", flush=True)
        continue
    if oname in s3_outputs:
        bucket, key = _split_s3_uri(s3_prefix)
        key = f"{key.rstrip('/')}/{os.path.basename(cpath)}"
        s3.upload_file(cpath, bucket, key)
        print(f"@@S3:{oname}:s3://{bucket}/{key}@@", flush=True)
    else:
        with open(cpath, "rb") as f:
            blob = base64.b64encode(f.read()).decode()
        print(f"@@ARTIFACT:{oname}:{blob}@@", flush=True)
sys.exit(exit_code)
"""

STANDARD_LABELS = {
    "app.kubernetes.io/managed-by": "npa",
    "app.kubernetes.io/part-of": "npa-workbench",
    "purpose": "manifest-mvp",
}


class JobTimeoutError(TimeoutError):
    """_wait_done timed out; carries the job's partial logs for the record."""

    def __init__(self, message: str, partial_logs: str = ""):
        super().__init__(message)
        self.partial_logs = partial_logs


class NebiusBackend:
    name = "nebius"
    s3_upload = True

    def __init__(
        self,
        kubectl: str = "kubectl",
        namespace: str = "default",
        kube_context: str | None = None,
        timeout_s: int = 2400,
        s3_profile: str | None = None,
        s3_endpoint_url: str = "",
        control_plane_timeout_s: int | None = 300,
    ):
        self.kubectl = kubectl
        self.namespace = namespace
        self.kube_context = kube_context
        self.timeout_s = timeout_s
        self.s3_profile = s3_profile
        self.s3_endpoint_url = s3_endpoint_url
        # Bound on individual kubectl control-plane calls (create/get/logs/
        # delete). This is NOT a job-duration limit: the job itself is
        # bounded only by the operator's timeout_s. Without it, a hung API
        # server would wedge teardown forever and leak the Job/ConfigMap/
        # Secret. None means truly unbounded (operator opt-out).
        self.control_plane_timeout_s = control_plane_timeout_s

    def _kc(
        self,
        *args: str,
        input_text: str | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess:
        if timeout is None:
            timeout = self.control_plane_timeout_s
        cmd = [self.kubectl]
        if self.kube_context:
            cmd += ["--context", self.kube_context]
        cmd += ["-n", self.namespace, *args]
        return subprocess.run(
            cmd, input=input_text, capture_output=True, text=True, timeout=timeout
        )

    def _s3_env(self) -> dict[str, str]:
        """Load S3 credentials for in-pod use from the configured profile.

        Read-only access to the operator's own credential store. The two
        credential values travel to the pod via a per-run Secret referenced
        with secretKeyRef (created in run(), deleted with the run) -- never
        as plaintext env values in the Job spec, where anyone with
        get job/get pod in the namespace could read them. Nothing here is
        logged.
        """
        if not self.s3_profile:
            return {}
        import boto3

        session = boto3.Session(profile_name=self.s3_profile)
        creds = session.get_credentials()
        if creds is None:
            return {}
        return {
            "AWS_ACCESS_KEY_ID": creds.access_key,
            "AWS_SECRET_ACCESS_KEY": creds.secret_key,
            "AWS_ENDPOINT_URL": self.s3_endpoint_url,
        }

    def run(
        self,
        image_pinned: str,
        argv: list[str],
        gpu: int,
        outputs: list[tuple[str, str, str, str]],
        payload: dict[str, str] | None = None,
        memory_gb: int = 0,
        s3_inputs: list[tuple[str, str]] | None = None,
        s3_output_names: list[str] | None = None,
        s3_prefix: str = "",
        s3_env: dict[str, str] | None = None,
        pip_packages: list[str] | None = None,
        apt_packages: list[str] | None = None,
    ) -> BackendResult:
        t0 = time.time()
        run_id = uuid.uuid4().hex[:8]
        cm_name = f"manifest-{run_id}-cm"
        job_name = f"manifest-{run_id}"
        s3_output_names = list(s3_output_names or [])
        if s3_output_names and not s3_prefix:
            raise ValueError("s3 outputs require s3_prefix (artifact_store)")
        env = self._s3_env()
        if s3_env:
            env.update(s3_env)
        # Fail fast before anything is created or scheduled: without S3
        # credentials the runner would die with a bare KeyError on
        # AWS_ENDPOINT_URL after the job already took a GPU node.
        if s3_output_names or s3_inputs:
            missing = [
                k
                for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
                if not env.get(k)
            ]
            if "AWS_ENDPOINT_URL" not in env:
                missing.append("AWS_ENDPOINT_URL")
            if missing:
                raise RuntimeError(
                    "S3 transport requested but S3 credentials are incomplete "
                    f"(missing: {', '.join(missing)}; "
                    f"s3_profile={self.s3_profile!r}); refusing to schedule the job"
                )
        # Credentials travel via a per-run Secret referenced with secretKeyRef
        # (never as plaintext env values in the Job spec); the Secret is
        # deleted with the run in the finally below. The endpoint URL is not
        # a secret and stays a plain env value.
        secret_name = f"manifest-{run_id}-s3"
        secret_keys = [
            k for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") if k in env
        ]
        secret_data = {k: env.pop(k) for k in secret_keys}
        try:
            if secret_data:
                self._create_secret(secret_name, secret_data)
            payload_map = self._create_configmap(cm_name, payload or {})
            self._create_job(
                job_name,
                cm_name,
                image_pinned,
                argv,
                gpu,
                memory_gb,
                outputs,
                payload_map,
                s3_inputs or [],
                s3_output_names,
                s3_prefix,
                env,
                pip_packages or [],
                apt_packages or [],
                secret_name=secret_name if secret_data else "",
                secret_keys=tuple(secret_keys),
            )
            try:
                self._wait_done(job_name)
            except JobTimeoutError as e:
                # The hung-job case: carry the partial logs on the exception
                # so the verification record keeps the evidence instead of
                # an empty log. Teardown still runs in the outer finally.
                e.partial_logs = self._logs_or_empty(job_name)
                raise
            logs = self._logs_or_empty(job_name)
            return self._parse(logs, time.time() - t0)
        finally:
            self._kc("delete", "job", job_name, "--wait=false")
            self._kc("delete", "configmap", cm_name, "--wait=false")
            if secret_data:
                self._kc("delete", "secret", secret_name, "--wait=false")

    def _logs_or_empty(self, job_name: str) -> str:
        """Fetch job logs, returning "" when the fetch itself fails.

        Used in the teardown path so a hung API server can't mask the
        original failure (or hang teardown behind a second unbounded call).
        """
        try:
            r = self._kc("logs", f"job/{job_name}")
        except Exception:
            return ""
        return r.stdout if r.returncode == 0 else ""

    def _create_secret(self, secret_name: str, data: dict[str, str]) -> None:
        secret = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": secret_name, "labels": dict(STANDARD_LABELS)},
            "stringData": data,
        }
        r = self._kc("create", "-f", "-", input_text=json.dumps(secret))
        if r.returncode != 0:
            raise RuntimeError(f"secret create failed: {r.stderr[:500]}")

    def _create_configmap(
        self, cm_name: str, payload: dict[str, str]
    ) -> dict[str, str]:
        """Create the ConfigMap; return {container_path: configmap key}.

        Keys preserve the payload file's basename (sanitized) so that
        multi-file payloads can import each other via
        ``sys.path.insert(0, os.path.dirname(__file__))``.
        """
        import re

        payload_map = {}
        data = {"runner.py": RUNNER_PY}
        used_keys = set()
        for cpath, content in payload.items():
            base = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(cpath).name) or "payload"
            key, i = base, 0
            while key in used_keys:
                i += 1
                key = f"{base}.{i}"
            used_keys.add(key)
            payload_map[cpath] = key
            data[key] = content
        cm = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": cm_name, "labels": dict(STANDARD_LABELS)},
            "data": data,
        }
        r = self._kc("create", "-f", "-", input_text=json.dumps(cm))
        if r.returncode != 0:
            raise RuntimeError(f"configmap create failed: {r.stderr[:500]}")
        return payload_map

    def _create_job(
        self,
        job_name,
        cm_name,
        image_pinned,
        argv,
        gpu,
        memory_gb,
        outputs,
        payload_map,
        s3_inputs,
        s3_output_names,
        s3_prefix,
        env,
        pip_packages,
        apt_packages,
        secret_name: str = "",
        secret_keys: tuple[str, ...] = (),
    ) -> None:
        # Rewrite payload container paths to their ConfigMap mount locations.
        rewritten = []
        for token in argv:
            for cpath, key in payload_map.items():
                token = token.replace(cpath, f"/payload/{key}")
            rewritten.append(token)
        file_outputs = [
            [oname, cpath, fmt]
            for oname, cpath, fmt, source in outputs
            if source == "file"
        ]
        container: dict = {
            "name": "job",
            "image": image_pinned,
            "command": ["python3", "/runner/runner.py"],
            "env": [
                {"name": "MANIFEST_ARGV", "value": json.dumps(rewritten)},
                {"name": "MANIFEST_OUTPUTS", "value": json.dumps(file_outputs)},
                {
                    "name": "MANIFEST_S3_INPUTS",
                    "value": json.dumps(s3_inputs),
                },
                {
                    "name": "MANIFEST_S3_OUTPUTS",
                    "value": json.dumps(s3_output_names),
                },
                {"name": "MANIFEST_S3_PREFIX", "value": s3_prefix},
                {"name": "MANIFEST_PIP", "value": json.dumps(pip_packages)},
                {"name": "MANIFEST_APT", "value": json.dumps(apt_packages)},
            ]
            + [{"name": k, "value": v} for k, v in env.items()]
            # Secret-backed credentials: valueFrom, never plaintext value.
            + [
                {
                    "name": k,
                    "valueFrom": {"secretKeyRef": {"name": secret_name, "key": k}},
                }
                for k in secret_keys
            ],
            "volumeMounts": [
                {"name": "runner", "mountPath": "/runner"},
                {"name": "payload", "mountPath": "/payload"},
            ],
        }
        resources: dict = {}
        if gpu > 0:
            resources.setdefault("limits", {})["nvidia.com/gpu"] = gpu
        if memory_gb > 0:
            resources.setdefault("limits", {})["memory"] = f"{memory_gb}Gi"
            resources.setdefault("requests", {})["memory"] = f"{memory_gb}Gi"
        if resources:
            container["resources"] = resources
        pod_spec: dict = {
            "restartPolicy": "Never",
            "activeDeadlineSeconds": self.timeout_s,
            "containers": [container],
            "volumes": [
                {
                    "name": "runner",
                    "configMap": {
                        "name": cm_name,
                        "items": [{"key": "runner.py", "path": "runner.py"}],
                    },
                },
                {
                    "name": "payload",
                    "configMap": {
                        "name": cm_name,
                        "items": [{"key": k, "path": k} for k in payload_map.values()],
                    },
                },
            ],
        }
        # GPU node selection only applies when the job actually wants a GPU.
        if gpu > 0:
            pod_spec["nodeSelector"] = {"nvidia.com/gpu.present": "true"}
        job = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_name, "labels": dict(STANDARD_LABELS)},
            "spec": {
                "backoffLimit": 0,
                "ttlSecondsAfterFinished": 86400,
                "template": {
                    "metadata": {"labels": dict(STANDARD_LABELS)},
                    "spec": pod_spec,
                },
            },
        }
        r = self._kc("create", "-f", "-", input_text=json.dumps(job))
        if r.returncode != 0:
            raise RuntimeError(f"job create failed: {r.stderr[:500]}")

    def _wait_done(self, job_name: str) -> None:
        # Poll the job status instead of `kubectl wait`: multiple --for
        # conditions are not an OR (a failed job never gains Complete, so
        # waiting on both hangs), and a failed job must be detected
        # promptly rather than burning the whole timeout.
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            # Bound each status call by the operator's own remaining deadline,
            # not by a hardcoded cap: the only time limit in the system is
            # the one the operator passed as timeout_s.
            remaining = max(1, int(deadline - time.time()))
            try:
                r = self._kc("get", "job", job_name, "-o", "json", timeout=remaining)
            except subprocess.TimeoutExpired:
                continue  # loop re-checks the deadline; raises TimeoutError below
            if r.returncode == 0:
                try:
                    data = json.loads(r.stdout or "{}")
                except json.JSONDecodeError:
                    data = {}
                conds = {
                    c.get("type"): c.get("status")
                    for c in data.get("status", {}).get("conditions", [])
                }
                if conds.get("Complete") == "True" or conds.get("Failed") == "True":
                    return
            time.sleep(15)
        raise JobTimeoutError(
            f"job {job_name} reached no terminal condition in {self.timeout_s}s"
        )

    @staticmethod
    def _strip_marker(line: str, prefix: str) -> str | None:
        """Return the marker body, or None if the line isn't that marker.

        The trailing @@ is stripped only when present; a truncated marker
        line still yields its body rather than silently mis-parsing.
        """
        if not line.startswith(prefix):
            return None
        body = line[len(prefix) :]
        if body.endswith("@@"):
            body = body[:-2]
        return body

    def _parse(self, logs: str, elapsed: float) -> BackendResult:
        exit_code = 1
        artifacts: dict[str, str] = {}
        for line in logs.splitlines():
            body = self._strip_marker(line, "@@EXIT:")
            if body is not None:
                try:
                    exit_code = int(body)
                except ValueError:
                    pass
                continue
            if self._strip_marker(line, "@@ARTIFACT_MISSING:") is not None:
                continue  # artifact stays absent; required-check catches it
            if self._strip_marker(line, "@@S3IN:") is not None:
                continue
            if self._strip_marker(line, "@@PIP:") is not None:
                continue
            if self._strip_marker(line, "@@PIPFAIL:") is not None:
                exit_code = 3
                continue
            if self._strip_marker(line, "@@APT:") is not None:
                continue
            if self._strip_marker(line, "@@APTFAIL:") is not None:
                exit_code = 3
                continue
            if self._strip_marker(line, "@@APTFAIL@@") is not None:
                exit_code = 3
                continue
            body = self._strip_marker(line, "@@S3:")
            if body is not None:
                oname, _, uri = body.partition(":")
                artifacts[oname] = uri
                continue
            body = self._strip_marker(line, "@@ARTIFACT:")
            if body is not None:
                oname, _, blob = body.partition(":")
                try:
                    artifacts[oname] = base64.b64decode(blob).decode()
                except (ValueError, UnicodeDecodeError):
                    pass
        # stdout for source=stdout outputs: marker lines are stripped; any
        # other line beginning with @@ passes through untouched.
        stdout_lines = [
            line
            for line in logs.splitlines()
            if self._strip_marker(line, "@@EXIT:") is None
            and self._strip_marker(line, "@@ARTIFACT:") is None
            and self._strip_marker(line, "@@ARTIFACT_MISSING:") is None
            and self._strip_marker(line, "@@S3:") is None
            and self._strip_marker(line, "@@S3IN:") is None
            and self._strip_marker(line, "@@PIP:") is None
            and self._strip_marker(line, "@@PIPFAIL:") is None
            and self._strip_marker(line, "@@APT:") is None
            and self._strip_marker(line, "@@APTFAIL:") is None
            and self._strip_marker(line, "@@APTFAIL@@") is None
        ]
        return BackendResult(
            exit_code=exit_code,
            logs=logs,
            artifacts_raw=artifacts,
            elapsed_s=elapsed,
            stdout="\n".join(stdout_lines),
        )
