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
        [sys.executable, "-m", "pip", "install", "-q", "boto3"], check=False
    )
    import boto3

    s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])

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
    ):
        self.kubectl = kubectl
        self.namespace = namespace
        self.kube_context = kube_context
        self.timeout_s = timeout_s
        self.s3_profile = s3_profile
        self.s3_endpoint_url = s3_endpoint_url

    def _kc(
        self, *args: str, input_text: str | None = None
    ) -> subprocess.CompletedProcess:
        cmd = [self.kubectl]
        if self.kube_context:
            cmd += ["--context", self.kube_context]
        cmd += ["-n", self.namespace, *args]
        return subprocess.run(
            cmd, input=input_text, capture_output=True, text=True, timeout=120
        )

    def _s3_env(self) -> dict[str, str]:
        """Load S3 credentials for in-pod use from the configured profile.

        Read-only access to the operator's own credential store; values are
        passed to the pod as env vars, never logged or persisted.
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
        try:
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
            )
            self._wait_done(job_name)
            logs = self._kc("logs", f"job/{job_name}").stdout
            return self._parse(logs, time.time() - t0)
        finally:
            self._kc("delete", "job", job_name, "--wait=false")
            self._kc("delete", "configmap", cm_name, "--wait=false")

    def _create_configmap(
        self, cm_name: str, payload: dict[str, str]
    ) -> dict[str, str]:
        """Create the ConfigMap; return {container_path: configmap key}."""
        payload_map = {}
        data = {"runner.py": RUNNER_PY}
        for i, (cpath, content) in enumerate(payload.items()):
            key = f"payload{i}.py"
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
            ]
            + [{"name": k, "value": v} for k, v in env.items()],
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
        r = self._kc(
            "wait",
            f"job/{job_name}",
            "--for=condition=complete",
            f"--timeout={self.timeout_s}s",
        )
        if r.returncode != 0:
            raise TimeoutError(
                f"job {job_name} did not complete in {self.timeout_s}s: "
                f"{r.stderr[:300]}"
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
        ]
        return BackendResult(
            exit_code=exit_code,
            logs=logs,
            artifacts_raw=artifacts,
            elapsed_s=elapsed,
            stdout="\n".join(stdout_lines),
        )
