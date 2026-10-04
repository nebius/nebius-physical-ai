"""Nebius backend: translates a validated invocation into an ephemeral
mk8s pod via kubectl, then tears it down.

Reuse, don't rebuild: the cluster, GPU operator, and image pulls are
existing machinery. This backend only renders the pod spec from
(image, argv, resources, outputs, payload) and parses the results.

Artifact transport: the runner prints base64-encoded artifacts to stdout
between markers; the backend parses `kubectl logs`. No sleep-holds, no
kubectl cp. Pod + ConfigMap are always deleted afterwards.
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
outputs = json.loads(os.environ["MANIFEST_OUTPUTS"])  # [[name, path], ...]

for _oname, cpath in outputs:
    d = os.path.dirname(cpath)
    if d:
        os.makedirs(d, exist_ok=True)

proc = subprocess.run(argv)
exit_code = proc.returncode
print(f"@@EXIT:{exit_code}@@", flush=True)

for oname, cpath in outputs:
    if os.path.exists(cpath):
        with open(cpath, "rb") as f:
            blob = base64.b64encode(f.read()).decode()
        print(f"@@ARTIFACT:{oname}:{blob}@@", flush=True)
    else:
        print(f"@@ARTIFACT_MISSING:{oname}@@", flush=True)
sys.exit(exit_code)
"""


class NebiusBackend:
    name = "nebius"

    def __init__(
        self,
        kubectl: str = "kubectl",
        namespace: str = "default",
        timeout_s: int = 2400,
        poll_s: int = 10,
    ):
        self.kubectl = kubectl
        self.namespace = namespace
        self.timeout_s = timeout_s
        self.poll_s = poll_s

    def _kc(
        self, *args: str, input_text: str | None = None
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.kubectl, "-n", self.namespace, *args],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def run(self, image_pinned, argv, gpu, outputs, payload=None):
        t0 = time.time()
        run_id = uuid.uuid4().hex[:8]
        cm_name = f"manifest-{run_id}-cm"
        pod_name = f"manifest-{run_id}-pod"
        payload = payload or {}
        try:
            self._create_configmap(cm_name, payload)
            self._create_pod(pod_name, cm_name, image_pinned, argv, gpu, outputs)
            self._wait_done(pod_name)
            logs = self._kc("logs", pod_name).stdout
            return self._parse(logs, outputs, time.time() - t0)
        finally:
            self._kc("delete", "pod", pod_name, "--wait=false")
            self._kc("delete", "configmap", cm_name, "--wait=false")

    def _create_configmap(self, cm_name: str, payload: dict[str, str]) -> None:
        # ConfigMap keys must be valid path segments: use payload0, payload1...
        self._payload_keys = [f"payload{i}.py" for i in range(len(payload))]
        data = {"runner.py": RUNNER_PY}
        for key, content in zip(self._payload_keys, payload.values()):
            data[key] = content
        cm = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": cm_name},
            "data": data,
        }
        # Also stash container-path mapping for the runner env.
        r = self._kc("create", "-f", "-", input_text=json.dumps(cm))
        if r.returncode != 0:
            raise RuntimeError(f"configmap create failed: {r.stderr[:500]}")
        self._payload_map = dict(zip(payload.keys(), self._payload_keys))

    def _create_pod(self, pod_name, cm_name, image_pinned, argv, gpu, outputs):
        # Rewrite payload container paths to their ConfigMap mount locations.
        rewritten = []
        for token in argv:
            for cpath, key in self._payload_map.items():
                token = token.replace(cpath, f"/payload/{key}")
            rewritten.append(token)
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": pod_name, "labels": {"purpose": "manifest-mvp"}},
            "spec": {
                "restartPolicy": "Never",
                "activeDeadlineSeconds": self.timeout_s,
                "nodeSelector": {"nvidia.com/gpu.present": "true"},
                "containers": [
                    {
                        "name": "job",
                        "image": image_pinned,
                        "command": ["python3", "/runner/runner.py"],
                        "env": [
                            {"name": "MANIFEST_ARGV", "value": json.dumps(rewritten)},
                            {"name": "MANIFEST_OUTPUTS", "value": json.dumps(outputs)},
                        ],
                        "resources": {"limits": {"nvidia.com/gpu": gpu}} if gpu else {},
                        "volumeMounts": [
                            {"name": "runner", "mountPath": "/runner"},
                            {"name": "payload", "mountPath": "/payload"},
                        ],
                    }
                ],
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
                            "items": [
                                {"key": k, "path": k} for k in self._payload_keys
                            ],
                        },
                    },
                ],
            },
        }
        r = self._kc("create", "-f", "-", input_text=json.dumps(pod))
        if r.returncode != 0:
            raise RuntimeError(f"pod create failed: {r.stderr[:500]}")

    def _wait_done(self, pod_name: str) -> None:
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            r = self._kc("get", "pod", pod_name, "-o", "jsonpath={.status.phase}")
            phase = r.stdout.strip()
            if phase in ("Succeeded", "Failed"):
                return
            time.sleep(self.poll_s)
        raise TimeoutError(f"pod {pod_name} did not finish in {self.timeout_s}s")

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

    def _parse(self, logs: str, outputs, elapsed: float) -> BackendResult:
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
            missing = self._strip_marker(line, "@@ARTIFACT_MISSING:")
            if missing is not None:
                continue  # artifact stays absent; required-check catches it
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
        ]
        return BackendResult(
            exit_code=exit_code,
            logs=logs,
            artifacts_raw=artifacts,
            elapsed_s=elapsed,
            stdout="\n".join(stdout_lines),
        )
