"""Capture real local team API checks in a standalone illustrative HTML example."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import threading
import time
from types import SimpleNamespace

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from npa.workbench.team.api import create_app
from npa.workbench.team.authentication import TokenVerifier
from npa.workbench.team.authorization import bind_execution
from npa.workbench.team.errors import AuthorizationError
from npa.workbench.team.manifests import execution_manifests
from npa.workbench.team.models import Actor, TeamConfig
from npa.workbench.team.service import TeamService
from npa.workbench.team.storage import authorize_uri
from npa.workbench.team.workflow_policy import prepare_document


def _configuration(root):
    return TeamConfig.model_validate(
        {
            "identity": {
                "issuer": "https://identity.example.test",
                "audience": "workbench",
                "jwks_url": "https://identity.example.test/jwks",
            },
            "state_dir": root / "state",
            "sky_python": root / "unused-interpreter",
            "sky_endpoint": "http://127.0.0.1:46581",
            "clusters": {
                name: {
                    "context": name,
                    "kubeconfig": root / f"{name}.yaml",
                    "api_server_url": "https://192.0.2.1",
                    "api_server_cidr": "192.0.2.1/32",
                }
                for name in ("east", "west")
            },
            "workspaces": {
                "robotics": {
                    "grants": [
                        {"kind": "group", "value": "researchers", "role": "runner"},
                        {"kind": "group", "value": "reviewers", "role": "reader"},
                    ],
                    "gpu_limit": 6,
                    "gpu_limits": {"east": 4, "west": 2},
                    "allocations": [
                        _allocation(root, name) for name in ("alice", "bob")
                    ],
                }
            },
        }
    )


def _allocation(root, name):
    return {
        "subject": name,
        "gpu_limit": 3,
        "clusters": {"east": 2, "west": 1},
        "storage": {
            "endpoint": "https://objects.example.test",
            "bucket": f"example-{name}",
            "prefix": "personal",
            "principal": f"example-principal-{name}",
            "credentials_file": root / f"unused-{name}.json",
        },
    }


def _workflow():
    return {
        "apiVersion": "npa.workflow/v0.0.1",
        "kind": "Workflow",
        "metadata": {"name": "team-example"},
        "config": {},
        "resources": {
            "cpu": {"cloud": "kubernetes", "cpus": 1, "image": "ubuntu:24.04"}
        },
        "initial": "hello",
        "states": {
            "hello": {
                "run": {"shell": "echo example"},
                "resources": "cpu",
                "terminal": True,
            }
        },
    }


def _identity(provider):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    keys = SimpleNamespace(
        get_signing_key_from_jwt=lambda token: SimpleNamespace(key=private.public_key())
    )
    verifier = TokenVerifier(provider, jwks_client=keys)

    def headers(subject="alice", groups=None, **overrides):
        claims = {
            "iss": provider.issuer,
            "aud": provider.audience,
            "sub": subject,
            "iat": int(time.time()),
            "exp": int(time.time()) + 300,
            "groups": ["researchers"] if groups is None else groups,
            **overrides,
        }
        return {
            "Authorization": "Bearer " + jwt.encode(claims, private, algorithm="RS256")
        }

    return verifier, headers


def _record(checks, title, observed, expected, detail):
    checks.append(
        {
            "title": title,
            "observed": str(observed),
            "passed": observed == expected,
            "detail": detail,
        }
    )
    if observed != expected:
        raise AssertionError(f"{title}: expected {expected}, observed {observed}")


def _http(checks, client, title, method, path, expected, detail, **kwargs):
    response = client.request(method, path, **kwargs)
    _record(checks, title, f"HTTP {response.status_code}", f"HTTP {expected}", detail)
    return response


def _api_checks(client, headers, checks, policy):
    request = {
        "workspace": "robotics",
        "cluster": "east",
        "idempotency_key": "retained-key",
        "workflow": _workflow(),
    }
    run_id = _submission_checks(client, headers, checks, request)
    _identity_checks(client, headers, checks, request)
    _ownership_checks(client, headers, checks, run_id)
    _http(
        checks,
        client,
        "Explicit second cluster allocation",
        "POST",
        "/v1/runs",
        202,
        "The same external person can target a separately allocated cluster; no cloud account is required.",
        headers=headers(),
        json={**request, "cluster": "west", "idempotency_key": "second-cluster"},
    )
    policy[0] = policy[0].model_copy(update={"disabled_subjects": ("alice",)})
    _http(
        checks,
        client,
        "Local offboarding takes effect",
        "GET",
        f"/v1/runs/{run_id}",
        403,
        "The actual service checks the current denylist. Already issued cloud keys are not revoked by this check.",
        headers=headers(),
    )


def _submission_checks(client, headers, checks, request):
    response = _http(
        checks,
        client,
        "Verified identity can submit",
        "POST",
        "/v1/runs",
        202,
        "Actual signed JWT verification, authorization, parser, and run ledger; execution adapter is simulated.",
        headers=headers(),
        json=request,
    )
    run_id = response.json()["id"]
    retried = client.post("/v1/runs", headers=headers(), json=request)
    _record(
        checks,
        "Identical retry returns the same run",
        retried.json().get("id") == run_id,
        True,
        "The actual private ledger deduplicates the same identity, workspace, request key, and body.",
    )
    changed = copy.deepcopy(request)
    changed["workflow"]["metadata"]["name"] = "changed-document"
    _http(
        checks,
        client,
        "Changed retry is rejected",
        "POST",
        "/v1/runs",
        409,
        "An existing idempotency key cannot silently become a different workflow.",
        headers=headers(),
        json=changed,
    )
    return run_id


def _identity_checks(client, headers, checks, request):
    cases = [
        ("Proxy headers do not authenticate", {"X-Auth-Request-Email": "alice"}, 401),
        ("Wrong token audience is rejected", headers(aud="another-application"), 401),
        ("Missing workspace group is rejected", headers("eve", []), 403),
        ("Reader cannot submit", headers("casey", ["reviewers"]), 403),
        ("Group membership without allocation", headers("dana"), 403),
    ]
    for title, authentication, expected in cases:
        _http(
            checks,
            client,
            title,
            "POST",
            "/v1/runs",
            expected,
            "Recorded response from the actual team API with a synthetic identity.",
            headers=authentication,
            json=request,
        )


def _ownership_checks(client, headers, checks, run_id):
    for method, suffix in (
        ("GET", ""),
        ("GET", "/logs"),
        ("GET", "/artifacts"),
        ("GET", "/artifacts/output.txt"),
        ("POST", "/cancel"),
        ("POST", "/resume"),
    ):
        name = suffix.strip("/") or "status"
        _http(
            checks,
            client,
            f"Another person cannot access {name}",
            method,
            f"/v1/runs/{run_id}{suffix}",
            404,
            "Bob cannot operate Alice's run, even though both have the same workspace group.",
            headers=headers("bob"),
        )


def _boundary_checks(config, checks):
    actor = Actor(
        issuer=config.identity.issuer, subject="alice", groups={"researchers"}
    )
    binding = bind_execution(config, actor, "robotics", "east")
    _manifest_checks(binding, checks)
    _artifact_checks(binding, checks)
    prepared = prepare_document(_workflow(), binding, "example-run")
    _record(
        checks,
        "Placement is server-bound",
        prepared["resources"]["cpu"]["region"].startswith(binding.namespace),
        True,
        "The actual workflow policy binds resources to the authenticated person's cluster context.",
    )


def _manifest_checks(binding, checks):
    manifests = execution_manifests(binding)
    quota = next(item for item in manifests if item["kind"] == "ResourceQuota")
    _record(
        checks,
        "GPU quota is rendered",
        quota["spec"]["hard"]["requests.nvidia.com/gpu"],
        "2",
        "Actual generated Kubernetes manifest, not evidence of live quota enforcement.",
    )
    worker = next(item for item in manifests if item["kind"] == "ServiceAccount")
    _record(
        checks,
        "Worker Kubernetes token disabled",
        worker["automountServiceAccountToken"],
        False,
        "Actual worker account manifest. Live admission and escalation checks remain required.",
    )


def _artifact_checks(binding, checks):
    try:
        authorize_uri(
            "s3://example-bob/personal/output.txt", binding.allocation.storage
        )
    except AuthorizationError:
        denied = True
    else:
        denied = False
    _record(
        checks,
        "Foreign artifact scope rejected",
        denied,
        True,
        "Actual URI authorization policy. Cloud IAM enforcement is a separate live check.",
    )


def _capture(root):
    policy = [_configuration(root)]
    verifier, headers = _identity(policy[0].identity)
    release = threading.Event()

    def engine(*args, **kwargs):
        release.wait()
        return SimpleNamespace(status="succeeded")

    def backend(*args):
        return SimpleNamespace(root=root)

    service = TeamService(
        lambda: policy[0],
        engine=engine,
        backend_factory=backend,
        enrollment_check=lambda binding: None,
    )
    checks = []
    try:
        with TestClient(create_app(None, service=service, verifier=verifier)) as client:
            _boundary_checks(policy[0], checks)
            _api_checks(client, headers, checks, policy)
    finally:
        workers = list(service._workers.values())
        release.set()
        for worker in workers:
            worker.join()
    return checks


def main():
    """Build the example from real policy/API receipts and a self-contained HTML template.

    Args:
        None; command-line arguments select the template and output file.
    Returns:
        None; writes the HTML and prints only its path and number of verified checks.
    Raises:
        AssertionError, OSError: A real check fails or the output cannot be written.
    """
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument(
        "--template", type=Path, default=repository / "docs/demos/team-access.html"
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="npa-team-example-") as directory:
        checks = _capture(Path(directory))
    sources = repository / "npa/src/npa/workbench/team"
    fingerprint = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(sources.glob("*.py")))
    ).hexdigest()
    evidence = {
        "checks": checks,
        "source_sha256": fingerprint,
        "scope": f"{len(checks)} real local API and policy checks passed. Synthetic JWT issuer; simulated execution and enrollment. No Kubernetes or cloud-storage calls.",
    }
    encoded = json.dumps(evidence, indent=2).replace("<", "\\u003c")
    document, count = re.subn(
        r'(<script type="application/json" id="evidence-data">).*?(</script>)',
        lambda match: match[1] + encoded + match[2],
        args.template.read_text(),
        flags=re.S,
    )
    if count != 1:
        raise ValueError("HTML must contain exactly one evidence-data block")
    args.output_path.write_text(document)
    print(json.dumps({"path": str(args.output_path), "checks_passed": len(checks)}))


if __name__ == "__main__":
    main()
