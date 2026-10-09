"""Capture offline local team API checks in a standalone illustrative HTML example."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient

from npa.workbench.team.api import create_app
from npa.workbench.team.accounts import Accounts
from npa.workbench.team.authorization import bind_execution
from npa.workbench.team.errors import AuthorizationError
from npa.workbench.team.manifests import execution_manifests
from npa.workbench.team.models import Actor, TeamConfig
from npa.workbench.team.service import TeamService
from npa.workbench.team.storage import authorize_uri
from npa.workbench.team.workflow_policy import prepare_document


def _configuration(root, subjects):
    return TeamConfig.model_validate(
        {
            "account_namespace": "00000000-0000-4000-8000-000000000001",
            "state_dir": root / "state",
            "sky_python": root / "unused-interpreter",
            "sky_endpoint": "http://127.0.0.1:46580",
            "clusters": {
                name: {
                    "context": name,
                    "kubeconfig": root / f"{name}.yaml",
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
                        _allocation(root, name, subjects[name])
                        for name in ("runner_a", "runner_b")
                    ],
                }
            },
        }
    )


def _allocation(root, name, subject):
    return {
        "subject": subject,
        "gpu_limit": 3,
        "clusters": {"east": 2, "west": 1},
        "storage": {
            "endpoint": "https://objects.example.test",
            "bucket": f"example-{name.replace('_', '-')}",
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


def _local_credentials(config):
    accounts = Accounts(config)
    users = {
        "runner_a": accounts.create("sample-runner-a", ["researchers"]),
        "runner_b": accounts.create("sample-runner-b", ["researchers"]),
        "reader": accounts.create("sample-reader", ["reviewers"]),
        "pending": accounts.create("sample-pending", ["researchers"]),
        "unassigned": accounts.create("sample-unassigned"),
    }
    keys = {
        name: accounts.issue_key(user["id"])[1] for name, user in users.items()
    }

    def headers(name="runner_a"):
        return {"Authorization": "Bearer " + keys[name]}

    return accounts, users, headers


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


def _api_checks(client, headers, checks, accounts, users):
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
        "The same local account can target a separately allocated cluster; no cloud account is required.",
        headers=headers(),
        json={**request, "cluster": "west", "idempotency_key": "second-cluster"},
    )
    accounts.update(users["runner_a"]["id"], disabled=True)
    _http(
        checks,
        client,
        "Local offboarding takes effect",
        "GET",
        f"/v1/runs/{run_id}",
        401,
        "The actual account registry rejects disabled local users. Already issued cloud keys are not revoked by this check.",
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
        "Actual local-key verification, authorization, parser, and run ledger; execution adapter is simulated.",
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
        ("Missing access key is rejected", {}, 401),
        (
            "Unknown personal key is rejected",
            {"Authorization": "Bearer npa_wb_" + "x" * 43},
            401,
        ),
        ("Unassigned local user is rejected", headers("unassigned"), 403),
        ("Reader cannot submit", headers("reader"), 403),
        ("Group membership without allocation", headers("pending"), 403),
    ]
    for title, authentication, expected in cases:
        _http(
            checks,
            client,
            title,
            "POST",
            "/v1/runs",
            expected,
            "Recorded response from the actual team API with a synthetic local key.",
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
            "A different local user cannot operate this sample runner's run, even with the same workspace group.",
            headers=headers("runner_b"),
        )


def _boundary_checks(config, checks, actor):
    binding = bind_execution(config, actor, "robotics", "east")
    _manifest_checks(binding, checks)
    _artifact_checks(binding, checks)
    prepared = prepare_document(_workflow(), binding, "example-run")
    _record(
        checks,
        "Placement is server-bound",
        prepared["resources"]["cpu"]["region"].startswith(binding.namespace),
        True,
        "The actual workflow policy binds resources to the authenticated local user's cluster context.",
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
            "s3://example-runner-b/personal/output.txt", binding.allocation.storage
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


def _active_policy(root, users):
    policy = _configuration(
        root,
        {
            "runner_a": users["runner_a"]["id"],
            "runner_b": users["runner_b"]["id"],
        },
    )
    actor = Actor(
        issuer=policy.principal_issuer,
        subject=users["runner_a"]["id"],
        groups=frozenset({"researchers"}),
    )
    return policy, actor


def _simulated_service(root, policy, release):
    def engine(*args, **kwargs):
        release.wait()
        return SimpleNamespace(status="succeeded")

    def backend(*args):
        return SimpleNamespace(root=root)

    return TeamService(
        lambda: policy,
        engine=engine,
        backend_factory=backend,
        enrollment_check=lambda binding: None,
    )


def _capture(root):
    provisional = _configuration(
        root, {"runner_a": "pending-runner-a", "runner_b": "pending-runner-b"}
    )
    accounts, users, headers = _local_credentials(provisional)
    policy, actor = _active_policy(root, users)
    release = threading.Event()
    service = _simulated_service(root, policy, release)
    checks = []
    try:
        with TestClient(create_app(None, service=service)) as client:
            _boundary_checks(policy, checks, actor)
            _api_checks(client, headers, checks, accounts, users)
    finally:
        workers = list(service._workers.values())
        release.set()
        for worker in workers:
            worker.join()
    return checks


def _evidence(repository, checks):
    sources = repository / "npa/src/npa/workbench/team"
    fingerprint = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(sources.glob("*.py")))
    ).hexdigest()
    return {
        "checks": checks,
        "source_sha256": fingerprint,
        "scope": f"{len(checks)} local API and policy checks passed. Synthetic local accounts and access keys; simulated execution and enrollment. No Kubernetes or cloud-storage calls.",
    }


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
    evidence = _evidence(repository, checks)
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
