import json
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest

from npa.clients.project_credentials import CredentialPair
from npa.errors import ScopedCredentialError
from npa.guardrails.pytest_collection import assert_nonzero_collection

os.environ.setdefault("NPA_PROJECT_ID", "project-test-00000000")
os.environ.setdefault("NPA_S3_BUCKET", "test-bucket-00000000")

# Live markers whose tests intentionally use real ambient credentials.
_LIVE_MARKERS = frozenset(
    {
        "agent_live",
        "byovm_live",
        "e2e",
        "e2e_pipeline",
        "e2e_serverless",
        "e2e_skypilot",
        "gpu",
        "multi_gpu",
        "ngc_e2e",
        "token_factory_e2e",
    }
)

# Credential/storage env vars that override file config. A contributor who has
# followed the quickstart and exported real Nebius creds must still get a
# hermetic unit suite, so these are scrubbed for non-live tests. Without this,
# e.g. an exported AWS_ENDPOINT_URL leaks into deploy-config assertions.
_AMBIENT_CREDENTIAL_ENV_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_S3",
    "AWS_PROFILE",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "NEBIUS_S3_ENDPOINT",
    "NEBIUS_S3_BUCKET",
    "NPA_STORAGE_ENDPOINT",
    "NPA_CHECKPOINT_BUCKET",
    "NPA_S3_BUCKET",
    "NPA_S3_PREFIX",
    "NPA_INPUT_PATH",
    "NPA_OUTPUT_PATH",
    "DETECTION_TRAINING_STATE_DIR",
    "DETECTION_TRAINING_AUTH_MODE",
    "DETECTION_TRAINING_TOKEN",
    "NEBIUS_PROJECT_ID",
    "NEBIUS_TENANT_ID",
    "NPA_AGENT_DATASET_TENANT_ID",
    "NPA_AGENT_DATASET_URI",
    "NPA_AGENT_DATASET_OUTBOX",
    "NPA_AGENT_DATASET_REDACTION_FILE",
    "NPA_REGISTRY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "FOXGLOVE_API_TOKEN",
    "NEBIUS_TOKEN_FACTORY_KEY",
    "NEBIUS_TOKEN_FACTORY_BASE_URL",
    "NEBIUS_BASE_URL",
    "NGC_API_KEY",
    "NGC_ORG",
    "NGC_TEAM",
    "NPA_BYOVM_HOST",
    "NPA_SSH_HOST",
    "NPA_BYOVM_SSH_USER",
    "NPA_SSH_USER",
    "NPA_BYOVM_SSH_KEY",
    "NPA_SSH_KEY",
    # SkyPilot private-registry docker creds. When an operator/dev VM exports the
    # real values (e.g. for live submits), the non-live redaction/render unit
    # tests must still behave as on a pristine runner.
    "SKYPILOT_DOCKER_SERVER",
    "SKYPILOT_DOCKER_USERNAME",
    "SKYPILOT_DOCKER_PASSWORD",
    # Operator knobs that product code reads and that therefore change render or
    # tool behavior. A dev VM sources these from ~/.npa/live-e2e.env for live
    # submits; without scrubbing, e.g. GPU accelerator or source-overlay values
    # make render tests assert against the operator's live profile.
    "NPA_E2E_CLEAR_WORKBENCH_IMAGES",
    "NPA_SRC_S3_URI",
    "NPA_SRC_OVERLAY",
    "NPA_WORKFLOW_GPU_ACCELERATOR",
    "NPA_COSMOS_CONDITION_ON_INPUT",
    "NPA_COSMOS_VARIANT_PARALLELISM",
    "NPA_COSMOS_CONTROL",
    "NPA_COSMOS_CONTROL_WEIGHT",
    "NPA_COSMOS_CONTROL_ASSET",
    "NPA_COSMOS_CONTROL_PROMPT",
    "NPA_COSMOS_MASK_ASSET",
    "NPA_COSMOS_MASK_PROMPT",
    "NPA_COSMOS_SHARD_JOIN_TIMEOUT_S",
    # Upstream Cosmos OSS checkouts: a dev VM with these exported would let unit
    # tests import a real checkout and take the upstream code path.
    "NPA_COSMOS_CURATE_SRC",
    "NPA_COSMOS_CURATE_WEIGHTS_DIR",
    "NPA_COSMOS_EVALUATOR_SRC",
)

# Ambient infra-targeting env vars an operator/dev VM exports (a live kube
# context/config and the persisted SkyPilot bin) that would otherwise leak into
# non-live tests and make them resolve real clusters/binaries instead of their
# mocked ones — e.g. `resolve_byof_kubernetes_target` reads KUBECONTEXT, and the
# `sky` bin resolver reads NPA_SKYPILOT_BIN. A contributor who has provisioned a
# cluster and exported these must still get the same hermetic suite as CI, where
# they are unset. Non-live tests that need a value set them via monkeypatch after
# this scrub runs; live-marked tests are exempt and keep the real context.
_AMBIENT_INFRA_TARGET_ENV_VARS = (
    # Operator configuration can select private runtime roots. Non-live tests must
    # resolve their temporary HOME, never the operator's config or journals.
    "NPA_CONFIG_DIR",
    "NPA_OPERATION_JOURNAL_DIR",
    "NPA_SKYPILOT_ISOLATED_CONFIG_DIR",
    "KUBECONFIG",
    "KUBECONTEXT",
    "NPA_K8S_CONTEXT",
    "NPA_K8S_NAMESPACE",
    "NPA_KUBECONFIG",
    "NPA_BYOF_K8S_CONTEXT",
    "NPA_BYOF_K8S_NAMESPACE",
    "NPA_BYOF_KUBECONFIG",
    "NPA_BYOF_CLUSTER_NAME",
    "NPA_SKYPILOT_BIN",
    # Nebius CLI profile selectors. Product code prepends `--profile <name>` to
    # the argv it builds whenever either is set, so an operator who has selected
    # a profile (the normal state on a machine that actually runs npa) shifts
    # every argv assertion by two elements. Tests of the profile behavior itself
    # set these via monkeypatch after this scrub.
    "NPA_NEBIUS_PROFILE",
    "NEBIUS_PROFILE",
)

_CI_TIMING_MANIFEST = Path(__file__).with_name("ci_test_durations.json")
_CI_RECORDED_DURATIONS: dict[str, float] = defaultdict(float)


def _ci_shard_coordinates() -> tuple[int, int] | None:
    """Return the zero-based CI shard index and total.

    Args:
        None.
    Returns:
        The shard coordinates, or ``None`` outside sharded CI.
    Raises:
        pytest.UsageError: Shard environment variables are incomplete or invalid.
    """

    raw_index = os.environ.get("NPA_CI_SHARD_INDEX")
    raw_total = os.environ.get("NPA_CI_TOTAL_SHARDS")
    if raw_index is None and raw_total is None:
        return None
    if raw_index is None or raw_total is None:
        raise pytest.UsageError(
            "NPA_CI_SHARD_INDEX and NPA_CI_TOTAL_SHARDS must be set together"
        )

    try:
        index = int(raw_index)
        total = int(raw_total)
    except ValueError as error:
        raise pytest.UsageError("CI shard coordinates must be integers") from error
    if total < 1 or index < 1 or index > total:
        raise pytest.UsageError("CI shard index must be between 1 and the shard total")
    return index - 1, total


def _ci_timing_weights() -> dict[str, float]:
    """Load positive module-duration weights for CI partitioning.

    Args:
        None.
    Returns:
        Duration in seconds by test module.
    Raises:
        pytest.UsageError: The committed timing manifest is invalid.
    """

    try:
        raw_weights = json.loads(_CI_TIMING_MANIFEST.read_text(encoding="utf-8"))
        weights = {path: float(seconds) for path, seconds in raw_weights.items()}
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise pytest.UsageError("CI timing manifest is unreadable") from error
    if any(seconds <= 0 for seconds in weights.values()):
        raise pytest.UsageError("CI timing weights must be positive")
    return weights


def _items_for_ci_shard(
    items: list[pytest.Item], shard_index: int, shard_total: int
) -> list[pytest.Item]:
    """Select a deterministic shard balanced by recorded module durations.

    Args:
        items: Collected pytest items.
        shard_index: Zero-based shard index.
        shard_total: Number of shards.
    Returns:
        Items assigned to this shard in stable node-id order.
    Raises:
        None.
    """

    weights = _ci_timing_weights()
    module_counts: dict[str, int] = defaultdict(int)
    for item in items:
        module_counts[item.nodeid.split("::", 1)[0]] += 1
    ranked_items = sorted(
        items,
        key=lambda item: (
            -weights.get(item.nodeid.split("::", 1)[0], 1.0)
            / module_counts[item.nodeid.split("::", 1)[0]],
            item.nodeid,
        ),
    )
    shards: list[list[pytest.Item]] = [[] for _ in range(shard_total)]
    totals = [0.0] * shard_total
    for item in ranked_items:
        module = item.nodeid.split("::", 1)[0]
        item_weight = weights.get(module, 1.0) / module_counts[module]
        destination = min(range(shard_total), key=lambda index: (totals[index], index))
        shards[destination].append(item)
        totals[destination] += item_weight
    return sorted(shards[shard_index], key=lambda item: item.nodeid)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Partition the full suite when GitHub Actions supplies shard coordinates.

    Args:
        config: Active pytest configuration.
        items: Mutable collection of discovered tests.
    Returns:
        None.
    Raises:
        pytest.UsageError: CI shard coordinates are invalid.
    """

    coordinates = _ci_shard_coordinates()
    if coordinates is None:
        return
    selected_items = _items_for_ci_shard(items, *coordinates)
    selected_node_ids = {item.nodeid for item in selected_items}
    deselected_items = [item for item in items if item.nodeid not in selected_node_ids]
    config.hook.pytest_deselected(items=deselected_items)
    items[:] = selected_items


def pytest_collection_finish(session: pytest.Session) -> None:
    assert_nonzero_collection(len(session.items))


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Accumulate module timings for a trusted main-run artifact.

    Args:
        report: One setup, call, or teardown timing report.
    Returns:
        None.
    Raises:
        None.
    """

    if os.environ.get("NPA_CI_TIMING_OUTPUT"):
        module = report.nodeid.split("::", 1)[0]
        _CI_RECORDED_DURATIONS[module] += report.duration


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write measured module durations from the xdist controller.

    Args:
        session: Completed pytest session.
        exitstatus: Pytest process status, retained for the hook contract.
    Returns:
        None.
    Raises:
        OSError: The requested CI timing artifact cannot be written.
    """

    del exitstatus
    output = os.environ.get("NPA_CI_TIMING_OUTPUT")
    if not output or hasattr(session.config, "workerinput"):
        return
    Path(output).write_text(
        json.dumps(dict(sorted(_CI_RECORDED_DURATIONS.items())), indent=2) + "\n",
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def scrub_ambient_credential_env(monkeypatch, request):
    """Isolate non-live tests from real credentials exported in the shell."""
    if any(request.node.get_closest_marker(marker) for marker in _LIVE_MARKERS):
        return
    for env_var in (*_AMBIENT_CREDENTIAL_ENV_VARS, *_AMBIENT_INFRA_TARGET_ENV_VARS):
        monkeypatch.delenv(env_var, raising=False)


@pytest.fixture(autouse=True)
def isolate_home_config(monkeypatch, tmp_path_factory, request):
    """Isolate non-live tests from the operator's real ~/.npa, ~/.aws, ~/.ssh.

    A dev machine or workbench VM carries a real ~/.npa/config.yaml and
    credentials.yaml; the unit suite must behave identically there and on a
    pristine CI runner, so every non-live test gets an empty HOME. Several
    modules capture home-derived paths in module-level constants at import
    time, so those are repointed as well — new home-derived constants belong
    in this list.
    """
    if any(request.node.get_closest_marker(marker) for marker in _LIVE_MARKERS):
        return
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    # The agent preflight's outbound-tcp/22 probe is a real network call; unit
    # tests must not make one (tests of the probe itself inject a connector).
    monkeypatch.setenv("NPA_SSH_EGRESS_PROBE", "off")

    import npa.cli.cluster.terraform_lifecycle
    import npa.cli.skypilot
    import npa.clients.config
    import npa.clients.credentials
    import npa.cluster.state
    import npa.controller_ownership
    import npa.deploy.provisioner
    import npa.orchestration.skypilot._bin

    npa_dir = home / ".npa"
    monkeypatch.setattr(npa.clients.config, "CONFIG_PATH", npa_dir / "config.yaml")
    monkeypatch.setattr(
        npa.clients.credentials, "CREDENTIALS_PATH", npa_dir / "credentials.yaml"
    )
    monkeypatch.setattr(
        npa.controller_ownership, "CONFIG_PATH", npa_dir / "config.yaml"
    )
    monkeypatch.setattr(
        npa.orchestration.skypilot._bin, "CONFIG_PATH", npa_dir / "config.yaml"
    )
    monkeypatch.setattr(
        npa.deploy.provisioner, "_WORKBENCH_BASE", npa_dir / "workbenches"
    )
    monkeypatch.setattr(
        npa.deploy.provisioner,
        "_TF_PLUGIN_CACHE_DIR",
        npa_dir / "terraform-plugin-cache",
    )
    monkeypatch.setattr(npa.cluster.state, "CLUSTERS_DIR", npa_dir / "clusters")
    monkeypatch.setattr(
        npa.cli.skypilot, "DEFAULT_VENV_PATH", npa_dir / "skypilot-venv"
    )
    monkeypatch.setattr(
        npa.cli.cluster.terraform_lifecycle,
        "_DEFAULT_SKYPILOT_BIN",
        npa_dir / "skypilot-venv" / "bin" / "sky",
    )


def _is_huggingface_url(url: object) -> bool:
    host = urlparse(str(url)).hostname or ""
    return host == "huggingface.co" or host.endswith(".huggingface.co")


@pytest.fixture(autouse=True)
def block_live_huggingface_http(monkeypatch, request):
    """Keep unit tests from depending on live Hugging Face availability."""
    live_markers = {
        "byovm_live",
        "e2e",
        "e2e_pipeline",
        "e2e_serverless",
        "gpu",
        "multi_gpu",
        "ngc_e2e",
        "token_factory_e2e",
    }
    if any(request.node.get_closest_marker(marker) for marker in live_markers):
        return

    def blocked(method: str, url: object) -> None:
        if _is_huggingface_url(url):
            raise AssertionError(
                f"Live Hugging Face HTTP is blocked in unit tests: {method} {url}. "
                "Mock the access check instead."
            )

    original_head = httpx.head
    original_get = httpx.get
    original_post = httpx.post
    original_request = httpx.request
    original_client_request = httpx.Client.request
    original_async_client_request = httpx.AsyncClient.request

    def guarded_head(url, *args, **kwargs):
        blocked("HEAD", url)
        return original_head(url, *args, **kwargs)

    def guarded_get(url, *args, **kwargs):
        blocked("GET", url)
        return original_get(url, *args, **kwargs)

    def guarded_post(url, *args, **kwargs):
        blocked("POST", url)
        return original_post(url, *args, **kwargs)

    def guarded_request(method, url, *args, **kwargs):
        blocked(str(method).upper(), url)
        return original_request(method, url, *args, **kwargs)

    def guarded_client_request(self, method, url, *args, **kwargs):
        blocked(str(method).upper(), url)
        return original_client_request(self, method, url, *args, **kwargs)

    async def guarded_async_client_request(self, method, url, *args, **kwargs):
        blocked(str(method).upper(), url)
        return await original_async_client_request(self, method, url, *args, **kwargs)

    monkeypatch.setattr(httpx, "head", guarded_head)
    monkeypatch.setattr(httpx, "get", guarded_get)
    monkeypatch.setattr(httpx, "post", guarded_post)
    monkeypatch.setattr(httpx, "request", guarded_request)
    monkeypatch.setattr(httpx.Client, "request", guarded_client_request)
    monkeypatch.setattr(httpx.AsyncClient, "request", guarded_async_client_request)


@pytest.fixture
def tmp_workspace(tmp_path):
    """A clean temp directory simulating a workspace."""
    return tmp_path


@pytest.fixture
def sample_config(tmp_path):
    """Write a minimal valid config YAML and return its path."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "tenant: test-tenant\n"
        "project: test-project\n"
        "region: eu-north1\n"
        "bucket: test-bucket\n"
    )
    return cfg


@pytest.fixture
def mock_ssh(mocker):
    """Patch paramiko.SSHClient universally."""
    mock_client = mocker.MagicMock()
    mock_client.exec_command.return_value = (
        mocker.MagicMock(),  # stdin
        mocker.MagicMock(read=lambda: b"ok\n"),  # stdout
        mocker.MagicMock(read=lambda: b""),  # stderr
    )
    mocker.patch("paramiko.SSHClient", return_value=mock_client)
    return mock_client


@pytest.fixture
def mock_s3(mocker):
    """Patch boto3 S3 client."""
    mock_client = mocker.MagicMock()
    mocker.patch("boto3.client", return_value=mock_client)
    return mock_client


@pytest.fixture
def mock_cross_project_creds(monkeypatch):
    """Mock credential resolution to simulate distinct credentials per project."""
    creds_by_project = {
        "project-source": CredentialPair(
            project="project-source",
            endpoint_url="https://source-storage.example",
            aws_access_key_id="src-key",
            aws_secret_access_key="src-secret",
        ),
        "project-target": CredentialPair(
            project="project-target",
            endpoint_url="https://target-storage.example",
            aws_access_key_id="tgt-key",
            aws_secret_access_key="tgt-secret",
        ),
        None: CredentialPair(
            project=None,
            endpoint_url="https://source-storage.example",
            aws_access_key_id="src-key",
            aws_secret_access_key="src-secret",
        ),
    }

    def fake_resolve(project, allow_host_creds=False):
        if project in creds_by_project:
            return creds_by_project[project]
        if allow_host_creds:
            return CredentialPair(
                project=project,
                endpoint_url="https://host-storage.example",
                aws_access_key_id="",
                aws_secret_access_key="",
                uses_host_credentials=True,
            )
        raise ScopedCredentialError(
            project or "default",
            f"resolve storage credentials for project '{project or 'default'}'",
            failed_project=project or "default",
        )

    monkeypatch.setattr(
        "npa.clients.project_credentials.resolve_credentials", fake_resolve
    )
    return creds_by_project
