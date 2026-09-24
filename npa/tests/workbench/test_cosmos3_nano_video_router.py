"""Regression contracts for the all-replica Ray routing policy."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from npa.workbench.cosmos import nano_video_server


@pytest.fixture
def router_class(monkeypatch):
    class FIFOMixin:
        def _fulfill_next_pending_request(self, replica, request_metadata=None):
            self.fifo_assignment = (replica, request_metadata)
            return "assigned-through-public-fifo"

    class BaseRouter:
        def __init__(self, *args, **kwargs):
            self.base_args = args
            self.base_kwargs = kwargs

        def _fulfill_next_pending_request(self, replica, request_metadata=None):
            pytest.fail("The exact-ID-only base method can orphan a pending request")

    fake = ModuleType("ray.serve.request_router")
    fake.FIFOMixin = FIFOMixin
    fake.RequestRouter = BaseRouter
    monkeypatch.setitem(sys.modules, "ray.serve.request_router", fake)
    path = Path(nano_video_server.__file__).with_name("nano_video_router.py")
    spec = importlib.util.spec_from_file_location("nano_router_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LeastOutstandingRouter


@pytest.mark.parametrize("inherited_cache", [True, False, None])
def test_routing_always_awaits_fresh_queue_snapshots(router_class, inherited_cache):
    kwargs = {"initial_backoff_s": 0.025, "deployment_id": "synthetic"}
    if inherited_cache is not None:
        kwargs["use_replica_queue_len_cache"] = inherited_cache
    router = router_class(**kwargs)
    assert router.base_kwargs["use_replica_queue_len_cache"] is False
    assert router.base_kwargs["initial_backoff_s"] == 0.025
    assert router.base_kwargs["deployment_id"] == "synthetic"


@pytest.mark.parametrize("initial_backoff", [False, True])
def test_full_rank_retries_enable_base_backoff(router_class, initial_backoff):
    router = router_class()
    pending = SimpleNamespace(
        routing_context=SimpleNamespace(should_backoff=initial_backoff)
    )
    replicas = [object() for _ in range(16)]

    async def repeated_choices():
        for _ in range(3):
            chosen = await router.choose_replicas(replicas, pending)
            assert chosen == [replicas]
            assert chosen[0] is not replicas
            assert pending.routing_context.should_backoff is True

    asyncio.run(repeated_choices())


def test_idle_router_and_empty_replicas_preserve_base_contract(router_class):
    router = router_class()
    assert asyncio.run(router.choose_replicas([], None)) == [[]]


@pytest.mark.parametrize("metadata", [None, "retired-scheduler-request"])
def test_unmatched_assignment_delegates_to_public_fifo_fallback(router_class, metadata):
    # The real pinned Ray FIFO/scheduler loops are also exercised in the image
    # gate. Here a dependency stub proves the integration uses its public hook,
    # including when a retired scheduler leaves no matching request metadata.
    router = router_class()
    replica = object()
    assert (
        router._fulfill_next_pending_request(replica, metadata)
        == "assigned-through-public-fifo"
    )
    assert router.fifo_assignment == (replica, metadata)


_ABSENT = object()


def _load_router_with_ray(monkeypatch, ray_version=_ABSENT):
    """Exec nano_video_router.py with a controlled top-level ray stub."""
    fake_router = ModuleType("ray.serve.request_router")
    fake_router.FIFOMixin = type("FIFOMixin", (), {})
    fake_router.RequestRouter = type("RequestRouter", (), {})
    monkeypatch.setitem(sys.modules, "ray.serve.request_router", fake_router)
    if ray_version is _ABSENT:
        monkeypatch.delitem(sys.modules, "ray", raising=False)
    else:
        fake_ray = ModuleType("ray")
        fake_ray.__version__ = ray_version
        monkeypatch.setitem(sys.modules, "ray", fake_ray)
    path = Path(nano_video_server.__file__).with_name("nano_video_router.py")
    spec = importlib.util.spec_from_file_location("nano_router_version_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_guard_rejects_mismatched_ray(monkeypatch):
    # A rebuild against Ray 2.58.0 must fail loudly, not silently change
    # routing semantics via drifted Serve internals.
    with pytest.raises(RuntimeError, match="requires ray==2.56.0"):
        _load_router_with_ray(monkeypatch, ray_version="2.58.0")


def test_version_guard_accepts_pinned_ray(monkeypatch):
    module = _load_router_with_ray(monkeypatch, ray_version="2.56.0")
    assert module.EXPECTED_RAY_VERSION == "2.56.0"
    assert hasattr(module, "LeastOutstandingRouter")


def test_version_guard_skipped_when_ray_not_installed(monkeypatch):
    # Unit-test harnesses fake ray.serve.request_router without ray installed;
    # the guard must not break that.
    module = _load_router_with_ray(monkeypatch)
    assert hasattr(module, "LeastOutstandingRouter")
