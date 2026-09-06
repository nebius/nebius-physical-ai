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
    class BaseRouter:
        def __init__(self, *args, **kwargs):
            self.base_args = args
            self.base_kwargs = kwargs

    fake = ModuleType("ray.serve.request_router")
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
