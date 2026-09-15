"""Episode context is nested, thread/task-local and reset on product failure."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from npa.agent_backend import trajectory


@pytest.fixture
def emission(monkeypatch):
    records = []
    monkeypatch.setattr(trajectory, "flush_outbox", lambda **kwargs: None)
    monkeypatch.setattr(trajectory, "emit_trajectory", lambda **kwargs: records.append(kwargs))
    return records


def test_nested_boundaries_restore_parent_and_clear_after_return(emission):
    contexts = []

    @trajectory.goal_episode_boundary()
    def inner(payload):
        contexts.append((trajectory.current_episode_id(), trajectory.current_session_id()))
        return {"ok": True}

    @trajectory.goal_episode_boundary()
    def outer(payload):
        parent = (trajectory.current_episode_id(), trajectory.current_session_id())
        inner({"session_id": "inner-session"})
        assert (trajectory.current_episode_id(), trajectory.current_session_id()) == parent
        contexts.append(parent)
        return {"ok": True}

    outer({"session_id": "outer-session"})
    assert contexts[0][0] != contexts[1][0]
    assert contexts[0][1] == "inner-session" and contexts[1][1] == "outer-session"
    assert trajectory.current_episode_id() == trajectory.current_session_id() == ""
    assert {record["episode_id"] for record in emission} == {context[0] for context in contexts}


def test_product_and_emitter_failure_preserve_exception_and_reset_context(monkeypatch, emission):
    original = RuntimeError("synthetic product failure")

    def failed_emitter(**kwargs):
        raise ValueError("synthetic collection failure")

    monkeypatch.setattr(trajectory, "emit_trajectory", failed_emitter)

    @trajectory.goal_episode_boundary()
    def action(payload):
        assert trajectory.current_episode_id()
        raise original

    with pytest.raises(RuntimeError) as captured:
        action({})
    assert captured.value is original
    assert trajectory.current_episode_id() == trajectory.current_session_id() == ""


def test_concurrent_threads_keep_distinct_episode_context(emission):
    barrier = threading.Barrier(2)

    @trajectory.goal_episode_boundary()
    def action(payload):
        episode = trajectory.current_episode_id()
        barrier.wait()
        assert trajectory.current_episode_id() == episode
        return {"ok": True, "episode": episode, "session": trajectory.current_session_id()}

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(action, [{"session_id": "first"}, {"session_id": "second"}]))
    assert results[0]["episode"] != results[1]["episode"]
    assert [result["session"] for result in results] == ["first", "second"]
    assert trajectory.current_episode_id() == ""


def test_async_tasks_do_not_inherit_finished_episode(emission):
    @trajectory.goal_episode_boundary()
    def action(payload):
        return {"ok": True, "episode": trajectory.current_episode_id()}

    async def task(session):
        await asyncio.sleep(0)
        response = action({"session_id": session})
        await asyncio.sleep(0)
        assert trajectory.current_episode_id() == ""
        return response["episode"]

    async def run():
        return await asyncio.gather(task("first"), task("second"))

    first, second = asyncio.run(run())
    assert first != second
