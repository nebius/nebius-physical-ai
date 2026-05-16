from __future__ import annotations

from types import SimpleNamespace

import pytest

import npa.viz.backends as backends
from npa.viz.backends import BackendUnavailable, get_backend


def test_backend_unavailable_message_is_flag_agnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backends, "rerun", SimpleNamespace(), raising=False)

    with pytest.raises(BackendUnavailable) as excinfo:
        get_backend("rerun")

    message = str(excinfo.value)
    assert "--backend" not in message
    assert "--renderer" not in message
    assert "Use renderer 'matplotlib'." in message
