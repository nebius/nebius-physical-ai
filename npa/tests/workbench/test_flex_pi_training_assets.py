"""Enforce HTTPS source transport before immutable-content verification."""

import pytest

from npa.workbench.flex_pi.training_assets import _source


def test_upstream_source_fetch_rejects_redirects_before_persisting_payload(
    tmp_path, monkeypatch
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    requests = []

    def request(method, url, **options):
        requests.append((method, url, options))
        return SimpleNamespace(status=302, data=b"untrusted redirected response")

    monkeypatch.setattr(
        "urllib3.PoolManager", lambda: nullcontext(SimpleNamespace(request=request))
    )
    entry = {
        "repository": "example/source",
        "revision": "a" * 40,
        "files": [{"path": "config.yaml", "sha256": "b" * 64, "size": 1}],
    }
    with pytest.raises(RuntimeError, match="source fetch failed"):
        _source(entry, tmp_path)
    assert requests[0][1].startswith("https://raw.githubusercontent.com/")
    assert requests[0][2] == {"redirect": False}
    assert not (tmp_path / "config.yaml").exists()
