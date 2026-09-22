"""Keep the exact MCAP grant and its readable runtime COPY in SeedVR images."""

from __future__ import annotations

import hashlib
from pathlib import Path


def test_seedvr_delivers_the_mcap_release_grant() -> None:
    root = Path(__file__).resolve().parents[3]
    directory = root / "npa/docker/workbench/seedvr2"
    notice = directory / "MCAP-1.4.0-LICENSE.txt"
    expected = "da11235665c17d4c1634072dae92b8ba1b38d6fdde2ccf19a6bbede33253f58d"
    assert hashlib.sha256(notice.read_bytes()).hexdigest() == expected
    assert (
        "COPY --chmod=0444 docker/workbench/seedvr2/MCAP-1.4.0-LICENSE.txt "
        "/usr/share/doc/npa-seedvr2/MCAP-1.4.0-LICENSE.txt"
    ) in (directory / "Dockerfile").read_text()
    attribution = (directory / "THIRD_PARTY_NOTICES.md").read_text()
    assert "b33fa682a5c517b1d213faeabd118e0b4f9d9d93/LICENSE" in attribution
    assert expected in attribution
