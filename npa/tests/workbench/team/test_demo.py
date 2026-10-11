"""Keep the illustrated example's receipts tied to actual team policy checks."""

import hashlib
import json
from pathlib import Path
import re
import runpy


ROOT = Path(__file__).resolve().parents[4]


def test_demo_receipts_are_reproducible_and_match_current_team_source(tmp_path):
    module = runpy.run_path(str(ROOT / "npa/scripts/build_team_demo.py"))
    recorded = module["_capture"](tmp_path)
    html = (ROOT / "docs/demos/team-access.html").read_text()
    match = re.search(
        r'<script type="application/json" id="evidence-data">(.*?)</script>', html, re.S
    )
    assert match is not None
    evidence = json.loads(match[1])
    assert evidence["checks"] == recorded
    assert all(check["passed"] for check in recorded)
    sources = ROOT / "npa/src/npa/workbench/team"
    expected = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(sources.glob("*.py")))
    ).hexdigest()
    assert evidence["source_sha256"] == expected
    assert "simulated execution and enrollment" in evidence["scope"]


def test_demo_declares_offline_boundaries_and_contains_no_tokens():
    html = (ROOT / "docs/demos/team-access.html").read_text()
    assert "connect-src 'none'" in html
    assert not re.search(r"<script[^>]+src=", html)
    assert not re.search(r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.", html)
    assert "No Kubernetes or cloud-storage calls" in html
