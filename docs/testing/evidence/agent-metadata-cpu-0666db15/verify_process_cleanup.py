"""Reproduce the CPU-only process control on Linux at PR #654's pinned source.

Run using npa/.venv/bin/python after checking out the exact commit named below.
No provider, network, credentials, GPU, or VLM is used.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from npa.cli import agent_resources as runtime

EXPECTED = "0666db15f7c5a715ceba74b9dbfa57fe93054e26"
root = Path(runtime.__file__).resolve().parents[4]
head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
assert sys.platform == "linux", "This control exercises Linux /proc and waitid."
assert head == EXPECTED, "Use the exact reviewed source commit."
assert hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest() == (
    "19438686adad42371c5b5e9a47099fe5ca6014aff12712666eb0d7cf0c4f8bcb"
), "Imported process helper must match the reviewed bytes."
assert not runtime._AGENT_ACTIVE_PROCESSES and not runtime._AGENT_COMMAND_BREAKER_OPEN
child = (
    "import subprocess,sys,time;"
    "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
    "time.sleep(60)"
)
start = time.monotonic()
try:
    runtime.run_bounded_agent_command([sys.executable, "-c", child], timeout_s=0.1)
    raise AssertionError("Expected deadline failure")
except TimeoutError as error:
    assert str(error) == "agent command timed out"
elapsed = time.monotonic() - start
assert elapsed < 2
cleanup_deadline = time.monotonic() + 5
while runtime._AGENT_COMMAND_BREAKER_OPEN and time.monotonic() < cleanup_deadline:
    time.sleep(0.01)
assert not runtime._AGENT_COMMAND_BREAKER_OPEN
assert not runtime._AGENT_ABANDONED_PROCESS_GROUPS
assert not runtime._AGENT_ACTIVE_PROCESSES
healthy = runtime.run_bounded_agent_command(
    [sys.executable, "-c", "print('healthy')"], timeout_s=2
)
assert healthy.returncode == 0 and healthy.stdout.strip() == "healthy"
print(json.dumps({"source_commit": head, "timeout_return_seconds": elapsed,
                  "cleanup_and_recovery_passed": True}, indent=2))
