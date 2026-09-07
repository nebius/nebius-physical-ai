#!/usr/bin/env bash
set -euo pipefail
umask 077

: "${HF_TOKEN:?HF_TOKEN is required for guarded Cosmos3-Nano serving}"
: "${NPA_COSMOS3_RAY_TOKEN:?NPA_COSMOS3_RAY_TOKEN is required}"

work="$(mktemp -d /tmp/npa-cosmos3-ray-smoke.XXXXXX)"
server_log="${work}/server.log"
parent_start_ticks="$(/opt/npa/.venv/bin/python - "$$" <<'PY'
import sys
from pathlib import Path

print(Path(f"/proc/{sys.argv[1]}/stat").read_text().rsplit(")", 1)[1].split()[19])
PY
)"
cleanup() {
  result=$?
  trap - EXIT
  trap '' INT TERM
  if [ -n "${server_pid:-}" ]; then
    signal_error=0
    /opt/npa/.venv/bin/python - "${server_pid}" "${work}/server-start.json" <<'PY' || signal_error=$?
import json
import os
import select
import signal
import sys
from pathlib import Path

pid = int(sys.argv[1])
identity_path = Path(sys.argv[2])
try:
    descriptor = os.pidfd_open(pid)
except ProcessLookupError:
    pass
else:
    try:
        # Cancellation may arrive before the launcher records its birth. Wait
        # for that record or exact pidfd exit, without following a reused PID.
        while not identity_path.exists():
            if select.select([descriptor], [], [], 0.05)[0]:
                sys.exit(0)
        identity = json.loads(identity_path.read_text())
        assert identity["pid"] == pid
        expected = identity["start_ticks"]
        actual = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        if actual == expected:
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
    except (FileNotFoundError, ProcessLookupError):
        pass
    finally:
        os.close(descriptor)
PY
    server_exit=0
    wait "${server_pid}" || server_exit=$?
    /opt/npa/.venv/bin/python - "${work}" "${server_pid}" \
      "$$" "${parent_start_ticks}" \
      "${server_exit}" "${signal_error}" <<'PY'
import json
import sys
from pathlib import Path

root, pid, parent_pid, parent_ticks, code, signal_error = sys.argv[1:]
start_file = Path(root) / "server-start.json"
identity = json.loads(start_file.read_text()) if start_file.exists() else {}
(Path(root) / "server-join.json").write_text(json.dumps({
    "pid": int(pid), "start_ticks": identity.get("start_ticks"),
    "parent_pid": int(parent_pid),
    "parent_start_ticks": parent_ticks,
    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    "signal_helper_exit_code": int(signal_error),
    "wait_exit_code": int(code), "state": "joined",
}) + "\n")
PY
    if [ "${signal_error}" -ne 0 ] && [ "${result}" -eq 0 ]; then result=1; fi
  fi
  exit "${result}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

(
  /opt/npa/.venv/bin/python - "${BASHPID}" "${work}/server-start.json" <<'PY'
import json
import sys
from pathlib import Path

pid = int(sys.argv[1])
ticks = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
Path(sys.argv[2]).write_text(json.dumps({"pid": pid, "start_ticks": ticks}) + "\n")
PY
  exec /usr/local/bin/cosmos3-ray-serve-entrypoint
) >"${server_log}" 2>&1 &
server_pid=$!

for _ in $(seq 1 180); do
  if npa workbench cosmos3 ray-health --endpoint http://127.0.0.1:8000 >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    tail -n 200 "${server_log}" >&2
    exit 1
  fi
  sleep 10
done
npa workbench cosmos3 ray-health --endpoint http://127.0.0.1:8000 >/dev/null

printf '%s\n' '{"model":"Cosmos3-Nano","samples":[{"name":"ray-smoke-a","model_mode":"text2image","prompt":"a red cube on a robotics workbench","seed":17,"num_steps":4},{"name":"ray-smoke-b","model_mode":"text2image","prompt":"a blue cube on a robotics workbench","seed":23,"num_steps":4}]}' >"${work}/batch.json"
npa workbench cosmos3 ray-batch \
  --input-path "${work}/batch.json" \
  --output-path "${work}/result" \
  --endpoint http://127.0.0.1:8000 \
  --timeout 1800 >"${work}/result.json"

/opt/npa/.venv/bin/python - "${work}" <<'PY'
import json
import sys
from pathlib import Path
from PIL import Image

root = Path(sys.argv[1])
result = json.loads((root / "result.json").read_text())
assert result["status"] == "completed"
assert result["backend"] == "cosmos-framework-native-ray-serve"
assert result["batch_size"] == 2
assert result["guardrails"] is True
assert result["max_batch_size"] >= 2
images = list((root / "result" / "artifacts").rglob("*.jpg")) + list(
    (root / "result" / "artifacts").rglob("*.png")
)
assert len(images) >= 2, images
for path in images:
    with Image.open(path) as image:
        image.verify()
print(json.dumps({"status": "PASS", "batch_size": 2, "artifacts": len(images)}))
PY
