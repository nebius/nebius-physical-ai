#!/usr/bin/env bash
# Stage industrial customer assets on S3 (UR5e MJCF, scene fixtures, part mesh, specs).
#
# Usage:
#   CUSTOMER_TASK_ID=industrial-prod-20260614 bash seed-industrial-production-assets.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"
ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
PY="${ROOT}/npa/.venv/bin/python"
TASK_ID="${CUSTOMER_TASK_ID:-industrial-prod-$(date -u +%Y%m%dT%H%M%SZ | tr '[:upper:]' '[:lower:]')}"

_npa_cfg=()
while IFS= read -r _line; do
  _npa_cfg+=("${_line}")
done < <(operator_read_config "${ROOT}" 2>/dev/null || true)
BUCKET="${S3_BUCKET:-${_npa_cfg[0]:-}}"
ENDPOINT="${S3_ENDPOINT:-${_npa_cfg[1]:-https://storage.eu-north1.nebius.cloud}}"

if [ -z "${BUCKET}" ]; then
  echo "Set S3_BUCKET or storage.bucket in ~/.npa/config.yaml" >&2
  exit 1
fi

STAGING="$(mktemp -d)"
trap 'rm -rf "${STAGING}"' EXIT

echo "=== Seed industrial production assets ==="
echo "  task_id=${TASK_ID}"
echo "  bucket=${BUCKET}"

"${PY}" - "${STAGING}" <<'PY'
import shutil
import subprocess
import sys
from pathlib import Path

staging = Path(sys.argv[1])
repo = staging / "ur5e"
repo.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        "git",
        "clone",
        "--depth",
        "1",
        "--filter=blob:none",
        "--sparse",
        "https://github.com/google-deepmind/mujoco_menagerie.git",
        str(staging / "menagerie"),
    ],
    check=True,
)
subprocess.run(
    [
        "git",
        "-C",
        str(staging / "menagerie"),
        "sparse-checkout",
        "set",
        "universal_robots_ur5e",
    ],
    check=True,
)
src = staging / "menagerie" / "universal_robots_ur5e"
dest = staging / "robot" / "ur5e"
shutil.copytree(src, dest)
print(f"UR5e menagerie -> {dest}")


def write_box_obj(path: Path, *, sx: float, sy: float, sz: float) -> None:
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    verts = [
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]
    faces = [
        (1, 2, 3, 4),
        (5, 8, 7, 6),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 8, 4),
        (5, 1, 4, 8),
    ]
    lines = ["# minimal box OBJ"]
    for x, y, z in verts:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for f in faces:
        lines.append("f " + " ".join(str(i) for i in f))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


meshes = staging / "meshes"
write_box_obj(meshes / "conveyor.obj", sx=1.2, sy=0.4, sz=0.08)
write_box_obj(meshes / "part.obj", sx=0.06, sy=0.06, sz=0.04)
print(f"meshes -> {meshes}")
PY

PREFIX="sim2real-assets/${TASK_ID}"
S3_BASE="s3://${BUCKET}/${PREFIX}"

"${PY}" - "${STAGING}" "${BUCKET}" "${PREFIX}" "${ENDPOINT}" "${TASK_ID}" <<'PY'
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
import yaml
from botocore.client import Config

staging, bucket, prefix, endpoint, task_id = sys.argv[1:6]
root = Path(staging)
creds = yaml.safe_load((Path.home() / ".npa" / "credentials.yaml").read_text()) or {}
storage = creds.get("storage") or creds.get("aws") or {}
ak = storage.get("aws_access_key_id") or storage.get("access_key_id")
sk = storage.get("aws_secret_access_key") or storage.get("secret_access_key")
client = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=ak,
    aws_secret_access_key=sk,
    config=Config(signature_version="s3v4"),
    region_name="eu-north1",
)

def upload(local: Path, key: str) -> str:
    client.upload_file(str(local), bucket, key)
    uri = f"s3://{bucket}/{key}"
    print(f"  uploaded {uri}")
    return uri

robot_prefix = f"{prefix}/robot/ur5e"
for path in sorted((root / "robot" / "ur5e").rglob("*")):
    if path.is_file():
        upload(path, f"{robot_prefix}/{path.relative_to(root / 'robot' / 'ur5e').as_posix()}")

robot_uri = upload(
    root / "robot" / "ur5e" / "ur5e.xml",
    f"{robot_prefix}/ur5e.xml",
)
for name in ("conveyor.obj", "part.obj"):
    upload(root / "meshes" / name, f"{prefix}/{name}")

base = f"s3://{bucket}/{prefix}"
scene_spec = {
    "objects": [
        {
            "name": "conveyor",
            "role": "static",
            "asset_source": "byo_mesh",
            "uri": f"{base}/conveyor.obj",
            "pos": [0.6, 0.0, 0.04],
            "fixed": True,
            "friction": 0.8,
        },
        {
            "name": "part",
            "role": "manipuland",
            "asset_source": "byo_mesh",
            "uri": f"{base}/part.obj",
            "pos": [0.5, 0.0, 0.06],
            "mass": 0.05,
            "friction": 0.9,
        },
    ],
    "cameras": {
        "workspace": {
            "placement": "custom",
            "pos": [1.2, 0.0, 1.5],
            "look_at": [0.5, 0.0, 0.0],
            "resolution": [640, 480],
            "dtype": "uint8",
        },
        "wrist": {
            "placement": "stock_ee_mounted",
            "resolution": [640, 480],
            "dtype": "uint8",
        },
    },
}
robot_spec = {"preset": "ur5e", "robot_uri": robot_uri}
spec_dir = root / "specs"
spec_dir.mkdir()
scene_path = spec_dir / "scene-spec.json"
robot_path = spec_dir / "robot-spec.json"
scene_path.write_text(json.dumps(scene_spec, indent=2) + "\n", encoding="utf-8")
robot_path.write_text(json.dumps(robot_spec, indent=2) + "\n", encoding="utf-8")
scene_uri = upload(scene_path, f"{prefix}/scene-spec.json")
robot_spec_uri = upload(robot_path, f"{prefix}/robot-spec.json")
print(json.dumps({"task_id": task_id, "robot_spec_uri": robot_spec_uri, "scene_spec_uri": scene_uri}, indent=2))
PY

echo ""
echo "=== Industrial assets ready ==="
echo "  CUSTOMER_TASK_ID=${TASK_ID}"
echo "  export CUSTOMER_ASSET_PROFILE=industrial"
echo "  export CUSTOMER_TASK_ID=${TASK_ID}"
