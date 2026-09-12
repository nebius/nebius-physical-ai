"""Fail-closed verifier for a neutral Gymnasium-Robotics bootstrap image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

LOCK_ROOT = Path("/opt/npa/gymnasium-robotics")
EXPECTED_SOURCE = "4d1ebecbc6436806cfbc0e42ebc36f594d05844e"
EXPECTED_MUJOCO = "3.12.0"
EXPECTED_ASSET_LOCK = "e22eb62fc690a5e1d1ea931bab950392ca480caf3d51c7f16fd8cb4133d65568"
EXPECTED_NEUTRAL_FILE_SHA256: dict[str, str | None] = {
    "source-lock.json": "3318043e3d3fec10b233b212b8e7bd97391f48f20b629dbdb3319981010b6ca9",
    "apt-runtime.lock.json": "6e1df9be2187010e9d4ee12dc2a4d95e4f0aa799ff321c70d86ec2d8772b855e",
    "corresponding-source.lock.json": "7a097851d8c9eae45bb663d7d8d989f507afc0fcdc12e721d7431dd27aa9a3be",
    "requirements.lock": "30d48e4b2bfcf0c590b47ed569393104dd759476d720a608aa9f441cd9976e4a",
    "runtime-bootstrap.py": "1f127f8b67dbee7049c3ceda98d2a7894168ad974aba1278cf084fc687f3477b",
    "capability_smoke.py": "c3707490a49224bb262bceab8548c5ee04aa5ce9d5a41062327c5140c236f6bf",
}
KNOWN_FORBIDDEN_CONTENT_SHA256 = frozenset(
    {
        # Complete upstream source archive and MuJoCo runtime wheel.
        "ad8771ed6e9dd772b1101a25310ea46dd0f6f0044fbd6ea9af01af7b7c52c2c7",
        "7ec16ce408871a0a9157cc556958ab66cd34db9fc1dccd3ef07717170163a4e0",
        # Every directly loaded Shadow Hand XML/mesh/texture byte.
        "c1004adf05daea7b57ee57643f8e94945393a2e12b34b18a244be3c1776776b2",
        "9b6f70c49c8bb043ce3e52d181a0797c514a3b85f44f0cf59600d9f906df9f64",
        "fbc404e52def67e38222a395fcc39c051eacc503dbf8e22a22a5bcce36992524",
        "95a9153e6bba4c555ad7cdcea746520eab38e0d57415fa9b3bfaf987dd0418f1",
        "248bc2cb73920c88903786842aaa5476262d550bd01c2c716dd5f2ee642f3318",
        "a2d35742067f71e4888954d1aa55d043dbc6ec0c63f4cc715f339cb45fd13734",
        "0eeb932dcc102dd1fc6bef55fe83f6a74d97aebd32c34d6ee7020c19647306c5",
        "d11a521ae498e947491ba947df8f999136fdf1401d1ba86d594eec20da656ad1",
        "83fd93c9e4c1bf240aa6def2e5fdf5b1adcf4e341c146b05313aeaa8c54fd36a",
        "98e26cf013cd8d7f6ed890d3402ba0aceb11ec5c78779e388390ca2cddc7daf2",
        "80aae0002a6684428278cf214a31040f7cd7aeb3e2e648bd4b68baa375c2d2ad",
        "01aca61837d9db13c52dd170245cacb6f1cedc8238d66fe7239602c80c7a0130",
        "874ae2ac813c6c9d873d04f872cb13de3cabd427cb24a8924619d0a1719d6da7",
        "a8a9c1659aa4391291531b46997ba5eeda1b36e597070ae8a3cd824159937aae",
        "b6901da4d4b93059c02b8803650ad484b3f31aebce46badfa411897d521a4877",
        "12593bcba03bbf278cd7d7db3ca79ba753ada3cb58ee93e725603e9b5e29fd0c",
        "47d0f252ae456541993746ff76d7022135b1bf54263f8f83b19d61f6f94871c9",
        "d39ffb85a87d00c346764191e38ecac3135f2d6f690f64a8ee5da4783ef18e76",
        "3649cb94a9a5f74751d15c0f38291dd666b7eecc286977888b64b6c3c626c9d3",
    }
)
EXPECTED_SYSTEM_WHEEL_FILES = {
    "usr/share/python-wheels/pip-24.0-py3-none-any.whl": {
        "sha256": "e995a37590643450898cfa5bd5113831a547506cd545c335a409339d0c1e87ab",
        "package": "python3-pip-whl",
        "package_sha256": "4b7c50db8f261b208c1d9cde8db148c1f682cc516957b986ada5088cfcee1359",
    },
    "usr/share/python-wheels/setuptools-68.1.2-py3-none-any.whl": {
        "sha256": "fcfc63a09d24f6195a4c89e8e55323331857ff3711f9f0f574152e76b6f7d8ba",
        "package": "python3-setuptools-whl",
        "package_sha256": "edfa94cc1f6a33af99cfaf6ebfe35dbcd9c4bdd8555b90c0d8e78479faf5c8f0",
    },
}
FORBIDDEN_ROOTS = (
    Path("/opt/venv"),
    Path("/usr/local/cuda"),
    Path("/opt/nvidia"),
    Path("/usr/share/source/npa-gymnasium-robotics"),
    Path("/workspace/.cache"),
    Path("/workspace/byof-runs"),
    Path("/home/ubuntu/.cache"),
    Path("/root/.cache"),
    Path("/root/.docker"),
    Path("/root/.ssh"),
)
FORBIDDEN_NAME = re.compile(
    r"(?:^|/)(?:gymnasium_robotics|shadow[_-]?hand|mujoco[^/]*)"
    r"(?:/|$)|\.(?:whl|pt|pth|ckpt|safetensors|onnx|engine)$",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root: Path = Path("/")) -> dict[str, object]:
    def at(path: Path) -> Path:
        return root / path.relative_to("/")

    for name, expected in EXPECTED_NEUTRAL_FILE_SHA256.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"reviewed neutral file digest is not configured: {name}")
        path = at(LOCK_ROOT / name)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"reviewed neutral image file changed: {name}")
    if _sha256(at(LOCK_ROOT / "asset-lock.json")) != EXPECTED_ASSET_LOCK:
        raise ValueError("approved asset provenance metadata changed")
    source = json.loads(at(LOCK_ROOT / "source-lock.json").read_text(encoding="utf-8"))
    if (
        source.get("source_commit") != EXPECTED_SOURCE
        or source.get("mujoco_version") != EXPECTED_MUJOCO
        or source.get("delivery", {}).get("data_assets") != "runtime-cache-only"
    ):
        raise ValueError("runtime source identity or delivery boundary changed")
    for forbidden in FORBIDDEN_ROOTS:
        candidate = at(forbidden)
        if candidate.exists() and (not candidate.is_dir() or any(candidate.iterdir())):
            raise ValueError(f"forbidden baked payload or state: {forbidden}")
    for relative, record in EXPECTED_SYSTEM_WHEEL_FILES.items():
        wheel = root / relative
        if not wheel.is_file() or _sha256(wheel) != record["sha256"]:
            raise ValueError(f"reviewed system bootstrap wheel changed: /{relative}")
    system_wheel_hashes = {
        record["sha256"]: path for path, record in EXPECTED_SYSTEM_WHEEL_FILES.items()
    }
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if FORBIDDEN_NAME.search(relative) and relative not in EXPECTED_SYSTEM_WHEEL_FILES:
            raise ValueError(f"forbidden upstream/runtime path: /{relative}")
        try:
            digest = _sha256(path)
        except PermissionError as error:
            raise ValueError(f"unreadable final image byte: /{relative}") from error
        if digest in KNOWN_FORBIDDEN_CONTENT_SHA256:
            raise ValueError(f"forbidden upstream/runtime byte: /{relative}")
        reviewed_path = system_wheel_hashes.get(digest)
        if reviewed_path is not None and relative != reviewed_path:
            raise ValueError(f"system bootstrap wheel at unauthorized path: /{relative}")
    if root == Path("/") and os.geteuid() == 0:
        raise ValueError("image verifier must run as the non-root runtime user")
    return {
        "schema": "npa.gymnasium-robotics.neutral-image-verification.v2",
        "status": "passed",
        "source_commit": EXPECTED_SOURCE,
        "runtime_payload_present": False,
        "shadow_asset_present": False,
        "credential_present": False,
        "release_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/"))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
