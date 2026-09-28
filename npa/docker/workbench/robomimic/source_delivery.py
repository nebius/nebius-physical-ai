"""Deliver locked corresponding source and verify it against every image layer."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import re
import urllib.parse


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = "usr/share/npa/robomimic/corresponding-source"
ALLOWED_HOSTS = {"snapshot.debian.org", "www.python.org"}


def _artifacts(lock: dict) -> dict[str, dict]:
    result = {}
    artifacts = [lock["cpython"]]
    for source in lock["sources"]:
        artifacts.extend(source["artifacts"])
    for item in artifacts:
        name, digest = item["filename"], item["sha256"]
        if not re.fullmatch(r"[A-Za-z0-9_+.%~-]+", name):
            raise ValueError("unsafe corresponding-source filename")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or item["size"] <= 0:
            raise ValueError("invalid corresponding-source identity")
        result[f"{digest}/{name}"] = item
    return result


def _download(item: dict, destination: Path) -> None:
    url = item["url"]
    for _ in range(6):
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in ALLOWED_HOSTS
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.fragment
        ):
            raise ValueError("unapproved corresponding-source origin")
        connection = http.client.HTTPSConnection(parsed.hostname, timeout=120)
        try:
            connection.request("GET", parsed.path)
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                url = urllib.parse.urljoin(url, response.getheader("Location", ""))
                continue
            if response.status != 200:
                raise ValueError("corresponding-source download failed")
            _write_download(response, destination, item)
            return
        finally:
            connection.close()
    raise ValueError("corresponding-source redirect limit exceeded")


def _write_download(response, destination: Path, item: dict) -> None:
    digest = hashlib.sha256()
    total = 0
    with destination.open("xb") as stream:
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > item["size"]:
                raise ValueError("corresponding-source size exceeds lock")
            digest.update(chunk)
            stream.write(chunk)
    if total != item["size"] or digest.hexdigest() != item["sha256"]:
        raise ValueError("corresponding-source content differs from lock")


def _fetch(lock: dict, lock_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    artifacts = _artifacts(lock)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for relative, item in artifacts.items():
            target = output / relative
            target.parent.mkdir(exist_ok=True)
            futures.append(executor.submit(_download, item, target))
        for future in futures:
            future.result()
    (output / "corresponding-source.lock.json").write_bytes(lock_path.read_bytes())
    return {
        "archives": len(artifacts),
        "bytes": sum(x["size"] for x in artifacts.values()),
    }


def _inventory(archive: Path) -> dict:
    helper = ROOT.parent / "ncore" / "base_sources.py"
    spec = importlib.util.spec_from_file_location("npa_base_sources", helper)
    if spec is None or spec.loader is None:
        raise ValueError("image inventory implementation unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inventory(archive)


def _verify_packages(lock: dict, inventory: dict) -> None:
    parent_ids = lock["parent_diff_ids"]
    if [row["diff_id"] for row in inventory["layers"][: len(parent_ids)]] != parent_ids:
        raise ValueError("image does not retain the inspected Python parent layers")
    expected = {tuple(sorted(package.items())) for package in lock["packages"]}
    observed = {
        tuple(sorted(package.items()))
        for row in inventory["layers"]
        for package in row["debian_packages"]
    }
    if observed != expected:
        raise ValueError(
            "distributed Debian package identities differ from source lock"
        )
    sources = {(item["name"], item["version"]) for item in lock["sources"]}
    required = {(item["source"], item["source_version"]) for item in lock["packages"]}
    if sources != required:
        raise ValueError("corresponding-source package closure is incomplete")


def _verify(lock: dict, lock_path: Path, archive: Path) -> dict:
    inventory = _inventory(archive)
    _verify_packages(lock, inventory)
    artifacts = _artifacts(lock)
    artifacts["corresponding-source.lock.json"] = {
        "size": lock_path.stat().st_size,
        "sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    }
    for relative, expected in artifacts.items():
        actual = inventory["final_files"].get(f"{SOURCE_ROOT}/{relative}", {})
        if (
            actual.get("type") != "0"
            or actual.get("size") != expected["size"]
            or actual.get("sha256") != expected["sha256"]
        ):
            raise ValueError("image corresponding-source artifact is absent or changed")
    return {
        "schema": "npa.robomimic.source-delivery-verification.v1",
        "archive_sha256": inventory["archive_sha256"],
        "source_lock_sha256": artifacts["corresponding-source.lock.json"]["sha256"],
        "distributed_package_identities": len(lock["packages"]),
        "source_packages": len(lock["sources"]),
        "delivered_archives": len(artifacts) - 1,
    }


def main() -> None:
    """Fetch locked sources or verify their delivery in a saved image.

    Args:
        None. Arguments are parsed from the command line.
    Returns:
        None.
    Raises:
        ValueError, OSError: Download, source identity or image correspondence fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fetch", "verify"))
    parser.add_argument(
        "--lock", type=Path, default=ROOT / "corresponding-source.lock.json"
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--image-archive", type=Path)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_bytes())
    if lock.get("schema") != "npa.robomimic.corresponding-source.v1":
        raise ValueError("unsupported source-delivery lock")
    if args.mode == "fetch":
        if args.output_root is None:
            parser.error("fetch requires --output-root")
        result = _fetch(lock, args.lock, args.output_root)
    else:
        if args.image_archive is None:
            parser.error("verify requires --image-archive")
        result = _verify(lock, args.lock, args.image_archive)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
