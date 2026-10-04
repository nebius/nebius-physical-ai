"""Install only two reviewed OpenSSL binaries inside the image's apt RUN."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import urlsplit
import urllib.request


def verify_manifest(manifest: dict) -> list[dict]:
    if manifest.get("schema") != "npa.curobo.security-apt.v1":
        raise ValueError("Unexpected security package schema")
    rows = manifest["packages"]
    if [row["name"] for row in rows] != ["libssl3t64", "openssl"]:
        raise ValueError("Unexpected security package population/order")
    for row in rows:
        expected = (
            "https://snapshot.ubuntu.com/ubuntu/20261004T000000Z/pool/main/o/openssl/"
            f"{row['name']}_3.0.13-0ubuntu3.16_amd64.deb"
        )
        if (
            row["version"] != "3.0.13-0ubuntu3.16"
            or row["architecture"] != "amd64"
            or row["url"] != expected
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
        ):
            raise ValueError("Unexpected security package identity")
    return rows


def install(manifest: dict) -> dict:
    rows = verify_manifest(manifest)
    # The directory is owned by this invocation and never exported as a layer.
    with tempfile.TemporaryDirectory(prefix="npa-curobo-security-") as scratch:
        paths = []
        for row in rows:
            with urllib.request.urlopen(row["url"]) as response:
                data = response.read()
            if (
                len(data) != row["bytes"]
                or hashlib.sha256(data).hexdigest() != row["sha256"]
            ):
                raise ValueError("Security package bytes differ")
            path = Path(scratch) / Path(urlsplit(row["url"]).path).name
            path.write_bytes(data)
            fields = subprocess.check_output(
                [
                    "dpkg-deb",
                    "--field",
                    str(path),
                    "Package",
                    "Version",
                    "Architecture",
                ],
                text=True,
            )
            if fields != (
                f"Package: {row['name']}\nVersion: {row['version']}\n"
                f"Architecture: {row['architecture']}\n"
            ):
                raise ValueError("Security package control identity differs")
            paths.append(str(path))
        # No resolver, shell interpolation, unrelated package upgrade or IAM.
        subprocess.run(["dpkg", "--install", *paths], check=True)
    installed = subprocess.check_output(
        [
            "dpkg-query",
            "--show",
            "--showformat=${Package}\t${Version}\t${Architecture}\n",
            *[row["name"] for row in rows],
        ],
        text=True,
    )
    expected = {
        f"{row['name']}\t{row['version']}\t{row['architecture']}" for row in rows
    }
    if set(installed.splitlines()) != expected:
        raise ValueError("Installed security package identity differs")
    return {"schema": manifest["schema"], "packages": rows, "installed_verified": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(install(json.loads(args.manifest.read_text())), sort_keys=True))


if __name__ == "__main__":
    main()
