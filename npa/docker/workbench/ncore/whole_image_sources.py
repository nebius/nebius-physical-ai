"""Offline source inventory of Docker-save layers and the merged filesystem.

This is a source correspondence tool, not a security scanner or release gate.
No image code is executed and archive paths are never extracted.
"""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import gzip
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import urllib.request


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe path: {name}")
    return str(path)


def deb822(raw: bytes) -> list[dict]:
    result = []
    for paragraph in raw.decode().split("\n\n"):
        fields = {}
        key = None
        for line in paragraph.splitlines():
            if line.startswith((" ", "\t")) and key:
                fields[key] += "\n" + line[1:]
            elif ": " in line:
                key, value = line.split(": ", 1)
                fields[key] = value
            elif line.endswith(":"):
                key = line[:-1]
                fields[key] = ""
        if fields:
            result.append(fields)
    return result


def source_identity(fields: dict) -> tuple[str, str]:
    match = re.fullmatch(
        r"([^ ]+)(?: \(([^)]+)\))?", fields.get("Source", fields["Package"])
    )
    if not match:
        raise ValueError("invalid Debian Source field")
    return match[1], match[2] or fields["Version"]


def debian_packages(raw: bytes) -> list[dict]:
    result = []
    for fields in deb822(raw):
        if fields.get("Status") != "install ok installed":
            continue
        source, version = source_identity(fields)
        result.append(
            {
                "name": fields["Package"],
                "version": fields["Version"],
                "architecture": fields["Architecture"],
                "source": source,
                "source_version": version,
            }
        )
    return result


def inventory(archive_path: Path) -> dict:
    result = {"schema": 1, "archive_sha256": file_digest(archive_path), "layers": []}
    merged = {}
    with tarfile.open(archive_path) as outer:
        manifests = json.load(outer.extractfile("manifest.json"))
        if len(manifests) != 1:
            raise ValueError("save exactly one image")
        manifest = manifests[0]
        config_raw = outer.extractfile(safe_path(manifest["Config"])).read()
        config = json.loads(config_raw)
        result["config_sha256"] = digest(config_raw)
        diff_ids = config["rootfs"]["diff_ids"]
        if len(diff_ids) != len(manifest["Layers"]):
            raise ValueError("layer count mismatch")
        for ordinal, name in enumerate(manifest["Layers"]):
            name = safe_path(name)
            with outer.extractfile(name) as blob:
                compressed_sha = hashlib.file_digest(blob, "sha256").hexdigest()
            if (
                name.startswith("blobs/sha256/")
                and name.split("/")[-1] != compressed_sha
            ):
                raise ValueError("layer blob SHA256 mismatch")
            with outer.extractfile(name) as blob:
                magic = blob.read(2)
                blob.seek(0)
                stream = gzip.GzipFile(fileobj=blob) if magic == b"\x1f\x8b" else blob
                diff_sha = hashlib.file_digest(stream, "sha256").hexdigest()
            if diff_ids[ordinal] != "sha256:" + diff_sha:
                raise ValueError("layer diff_id mismatch")
            row = {
                "ordinal": ordinal,
                "blob_sha256": compressed_sha,
                "diff_id": diff_ids[ordinal],
                "files": {},
                "debian_packages": [],
                "python_packages": [],
            }
            removals = []
            with (
                outer.extractfile(name) as blob,
                tarfile.open(fileobj=blob, mode="r|*") as layer,
            ):
                for member in layer:
                    if member.name in (".", "./"):
                        continue
                    path = safe_path(member.name)
                    if path in row["files"]:
                        raise ValueError(f"duplicate layer path: {path}")
                    basename = PurePosixPath(path).name
                    if basename.startswith(".wh."):
                        parent = str(PurePosixPath(path).parent)
                        prefix = "" if parent == "." else parent + "/"
                        removals.append((prefix, basename))
                    item = {
                        "type": member.type.decode("ascii"),
                        "size": member.size,
                        "mode": member.mode,
                    }
                    if member.issym() or member.islnk():
                        item["link"] = member.linkname
                    if member.isfile():
                        with layer.extractfile(member) as stream:
                            head = stream.read(4)
                            hasher = hashlib.sha256(head)
                            # Only package metadata is retained as text. Other bytes are hashed.
                            retain = path == "var/lib/dpkg/status" or path.endswith(
                                ".dist-info/METADATA"
                            )
                            body = bytearray(head) if retain else None
                            while chunk := stream.read(1024 * 1024):
                                hasher.update(chunk)
                                if body is not None:
                                    body.extend(chunk)
                        item.update(sha256=hasher.hexdigest(), elf=head == b"\x7fELF")
                        if path == "var/lib/dpkg/status":
                            row["debian_packages"] = debian_packages(bytes(body))
                        elif path.endswith(".dist-info/METADATA"):
                            meta = BytesParser().parsebytes(bytes(body))
                            row["python_packages"].append(
                                {
                                    "name": meta["Name"],
                                    "version": meta["Version"],
                                    "metadata_path": path,
                                    "metadata_sha256": item["sha256"],
                                }
                            )
                    row["files"][path] = item
            # Whiteouts affect lower layers, including opaque directories, independent of tar order.
            for prefix, basename in removals:
                target = prefix + basename[4:]
                for path in list(merged):
                    if (
                        (basename == ".wh..wh..opq" and path.startswith(prefix))
                        or path == target
                        or path.startswith(target + "/")
                    ):
                        del merged[path]
            lower_directories = {
                str(parent) for path in merged for parent in PurePosixPath(path).parents
            }
            for path, item in row["files"].items():
                if not PurePosixPath(path).name.startswith(".wh."):
                    if item["type"] != "5" and path in lower_directories:
                        for descendant in [
                            p for p in merged if p.startswith(path + "/")
                        ]:
                            del merged[descendant]
                    merged[path] = {**item, "layer": ordinal}
            result["layers"].append(row)
    result["final_files"] = merged
    result["counts"] = {
        "layers": len(result["layers"]),
        "final_entries": len(merged),
        "layer_entries": sum(len(row["files"]) for row in result["layers"]),
        "layer_regular_bytes": sum(
            item["size"]
            for row in result["layers"]
            for item in row["files"].values()
            if "sha256" in item
        ),
    }
    return result


def artifact_path(item: dict, output: Path, native: Path) -> Path:
    name = safe_path(item["filename"])
    if "/" in name:
        raise ValueError("unsafe artifact filename")
    return (
        native / name
        if item.get("delivery") == "native"
        else output / safe_path(item["path"])
    )


def verify_artifacts(lock: dict, output: Path, native: Path) -> dict:
    size = 0
    for item in lock["artifacts"]:
        target = artifact_path(item, output, native)
        if not target.is_file() or file_digest(target) != item["sha256"]:
            raise ValueError(f"SHA256 mismatch or missing source: {item['path']}")
        size += target.stat().st_size
    return {"artifacts": len(lock["artifacts"]), "bytes": size}


def assemble(
    lock: dict, output: Path, native: Path, caches: list[Path], *, offline: bool = False
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    for item in lock["artifacts"]:
        target = artifact_path(item, output, native)
        if item.get("delivery") == "native" or target.exists():
            if not target.is_file() or file_digest(target) != item["sha256"]:
                raise ValueError(f"SHA256 mismatch or missing source: {item['path']}")
            continue
        cached = next(
            (
                p
                for root in caches
                for p in (root / item["path"], root / item["filename"])
                if p.is_file()
            ),
            None,
        )
        if cached is None and offline:
            raise ValueError(f"offline source missing: {item['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        try:
            if cached:
                shutil.copyfile(cached, temporary)
            else:
                if not item["url"].startswith("https://"):
                    raise ValueError("source URL must use HTTPS")
                with (
                    urllib.request.urlopen(item["url"]) as source,
                    temporary.open("wb") as destination,
                ):
                    shutil.copyfileobj(source, destination)
            if file_digest(temporary) != item["sha256"]:
                raise ValueError(f"SHA256 mismatch: {item['path']}")
            temporary.replace(target)
            target.chmod(0o644)
        finally:
            temporary.unlink(missing_ok=True)
    return verify_artifacts(lock, output, native)


def verify_debian_metadata(
    lock: dict, output: Path, native: Path, keyring: Path
) -> dict:
    """Reauthenticate exact binary -> source -> archive mappings offline."""
    if file_digest(keyring) != lock["debian_keyring_sha256"]:
        raise ValueError("Debian archive keyring SHA256 mismatch")
    artifacts = {item["sha256"]: item for item in lock["artifacts"]}
    binaries = {
        (p["name"], p["version"], p["architecture"]): p for p in lock["debian_binaries"]
    }
    components = {
        c["id"]: c for c in lock["components"] if c["kind"] == "debian-source"
    }
    seen_binary, seen_source = set(), set()
    for repo in lock["debian_repositories"]:
        release = artifact_path(artifacts[repo["inrelease"]], output, native)
        checked = subprocess.run(
            [
                "gpgv",
                "--keyring",
                str(keyring.resolve()),
                "--output",
                "-",
                str(release),
            ],
            capture_output=True,
            check=True,
        )
        fields = deb822(checked.stdout)[0]
        signed = {
            line.split()[2]: (line.split()[0], int(line.split()[1]))
            for line in fields["SHA256"].splitlines()
            if line.strip()
        }
        for name, entry in repo["indexes"].items():
            path = artifact_path(artifacts[entry["artifact"]], output, native)
            if signed[entry["release_path"]] != (
                file_digest(path),
                path.stat().st_size,
            ):
                raise ValueError("signed index SHA256/size mismatch")
            for record in deb822(lzma.decompress(path.read_bytes())):
                if name == "Packages.xz":
                    identity = (
                        record["Package"],
                        record["Version"],
                        record["Architecture"],
                    )
                    expected = binaries.get(identity)
                    if (
                        expected is None
                        or expected["package_index"] != entry["artifact"]
                    ):
                        continue
                    source, version = source_identity(record)
                    if (
                        expected["sha256"] != record["SHA256"]
                        or expected["source"] != f"debian:{source}@{version}"
                        or not expected["url"].endswith("/" + record["Filename"])
                    ):
                        raise ValueError(
                            f"authenticated binary/source mismatch: {identity}"
                        )
                    seen_binary.add(identity)
                elif name == "Sources.xz":
                    identity = f"debian:{record['Package']}@{record['Version']}"
                    expected = components.get(identity)
                    if (
                        expected is None
                        or expected["source_index"] != entry["artifact"]
                    ):
                        continue
                    source_files = {
                        line.split()[0]: line.split()[2]
                        for line in record["Checksums-Sha256"].splitlines()
                        if line.strip()
                    }
                    if set(source_files) != set(expected["artifacts"]):
                        raise ValueError(f"incomplete authenticated source: {identity}")
                    for sha, filename in source_files.items():
                        if artifacts[sha]["filename"] != filename:
                            raise ValueError(f"source filename mismatch: {identity}")
                    seen_source.add(identity)
                else:
                    raise ValueError(f"unexpected index kind: {name}")
    if seen_binary != set(binaries) or seen_source != set(components):
        raise ValueError("unverified Debian binary/source mappings")
    return {
        "authenticated_binary_versions": len(seen_binary),
        "authenticated_source_versions": len(seen_source),
    }


def verify_coverage(lock: dict, inv: dict, native: Path) -> dict:
    """Fail on unknown package identities or ELF bytes, including ancestors."""
    if [row["diff_id"] for row in inv["layers"][: len(lock["base_diff_ids"])]] != lock[
        "base_diff_ids"
    ]:
        raise ValueError("digest-pinned base layers changed")
    debian = {
        (p["name"], p["version"], p["architecture"]): p for p in lock["debian_binaries"]
    }
    python = {
        (p["name"].lower().replace("_", "-"), p["version"])
        for p in lock["python_distributions"]
    }
    known = {
        (item["path"], item["sha256"])
        for package in lock["debian_binaries"]
        for item in package["elf_files"]
    }
    known.update((item["path"], item["sha256"]) for item in lock["cpython_elf_files"])
    wheels = {
        (item["path"], item["sha256"])
        for wheel in lock["python_wheels"]
        for item in wheel["elf_files"]
    }
    for name in ("native-reader.json", "native-converter.json"):
        if (native / name).is_file():
            receipt = json.loads((native / name).read_text())
            known.update(
                (path.lstrip("/"), row["sha256"])
                for path, row in receipt["libraries"].items()
            )
    count = 0
    for row in inv["layers"]:
        for package in row["debian_packages"]:
            identity = (package["name"], package["version"], package["architecture"])
            expected = debian.get(identity)
            if (
                expected is None
                or expected["source"]
                != f"debian:{package['source']}@{package['source_version']}"
            ):
                raise ValueError(f"uncovered Debian source identity: {identity}")
        for package in row["python_packages"]:
            identity = (package["name"].lower().replace("_", "-"), package["version"])
            if identity not in python:
                raise ValueError(f"uncovered Python source identity: {identity}")
        for path, item in row["files"].items():
            if not item.get("elf"):
                continue
            if (path, item["sha256"]) not in known:
                suffix = path.split("/site-packages/", 1)
                if len(suffix) != 2 or (suffix[1], item["sha256"]) not in wheels:
                    raise ValueError(f"unmapped ELF: {path}")
            count += 1
    return {"elf_occurrences": count, "inventoried_layers": len(inv["layers"])}


def verify_preferred_source(
    lock: dict, files: dict, output: Path, native: Path
) -> dict:
    """Compare installed copyleft source to the complete accompanying sdists.

    Wheels can normalize CRLF to LF. No other source transformation is accepted.
    Binary extensions are covered separately by wheel hashes and their build source.
    """
    artifacts = {item["sha256"]: item for item in lock["artifacts"]}
    hashes = {}
    count = 0
    for proof in lock["preferred_source"]:
        sha = proof["artifact"]
        if sha not in hashes:
            hashes[sha] = set()
            with tarfile.open(artifact_path(artifacts[sha], output, native)) as archive:
                for member in archive:
                    if member.isfile() and member.name.endswith(
                        (
                            ".py",
                            ".pyx",
                            ".pxd",
                            ".c",
                            ".cpp",
                            ".h",
                            ".hpp",
                            ".sh",
                            ".pem",
                        )
                    ):
                        raw = archive.extractfile(member).read()
                        hashes[sha].update(
                            (digest(raw), digest(raw.replace(b"\r\n", b"\n")))
                        )
        found = 0
        for path, item in files.items():
            if path.startswith(proof["prefix"]) and path.endswith(
                (".py", ".pyx", ".pxd", ".c", ".cpp", ".h", ".hpp", ".sh", ".pem")
            ):
                if item.get("sha256") not in hashes[sha]:
                    raise ValueError(f"unmatched preferred source: {path}")
                found += 1
        if not found:
            raise ValueError(f"missing preferred source: {proof['prefix']}")
        count += found
    return {"preferred_source_files": count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory")
    inv.add_argument("--image-archive", type=Path, required=True)
    inv.add_argument("--output", type=Path, required=True)
    for name in ("assemble", "verify"):
        child = commands.add_parser(name)
        child.add_argument("--lock", type=Path, required=True)
        child.add_argument("--annex", type=Path, required=True)
        child.add_argument("--native", type=Path, required=True)
        if name == "assemble":
            child.add_argument("--cache", type=Path, action="append", default=[])
            child.add_argument("--offline", action="store_true")
        else:
            child.add_argument(
                "--keyring",
                type=Path,
                default=Path("/usr/share/keyrings/debian-archive-keyring.gpg"),
            )
            child.add_argument("--inventory", type=Path)
            child.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.command == "inventory":
        result = inventory(args.image_archive)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["counts"], sort_keys=True))
    else:
        lock = json.loads(args.lock.read_text())
        if args.command == "assemble":
            result = assemble(
                lock, args.annex, args.native, args.cache, offline=args.offline
            )
        else:
            result = verify_artifacts(lock, args.annex, args.native)
            result.update(
                verify_debian_metadata(lock, args.annex, args.native, args.keyring)
            )
            if args.inventory:
                inv = json.loads(args.inventory.read_text())
                result.update(verify_coverage(lock, inv, args.native))
                result.update(
                    verify_preferred_source(
                        lock, inv["final_files"], args.annex, args.native
                    )
                )
            if args.root:
                files = {}
                for proof in lock["preferred_source"]:
                    for path in (args.root / safe_path(proof["prefix"])).rglob("*"):
                        if path.is_file():
                            files[path.relative_to(args.root).as_posix()] = {
                                "sha256": file_digest(path)
                            }
                result.update(
                    verify_preferred_source(lock, files, args.annex, args.native)
                )
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
