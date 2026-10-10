"""Deliver hash-locked source and license texts for PyAV's native dependencies."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import tarfile
from urllib.parse import urlsplit
from urllib.request import urlopen


HOSTS = frozenset(
    {
        "www.alsa-project.org",
        "github.com",
        "codeload.github.com",
        "ffmpeg.org",
        "ftp.gnu.org",
        "www.gnupg.org",
        "deb.debian.org",
        "ftp.osuosl.org",
        "code.videolan.org",
        "download.videolan.org",
        "gitlab.com",
        "downloads.sourceforge.net",
        "bitbucket.org",
        "www.nasm.us",
        "files.pythonhosted.org",
        "www.python.org",
    }
)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate(lock: dict) -> None:
    if lock.get("schema") != "npa.mjlab.python-corresponding-source.v1":
        raise ValueError("unsupported Python source lock")
    names = set()
    for item in lock["artifacts"]:
        name = item["filename"]
        if not re.fullmatch(r"[a-zA-Z0-9_.+-]+", name) or name in names:
            raise ValueError("unsafe or duplicate source filename")
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) or item["size"] <= 0:
            raise ValueError("invalid source identity")
        url = urlsplit(item["url"])
        if (
            url.scheme != "https"
            or url.hostname not in HOSTS
            or url.username
            or url.password
            or url.port not in (None, 443)
            or url.fragment
        ):
            raise ValueError("unapproved source origin")
        names.add(name)


def _fetch(item: dict, root: Path) -> None:
    target = root / item["filename"]
    total = 0
    with urlopen(item["url"]) as response, target.open("xb") as stream:
        if urlsplit(response.geturl()).scheme != "https":
            raise ValueError("source download left HTTPS")
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > item["size"]:
                raise ValueError("source download exceeds its locked size")
            stream.write(chunk)
    if target.stat().st_size != item["size"] or _digest(target) != item["sha256"]:
        raise ValueError("source archive differs from the reviewed lock")


def _license_texts(item: dict, root: Path) -> list[dict]:
    notices = []
    with tarfile.open(root / item["filename"], mode="r|*") as archive:
        for member in archive:
            name = Path(member.name).name.lower()
            if not member.isfile() or len(Path(member.name).parts) > 3:
                continue
            if not name.startswith(("copying", "license", "licence", "copyright")):
                continue
            content = archive.extractfile(member).read()
            digest = hashlib.sha256(content).hexdigest()
            target = root / "notices" / (digest + ".txt")
            target.write_bytes(content)
            notices.append({"member": member.name, "sha256": digest})
    return notices


def main() -> None:
    """Fetch and verify source archives and retain their original license texts.

    Args:
        None. Paths are supplied through command-line arguments.
    Returns:
        None.
    Raises:
        ValueError, OSError: A source identity, notice or download is invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_bytes())
    _validate(lock)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "notices").mkdir()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(_fetch, item, args.output) for item in lock["artifacts"]
        ]
        for future in futures:
            future.result()
    notices = {
        item["name"]: _license_texts(item, args.output) for item in lock["artifacts"]
    }
    (args.output / "python-sources.lock.json").write_bytes(args.lock.read_bytes())
    (args.output / "notices.json").write_text(json.dumps(notices, indent=2) + "\n")
    print(f"Delivered {len(lock['artifacts'])} verified Python/native source archives")


if __name__ == "__main__":
    main()
