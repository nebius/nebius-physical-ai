"""Build an immutable deterministic zip payload without embedding tests or private notes."""

import json
import zipfile
from pathlib import Path

from regression_contract import _sha256

FILES = (
    "__main__.py",
    "regression_contract.py",
    "object_storage.py",
    "artifact_validation.py",
    "train.py",
    "verify.py",
)


def main():
    """Write the archive and a manifest describing its exact source bytes.

    Args: None; all paths resolve beside this program.
    Returns: None.
    Raises: OSError on local read/write failure.
    """
    root = Path(__file__).resolve().parent
    source_hashes = {}
    target = root / "payload.pyz"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            data = (root / name).read_bytes()
            source_hashes[name] = _sha256(data)
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED)
    result = {
        "schema": "cuda-regression-payload/v1",
        "sha256": _sha256(target.read_bytes()),
        "size": target.stat().st_size,
        "files": source_hashes,
    }
    (root / "payload-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
