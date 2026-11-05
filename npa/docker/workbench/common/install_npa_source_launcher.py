"""Install NPA's source entrypoint in a separately locked vendor environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def install_launcher(source_root: Path) -> Path:
    """Write a source launcher without installing the full client distribution.

    Args:
        source_root: Directory containing the checked-in npa package.
    Returns:
        Launcher in the invoking interpreter's environment.
    Raises:
        FileNotFoundError: The source entrypoint or environment is absent.
        OSError: The launcher cannot be written.
    """
    source_root = source_root.resolve(strict=True)
    if not (source_root / "npa/cli/entry.py").is_file():
        raise FileNotFoundError("NPA source entrypoint is absent")
    launcher = Path(sys.prefix) / "bin/npa"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"sys.path.insert(0, {str(source_root)!r})\n"
        "from npa.cli.entry import main\n"
        "main()\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    install_launcher(parser.parse_args().source_root)
