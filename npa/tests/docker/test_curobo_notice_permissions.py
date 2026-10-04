"""The non-root runtime must traverse the final complete notice directory."""

from pathlib import Path


def test_notice_directory_is_traversable_after_every_notice_copy():
    dockerfile = (
        Path(__file__).resolve().parents[2] / "docker/workbench/curobo/Dockerfile"
    ).read_text()
    permission = dockerfile.index("chmod 0755 /usr/share/doc/npa-curobo")
    copies = [
        dockerfile.index(line)
        for line in dockerfile.splitlines()
        if line.startswith("COPY ") and "/usr/share/doc/npa-curobo/" in line
    ]
    assert len(copies) == 3
    assert max(copies) < permission < dockerfile.rindex("USER ubuntu")
    assert "--chmod=0444" in dockerfile
