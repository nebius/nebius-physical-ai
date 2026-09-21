"""What the build context actually admits, asked of Docker rather than of the file.

A complete-byte scan of a shipped image found
`opt/npa/src/npa/cli/cluster/__pycache__/terraform_lifecycle.cpython-312.pyc` in a layer.
The runtime in that image is Python 3.11 and runs with `PYTHONDONTWRITEBYTECODE`, so it
did not compile a 3.12 file: the bytes were copied in from the build host, through
`COPY src/npa`, and once they are in an ancestor layer no later `RUN rm` takes them back
out. Deleting at ingress is the only place that works.

`npa/.dockerignore` already said `__pycache__/` and `*.py[cod]`, which is why this went
unnoticed. Those patterns are matched against the whole path relative to the context
root, so they describe the root directory and nothing below it; Docker's context
documentation is explicit that `**` is what matches any number of directories. The
distinction is invisible by reading and obvious by building, so these tests build.

The context is wholly synthetic -- four files under a `FROM scratch` stage, no NPA
source, no production image. It carries the same shapes the real one does: nested
bytecode, nested legitimate source, packaged data a tool needs at runtime, and one
root-level `.pyc` that even the old rules caught. Each case is run twice, once under the
old rules and once under the rules now committed, because a test that only shows the fix
passing cannot tell you the fix was needed.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

DOCKERIGNORE = Path(__file__).resolve().parents[2] / ".dockerignore"

#: The rules as they stood when the scan found the bytecode.
ROOT_ONLY_RULES = "__pycache__/\n*.py[cod]\n"

#: Synthetic content at the path shape the scan actually found, so this stays tied to
#: the defect rather than to bytecode in general. Alongside it, the nested source and
#: the packaged data a recursive rule could easily take with it. The root-level `.pyc`
#: is the case the old rules did handle, kept so the fix cannot be credited for it.
SYNTHETIC_CONTEXT = {
    "src/npa/cli/cluster/terraform_lifecycle.py": "VALUE = 1\n",
    "src/npa/smoke/golden_evals.yaml": "tools: {}\n",
    "src/npa/cli/cluster/__pycache__/terraform_lifecycle.cpython-312.pyc": (
        "nested-host-bytecode-sentinel\n"
    ),
    "__pycache__/root.cpython-312.pyc": "root-bytecode-sentinel\n",
}

NESTED_BYTECODE = (
    "ctx/src/npa/cli/cluster/__pycache__/terraform_lifecycle.cpython-312.pyc"
)
ROOT_BYTECODE = "ctx/__pycache__/root.cpython-312.pyc"
NESTED_SOURCE = "ctx/src/npa/cli/cluster/terraform_lifecycle.py"
PACKAGED_DATA = "ctx/src/npa/smoke/golden_evals.yaml"


def _committed_rules() -> str:
    """The bytecode rules this repository now ships, comments and all."""

    return DOCKERIGNORE.read_text()


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        timeout=30,
    )
    return probe.returncode == 0


requires_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="asks Docker what its own matcher does; there is nothing to ask without a daemon",
)


def _context_admits(tmp_path: Path, rules: str) -> set[str]:
    """Build the synthetic context under `rules` and return what reached the image."""

    context = tmp_path / "context"
    for relative, content in SYNTHETIC_CONTEXT.items():
        path = context / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (context / ".dockerignore").write_text(rules)
    (context / "Dockerfile").write_text("FROM scratch\nCOPY . /ctx\n")

    exported = tmp_path / "exported.tar"
    subprocess.run(
        [
            "docker",
            "build",
            "--no-cache",
            "--output",
            f"type=tar,dest={exported}",
            str(context),
        ],
        check=True,
        capture_output=True,
        timeout=600,
    )
    with tarfile.open(exported) as archive:
        return {member.name for member in archive.getmembers() if member.isfile()}


@pytest.fixture(scope="module")
def under_old_rules(tmp_path_factory) -> set[str]:
    return _context_admits(tmp_path_factory.mktemp("old"), ROOT_ONLY_RULES)


@pytest.fixture(scope="module")
def under_committed_rules(tmp_path_factory) -> set[str]:
    return _context_admits(tmp_path_factory.mktemp("committed"), _committed_rules())


@requires_docker
def test_root_only_rules_let_nested_bytecode_into_the_image(under_old_rules):
    # The defect, reproduced: this is how a .pyc compiled by the host's interpreter
    # reached a layer of an image whose own runtime never writes one.
    assert NESTED_BYTECODE in under_old_rules
    # And the reason it looked handled. The root-level case always worked, so the rules
    # read as though they covered bytecode generally.
    assert ROOT_BYTECODE not in under_old_rules


@requires_docker
def test_committed_rules_keep_nested_bytecode_out(under_committed_rules):
    assert NESTED_BYTECODE not in under_committed_rules
    assert ROOT_BYTECODE not in under_committed_rules


@requires_docker
def test_committed_rules_still_admit_source_and_packaged_data(under_committed_rules):
    # A recursive exclusion is easy to overshoot into the tree it is protecting. Nested
    # source and the data files tools read at runtime have to arrive unchanged.
    assert NESTED_SOURCE in under_committed_rules
    assert PACKAGED_DATA in under_committed_rules


@requires_docker
def test_the_fix_changes_only_the_nested_bytecode(
    under_old_rules, under_committed_rules
):
    # Scope, measured rather than asserted: the two builds differ by exactly the file
    # the scan found, so nothing else was quietly dropped from the context.
    assert under_old_rules - under_committed_rules == {NESTED_BYTECODE}
    assert under_committed_rules - under_old_rules == set()
