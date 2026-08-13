"""Workbench images must stay schedulable by SkyPilot on Kubernetes.

SkyPilot's Kubernetes runtime bootstrap runs inside the task container and needs a
system ``python3`` (plus ``rsync``). A vendor image that ships only its own interpreter
cannot host a task at all: provisioning fails with

    KubernetesError: Failed to get ssh user for pod ...: container not found ("ray-node")

which is what the Isaac Lab image did on npa-rtxpro-mk8s until the prerequisites were
added. These are cheap textual guards so the requirement cannot silently regress, and
so a reviewer can see *why* the lines exist.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

DOCKER_ROOT = Path(__file__).resolve().parents[2] / "docker" / "workbench"


def _build_text(tool: str) -> str:
    """Everything a tool's build actually executes: its Dockerfile plus the scripts it runs.

    The prerequisites below used to sit literally in isaac-lab/Dockerfile. They now live in
    docker/workbench/common/install_isaac_runtime_base.sh, which the Dockerfile COPYs and
    RUNs, because isaac-lab and sonic share that layer. A guard that only read the
    Dockerfile would have started passing vacuously the moment the layer was factored out -
    so it follows the Dockerfile into the scripts instead, and keeps checking the same
    facts about the same image.
    """
    dockerfile = DOCKER_ROOT / tool / "Dockerfile"
    assert dockerfile.is_file(), dockerfile
    text = dockerfile.read_text(encoding="utf-8")
    parts = [text]
    for script in sorted((DOCKER_ROOT / "common").glob("*.sh")):
        # Only scripts this Dockerfile actually invokes count towards the guard.
        if script.name in text:
            parts.append(script.read_text(encoding="utf-8"))
    return "\n".join(parts)


# Images whose stages are submitted through SkyPilot (npa.workflow / workbench
# workflows) and therefore must be schedulable in a pod. This list grows as the raw
# SkyPilot task catalog is retired: once a tool's only workflow surface is an
# npa.workflow spec, its image MUST be able to host a SkyPilot task.
SKYPILOT_HOSTED_IMAGES = (
    "cosmos2-transfer",
    "cosmos3-reason",
    "groot",
    "isaac-lab",
    "lerobot",
    "sim2real-control",
    "sim2real-envgen",
    "sim2real-eval",
    "sonic",
)

# Only these images publish a lightweight repair Dockerfile for an existing
# vendor base.  The purpose-built canonical Sim2Real images above must satisfy
# the same runtime contract, but rebuilding their primary Dockerfile is the
# supported repair path.
DERIVED_PREREQ_IMAGES = (
    "cosmos3-reason",
    "groot",
    "isaac-lab",
    "lerobot",
    "sim2real-control",
    "sonic",
)

#: Images built on an Isaac base, where /isaac-sim is mode 750 owned by
#: isaac-sim:isaac-sim and the runtime user therefore has to join that GROUP (a
#: recursive chmod would rewrite multi-GB layers). Not universal: the lerobot image has
#: no /isaac-sim at all, so requiring the usermod there would pin a no-op.
ISAAC_BASED_IMAGES = ("isaac-lab", "sonic")


#: What every SkyPilot-hosted image needs, established by bisecting derived images against a
#: live Kubernetes GPU cluster. Missing any one makes provisioning fail with
#: `container not found ("ray-node")`, which names none of them.
REQUIRED_INGREDIENTS = (
    ("python3", "SkyPilot's k8s runtime bootstrap needs a system python3"),
    ("rsync", "SkyPilot syncs files with rsync"),
    ("NOPASSWD", "SkyPilot's in-pod setup shells out to sudo without a password"),
)

#: NOT universal, despite being required wherever it applies. An Isaac image's default python3
#: is a kit interpreter that cannot import its own site-packages outside python.sh, so the
#: system one has to win. Forcing the same ordering on an image whose OWN python carries npa
#: breaks it: cosmos3-reason failed setup with "npa is not importable after setup" (job 307).
PATH_ORDERING_INGREDIENT = (
    "ENV PATH=/usr/bin:$PATH",
    "an Isaac kit python cannot import its own site-packages, so the system one must precede it",
)


def _ingredients_for(tool: str) -> tuple[tuple[str, str], ...]:
    if tool in ISAAC_BASED_IMAGES:
        return REQUIRED_INGREDIENTS + (PATH_ORDERING_INGREDIENT,)
    return REQUIRED_INGREDIENTS


@pytest.mark.parametrize("tool", SKYPILOT_HOSTED_IMAGES)
def test_dockerfile_has_skypilot_runtime_prerequisites(tool: str) -> None:
    """Follows the Dockerfile into the shared script (#229), per-tool ingredients (this branch).

    The PATH ordering is an ISAAC requirement, not a universal one: forcing /usr/bin first on
    cosmos3-reason shadowed the vendor's own python and left npa unimportable (live job 307).
    """

    text = _build_text(tool)
    for token, why in _ingredients_for(tool):
        assert token in text, f"{tool}: {why}"


def test_the_prereq_guard_is_not_satisfied_by_the_dockerfile_alone() -> None:
    """Pin that the guard genuinely follows the Dockerfile into the shared script.

    Without this, someone could "fix" a failure by reading only the Dockerfile again and
    the guard would silently stop checking anything - which is exactly what happened when
    the layer was factored out.
    """
    dockerfile_only = (DOCKER_ROOT / "isaac-lab" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    combined = _build_text("isaac-lab")
    assert len(combined) > len(dockerfile_only), "no scripts were followed"
    moved = [
        token
        for token, _ in _ingredients_for("isaac-lab")
        if token not in dockerfile_only and token in combined
    ]
    assert moved, (
        "expected at least one prerequisite to live in the shared script rather than the "
        "Dockerfile; if they have all moved back, simplify this guard deliberately"
    )


def test_groot_enables_the_shared_skypilot_prerequisite_layer() -> None:
    """GR00T invokes the shared installer conditionally, so pin the enabled branch."""

    dockerfile = (DOCKER_ROOT / "groot" / "Dockerfile").read_text(encoding="utf-8")
    assert "NPA_INSTALL_SKYPILOT_PREREQS=1" in dockerfile
    assert "NPA_INSTALL_SKYPILOT_PREREQS=0" not in dockerfile
    assert "dpkg --purge --force-depends linux-libc-dev" not in dockerfile

    derived = (DOCKER_ROOT / "groot" / "Dockerfile.k8s-prereqs").read_text(
        encoding="utf-8"
    )
    assert "--fix-broken" in derived
    for token in (
        "openssh-server",
        "rsync",
        "sudo",
        "ubuntu ALL=(ALL) NOPASSWD:ALL",
        "/etc/sudoers.d/99-npa-runtime-user",
        'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"',
        "ENV HOME=/home/ubuntu",
        "ssh-keygen -A",
        "rm -f /etc/ssh/ssh_host_*",
        r'exec \"$@\"',
    ):
        assert token in derived, f"derived GR00T image missing {token!r}"
    service_keygen = (
        r"&& sed -i '/^  start)$/a\    ssh-keygen -A' /etc/init.d/ssh"
    )
    remove_build_keys = "&& rm -f /etc/ssh/ssh_host_*"
    assert service_keygen in derived
    assert derived.index(service_keygen) < derived.index(
        remove_build_keys
    ), "service start must regenerate host keys before the build-time keys are removed"
    user_lines = [
        line.strip()
        for line in derived.splitlines()
        if line.strip().startswith("USER ")
    ]
    assert user_lines[-1] == "USER ubuntu"


@pytest.mark.parametrize("tool", SKYPILOT_HOSTED_IMAGES)
def test_skypilot_hosted_image_stays_non_root(tool: str) -> None:
    """The prerequisites make a NON-root image schedulable; keep it that way.

    An image that simply ends as root also works, but that is a needless privilege
    escalation for every stage the workbench runs.
    """

    text = (DOCKER_ROOT / tool / "Dockerfile").read_text(encoding="utf-8")
    user_lines = [
        line.strip() for line in text.splitlines() if line.strip().startswith("USER ")
    ]
    assert user_lines and user_lines[-1] != "USER root", (
        f"{tool}: image must not end as root; the sudo/group/PATH ingredients exist "
        "precisely so a non-root image can host a SkyPilot task"
    )


def test_isaac_lab_grants_its_runtime_user_access_to_isaac_sim() -> None:
    """The runtime user must be able to traverse /isaac-sim and reach the interpreter.

    Historically /isaac-sim came from NVIDIA's base image at mode 750 isaac-sim:isaac-sim,
    unreadable by ``ubuntu``, and group membership was the cheap fix (a recursive
    chown/chmod would have rewritten multi-GB Isaac layers). The image no longer bakes
    Isaac Sim, so /isaac-sim is our own directory holding only the bootstrap shim - but
    the requirement is unchanged and still asserted, because the failure it prevents is
    silent: /isaac-sim/python.sh resolves empty and every Isaac job exits 127.
    """

    text = _build_text("isaac-lab")
    assert "usermod -aG isaac-sim" in text, (
        "the runtime user must be in the isaac-sim group so it can traverse /isaac-sim"
    )
    # Check instructions only: the rationale comments name the approach they avoid.
    instructions = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "chmod -R" not in instructions, (
        "a recursive chmod would rewrite multi-GB layers; use group membership"
    )
    # And the shim must actually be installed there, executable, for any of that to help.
    assert "/isaac-sim/python.sh" in text
    assert "install -d -m 0755 /isaac-sim" in text, (
        "/isaac-sim must be world-traversable now that it is our own directory"
    )


@pytest.mark.parametrize("tool", DERIVED_PREREQ_IMAGES)
def test_derived_prereq_dockerfile_matches_the_shipped_one(tool: str) -> None:
    """The derived recipe exists and applies the same prerequisites.

    Operators use it to repair an already-published tag without pulling the multi-GB
    base (scripts/build-workbench-image-in-cluster.sh); it must not drift from the
    image.
    """

    derived = DOCKER_ROOT / tool / "Dockerfile.k8s-prereqs"
    assert derived.is_file(), derived
    text = derived.read_text(encoding="utf-8")
    ingredients = tuple(
        item
        for item in _ingredients_for(tool)
        if not (tool == "groot" and item[0] == "NOPASSWD")
    )
    for token, _why in (*ingredients, ("ARG BASE_IMAGE", "derived build")):
        assert token in text, f"{tool}: derived prereq Dockerfile is missing {token!r}"
    if tool in ISAAC_BASED_IMAGES:
        assert "usermod -aG isaac-sim" in text, (
            f"{tool} derives from an Isaac base, where /isaac-sim is mode 750 "
            "isaac-sim:isaac-sim, so the runtime user must join that group"
        )
        # Scheduling is not enough: Kit also has to be able to WRITE. Without these three
        # directories Isaac boots, fails to save its user config, and then renders nothing
        # while burning CPU — live job 271 stalled for 45 minutes that way, which is far
        # harder to diagnose than a pod that never starts.
        for kit_dir in (
            "/isaac-sim/kit/data",
            "/isaac-sim/kit/logs",
            "/isaac-sim/kit/cache",
        ):
            assert kit_dir in text, (
                f"{tool}: {kit_dir} must exist and belong to the runtime user, or Kit stalls"
            )
        assert "chown -R ubuntu:ubuntu /isaac-sim/kit/data" in text
        # Newer bases fetch Isaac at run time and keep Kit's state in /tmp instead, so a
        # derived recipe must carry that mechanism too: it cannot know which base it repairs.
        assert "OMNI_USER_DIR" in text and "OMNI_LOG_DIR" in text


@pytest.mark.parametrize("tool", ("sim2real-envgen", "sim2real-eval"))
def test_genesis_derived_workflow_images_pin_the_bootstrap_closure(tool: str) -> None:
    """The two Genesis-derived canonical stages failed identically without sudo."""

    dockerfile = (DOCKER_ROOT / tool / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG UBUNTU_SNAPSHOT=20260801T053000Z" in dockerfile
    assert "install_workflow_runtime_prereqs.sh" in dockerfile
    assert 'install-workflow-runtime-prereqs "${UBUNTU_SNAPSHOT}"' in dockerfile

    installer = (
        DOCKER_ROOT / "common" / "install_workflow_runtime_prereqs.sh"
    ).read_text(encoding="utf-8")
    assert "snapshot.ubuntu.com/ubuntu/${snapshot}" in installer
    assert "ubuntu:22.04" in installer
    assert "apt-get --fix-broken install -y --no-install-recommends" in installer
    assert "sudo" in installer and "rsync" in installer
    assert "NOPASSWD" in installer
    assert "sudo -n true" in installer


def test_in_cluster_build_script_is_executable_and_generic() -> None:
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "build-workbench-image-in-cluster.sh"
    )
    assert script.is_file(), script
    assert os.access(script, os.X_OK), script
    text = script.read_text(encoding="utf-8")
    # No hardcoded registry/bucket/project identifiers.
    assert "cr.us-central1" not in text and "cr.eu-north1" not in text
    for flag in ("--base", "--tag", "--dockerfile", "--pull-secret", "--namespace"):
        assert flag in text, f"build script should accept {flag}"
