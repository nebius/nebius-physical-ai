"""The shipped npa-ltx2 bootstrap script is executed, not just read.

``ltx_runtime.sh`` is the only thing standing between an operator who has
declared nothing and Lightricks' gated weights, so a test that merely greps it
for the word "refuse" would be worthless. These cases assemble the in-image
layout in a temporary directory and run the real script with ``bash``, which is
the same code path the container takes.

Two properties matter more than the individual exit codes:

* the refusal is not vacuous — a complete declaration passes the licence gate;
* the two vendors gate independently, and both gate *before* the fetch, so
  accepting Lightricks' terms never silently accepts NVIDIA's.

Nothing here reaches the network. The one case that gets far enough to make a
regression dangerous points the source remote at a nonexistent local path, so a
broken ordering fails immediately instead of quietly cloning from GitHub.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from npa.workbench.ltx2.licensing import (
    ACCEPT_ENV,
    COMMERCIAL_AGREEMENT_ENV,
    ENTITY_CLASS_ENV,
    ENTITY_COMMERCIAL,
    ENTITY_COMMUNITY,
    SOURCE_REF,
    USE_CLASS_ENV,
    USE_COMMERCIAL,
    USE_NON_COMMERCIAL,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "docker" / "workbench" / "ltx2"
LICENSING_MODULE = REPO_ROOT / "src" / "npa" / "workbench" / "ltx2" / "licensing.py"

EX_CONFIG = 78
EX_SOFTWARE = 70
NVIDIA_ACCEPT_ENV = "NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS"

# Enough of a declaration to clear the Lightricks gate. Deliberately the
# non-commercial community answer: it is the only combination that needs no
# paid agreement, and the tests that use it never fetch anything.
DECLARED = {
    ACCEPT_ENV: "YES",
    ENTITY_CLASS_ENV: ENTITY_COMMUNITY,
    USE_CLASS_ENV: USE_NON_COMMERCIAL,
}

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="the bootstrap script requires bash"
)


@pytest.fixture
def image(tmp_path: Path) -> Path:
    """Reproduce the parts of the image layout the script actually depends on."""

    root = tmp_path / "image"
    gate_dir = root / "opt" / "npa" / "ltx2"
    gate_dir.mkdir(parents=True)
    shutil.copy2(DOCKER_DIR / "ltx_gate.py", gate_dir / "ltx_gate.py")
    shutil.copy2(LICENSING_MODULE, gate_dir / "licensing.py")

    runtime = root / "usr" / "local" / "bin" / "ltx-runtime"
    runtime.parent.mkdir(parents=True)
    shutil.copy2(DOCKER_DIR / "ltx_runtime.sh", runtime)
    runtime.chmod(0o755)
    return root


def run(image: Path, *args: str, env: dict[str, str] | None = None):
    """Invoke the shipped script with only the environment a container would have."""

    root = image.parent
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(root),
        "NPA_LTX_GATE": str(image / "opt" / "npa" / "ltx2" / "ltx_gate.py"),
        "NPA_LTX_RUNTIME_CACHE": str(root / "cache"),
        "NPA_LTX_MODEL_CACHE": str(root / "model-cache"),
    }
    environment.update(env or {})
    return subprocess.run(
        ["bash", str(image / "usr" / "local" / "bin" / "ltx-runtime"), *args],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )


def nothing_was_fetched(image: Path) -> bool:
    root = image.parent
    for cache in (root / "cache", root / "model-cache"):
        if cache.exists() and any(cache.iterdir()):
            return False
    return True


class TestTheBuildTimeRefusalProof:
    def test_assert_refusal_passes_exactly_as_the_dockerfile_runs_it(
        self, image: Path
    ) -> None:
        """This is the Dockerfile's own build-time proof, run for real here.

        The image cannot be built in unit CI, so running the identical mode
        locally is what keeps the build's claim honest between builds.
        """

        result = run(image, "assert-refusal")

        assert result.returncode == 0, result.stderr
        assert "NPA_LTX_BOOTSTRAP_REFUSES_WITHOUT_DECLARATION_OK" in result.stdout
        assert nothing_was_fetched(image)

    def test_the_proof_ignores_acceptance_leaked_into_the_builder(
        self, image: Path
    ) -> None:
        """A builder with the variables set must not be able to pass vacuously."""

        result = run(image, "assert-refusal", env=DECLARED | {NVIDIA_ACCEPT_ENV: "YES"})

        assert result.returncode == 0, result.stderr
        assert "NPA_LTX_BOOTSTRAP_REFUSES_WITHOUT_DECLARATION_OK" in result.stdout
        assert nothing_was_fetched(image)


class TestTheProofItselfIsMutationTested:
    """Break each gate in turn; ``assert-refusal`` must notice every time.

    This is the case that gives the build-time proof its value. An earlier
    version of ``assert_refusal`` only checked that ``ensure`` exited 78, and a
    licence gate rewritten to accept everything sailed through it — because the
    NVIDIA gate downstream refused for its own reasons and produced the same 78.
    Three gates that all fail closed with one exit code will mask each other
    unless the proof also checks *which* one fired.
    """

    def _mutate_licence_gate(self, image: Path) -> None:
        """Make the licensing module accept any environment, declared or not."""

        module = image / "opt" / "npa" / "ltx2" / "licensing.py"
        module.chmod(0o644)
        with module.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n\ndef declaration_from_env(env):\n"
                f"    return LicenseDeclaration(entity_class={ENTITY_COMMUNITY!r},"
                f" use_class={USE_NON_COMMERCIAL!r})\n"
            )

    def _mutate_shell_gate(self, image: Path, function: str) -> None:
        """Make one shell gate return success instead of refusing."""

        script = image / "usr" / "local" / "bin" / "ltx-runtime"
        text = script.read_text(encoding="utf-8")
        needle = f"{function}() {{"
        assert needle in text, f"{function} is no longer defined as expected"
        script.write_text(
            text.replace(needle, f"{needle} return 0;", 1), encoding="utf-8"
        )

    def test_a_licence_gate_that_accepts_everything_is_caught(
        self, image: Path
    ) -> None:
        self._mutate_licence_gate(image)

        result = run(image, "assert-refusal")

        assert result.returncode == EX_SOFTWARE
        assert "NPA_LTX_BOOTSTRAP_REFUSES_WITHOUT_DECLARATION_OK" not in result.stdout
        assert "not on NPA_LTX_ACCEPT_COMMUNITY_LICENSE" in result.stderr

    @pytest.mark.parametrize(
        "function",
        ["require_nvidia_acceptance", "require_hf_token"],
    )
    def test_a_downstream_gate_that_accepts_everything_is_caught(
        self, image: Path, function: str
    ) -> None:
        self._mutate_shell_gate(image, function)

        result = run(image, "assert-refusal")

        assert result.returncode == EX_SOFTWARE
        assert "NPA_LTX_BOOTSTRAP_REFUSES_WITHOUT_DECLARATION_OK" not in result.stdout

    def test_an_unmutated_image_still_passes(self, image: Path) -> None:
        """The control: without it the three cases above could pass on any error."""

        assert run(image, "assert-refusal").returncode == 0


class TestTheLightricksGate:
    def test_an_undeclared_ensure_refuses_before_touching_anything(
        self, image: Path
    ) -> None:
        result = run(image, "ensure")

        assert result.returncode == EX_CONFIG
        assert "Nothing has been downloaded." in result.stderr
        assert nothing_was_fetched(image)

    def test_the_refusal_names_every_answer_the_operator_owes(
        self, image: Path
    ) -> None:
        """The refusal has to be actionable, or operators will bake a YES to escape it."""

        stderr = run(image, "ensure").stderr

        for variable in (ACCEPT_ENV, ENTITY_CLASS_ENV, USE_CLASS_ENV):
            assert variable in stderr
        assert "LTX-2.x Community License Agreement" in stderr
        assert "Attachment A(18)" in stderr
        # The threshold is the question nobody but the operator can answer.
        assert "10,000,000" in stderr

    @pytest.mark.parametrize(
        "declaration",
        [
            pytest.param({ACCEPT_ENV: "YES"}, id="accepted-but-unclassified"),
            pytest.param(
                {ACCEPT_ENV: "YES", ENTITY_CLASS_ENV: ENTITY_COMMUNITY},
                id="no-use-class",
            ),
            pytest.param(
                {
                    ACCEPT_ENV: "no",
                    **{k: v for k, v in DECLARED.items() if k != ACCEPT_ENV},
                },
                id="explicitly-declined",
            ),
            pytest.param(
                DECLARED | {ENTITY_CLASS_ENV: "probably-fine"}, id="unrecognised-entity"
            ),
            pytest.param(
                DECLARED | {USE_CLASS_ENV: "research-ish"}, id="unrecognised-use"
            ),
        ],
    )
    def test_a_partial_or_unrecognised_declaration_refuses(
        self, image: Path, declaration: dict[str, str]
    ) -> None:
        result = run(image, "ensure", env=declaration)

        assert result.returncode == EX_CONFIG
        assert nothing_was_fetched(image)

    def test_a_commercial_entity_in_commercial_use_needs_its_paid_agreement(
        self, image: Path
    ) -> None:
        """Section 2.1: this combination is prohibited outright without one."""

        declaration = {
            ACCEPT_ENV: "YES",
            ENTITY_CLASS_ENV: ENTITY_COMMERCIAL,
            USE_CLASS_ENV: USE_COMMERCIAL,
        }
        refused = run(image, "ensure", env=declaration)

        assert refused.returncode == EX_CONFIG
        assert COMMERCIAL_AGREEMENT_ENV in refused.stderr
        assert "ltxv-licensing@lightricks.com" in refused.stderr
        assert nothing_was_fetched(image)


class TestTheNvidiaGateIsSeparate:
    def test_accepting_lightricks_terms_does_not_accept_nvidias(
        self, image: Path
    ) -> None:
        """Also the ordering proof: both gates close before the fetch.

        The source remote is pointed at a path that does not exist, so if a
        future edit ever moves the fetch ahead of a gate this fails loudly here
        instead of reaching out to GitHub from a unit test.
        """

        result = run(
            image,
            "ensure",
            env=DECLARED | {"NPA_LTX_SOURCE_REPO": str(image.parent / "no-such-repo")},
        )

        assert result.returncode == EX_CONFIG
        assert NVIDIA_ACCEPT_ENV in result.stderr
        assert "docs.nvidia.com/cuda/eula" in result.stderr
        assert nothing_was_fetched(image)

    def test_the_lightricks_gate_runs_first(self, image: Path) -> None:
        """Accepting NVIDIA's terms alone must not get anywhere near a download."""

        result = run(image, "ensure", env={NVIDIA_ACCEPT_ENV: "YES"})

        assert result.returncode == EX_CONFIG
        assert ACCEPT_ENV in result.stderr
        assert nothing_was_fetched(image)


class TestTheWeightsGate:
    def test_a_complete_declaration_still_needs_the_operators_own_entitlement(
        self, image: Path
    ) -> None:
        """The weights are gated on Hugging Face; we hold no entitlement to lend."""

        result = run(image, "fetch-weights", env=DECLARED)

        assert result.returncode == EX_CONFIG
        assert "GATED repository" in result.stderr
        assert "HF_TOKEN" in result.stderr
        assert nothing_was_fetched(image)

    def test_an_undeclared_weight_fetch_refuses_on_the_licence_first(
        self, image: Path
    ) -> None:
        result = run(image, "fetch-weights", env={"HF_TOKEN": "hf_not-a-real-token"})

        assert result.returncode == EX_CONFIG
        assert ACCEPT_ENV in result.stderr
        assert nothing_was_fetched(image)


class TestTheRefusalIsNotVacuous:
    def test_a_complete_declaration_passes_the_licence_gate(self, image: Path) -> None:
        """Without this, every case above would also pass a script that always exits 78."""

        gate = image / "opt" / "npa" / "ltx2" / "ltx_gate.py"
        result = subprocess.run(
            ["python3", str(gate), "check"],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", **DECLARED},
            timeout=60,
        )

        assert result.returncode == 0, result.stderr
        assert "operator declaration accepted" in result.stderr
        assert "derived_model_training=non-commercial-only" in result.stderr

    def test_the_declared_disposition_reaches_the_provenance_record(
        self, image: Path
    ) -> None:
        """The commercial answer is the one that later stops the trainer."""

        gate = image / "opt" / "npa" / "ltx2" / "ltx_gate.py"
        result = subprocess.run(
            ["python3", str(gate), "provenance"],
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                ACCEPT_ENV: "YES",
                ENTITY_CLASS_ENV: ENTITY_COMMUNITY,
                USE_CLASS_ENV: USE_COMMERCIAL,
                "NPA_LTX_RUN_ID": "bootstrap-check",
            },
            timeout=60,
        )

        assert result.returncode == 0, result.stderr
        record = json.loads(result.stdout)
        assert record["restrictions"]["derived_model_training"] == "prohibited"
        assert record["license"]["osi_approved"] is False
        assert record["run_id"] == "bootstrap-check"


class TestModesThatNeverNeedADeclaration:
    def test_terms_prints_the_agreement_summary_without_one(self, image: Path) -> None:
        result = run(image, "terms")

        assert result.returncode == 0, result.stderr
        assert "LTX-2.x Community License Agreement" in result.stdout

    def test_status_reports_an_empty_image_as_machine_readable_json(
        self, image: Path
    ) -> None:
        result = run(image, "status")

        assert result.returncode == 0, result.stderr
        status = json.loads(result.stdout)
        assert status == {
            "source": "absent",
            "source_ref": SOURCE_REF,
            "weights": "absent",
            "cache": str(image.parent / "cache"),
            "model_cache": str(image.parent / "model-cache"),
        }

    def test_version_reports_the_pinned_source_and_the_provisioning_model(
        self, image: Path
    ) -> None:
        result = run(image, "version")

        assert result.returncode == 0, result.stderr
        assert SOURCE_REF in result.stdout
        assert "provisioning=runtime-fetch" in result.stdout

    def test_an_unknown_mode_refuses_rather_than_defaulting_to_a_fetch(
        self, image: Path
    ) -> None:
        result = run(image, "definitely-not-a-mode")

        assert result.returncode == EX_CONFIG
        assert nothing_was_fetched(image)
