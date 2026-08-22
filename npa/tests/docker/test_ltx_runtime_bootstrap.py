"""The shipped npa-ltx2 bootstrap script is executed, not just read.

``ltx_runtime.sh`` is the only thing standing between an operator with no
entitlement and Lightricks' gated source and weights, so a test that merely
greps it for the word "refuse" would be worthless. These cases assemble the
in-image layout in a temporary directory and run the real script with ``bash``,
which is the same code path the container takes.

Two properties matter more than the individual exit codes:

* the refusal is not vacuous — the modes that need no entitlement still work,
  and the script does not simply exit 78 at everything;
* the two vendors gate independently, and both gate *before* the fetch, so
  holding a Hugging Face entitlement never silently accepts NVIDIA's terms.

There is nothing here for an operator to declare. The LTX-2.x agreement binds by
conduct and access to the gated repository is granted by Lightricks after a human
accepts the terms there, so ``HF_TOKEN`` — required for the *source* as well as
the weights, since Section 1.9 makes the source licensed material — is the only
entitlement this script can check.

Nothing here reaches the network. The one case that gets far enough to make a
regression dangerous points the source remote at a nonexistent local path, so a
broken ordering fails immediately instead of quietly cloning from GitHub.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from npa.workbench.ltx2.licensing import (
    ACCEPTABLE_USE_POLICY_URL,
    COMMERCIAL_LICENSE_CONTACT,
    LICENSE_NAME,
    LICENSE_URL,
    SOURCE_REF,
    WEIGHTS_REPO_URL,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "docker" / "workbench" / "ltx2"

EX_CONFIG = 78
EX_SOFTWARE = 70
NVIDIA_ACCEPT_ENV = "NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS"
MARKER = "NPA_LTX_BOOTSTRAP_REFUSES_WITHOUT_ENTITLEMENT_OK"

#: Enough to clear the Hugging Face check. It is not a credential and cannot
#: fetch anything: the check tests for presence, and every case that sets it
#: stops at the next gate or at a source remote that does not exist.
ENTITLED = {"HF_TOKEN": "not-a-real-token"}

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="the bootstrap script requires bash"
)


def test_official_build_helper_only_pushes_full_sha_development_tags() -> None:
    build = (DOCKER_DIR / "build.sh").read_text(encoding="utf-8")
    assert '"${REGISTRY%/}" == "ghcr.io/nebius/nebius-physical-ai"' in build
    assert '"$TAG" == "dev-${SOURCE_COMMIT}"' in build
    assert "promote releases by digest" in build


@pytest.fixture
def image(tmp_path: Path) -> Path:
    """Reproduce the parts of the image layout the script actually depends on."""

    root = tmp_path / "image"
    opt = root / "opt" / "npa" / "ltx2"
    opt.mkdir(parents=True)
    # `health` asserts the video check shipped, which is the only Python of ours
    # the image still carries.
    shutil.copy2(
        REPO_ROOT / "src" / "npa" / "workbench" / "ltx2" / "video_check.py",
        opt / "video_check.py",
    )
    shutil.copy2(DOCKER_DIR / "validate_video.py", opt / "validate_video.py")
    (opt / "validate_video.py").chmod(0o755)

    # The image creates both cache mount points before anything runs. Leaving
    # them absent here made assert_refusal's "the refusal wrote nothing" checks
    # pass trivially, on paths that could not have been written to.
    (root.parent / "cache").mkdir(parents=True, exist_ok=True)
    (root.parent / "model-cache").mkdir(parents=True, exist_ok=True)

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


def test_runtime_sync_uses_the_final_checkout_path() -> None:
    """Editable package paths must survive publication of the runtime tree."""

    script = (DOCKER_DIR / "ltx_runtime.sh").read_text(encoding="utf-8")
    move = script.index('mv "$tmp" "$tree"')
    sync = script.index('( cd "$tree" && uv sync --extra "$UV_EXTRA" )')
    marker = script.index(': > "$tree/.complete"')

    assert move < sync < marker
    assert '( cd "$tmp" && uv sync' not in script


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
        assert MARKER in result.stdout
        assert nothing_was_fetched(image)

    def test_the_proof_ignores_entitlement_leaked_into_the_builder(
        self, image: Path
    ) -> None:
        """A builder with a token and NVIDIA acceptance set must not pass vacuously."""

        result = run(image, "assert-refusal", env=ENTITLED | {NVIDIA_ACCEPT_ENV: "YES"})

        assert result.returncode == 0, result.stderr
        assert MARKER in result.stdout
        assert nothing_was_fetched(image)

    def test_it_passes_against_a_cache_an_earlier_run_already_filled(
        self, image: Path
    ) -> None:
        """The invariant is "the refusal downloaded nothing", not "the cache is empty".

        Pointing `NPA_LTX_MODEL_CACHE` at the operator's durable weight cache
        (docs/workbench/model-weight-cache.md) is what makes the second run of
        this image a cache hit rather than another 22B download. An emptiness
        assertion against the shared cache fails every run after the first, so the
        gates are exercised against private directories instead.
        """

        weights = image.parent / "model-cache" / "vae"
        weights.mkdir(parents=True)
        preexisting = weights / "ltx-2.5-video-vae-bf16.safetensors"
        preexisting.write_bytes(b"fetched by an earlier run")

        result = run(image, "assert-refusal")

        assert result.returncode == 0, result.stderr
        assert MARKER in result.stdout
        assert preexisting.read_bytes() == b"fetched by an earlier run"

    def test_it_ignores_another_stage_writing_to_the_shared_cache(
        self, image: Path
    ) -> None:
        """A durable cache is shared, so writes to it are not attributable to us.

        Two LTX stages can run at once against one claim, and the other one
        fetching weights while this proof runs must not read as this proof's gate
        leaking. Simulated deterministically by writing to the shared cache from
        inside the proof, between the gates and the assertion.
        """

        script = image / "usr" / "local" / "bin" / "ltx-runtime"
        text = script.read_text(encoding="utf-8")
        anchor = '  [[ -z "$(find "$PROBE_RUNTIME_CACHE"'
        assert anchor in text, "assert_refusal no longer checks a private tree"
        concurrent = (
            '  touch "$MODEL_CACHE/another-stage-fetched-this.safetensors"\n'
            '  touch "$CACHE_ROOT/another-stage-built-this"\n'
        )
        script.write_text(text.replace(anchor, concurrent + anchor, 1), encoding="utf-8")

        result = run(image, "assert-refusal")

        assert result.returncode == 0, result.stderr
        assert MARKER in result.stdout


class TestTheProofItselfIsMutationTested:
    """Break each gate in turn; ``assert-refusal`` must notice every time.

    This is the case that gives the build-time proof its value. An earlier
    version of ``assert_refusal`` only checked that ``ensure`` exited 78, and a
    gate rewritten to accept everything sailed through it — because the next
    gate refused for its own reasons and produced the same 78. Gates that all
    fail closed with one exit code will mask each other unless the proof also
    checks *which* one fired.
    """

    def _mutate_shell_gate(self, image: Path, function: str) -> None:
        """Make one shell gate return success instead of refusing."""

        script = image / "usr" / "local" / "bin" / "ltx-runtime"
        text = script.read_text(encoding="utf-8")
        needle = f"{function}() {{"
        assert needle in text, f"{function} is no longer defined as expected"
        script.write_text(
            text.replace(needle, f"{needle} return 0;", 1), encoding="utf-8"
        )

    @pytest.mark.parametrize(
        ("function", "expected"),
        [
            # Assert *why* each mutant was caught. `rc == 70` alone is satisfied
            # by any error in the script — the same vacuity this proof exists to
            # avoid one level up.
            ("require_hf_token", "not on HF_TOKEN"),
            ("require_nvidia_acceptance", "the NVIDIA runtime gate"),
        ],
    )
    def test_a_gate_that_accepts_everything_is_caught(
        self, image: Path, function: str, expected: str
    ) -> None:
        self._mutate_shell_gate(image, function)

        result = run(image, "assert-refusal")

        assert result.returncode == EX_SOFTWARE
        assert MARKER not in result.stdout
        assert expected in result.stderr, result.stderr

    def test_dropping_the_token_check_from_the_source_fetch_is_caught(
        self, image: Path
    ) -> None:
        """The source is licensed material too, so its own gate is load-bearing.

        Removing `require_hf_token` from `fetch_source` alone leaves the weight
        path fully guarded, so a proof that only exercised `fetch-weights` would
        still pass while an unentitled `ensure` cloned Lightricks' repository.
        """

        script = image / "usr" / "local" / "bin" / "ltx-runtime"
        text = script.read_text(encoding="utf-8")
        guards = "  require_hf_token\n  require_nvidia_acceptance\n"
        assert guards in text, "fetch_source no longer opens with both gates"
        script.write_text(
            text.replace(guards, "  require_nvidia_acceptance\n", 1), encoding="utf-8"
        )

        result = run(image, "assert-refusal")

        assert result.returncode == EX_SOFTWARE
        assert MARKER not in result.stdout
        assert "'ensure' refused, but not on HF_TOKEN" in result.stderr

    def _fetch_before_the_gates(self, image: Path) -> None:
        script = image / "usr" / "local" / "bin" / "ltx-runtime"
        text = script.read_text(encoding="utf-8")
        guards = "  require_hf_token\n  require_nvidia_acceptance\n"
        assert guards in text, "fetch_source no longer opens with both gates"
        moved = text.replace(guards, "", 1).replace(
            '  git init -q "$tmp"', guards + '  git init -q "$tmp"', 1
        )
        script.write_text(moved, encoding="utf-8")

    def test_fetching_before_the_gates_run_is_caught(self, image: Path) -> None:
        """The ordering property, mutated rather than merely implied.

        Both gates must close before `fetch_source` reaches the network. Move
        the guard calls after the clone begins and the proof should notice — via
        the cache the aborted clone leaves behind, not via the exit code.
        """

        self._fetch_before_the_gates(image)

        result = run(
            image,
            "assert-refusal",
            env={"NPA_LTX_SOURCE_REPO": str(image.parent / "no-such-repo")},
        )

        assert result.returncode == EX_SOFTWARE
        assert MARKER not in result.stdout

    def test_that_ordering_check_still_bites_on_a_warm_cache(
        self, image: Path
    ) -> None:
        """Tolerating an already-filled cache must not tolerate a leaking gate.

        The proof no longer demands the shared caches be empty, so this is the
        case that says it still detects the thing it exists to detect: a run whose
        cache is already populated by an earlier run must still catch a fetch that
        beat the gates.
        """

        seeded = image.parent / "cache" / "src" / "from-an-earlier-run"
        seeded.mkdir(parents=True)
        (seeded / ".complete").write_text("", encoding="utf-8")
        self._fetch_before_the_gates(image)

        result = run(
            image,
            "assert-refusal",
            env={"NPA_LTX_SOURCE_REPO": str(image.parent / "no-such-repo")},
        )

        assert result.returncode == EX_SOFTWARE
        assert MARKER not in result.stdout
        assert "refusal wrote to" in result.stderr, result.stderr

    def test_failing_to_scrub_the_builders_environment_is_caught(
        self, image: Path
    ) -> None:
        """Pin the mechanism, not just the outcome.

        `test_the_proof_ignores_entitlement_leaked_into_the_builder` asserts the
        result but nothing held `unentitled()` to producing it, so the `-u`
        scrubbing could be dropped and only that outcome would change — silently,
        on builders where nothing is set.
        """

        script = image / "usr" / "local" / "bin" / "ltx-runtime"
        text = script.read_text(encoding="utf-8")
        assert 'args+=(-u "$name")' in text
        script.write_text(text.replace('args+=(-u "$name")', "args+=()", 1), "utf-8")

        result = run(image, "assert-refusal", env=ENTITLED | {NVIDIA_ACCEPT_ENV: "YES"})

        assert result.returncode == EX_SOFTWARE
        assert MARKER not in result.stdout

    def test_an_unmutated_image_still_passes(self, image: Path) -> None:
        """The control: without it the cases above could pass on any error."""

        assert run(image, "assert-refusal").returncode == 0


class TestTheEntitlementGate:
    def test_an_unentitled_ensure_refuses_before_touching_anything(
        self, image: Path
    ) -> None:
        """The source needs the token too: Section 1.9 makes it licensed material."""

        result = run(image, "ensure")

        assert result.returncode == EX_CONFIG
        assert "HF_TOKEN" in result.stderr
        assert "Nothing has been downloaded." in result.stderr
        assert nothing_was_fetched(image)

    def test_an_unentitled_weight_fetch_refuses(self, image: Path) -> None:
        result = run(image, "fetch-weights")

        assert result.returncode == EX_CONFIG
        assert "GATED repository" in result.stderr
        assert nothing_was_fetched(image)

    def test_the_refusal_says_where_the_entitlement_comes_from(
        self, image: Path
    ) -> None:
        """It has to be actionable, or operators will look for a way around it."""

        stderr = run(image, "ensure").stderr

        assert WEIGHTS_REPO_URL in stderr
        assert "read gated repos" in stderr
        # The non-obvious half: the token gates the source as well.
        assert "Section 1.9" in stderr
        assert LICENSE_NAME in stderr


class TestTheNvidiaGateIsSeparate:
    def test_holding_a_hugging_face_entitlement_does_not_accept_nvidias_terms(
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
            env=ENTITLED | {"NPA_LTX_SOURCE_REPO": str(image.parent / "no-such-repo")},
        )

        assert result.returncode == EX_CONFIG
        assert NVIDIA_ACCEPT_ENV in result.stderr
        assert "docs.nvidia.com/cuda/eula" in result.stderr
        assert nothing_was_fetched(image)

    def test_the_entitlement_check_runs_first(self, image: Path) -> None:
        """Accepting NVIDIA's terms alone must not get anywhere near a download."""

        result = run(image, "ensure", env={NVIDIA_ACCEPT_ENV: "YES"})

        assert result.returncode == EX_CONFIG
        assert "HF_TOKEN" in result.stderr
        assert nothing_was_fetched(image)


class TestTheEntrypointDispatch:
    """`docker run <image> ltx-runtime <mode>` must reach <mode>.

    Explicit LTX modes go through ``ltx-runtime``. Other argv must remain
    available to the container orchestrator before task secrets are injected;
    that cannot expose LTX bytes because the image carries none. The runbook,
    golden eval, and live refusal proof invoke the explicit form.
    """

    @pytest.fixture
    def entrypoint(self, tmp_path: Path) -> Path:
        """The shipped entrypoint, with its one absolute path pointed at a stub.

        ``ltx_runtime.sh`` is invoked by absolute path (deliberately — a PATH
        lookup would be a way around the checks), so exercising the dispatch
        means redirecting that one reference rather than shimming PATH.
        """

        stub = tmp_path / "ltx-runtime-stub"
        stub.write_text(
            '#!/usr/bin/env bash\nprintf "MODE:%s\\n" "$*"\n', encoding="utf-8"
        )
        stub.chmod(0o755)

        source = (DOCKER_DIR / "entrypoint.sh").read_text(encoding="utf-8")
        assert "/usr/local/bin/ltx-runtime" in source
        script = tmp_path / "entrypoint.sh"
        script.write_text(
            source.replace("/usr/local/bin/ltx-runtime", str(stub)), encoding="utf-8"
        )
        script.chmod(0o755)
        return script

    def _dispatch(self, entrypoint: Path, *args: str) -> str:
        result = subprocess.run(
            ["bash", str(entrypoint), *args],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    @pytest.mark.parametrize(
        "mode",
        ["health", "version", "status", "terms", "assert-refusal"],
    )
    def test_the_redundant_ltx_runtime_prefix_is_dropped(
        self, entrypoint: Path, mode: str
    ) -> None:
        assert self._dispatch(entrypoint, "ltx-runtime", mode) == f"MODE:{mode}"

    @pytest.mark.parametrize("mode", ["health", "status", "ensure", "fetch-weights"])
    def test_a_bare_mode_still_dispatches(self, entrypoint: Path, mode: str) -> None:
        assert self._dispatch(entrypoint, mode) == f"MODE:{mode}"

    def test_an_infrastructure_bootstrap_command_runs_without_a_fetch(
        self, entrypoint: Path
    ) -> None:
        """SkyPilot bootstrap runs before its task-level secrets are present."""

        assert self._dispatch(entrypoint, "printf", "BOOTSTRAP_OK") == "BOOTSTRAP_OK"


def test_prebuilt_image_carries_only_runtime_fetch_metadata() -> None:
    dockerfile = (DOCKER_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert '"source":"operator-runtime-fetch"' in dockerfile
    assert "> /opt/byof/npa_source_metadata.json" in dockerfile


class TestTheRunbookCommandsReachTheModeTheyClaim:
    """The runbook is the evidence, so make that a checkable statement.

    `docs/workbench/ltx2.md` tells an operator to prove the refusal with
    `docker run <image> ltx-runtime ensure` and says it "exits 78, names the
    entitlement, downloads nothing". Before the entrypoint was fixed it did exit
    78 — from the argument parser, not the gate. A vacuous 78, in the runbook
    that exists to show the 78 is real.

    This lifts the `docker run` lines out of the document, strips everything up
    to the image reference, and dispatches the remaining argv through the same
    stubbed entrypoint, so the prose cannot drift away from the behaviour.
    """

    RUNBOOK = Path(__file__).resolve().parents[3] / "docs" / "workbench" / "ltx2.md"

    def _documented_argv(self) -> list[list[str]]:
        commands: list[list[str]] = []
        for line in self.RUNBOOK.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("docker run ") or "npa-ltx2" not in stripped:
                continue
            parts = stripped.split()
            image = next(
                index for index, part in enumerate(parts) if "npa-ltx2" in part
            )
            argv = parts[image + 1 :]
            if argv:
                commands.append(argv)
        return commands

    def test_the_runbook_actually_documents_some_commands(self) -> None:
        assert self._documented_argv(), (
            "no `docker run ... npa-ltx2 ...` lines found; either the runbook "
            "moved or this test stopped checking anything"
        )

    def test_every_documented_command_reaches_a_real_mode(self, tmp_path: Path) -> None:
        stub = tmp_path / "ltx-runtime-stub"
        stub.write_text(
            '#!/usr/bin/env bash\nprintf "MODE:%s\\n" "$1"\n', encoding="utf-8"
        )
        stub.chmod(0o755)
        source = (DOCKER_DIR / "entrypoint.sh").read_text(encoding="utf-8")
        script = tmp_path / "entrypoint.sh"
        script.write_text(
            source.replace("/usr/local/bin/ltx-runtime", str(stub)), encoding="utf-8"
        )

        known = {
            "ensure",
            "warm",
            "fetch-weights",
            "assert-refusal",
            "status",
            "terms",
            "health",
            "version",
        }
        for argv in self._documented_argv():
            result = subprocess.run(
                ["bash", str(script), *argv],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, f"{argv}: {result.stderr}"
            mode = result.stdout.strip().removeprefix("MODE:")
            assert mode in known, (
                f"the runbook's `docker run ... {' '.join(argv)}` dispatches to "
                f"{mode!r}, which is not a mode. The operator would get an "
                "argument-parser error and, for a refusal check, could not tell "
                "it apart from the refusal the document promises."
            )


class TestTheImageLayoutIsReachableByTheRuntimeUser:
    """Directories the Dockerfile creates must stay traversable.

    BuildKit applies ``COPY --chmod`` to the parent directories it creates along
    the way, so a read-only file mode silently becomes a read-only *directory* —
    no execute bit, and the non-root runtime user cannot traverse into it. The
    first real build failed exactly this way: ``/opt/npa/ltx2`` came out
    ``dr--r--r--`` and every file inside it read as Permission denied. Nothing in
    a Dockerfile review shows that.
    """

    #: Directories the base image already provides, whose modes a COPY does not
    #: get to invent. Anything else a COPY lands in is created by that COPY.
    PREEXISTING = frozenset(
        {
            "/bin",
            "/etc",
            "/opt",
            "/sbin",
            "/usr/bin",
            "/usr/local/bin",
            "/usr/share/doc",
        }
    )

    def test_every_directory_a_readonly_copy_creates_is_normalized(self) -> None:
        dockerfile = (DOCKER_DIR / "Dockerfile").read_text(encoding="utf-8")

        created: set[str] = set()
        for match in re.finditer(
            r"^COPY\s+--chmod=(\d+)\s+\S+\s+(\S+)", dockerfile, re.MULTILINE
        ):
            mode, destination = match.group(1), match.group(2)
            if int(mode, 8) & 0o111:
                continue
            parent = str(PurePosixPath(destination).parent)
            if parent not in self.PREEXISTING:
                created.add(parent)

        assert created, "no read-only COPY destinations found; has the layout changed?"
        normalized = {
            directory
            for match in re.finditer(r"chmod\s+0?755\s+([^\n\\]+)", dockerfile)
            for directory in match.group(1).split()
        }
        assert created <= normalized, (
            f"directories created by a read-only COPY are never made traversable: "
            f"{sorted(created - normalized)}"
        )


class TestTheBuildsOwnEmptinessCheck:
    """The Dockerfile's "the refusal wrote nothing" test, run against the layout.

    This predicate is inside a `RUN`, so it only ever executes during a real
    build — and it was wrong: it searched `/workspace` by depth and so tripped
    over the empty cache mount points `install -d` creates two lines earlier.
    The bug stayed invisible because an earlier link in the same `&&` chain was
    failing first. Extracting the real expression and running it against a
    replica of the real layout gets that back under test without Docker.
    """

    def _predicate(self) -> str:
        dockerfile = (DOCKER_DIR / "Dockerfile").read_text(encoding="utf-8")
        match = re.search(r'test -z "\$\(find /workspace[^"]*\)"', dockerfile)
        assert match, "the build no longer checks that the refusal wrote nothing"
        return match.group(0)

    def _workspace(self, tmp_path: Path) -> Path:
        """The directories the Dockerfile's `install -d` creates, and nothing else."""

        workspace = tmp_path / "workspace"
        (workspace / ".cache" / "npa" / "ltx2" / "runtime").mkdir(parents=True)
        (workspace / "model-cache" / "ltx-2.5").mkdir(parents=True)
        return workspace

    def _run(self, predicate: str, workspace: Path) -> int:
        return subprocess.run(
            ["bash", "-c", predicate.replace("/workspace", str(workspace))],
            capture_output=True,
            timeout=60,
        ).returncode

    def test_it_passes_on_the_layout_the_build_actually_creates(
        self, tmp_path: Path
    ) -> None:
        assert self._run(self._predicate(), self._workspace(tmp_path)) == 0

    @pytest.mark.parametrize(
        "written",
        [
            ".cache/npa/ltx2/runtime/src/ltx_core.py",
            "model-cache/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors",
            "generated.mp4",
        ],
    )
    def test_it_still_fails_when_the_refusal_writes_anything(
        self, tmp_path: Path, written: str
    ) -> None:
        """Otherwise the fix would have turned the check into a no-op."""

        workspace = self._workspace(tmp_path)
        leaked = workspace / written
        leaked.parent.mkdir(parents=True, exist_ok=True)
        leaked.write_bytes(b"payload")

        assert self._run(self._predicate(), workspace) != 0


class TestModesThatNeverNeedAnEntitlement:
    def test_terms_prints_the_agreement_facts_without_one(self, image: Path) -> None:
        """The operator has to be able to read the terms before holding a token."""

        result = run(image, "terms")

        assert result.returncode == 0, result.stderr
        for fact in (
            LICENSE_NAME,
            LICENSE_URL,
            ACCEPTABLE_USE_POLICY_URL,
            WEIGHTS_REPO_URL,
            COMMERCIAL_LICENSE_CONTACT,
        ):
            assert fact in result.stdout, fact
        # The two obligations nothing here can check for them.
        assert "Section 2.1" in result.stdout
        assert "Attachment A(18)" in result.stdout

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
            # "unknown" rather than a repo default: nothing has been fetched, so
            # naming a revision here would claim bytes the image does not have.
            "weights_revision": "unknown",
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
