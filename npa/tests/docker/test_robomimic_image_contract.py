from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[3]
IMAGE_ROOT = ROOT / "npa" / "docker" / "workbench" / "robomimic"
DOCKERFILE = IMAGE_ROOT / "Dockerfile"
BUILD_SCRIPT = IMAGE_ROOT / "build.sh"
VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_robomimic_image_contract", IMAGE_ROOT / "verify_image.py"
)
assert VERIFY_SPEC and VERIFY_SPEC.loader
VERIFIER = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(VERIFIER)


def test_neutral_image_is_pinned_non_root_and_contains_no_cuda_install() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert (
        "FROM --platform=linux/amd64 python:3.11.16-slim-bookworm@sha256:"
        "528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84"
    ) in text
    assert "USER ubuntu" in text
    assert 'ENTRYPOINT ["/opt/npa/robomimic/entrypoint.sh"]' in text
    assert "org.nebius.npa.skypilot-bootstrap-contract" in text
    normalized = re.sub(r"\\\n\s*", " ", text)
    assert re.search(r"\b(?:nvidia/cuda|pytorch/pytorch):", normalized, re.I) is None
    assert (
        re.search(r"\bpip\s+install\b[^\n]*(?:torch|nvidia-)", normalized, re.I) is None
    )
    assert "robomimic-runtime assert-refusal" in normalized
    assert "robomimic-runtime ensure" not in normalized
    assert "robomimic-runtime exec" not in normalized
    assert "apt-get" not in normalized
    assert "git -C /opt/robomimic" not in normalized
    assert "source=build-inputs" in normalized
    assert "dpkg --unpack /mnt/robomimic-build-inputs/debian/*.deb" in normalized
    assert "verify_image.py build-inputs" in normalized
    assert "verify_image.py debian-install" in normalized


def test_neutral_image_boundaries_and_locks_are_explicit() -> None:
    baked = (IMAGE_ROOT / "baked-requirements.lock").read_bytes()
    baked_names = {
        line.split(b"==", 1)[0].decode("ascii")
        for line in baked.splitlines()
        if re.match(rb"^[A-Za-z0-9_.-]+==", line)
    }
    assert baked_names == {
        "anyio",
        "boto3",
        "botocore",
        "certifi",
        "charset-normalizer",
        "contourpy",
        "cycler",
        "diffusers",
        "filelock",
        "fonttools",
        "fsspec",
        "h11",
        "h5py",
        "hf-xet",
        "httpcore",
        "httpx",
        "huggingface-hub",
        "idna",
        "imageio",
        "importlib-metadata",
        "jmespath",
        "kiwisolver",
        "matplotlib",
        "numpy",
        "packaging",
        "pillow",
        "psutil",
        "pyparsing",
        "python-dateutil",
        "pyyaml",
        "regex",
        "requests",
        "s3transfer",
        "safetensors",
        "six",
        "termcolor",
        "tqdm",
        "typing-extensions",
        "urllib3",
        "zipp",
    }
    assert set(
        VERIFIER._locked_baked_packages(IMAGE_ROOT / "baked-requirements.lock")
    ) == (baked_names)
    # Import reachability at the pinned source revision is broader than the BC
    # code path itself: algo registration imports Diffusion Policy, while the
    # observation stack imports vis_utils and therefore matplotlib eagerly.
    assert {
        "diffusers",
        "huggingface-hub",
        "imageio",
        "matplotlib",
    } <= baked_names
    # These upstream install requirements are lazy and are not exercised by the
    # headless, non-language, non-rendering gate.
    assert {
        "egl-probe",
        "imageio-ffmpeg",
        "tensorboard",
        "tensorboardx",
        "transformers",
    }.isdisjoint(baked_names)
    lowered = baked.lower()
    for forbidden in (b"torch==", b"torchvision==", b"triton==", b"nvidia-"):
        assert forbidden not in lowered
    runtime = json.loads((IMAGE_ROOT / "runtime-requirements.lock").read_text())
    assert runtime["schema"] == "npa.robomimic.runtime-lock.v1"
    assert runtime["packages"]["torch"] == "2.7.1+cu128"
    assert runtime["packages"]["nvidia-cudnn-cu12"] == "9.7.1.26"
    assert runtime["artifact_hash_closure"] == "required-in-runtime-inventory"
    entitlement = runtime["customer_entitlement"]
    assert entitlement["schema"] == "npa.robomimic.customer-runtime-entitlement.v1"
    assert "field_of_use" not in entitlement
    assert entitlement["maximum_validity_seconds"] == 86_400
    assert entitlement["responsibilities"][-1] == "no-redistribution-grant"
    assert [term["url"] for term in entitlement["terms"]] == [
        "https://docs.nvidia.com/cuda/eula/index.html",
        (
            "https://www.nvidia.com/en-us/agreements/enterprise-software/"
            "nvidia-software-license-agreement/"
        ),
        (
            "https://docs.nvidia.com/deeplearning/cudnn/backend/latest/"
            "reference/eula.html"
        ),
    ]
    for filename in (
        "debian-packages.lock",
        "REDISTRIBUTION.md",
        "source-manifest.json",
        "THIRD_PARTY_NOTICES.md",
        "runtime_bootstrap.sh",
        "verify_image.py",
        "smoke.py",
    ):
        assert (IMAGE_ROOT / filename).is_file()


def test_no_unbound_boolean_consent_proxy_exists() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in IMAGE_ROOT.iterdir()
        if path.is_file()
    )
    assert "NPA_ROBOMIMIC_ACCEPT" not in combined
    assert "CUDA_ACCEPT" not in combined
    assert "CUDNN_ACCEPT" not in combined


def test_build_helper_defaults_local_and_uses_only_committed_context() -> None:
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert 'registry="${NPA_BYOF_ROBOMIMIC_REGISTRY:-local.invalid}"' in text
    assert "NPA_PUBLIC_REGISTRY" not in text
    assert 'archive "${revision}:npa/docker/workbench/robomimic"' in text
    assert '"${context}/verify_image.py" prepare-build-inputs' in text
    assert 'docker build --platform linux/amd64 --pull=false --iidfile' in text
    assert 'docker image tag "${image_id}" "${image}"' in text
    assert 'docker image ls --quiet --no-trunc --filter "reference=${reference}"' in text
    assert 'docker image rm "${image}"' in text
    assert 'flock -x "${tag_lock_fd}"' in text
    assert "npa.robomimic.neutral-build-unresolved.v1" in text
    assert '"consumer_image_ref": image_id' in text
    assert '--tag "${image}"' not in text
    assert '"${repo_root}/npa/docker/workbench/robomimic"' not in text


def _fake_build_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    bin_dir = tmp_path / "bin"
    temp_root = tmp_path / "tmp"
    receipt_dir = tmp_path / "receipts"
    lock_dir = tmp_path / "locks"
    for path in (bin_dir, temp_root, receipt_dir, lock_dir):
        path.mkdir(mode=0o700)

    python_wrapper = bin_dir / "python3"
    python_wrapper.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == */verify_image.py && "${2:-}" == "prepare-build-inputs" ]]; then
  while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "--output-root" ]]; then
      mkdir -p -- "$2"
      exit 0
    fi
    shift
  done
  exit 91
fi
if [[ "${FAKE_RECEIPT_FAILURE:-0}" == "1" && "${1:-}" == "-" ]]; then
  exit 97
fi
exec """
        + shlex.quote(sys.executable)
        + ' "$@"\n',
        encoding="utf-8",
    )
    python_wrapper.chmod(0o700)

    docker = bin_dir / "docker"
    docker.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
action="$1"
shift
case "${action}" in
  build)
    iidfile=""
    context=""
    while [[ "$#" -gt 0 ]]; do
      if [[ "$1" == "--iidfile" ]]; then
        iidfile="$2"
        shift 2
      else
        context="$1"
        shift
      fi
    done
    printf '%s' "${iidfile}" > "${FAKE_DOCKER_IID_RECORD}"
    printf '%s' "${context}" > "${FAKE_DOCKER_CONTEXT_RECORD}"
    if [[ "${FAKE_DOCKER_MODE:-}" == "signal" ]]; then
      kill -TERM "$PPID"
      exit 143
    fi
    if [[ "${FAKE_DOCKER_MODE:-}" == "missing-iid" ]]; then
      exit 0
    elif [[ "${FAKE_DOCKER_MODE:-}" == "malformed-iid" ]]; then
      printf 'not-an-image-id\n' > "${iidfile}"
    else
      printf '%s\n' "${FAKE_DOCKER_IMAGE_ID}" > "${iidfile}"
    fi
    chmod 600 -- "${iidfile}"
    ;;
  image)
    subaction="$1"
    shift
    case "${subaction}" in
      inspect)
        reference="${!#}"
        if [[ "${reference}" == sha256:* ]]; then
          if [[ -f "${FAKE_DOCKER_TAG_STATE}" \
            && "${FAKE_DOCKER_MODE:-}" == "post-assignment-built-inspect-failure" ]]; then
            exit 74
          elif [[ -f "${FAKE_DOCKER_TAG_STATE}" \
            && "${FAKE_DOCKER_MODE:-}" == "replace-before-rollback" ]]; then
            printf 'sha256:%064d\n' 7 > "${FAKE_DOCKER_TAG_STATE}"
            exit 74
          fi
          if [[ "${FAKE_DOCKER_MODE:-}" == "malformed-inspect" ]]; then
            printf 'malformed\n'
          elif [[ "${FAKE_DOCKER_MODE:-}" == "changed-inspect" ]]; then
            printf 'sha256:%064d\n' 9
          else
            printf '%s\n' "${reference}"
          fi
          if [[ "${FAKE_DOCKER_MODE:-}" == "changed-iid" ]]; then
            printf 'sha256:%064d\n' 8 > "$(cat "${FAKE_DOCKER_IID_RECORD}")"
          fi
          exit 0
        fi
        if [[ "${FAKE_DOCKER_MODE:-}" == "tag-inspect-error" ]]; then
          exit 74
        fi
        [[ -f "${FAKE_DOCKER_TAG_STATE}" ]] || exit 1
        if [[ "${FAKE_DOCKER_MODE:-}" == "changed-tag-inspect" ]]; then
          printf 'sha256:%064d\n' 7
        else
          cat "${FAKE_DOCKER_TAG_STATE}"
        fi
        ;;
      ls)
        [[ "${FAKE_DOCKER_MODE:-}" != "tag-list-failure" ]] || exit 75
        if [[ -f "${FAKE_DOCKER_TAG_STATE}" ]]; then
          cat "${FAKE_DOCKER_TAG_STATE}"
        fi
        ;;
      tag)
        source_id="$1"
        if [[ "${FAKE_DOCKER_MODE:-}" == "signal-tag" ]]; then
          kill -TERM "$PPID"
          exit 143
        fi
        printf 'tag\n' >> "${FAKE_DOCKER_ACTION_RECORD}"
        printf '%s\n' "${source_id}" > "${FAKE_DOCKER_TAG_STATE}"
        ;;
      rm)
        printf 'rm\n' >> "${FAKE_DOCKER_ACTION_RECORD}"
        [[ "${FAKE_DOCKER_MODE:-}" != "rollback-remove-failure" ]] || exit 76
        /usr/bin/rm -f -- "${FAKE_DOCKER_TAG_STATE}"
        ;;
      *) exit 92 ;;
    esac
    ;;
  *) exit 93 ;;
esac
""",
        encoding="utf-8",
    )
    docker.chmod(0o700)

    ln_wrapper = bin_dir / "ln"
    ln_wrapper.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
target="${!#}"
if [[ "${FAKE_RECEIPT_LINK_FAILURE:-0}" == "1" \
  && "${target}" != *.unresolved.json ]]; then
  exit 77
fi
exec /usr/bin/ln "$@"
""",
        encoding="utf-8",
    )
    ln_wrapper.chmod(0o700)

    rm_wrapper = bin_dir / "rm"
    rm_wrapper.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${FAKE_RM_RECEIPT_TEMP_FAILURE:-0}" == "1" \
  && ! -e "${FAKE_RM_FAILURE_MARKER}" ]]; then
  for candidate in "$@"; do
    if [[ "${candidate}" == *.receipt.* ]]; then
      : > "${FAKE_RM_FAILURE_MARKER}"
      exit 78
    fi
  done
fi
exec /usr/bin/rm "$@"
""",
        encoding="utf-8",
    )
    rm_wrapper.chmod(0o700)

    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TMPDIR": str(temp_root),
        "NPA_BYOF_ROBOMIMIC_RECEIPT_DIR": str(receipt_dir),
        "NPA_BYOF_ROBOMIMIC_LOCK_DIR": str(lock_dir),
        "FAKE_DOCKER_TAG_STATE": str(tmp_path / "tag-state"),
        "FAKE_DOCKER_IID_RECORD": str(tmp_path / "iid-path"),
        "FAKE_DOCKER_CONTEXT_RECORD": str(tmp_path / "context-path"),
        "FAKE_DOCKER_ACTION_RECORD": str(tmp_path / "docker-actions"),
        "FAKE_RM_FAILURE_MARKER": str(tmp_path / "rm-failure-marker"),
    }
    return environment, temp_root, receipt_dir, tmp_path / "tag-state"


def _run_fake_build(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(BUILD_SCRIPT)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_build_helper_records_immutable_id_in_owner_only_receipt(tmp_path: Path) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "1" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id

    result = _run_fake_build(environment)

    receipts = list(receipt_dir.glob("*.json"))
    assert result.returncode == 0, result.stderr
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["immutable_image_id"] == image_id
    assert receipt["consumer_image_ref"] == image_id
    assert receipt["revision"] == subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert receipt["tag_compare_and_set"] == "assigned"
    assert stat.S_IMODE(receipts[0].stat().st_mode) == 0o600
    assert tag_state.read_text(encoding="utf-8").strip() == image_id
    assert list(temp_root.glob("npa-robomimic-context.*")) == []


def test_build_helper_refuses_stale_shared_tag_without_retagging(tmp_path: Path) -> None:
    environment, _, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    stale_id = "sha256:" + "2" * 64
    tag_state.write_text(stale_id + "\n", encoding="utf-8")
    environment["FAKE_DOCKER_IMAGE_ID"] = "sha256:" + "3" * 64

    result = _run_fake_build(environment)

    assert result.returncode != 0
    assert "already names different bytes" in result.stderr
    assert tag_state.read_text(encoding="utf-8").strip() == stale_id
    assert list(receipt_dir.glob("*.json")) == []


def test_build_helper_refuses_ambiguous_inspection_without_retagging(
    tmp_path: Path,
) -> None:
    environment, _, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    stale_id = "sha256:" + "2" * 64
    tag_state.write_text(stale_id + "\n", encoding="utf-8")
    environment["FAKE_DOCKER_IMAGE_ID"] = "sha256:" + "3" * 64
    environment["FAKE_DOCKER_MODE"] = "tag-inspect-error"

    result = _run_fake_build(environment)

    assert result.returncode != 0
    assert "tag absence could not be proven" in result.stderr
    assert tag_state.read_text(encoding="utf-8").strip() == stale_id
    assert not (tmp_path / "docker-actions").exists()
    assert list(receipt_dir.glob("*.json")) == []


def test_build_helper_accepts_identical_shared_tag_idempotently(tmp_path: Path) -> None:
    environment, _, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "3" * 64
    tag_state.write_text(image_id + "\n", encoding="utf-8")
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id

    result = _run_fake_build(environment)

    assert result.returncode == 0, result.stderr
    receipt = json.loads(next(receipt_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert receipt["tag_compare_and_set"] == "existing-identical"
    assert tag_state.read_text(encoding="utf-8").strip() == image_id


def test_build_helper_serializes_concurrent_differing_ids(tmp_path: Path) -> None:
    environment, _, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    processes = []
    for index, digit in enumerate(("4", "5")):
        candidate = dict(environment)
        candidate["FAKE_DOCKER_IMAGE_ID"] = "sha256:" + digit * 64
        candidate["FAKE_DOCKER_IID_RECORD"] = str(tmp_path / f"iid-path-{index}")
        candidate["FAKE_DOCKER_CONTEXT_RECORD"] = str(tmp_path / f"context-{index}")
        processes.append(
            subprocess.Popen(
                ["bash", str(BUILD_SCRIPT)],
                cwd=ROOT,
                env=candidate,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    outcomes = [process.communicate(timeout=30) for process in processes]
    codes = [process.returncode for process in processes]
    assert sorted(codes) == [0, 1]
    winner = tag_state.read_text(encoding="utf-8").strip()
    assert winner in {"sha256:" + "4" * 64, "sha256:" + "5" * 64}
    assert len(list(receipt_dir.glob("*.json"))) == 1
    assert any("already names different bytes" in stderr for _, stderr in outcomes)


@pytest.mark.parametrize(
    "mode",
    (
        "missing-iid",
        "malformed-iid",
        "malformed-inspect",
        "changed-inspect",
        "changed-iid",
        "changed-tag-inspect",
    ),
)
def test_build_helper_refuses_malformed_or_changed_identity(
    tmp_path: Path, mode: str
) -> None:
    environment, _, receipt_dir, _ = _fake_build_environment(tmp_path)
    environment["FAKE_DOCKER_IMAGE_ID"] = "sha256:" + "6" * 64
    environment["FAKE_DOCKER_MODE"] = mode

    result = _run_fake_build(environment)

    assert result.returncode != 0
    assert list(receipt_dir.glob("*.json")) == []


def test_build_helper_cleans_transaction_context_on_signal(tmp_path: Path) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    environment["FAKE_DOCKER_IMAGE_ID"] = "sha256:" + "6" * 64
    environment["FAKE_DOCKER_MODE"] = "signal-tag"

    result = _run_fake_build(environment)

    assert result.returncode != 0
    assert list(temp_root.glob("npa-robomimic-context.*")) == []
    assert list(receipt_dir.glob("*.json")) == []
    assert not tag_state.exists()


def test_build_helper_receipt_failure_is_fail_closed_and_cleans_temp(
    tmp_path: Path,
) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "a" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id
    environment["FAKE_RECEIPT_FAILURE"] = "1"

    result = _run_fake_build(environment)

    assert result.returncode != 0
    assert not tag_state.exists()
    assert list(receipt_dir.iterdir()) == []
    assert list(temp_root.glob("npa-robomimic-context.*")) == []


def test_build_helper_post_assignment_failure_rolls_back_owned_tag(
    tmp_path: Path,
) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    environment["FAKE_DOCKER_IMAGE_ID"] = "sha256:" + "b" * 64
    environment["FAKE_DOCKER_MODE"] = "post-assignment-built-inspect-failure"

    result = _run_fake_build(environment)

    assert result.returncode != 0
    assert not tag_state.exists()
    assert (tmp_path / "docker-actions").read_text(encoding="utf-8").splitlines() == [
        "tag",
        "rm",
    ]
    assert list(receipt_dir.iterdir()) == []
    assert list(temp_root.glob("npa-robomimic-context.*")) == []


def test_build_helper_does_not_remove_replaced_tag_during_rollback(
    tmp_path: Path,
) -> None:
    environment, _, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    environment["FAKE_DOCKER_IMAGE_ID"] = "sha256:" + "c" * 64
    environment["FAKE_DOCKER_MODE"] = "replace-before-rollback"

    result = _run_fake_build(environment)

    assert result.returncode != 0
    assert tag_state.read_text(encoding="utf-8").strip() == "sha256:" + f"{7:064d}"
    assert (tmp_path / "docker-actions").read_text(encoding="utf-8").splitlines() == [
        "tag"
    ]
    assert "rollback skipped: shared tag identity changed" in result.stderr
    assert list(receipt_dir.iterdir()) == []


def test_build_helper_records_bounded_unresolved_rollback_failure(
    tmp_path: Path,
) -> None:
    environment, _, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "d" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id
    environment["FAKE_DOCKER_MODE"] = "rollback-remove-failure"
    environment["FAKE_RECEIPT_FAILURE"] = "1"

    result = _run_fake_build(environment)

    unresolved = list(receipt_dir.glob("*.unresolved.json"))
    assert result.returncode == 97
    assert tag_state.read_text(encoding="utf-8").strip() == image_id
    assert len(unresolved) == 1
    record = json.loads(unresolved[0].read_text(encoding="utf-8"))
    assert record["schema"] == "npa.robomimic.neutral-build-unresolved.v1"
    assert record["reason"] == "rollback-remove-failed"
    assert record["immutable_image_id"] == image_id
    assert record["intended_full_sha_tag"].endswith(
        ":dev-" + subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    )
    assert unresolved[0].stat().st_size < 1024
    assert stat.S_IMODE(unresolved[0].stat().st_mode) == 0o600
    assert "cleanup was incomplete" in result.stderr


def test_build_helper_receipt_temp_cleanup_failure_still_cleans_context(
    tmp_path: Path,
) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    environment["FAKE_DOCKER_IMAGE_ID"] = "sha256:" + "e" * 64
    environment["FAKE_RM_RECEIPT_TEMP_FAILURE"] = "1"

    result = _run_fake_build(environment)

    assert result.returncode == 78
    assert (tmp_path / "rm-failure-marker").is_file()
    assert not tag_state.exists()
    assert list(receipt_dir.iterdir()) == []
    assert list(temp_root.glob("npa-robomimic-context.*")) == []


def test_build_helper_receipt_link_failure_rolls_back_owned_tag(
    tmp_path: Path,
) -> None:
    environment, _, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    environment["FAKE_DOCKER_IMAGE_ID"] = "sha256:" + "f" * 64
    environment["FAKE_RECEIPT_LINK_FAILURE"] = "1"

    result = _run_fake_build(environment)

    assert result.returncode != 0
    assert not tag_state.exists()
    assert list(receipt_dir.iterdir()) == []


def test_dataset_notice_binds_exact_official_license_metadata() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "docs" / "workbench" / "byof-robomimic.md",
            IMAGE_ROOT / "THIRD_PARTY_NOTICES.md",
        )
    )
    for expected in (
        "74fa018461f479cd9fd15b924a16103012096203",
        "736f9c17ae642026c84d2b534119cc4dfea1548a",
        "e09a24720408bac08425dbaa0b7b55615e4440f1af0a61133303e4b5d5d6b09a",
        "2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540",
        "21,084,088",
        (
            "https://huggingface.co/datasets/robomimic/robomimic_datasets/raw/"
            "74fa018461f479cd9fd15b924a16103012096203/README.md"
        ),
        "https://huggingface.co/terms-of-service",
        "42020fcaac52b7b036bf7e816910ca45485ad04c2d31faf09e13636a5b48a36b",
    ):
        assert expected in text
    assert (
        "Public, anonymous reachability is access evidence, not a grant of rights"
        in " ".join(text.split())
    )


def test_immutable_build_locks_bind_exact_source_and_debian_bytes() -> None:
    debian = VERIFIER._debian_lock(IMAGE_ROOT / "debian-packages.lock")
    source = VERIFIER._source_manifest(IMAGE_ROOT / "source-manifest.json")
    assert len(debian["packages"]) == 78
    assert len(debian["sources"]) == 58
    assert sum(item["size"] for item in debian["packages"]) == 29_807_672
    assert debian["roots"] == [
        "ca-certificates",
        "openssh-server",
        "procps",
        "rsync",
        "sudo",
    ]
    assert all(item["debian_license_labels"] for item in debian["sources"])
    assert source["git_tree_sha1"] == "4c8ebe35dbef16126dadf59cf8b771b9203753ab"
    assert source["archive"] == {
        "format": "git-archive-tar",
        "path": "source/robomimic.tar",
        "size": 57_907_200,
        "sha256": "8dd695200bba3ca6043693a7db4b15713a740d5d6a91787984d4c0053f77fd8b",
        "member_count": 226,
        "regular_file_count": 201,
    }
    assert VERIFIER.verified_source_identity(IMAGE_ROOT / "source-manifest.json") == {
        "repository": "ARISE-Initiative/robomimic",
        "revision": "d309eaecc18acf4152a830a895a6984b8ac71b05",
        "observed_head": "d309eaecc18acf4152a830a895a6984b8ac71b05",
        "git_tree_sha1": "4c8ebe35dbef16126dadf59cf8b771b9203753ab",
        "tree_archive_sha256": (
            "8dd695200bba3ca6043693a7db4b15713a740d5d6a91787984d4c0053f77fd8b"
        ),
    }


def test_debian_install_verifier_accepts_exact_status_and_notice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package = {
        "name": "example-package",
        "version": "1:2.3-4",
        "notice_path": "/usr/share/doc/example-package/copyright",
    }
    notice = tmp_path / package["notice_path"].removeprefix("/")
    notice.parent.mkdir(parents=True)
    notice.write_text("reviewed notice\n", encoding="utf-8")
    monkeypatch.setattr(VERIFIER, "_debian_lock", lambda _: {"packages": [package]})
    monkeypatch.setattr(
        VERIFIER.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, f"ii |{package['version']}", ""
        ),
    )

    proof = VERIFIER.verify_debian_install(
        debian_lock_path=tmp_path / "lock", notice_root=tmp_path
    )

    assert proof["package_count"] == 1
    assert proof["notices_present"] == 1


def test_debian_install_verifier_rejects_old_literal_tab_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package = {
        "name": "example-package",
        "version": "1:2.3-4",
        "notice_path": "/usr/share/doc/example-package/copyright",
    }
    monkeypatch.setattr(VERIFIER, "_debian_lock", lambda _: {"packages": [package]})
    monkeypatch.setattr(
        VERIFIER.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, f"ii \\t{package['version']}", ""
        ),
    )

    with pytest.raises(VERIFIER.VerificationError, match="package mismatch"):
        VERIFIER.verify_debian_install(
            debian_lock_path=tmp_path / "lock", notice_root=tmp_path
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda lock: lock["packages"][0].update(url="http://snapshot.debian.org/a.deb"),
        lambda lock: lock["packages"][0]["depends"].append("missing-package"),
        lambda lock: lock["sources"][0].update(
            license_reference_url="https://example.invalid/copyright"
        ),
        lambda lock: lock["packages"][0].update(name="nvidia-cuda"),
        lambda lock: lock["packages"].pop(),
    ],
)
def test_debian_lock_parser_rejects_hostile_mutations(mutation) -> None:
    lock = json.loads((IMAGE_ROOT / "debian-packages.lock").read_text())
    mutation(lock)
    with pytest.raises(VERIFIER.VerificationError):
        VERIFIER._checked_debian_lock(lock)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision", "0" * 40),
        ("git_tree_sha1", "f" * 40),
        ("repository", "https://example.invalid/robomimic.git"),
    ],
)
def test_source_manifest_parser_rejects_hostile_mutations(field, value) -> None:
    manifest = json.loads((IMAGE_ROOT / "source-manifest.json").read_text())
    manifest[field] = value
    with pytest.raises(VERIFIER.VerificationError):
        VERIFIER._checked_source_manifest(manifest)


def _source_archive(tmp_path: Path, members: dict[str, bytes]) -> tuple[Path, dict]:
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        for name, raw in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    raw = archive_bytes.getvalue()
    path = tmp_path / "source.tar"
    path.write_bytes(raw)
    manifest = {
        "archive": {
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "member_count": len(members),
            "regular_file_count": len(members),
        },
        "license": {
            "path": "LICENSE",
            "sha256": hashlib.sha256(members["LICENSE"]).hexdigest(),
        },
    }
    return path, manifest


def test_source_archive_parser_rejects_path_traversal(tmp_path: Path) -> None:
    path, manifest = _source_archive(
        tmp_path, {"LICENSE": b"MIT\n", "../outside": b"hostile\n"}
    )
    with pytest.raises(VERIFIER.VerificationError, match="unsafe object"):
        VERIFIER._verify_source_archive(path, manifest)


def test_source_archive_parser_rejects_byte_mutation(tmp_path: Path) -> None:
    path, manifest = _source_archive(tmp_path, {"LICENSE": b"MIT\n"})
    manifest = copy.deepcopy(manifest)
    manifest["archive"]["sha256"] = "0" * 64
    with pytest.raises(VERIFIER.VerificationError, match="identity mismatch"):
        VERIFIER._verify_source_archive(path, manifest)
