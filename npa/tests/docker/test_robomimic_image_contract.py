from __future__ import annotations

import ast
import copy
import base64
import csv
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
import zipfile
from types import ModuleType, SimpleNamespace

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
    assert "PYTHONPATH=/opt/robomimic" in normalized
    assert "/opt/robomimic-deps" not in normalized
    assert "baked-requirements.lock" not in normalized
    assert "installer download" not in normalized
    assert "installer install" not in normalized
    assert "runtime-requirements.lock" in normalized
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
    assert len(runtime["packages"]) == 62
    assert set(baked_names) <= set(runtime["packages"])
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
    assert text.startswith(
        "#!/usr/bin/env bash\n"
        "# Build the neutral robomimic image with daemon-serialized tags and "
        "durable receipts.\n"
    )
    assert 'registry="${NPA_BYOF_ROBOMIMIC_REGISTRY:-local.invalid}"' in text
    assert "NPA_PUBLIC_REGISTRY" not in text
    assert "NPA_BYOF_ROBOMIMIC_LOCK_DIR" not in text
    assert 'archive "${revision}:npa/docker/workbench/robomimic"' in text
    assert '"${context_anchor}/verify_image.py" prepare-build-inputs' in text
    assert "docker build --platform linux/amd64 --pull=false --iidfile" in text
    assert 'docker image tag "${image_id}" "${image}"' in text
    assert (
        'docker image ls --quiet --no-trunc --filter "reference=${reference}"' in text
    )
    assert "docker image rm" not in text
    assert "docker container create --name" in text
    assert "docker container rm --volumes" in text
    assert "/run/user/$(id -u)" not in text
    assert 'local target_name="$1" inherited_directory_fd=9' in text
    assert '9<&"${receipt_dir_fd}"' in text
    assert "npa.robomimic.neutral-build-failure.v1" in text
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

    (bin_dir / "sitecustomize.py").write_text(
        """import os
import stat

_real_fsync = os.fsync


def _recorded_fsync(descriptor: int) -> None:
    value = os.fstat(descriptor)
    kind = "regular" if stat.S_ISREG(value.st_mode) else "directory"
    record_path = os.environ.get("FAKE_FSYNC_RECORD")
    failure_kind = os.environ.get("FAKE_FSYNC_FAILURE_KIND")
    failure_ordinal = int(os.environ.get("FAKE_FSYNC_FAILURE_ORDINAL", "1"))
    marker = os.environ.get("FAKE_FSYNC_FAILURE_MARKER")
    kind_count = 1
    if record_path and os.path.exists(record_path):
        with open(record_path, encoding="utf-8") as record:
            kind_count += sum(line.startswith(f"{kind}:") for line in record)
    should_fail = bool(
        failure_kind == kind
        and kind_count == failure_ordinal
        and marker
        and not os.path.exists(marker)
    )
    if record_path:
        with open(record_path, "a", encoding="utf-8") as record:
            record.write(f"{kind}:{'failed' if should_fail else 'ok'}\\n")
    if should_fail:
        marker_fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(marker_fd)
        raise OSError(f"injected {kind} fsync failure")
    _real_fsync(descriptor)


os.fsync = _recorded_fsync
""",
        encoding="utf-8",
    )

    python_wrapper = bin_dir / "python3"
    python_wrapper.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == */verify_image.py && "${2:-}" == "prepare-build-inputs" ]]; then
  context_path="$(readlink -f -- "${1%/verify_image.py}")"
  if [[ "${FAKE_PREBUILD_REPLACEMENT:-}" == "scratch-parent" ]]; then
    /usr/bin/mv -- "${TMPDIR}" "${TMPDIR}.retained"
    /usr/bin/mkdir -m 700 -- "${TMPDIR}"
    printf 'replacement\n' > "${TMPDIR}/replacement-marker"
  elif [[ "${FAKE_PREBUILD_REPLACEMENT:-}" == "context" ]]; then
    /usr/bin/mv -- "${context_path}" "${context_path}.retained"
    /usr/bin/mkdir -m 700 -- "${context_path}"
    printf 'replacement\n' > "${context_path}/replacement-marker"
  fi
  while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "--output-root" ]]; then
      mkdir -p -- "$2"
      exit 0
    fi
    shift
  done
  exit 91
fi
if [[ "${1:-}" == "-" && "${2:-}" == "receipt-link" ]]; then
  [[ "${3}" == "9" ]] || exit 78
  if [[ "${FAKE_RECEIPT_LINK_FAILURE:-0}" == "1" \
    && "${5}" != *.tag-terminal.json \
    && ! -e "${FAKE_RECEIPT_LINK_FAILURE_MARKER}" ]]; then
    : > "${FAKE_RECEIPT_LINK_FAILURE_MARKER}"
    exit 77
  fi
  if [[ "${FAKE_TERMINAL_RECEIPT_LINK_FAILURE:-0}" == "1" \
    && "${5}" == *.cleanup.json ]]; then
    exit 77
  fi
  status=0
        """
        + shlex.quote(sys.executable)
        + """ "$@" 9<&9 || status=$?
  [[ "${status}" -eq 0 ]] || exit "${status}"
  if [[ "${5}" == npa-robomimic-context.*.json \
    && "${5}" != *.cleanup*.json && "${5}" != *.failure*.json \
    && "${5}" != *.tag-terminal.json ]]; then
    if [[ -n "${FAKE_POST_PUBLICATION_SIGNAL:-}" ]]; then
      kill -"${FAKE_POST_PUBLICATION_SIGNAL}" "$PPID"
    fi
    [[ "${FAKE_POST_PUBLICATION_EXIT:-0}" == "0" ]] || exit 97
  fi
  if [[ "${FAKE_TERMINAL_RECEIPT_IDENTITY_FAILURE:-0}" == "1" \
    && "${5}" == *.cleanup.json ]]; then
        """
        + shlex.quote(sys.executable)
        + """ - "${3}" "${5}" 9<&9 <<'PY'
import os
import sys

directory_fd = int(sys.argv[1])
target_name = sys.argv[2]
os.rename(
    target_name,
    f"{target_name}.retained",
    src_dir_fd=directory_fd,
    dst_dir_fd=directory_fd,
)
replacement_fd = os.open(
    target_name,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
    0o600,
    dir_fd=directory_fd,
)
with os.fdopen(replacement_fd, "wb") as replacement:
    replacement.write(b'{"replacement":true}\\n')
PY
  fi
  if [[ -n "${FAKE_LIFECYCLE_RECORD:-}" ]]; then
    printf 'published:%s\n' "${5}" >> "${FAKE_LIFECYCLE_RECORD}"
  fi
  if [[ "${FAKE_REPLACE_FINAL_RECEIPT_TARGET:-0}" == "1" \
    && "${5}" == *.json \
    && "${5}" != *.failure.json \
    && "${5}" != *.failure-recovery.json \
    && "${5}" != *.tag-terminal.json \
    && ! -e "${FAKE_FINAL_TARGET_REPLACEMENT_MARKER}" ]]; then
        """
        + shlex.quote(sys.executable)
        + """ - "${3}" "${5}" "${FAKE_FINAL_TARGET_REPLACEMENT_MARKER}" 9<&9 <<'PY'
import os
import sys

directory_fd = int(sys.argv[1])
target_name = sys.argv[2]
marker = sys.argv[3]
os.rename(
    target_name,
    f"{target_name}.retained",
    src_dir_fd=directory_fd,
    dst_dir_fd=directory_fd,
)
replacement_fd = os.open(
    target_name,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
    0o600,
    dir_fd=directory_fd,
)
with os.fdopen(replacement_fd, "wb") as replacement:
    replacement.write(b'{"replacement":true}\\n')
marker_fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
os.close(marker_fd)
PY
  fi
  exit 0
fi
if [[ "${FAKE_TERMINAL_RECEIPT_FAILURE:-0}" == "1" \
  && "${1:-}" == "-" && "${#}" -eq 7 \
  && "${7}" == *.cleanup-journal.json ]]; then
  exit 97
fi
if [[ "${FAKE_RECEIPT_FAILURE:-0}" == "1" \
  && "${1:-}" == "-" && "${2:-}" =~ ^[0-9a-f]{40}$ ]]; then
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
replace_receipt_directory() {
  [[ ! -e "${FAKE_DIRECTORY_REPLACEMENT_MARKER}" ]] || return 0
  /usr/bin/mv -- "${NPA_BYOF_ROBOMIMIC_RECEIPT_DIR}" \
    "${NPA_BYOF_ROBOMIMIC_RECEIPT_DIR}.retained"
  /usr/bin/mkdir -m 700 -- "${NPA_BYOF_ROBOMIMIC_RECEIPT_DIR}"
  printf 'replacement\n' \
    > "${NPA_BYOF_ROBOMIMIC_RECEIPT_DIR}/replacement-marker"
  : > "${FAKE_DIRECTORY_REPLACEMENT_MARKER}"
}
action="$1"
shift
case "${action}" in
  info)
    printf '%s\\n' 'synthetic-daemon-identity'
    ;;
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
    if [[ "${FAKE_DOCKER_MODE:-}" == "signal-after-build-id" ]]; then
      kill -TERM "$PPID"
      exit 143
    fi
    ;;
  image)
    subaction="$1"
    shift
    case "${subaction}" in
      inspect)
        reference="${!#}"
        if [[ "${reference}" == sha256:* ]]; then
          if [[ "${FAKE_DOCKER_MODE:-}" == "replace-receipt-directory" ]]; then
            replace_receipt_directory
          fi
          if [[ -f "${FAKE_DOCKER_TAG_STATE}" \
            && "${FAKE_DOCKER_MODE:-}" == "post-assignment-built-inspect-failure" ]]; then
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
        if [[ "${FAKE_DOCKER_MODE:-}" == "replace-after-tag" \
          && -f "${FAKE_DOCKER_ACTION_RECORD}" ]]; then
          printf 'sha256:%064d\n' 7 > "${FAKE_DOCKER_TAG_STATE}"
        elif [[ "${FAKE_DOCKER_MODE:-}" == "remove-after-tag" \
          && -f "${FAKE_DOCKER_ACTION_RECORD}" ]]; then
          /usr/bin/rm -f -- "${FAKE_DOCKER_TAG_STATE}"
          exit 1
        elif [[ "${FAKE_DOCKER_MODE:-}" == "changed-tag-inspect" ]]; then
          printf 'sha256:%064d\n' 7
        elif [[ "${FAKE_DOCKER_MODE:-}" == "malformed-tag-inspect" ]]; then
          printf 'malformed\n'
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
  container)
    subaction="$1"
    shift
    case "${subaction}" in
      create)
        lock_name=""
        transaction=""
        while [[ "$#" -gt 1 ]]; do
          case "$1" in
            --name) lock_name="$2"; shift 2 ;;
            --label)
              transaction="${2#org.nebius.npa.robomimic.lock.transaction=}"
              shift 2
              ;;
            --entrypoint) shift 2 ;;
            *) exit 94 ;;
          esac
        done
        image_id="$1"
        caller="${FAKE_CALLER_UID:-same-uid}"
        lock_refused=0
        [[ "${FAKE_DOCKER_MODE:-}" != "lock-unavailable" ]] || lock_refused=1
        if [[ "${lock_refused}" -eq 0 ]]; then
          mkdir -- "${FAKE_DOCKER_LOCK_STATE}" 2>/dev/null || lock_refused=1
        fi
        if [[ "${lock_refused}" -eq 1 ]]; then
          printf '%s:%s:create-refused\n' \
            "${caller}" "${lock_name}" >> "${FAKE_DOCKER_LOCK_RECORD}"
          exit 80
        fi
        lock_id="$(printf '%s' "${transaction}|${image_id}" | sha256sum | cut -d ' ' -f 1)"
        mkdir -- "${FAKE_DOCKER_VOLUME_STATE}"
        printf '%s\n' "${lock_id}" > "${FAKE_DOCKER_LOCK_STATE}/id"
        printf '%s\n' "${lock_name}" > "${FAKE_DOCKER_LOCK_STATE}/name"
        printf '%s\n' "${transaction}" > "${FAKE_DOCKER_LOCK_STATE}/transaction"
        printf '%s\n' "${image_id}" > "${FAKE_DOCKER_LOCK_STATE}/image"
        printf '%s:%s:create-acquired\n' \
          "${caller}" "${lock_name}" >> "${FAKE_DOCKER_LOCK_RECORD}"
        printf '%s\n' "${lock_id}"
        ;;
      inspect)
        reference="${!#}"
        [[ -d "${FAKE_DOCKER_LOCK_STATE}" ]] || exit 1
        lock_id="$(cat "${FAKE_DOCKER_LOCK_STATE}/id")"
        lock_name="$(cat "${FAKE_DOCKER_LOCK_STATE}/name")"
        [[ "${reference}" == "${lock_id}" || "${reference}" == "${lock_name}" ]] \
          || exit 1
        printf '%s|%s|%s\n' "${lock_id}" \
          "$(cat "${FAKE_DOCKER_LOCK_STATE}/transaction")" \
          "$(cat "${FAKE_DOCKER_LOCK_STATE}/image")"
        ;;
      rm)
        remove_volumes=0
        if [[ "$1" == "--volumes" ]]; then
          remove_volumes=1
          shift
        fi
        reference="$1"
        [[ -d "${FAKE_DOCKER_LOCK_STATE}" ]] || exit 1
        [[ "${reference}" == "$(cat "${FAKE_DOCKER_LOCK_STATE}/id")" ]] || exit 1
        if [[ "${remove_volumes}" -eq 1 ]]; then
          /usr/bin/rm -rf -- "${FAKE_DOCKER_VOLUME_STATE}"
        fi
        /usr/bin/rm -rf -- "${FAKE_DOCKER_LOCK_STATE}"
        printf '%s:released\n' "${FAKE_CALLER_UID:-same-uid}" \
          >> "${FAKE_DOCKER_LOCK_RECORD}"
        ;;
      *) exit 95 ;;
    esac
    ;;
  *) exit 93 ;;
esac
""",
        encoding="utf-8",
    )
    docker.chmod(0o700)

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
if [[ "${FAKE_RM_CONTEXT_FAILURE:-0}" == "1" \
  && ! -e "${FAKE_RM_CONTEXT_FAILURE_MARKER}" ]]; then
  recorded_context="$(cat "${FAKE_DOCKER_CONTEXT_RECORD}" 2>/dev/null || true)"
  for candidate in "$@"; do
    if [[ "$1" == "-rf" \
      && -n "${recorded_context}" \
      && "${candidate}" == "${recorded_context}/"* ]]; then
      : > "${FAKE_RM_CONTEXT_FAILURE_MARKER}"
      exit 79
    fi
  done
fi
recorded_context="$(cat "${FAKE_DOCKER_CONTEXT_RECORD}" 2>/dev/null || true)"
if [[ -n "${FAKE_LIFECYCLE_RECORD:-}" && "$1" == "-rf" ]]; then
  for candidate in "$@"; do
    if [[ -n "${recorded_context}" && "${candidate}" == "${recorded_context}/"* ]]; then
      printf 'context-delete\n' >> "${FAKE_LIFECYCLE_RECORD}"
      break
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
        "PYTHONPATH": f"{bin_dir}:{os.environ.get('PYTHONPATH', '')}",
        "TMPDIR": str(temp_root),
        "NPA_BYOF_ROBOMIMIC_RECEIPT_DIR": str(receipt_dir),
        "NPA_BYOF_ROBOMIMIC_LOCK_DIR": str(lock_dir),
        "FAKE_DOCKER_TAG_STATE": str(tmp_path / "tag-state"),
        "FAKE_DOCKER_IID_RECORD": str(tmp_path / "iid-path"),
        "FAKE_DOCKER_CONTEXT_RECORD": str(tmp_path / "context-path"),
        "FAKE_DOCKER_ACTION_RECORD": str(tmp_path / "docker-actions"),
        "FAKE_DOCKER_LOCK_STATE": str(tmp_path / "daemon-lock"),
        "FAKE_DOCKER_VOLUME_STATE": str(tmp_path / "daemon-volume"),
        "FAKE_DOCKER_LOCK_RECORD": str(tmp_path / "daemon-lock-actions"),
        "FAKE_FSYNC_RECORD": str(tmp_path / "fsync-actions"),
        "FAKE_FSYNC_FAILURE_MARKER": str(tmp_path / "fsync-failure-marker"),
        "FAKE_RM_FAILURE_MARKER": str(tmp_path / "rm-failure-marker"),
        "FAKE_RM_CONTEXT_FAILURE_MARKER": str(tmp_path / "rm-context-marker"),
        "FAKE_RECEIPT_LINK_FAILURE_MARKER": str(tmp_path / "link-failure-marker"),
        "FAKE_DIRECTORY_REPLACEMENT_MARKER": str(
            tmp_path / "directory-replacement-marker"
        ),
        "FAKE_FINAL_TARGET_REPLACEMENT_MARKER": str(
            tmp_path / "final-target-replacement-marker"
        ),
        "FAKE_LIFECYCLE_RECORD": str(tmp_path / "lifecycle-actions"),
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


def _tag_terminal_module(tmp_path: Path) -> ModuleType:
    program = BUILD_SCRIPT.read_text().split(
        "tag_terminal_program=\"$(cat <<'PY'\n", 1
    )[1]
    program = program.split("\nPY\n)", 1)[0]
    path = tmp_path / "terminal_program.py"
    path.write_text(program)
    spec = importlib.util.spec_from_file_location("robomimic_tag_terminal_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _absent_owner_process(pid: int) -> str:
    raise FileNotFoundError


def _terminal_record(module: ModuleType) -> dict:
    return dict(
        schema=module.SCHEMA,
        phase="no-further-tag-writes",
        host=module.host_identity(),
        uid=os.geteuid(),
        pid=424242,
        start_ticks="12345",
        daemon="synthetic-daemon-identity",
        revision="a" * 40,
        transaction="npa-robomimic-context.abcdefgh",
        lock_id="b" * 64,
        tag="local.invalid/npa-robomimic:dev-" + "a" * 40,
        image="sha256:" + "c" * 64,
    )


def _terminal_case(tmp_path: Path) -> tuple[ModuleType, Path, dict, dict, list]:
    module = _tag_terminal_module(tmp_path)
    module.host_identity = lambda: ["synthetic-boot", "synthetic-pid-namespace"]
    module.process_start = _absent_owner_process
    record = _terminal_record(module)
    transaction = record["transaction"]
    path = tmp_path / (transaction + ".tag-terminal.json")
    path.write_text(json.dumps(record))
    path.chmod(0o600)
    name = (
        "npa-robomimic-tag-lock-" + hashlib.sha256(record["tag"].encode()).hexdigest()
    )
    state = dict(
        daemon=record["daemon"],
        lock_id=record["lock_id"],
        transaction=transaction,
        image=record["image"],
        tag=record["image"],
        running="false",
        name="/" + name,
        reads=0,
    )
    removed = []
    module.daemon_output = lambda *args: _terminal_daemon_read(state, args)
    module.subprocess = SimpleNamespace(
        DEVNULL=subprocess.DEVNULL,
        run=lambda args, **kwargs: removed.append(args),
    )
    return module, path, record, state, removed


def _terminal_daemon_read(state: dict, args: tuple) -> str:
    state["reads"] += 1
    if state.get("change_on_second_probe") and state["reads"] == 4:
        state["transaction"] = "npa-robomimic-context.successor"
    if args[0] == "info":
        return state["daemon"]
    if args[:2] == ("container", "inspect"):
        return "|".join(
            state[key] for key in ("lock_id", "transaction", "image", "running", "name")
        )
    assert args[:2] == ("image", "inspect")
    return state["tag"]


def test_terminal_fence_reconciliation_recovers_only_exact_dead_owner(
    tmp_path: Path,
) -> None:
    module, path, record, state, removed = _terminal_case(tmp_path)
    original = path.read_bytes()

    module.reconcile(path, record["revision"], record["tag"])

    assert state["reads"] == 6
    assert removed == [["docker", "container", "rm", "--volumes", record["lock_id"]]]
    assert path.read_bytes() == original
    state["lock_id"] = "d" * 64
    with pytest.raises(ValueError, match="lock changed"):
        module.reconcile(path, record["revision"], record["tag"])
    assert len(removed) == 1


@pytest.mark.parametrize("start", ("12345", "67890"), ids=("live-owner", "pid-reused"))
def test_terminal_fence_refuses_live_owner_and_pid_reuse(
    tmp_path: Path, start: str
) -> None:
    module, path, record, state, removed = _terminal_case(tmp_path)
    module.process_start = lambda pid: start
    with pytest.raises(ValueError, match="owner live, unreaped, or PID reused"):
        module.reconcile(path, record["revision"], record["tag"])
    assert state["reads"] == 0
    assert removed == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("host", ["foreign-boot", "synthetic-pid-namespace"]),
        ("host", ["synthetic-boot", "foreign-pid-namespace"]),
        ("uid", -1),
        ("phase", "tag-operation-in-flight"),
        ("revision", "d" * 40),
        ("tag", "local.invalid/other:tag"),
        ("transaction", "other-transaction"),
        ("start_ticks", "ambiguous"),
        ("pid", True),
    ],
)
def test_terminal_fence_refuses_ambiguous_or_foreign_owner(
    tmp_path: Path, field: str, value: object
) -> None:
    module, path, record, state, removed = _terminal_case(tmp_path)
    changed = {**record, field: value}
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        module.reconcile(path, record["revision"], record["tag"])
    assert state["reads"] == 0
    assert removed == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("daemon", "different-daemon"),
        ("lock_id", "d" * 64),
        ("transaction", "npa-robomimic-context.foreign1"),
        ("image", "sha256:" + "e" * 64),
        ("tag", "sha256:" + "f" * 64),
        ("running", "true"),
        ("name", "/foreign-lock"),
        ("change_on_second_probe", True),
    ],
)
def test_terminal_fence_refuses_changed_daemon_tag_transaction(
    tmp_path: Path, field: str, value: object
) -> None:
    module, path, record, state, removed = _terminal_case(tmp_path)
    state[field] = value
    with pytest.raises(ValueError):
        module.reconcile(path, record["revision"], record["tag"])
    assert removed == []


@pytest.mark.parametrize(
    "mutation",
    ("symlink", "hardlink", "permissive", "duplicate", "absent", "parent-symlink"),
)
def test_terminal_fence_refuses_untrusted_or_absent_evidence(
    tmp_path: Path, mutation: str
) -> None:
    module, path, record, state, removed = _terminal_case(tmp_path)
    if mutation == "symlink":
        retained = path.with_suffix(".retained")
        path.rename(retained)
        path.symlink_to(retained)
    elif mutation == "hardlink":
        os.link(path, path.with_suffix(".link"))
    elif mutation == "permissive":
        path.chmod(0o644)
    elif mutation == "duplicate":
        path.write_text(path.read_text()[:-1] + ',"pid":424242}')
    elif mutation == "absent":
        path.rename(path.with_suffix(".retained"))
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        path = alias / path.name
    with pytest.raises((OSError, ValueError)):
        module.reconcile(path, record["revision"], record["tag"])
    assert state["reads"] == 0
    assert removed == []


def test_terminal_fence_refuses_inaccessible_owner(tmp_path: Path) -> None:
    module, path, record, state, removed = _terminal_case(tmp_path)

    def inaccessible(pid: int) -> str:
        raise PermissionError

    module.process_start = inaccessible
    with pytest.raises(PermissionError):
        module.reconcile(path, record["revision"], record["tag"])
    assert state["reads"] == 0
    assert removed == []


def test_terminal_fence_refuses_receipt_replacement_during_probe(
    tmp_path: Path,
) -> None:
    module, path, record, state, removed = _terminal_case(tmp_path)

    def replace_receipt(*args: str) -> str:
        if state["reads"] == 0:
            path.rename(path.with_suffix(".retained"))
            path.write_text(json.dumps(record))
            path.chmod(0o600)
        return _terminal_daemon_read(state, args)

    module.daemon_output = replace_receipt
    with pytest.raises(ValueError, match="receipt changed"):
        module.reconcile(path, record["revision"], record["tag"])
    assert removed == []


def _receipt_records(receipt_dir: Path) -> list[tuple[Path, dict[str, object]]]:
    return [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(receipt_dir.glob("*.json"))
    ]


def _assert_failure_receipt(
    receipt_dir: Path,
    image_id: str,
    *,
    reason: str | None = None,
    context_disposition: str = "retained-owner-private-for-reconciliation",
) -> dict[str, object]:
    failures = [
        (path, record)
        for path, record in _receipt_records(receipt_dir)
        if record.get("schema") == "npa.robomimic.neutral-build-failure.v1"
    ]
    assert len(failures) == 1
    path, record = failures[0]
    assert record["status"] == "failure"
    assert record["immutable_image_id"] == image_id
    assert record["immutable_image_disposition"] == (
        "retained-no-exclusive-ownership-proof"
    )
    assert record["context_evidence_disposition"] == context_disposition
    assert (
        record["revision"]
        == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    )
    assert str(record["intended_full_sha_tag"]).endswith(
        ":dev-" + str(record["revision"])
    )
    assert str(record["transaction_id"]).startswith("npa-robomimic-context.")
    if reason is not None:
        assert record["failure_reason"] == reason
    assert path.stat().st_size < 2048
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    return record


def _docker_actions(tmp_path: Path) -> list[str]:
    path = tmp_path / "docker-actions"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def _records_with_schema(
    receipt_dir: Path, schema: str
) -> list[tuple[Path, dict[str, object]]]:
    return [
        (path, record)
        for path, record in _receipt_records(receipt_dir)
        if record.get("schema") == schema
    ]


def _assert_cleanup_receipt(
    receipt_dir: Path, image_id: str, *, status: str, disposition: str
) -> dict[str, object]:
    cleanups = _records_with_schema(
        receipt_dir, "npa.robomimic.neutral-build-cleanup.v1"
    )
    assert len(cleanups) == 1
    path, record = cleanups[0]
    builds = _records_with_schema(receipt_dir, "npa.robomimic.neutral-build-receipt.v1")
    assert len(builds) == 1
    assert record["transaction_id"] == builds[0][1]["transaction_id"]
    assert record["immutable_image_id"] == image_id
    assert record["status"] == status
    assert record["transaction_evidence_disposition"] == disposition
    assert (
        record["cleanup_journal"] == f"{record['transaction_id']}.cleanup-journal.json"
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    return record


def _assert_cleanup_journal(receipt_dir: Path, image_id: str) -> dict[str, object]:
    journals = _records_with_schema(
        receipt_dir, "npa.robomimic.neutral-build-cleanup-journal.v1"
    )
    assert len(journals) == 1
    path, record = journals[0]
    assert record["immutable_image_id"] == image_id
    assert record["cleanup_intent"] == "remove-transaction-context"
    assert record["intent_state"] == "durably-recorded-before-context-deletion"
    assert record["terminal_outcome_record"] == (
        f"{record['transaction_id']}.cleanup.json"
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    return record


def test_build_helper_records_immutable_id_in_owner_only_receipt(
    tmp_path: Path,
) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "1" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id

    result = _run_fake_build(environment)

    assert result.returncode == 0, result.stderr
    builds = _records_with_schema(receipt_dir, "npa.robomimic.neutral-build-receipt.v1")
    assert len(builds) == 1
    receipt_path, receipt = builds[0]
    assert receipt["immutable_image_id"] == image_id
    assert receipt["consumer_image_ref"] == image_id
    assert (
        receipt["revision"]
        == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    )
    assert receipt["tag_compare_and_set"] == "assigned"
    assert receipt["immutable_image_disposition"] == "retained-consumer-reference"
    assert receipt["transaction_evidence_disposition"] == (
        "cleanup-pending-terminal-result"
    )
    assert not (tmp_path / "daemon-volume").exists()
    _assert_cleanup_receipt(
        receipt_dir, image_id, status="completed", disposition="removed"
    )
    journal = _assert_cleanup_journal(receipt_dir, image_id)
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert (tmp_path / "fsync-actions").read_text().splitlines() == [
        "regular:ok",
        "directory:ok",
        "regular:ok",
        "directory:ok",
        "regular:ok",
        "directory:ok",
        "regular:ok",
        "directory:ok",
    ]
    assert (tmp_path / "lifecycle-actions").read_text().splitlines() == [
        f"published:{receipt['transaction_id']}.tag-terminal.json",
        f"published:{receipt['transaction_id']}.json",
        f"published:{journal['transaction_id']}.cleanup-journal.json",
        "context-delete",
        f"published:{receipt['transaction_id']}.cleanup.json",
    ]
    assert tag_state.read_text(encoding="utf-8").strip() == image_id
    assert list(temp_root.glob("npa-robomimic-context.*")) == []


@pytest.mark.parametrize(
    ("signal_name", "expected_status"), (("HUP", 129), ("INT", 130), ("TERM", 143))
)
def test_build_helper_signal_after_publication_preserves_success(
    tmp_path: Path, signal_name: str, expected_status: int
) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "d" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id
    environment["FAKE_POST_PUBLICATION_SIGNAL"] = signal_name

    result = _run_fake_build(environment)

    assert result.returncode == expected_status, result.stderr
    _assert_interrupted_success(tmp_path, temp_root, receipt_dir, tag_state, image_id)


def _assert_interrupted_success(
    tmp_path: Path, temp_root: Path, receipt_dir: Path, tag_state: Path, image_id: str
) -> None:
    builds = _records_with_schema(receipt_dir, "npa.robomimic.neutral-build-receipt.v1")
    assert len(builds) == 1
    assert builds[0][1]["consumer_image_ref"] == image_id
    assert not _records_with_schema(
        receipt_dir, "npa.robomimic.neutral-build-failure.v1"
    )
    _assert_cleanup_journal(receipt_dir, image_id)
    _assert_cleanup_receipt(
        receipt_dir,
        image_id,
        status="unresolved",
        disposition="retained-owner-private-for-reconciliation",
    )
    assert len(list(temp_root.glob("npa-robomimic-context.*"))) == 1
    assert tag_state.read_text().strip() == image_id
    assert _docker_actions(tmp_path) == ["tag"]
    assert not (tmp_path / "daemon-lock").exists()


def test_build_helper_error_after_publication_preserves_success(tmp_path: Path) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "d" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id
    environment["FAKE_POST_PUBLICATION_EXIT"] = "1"

    result = _run_fake_build(environment)

    assert result.returncode != 0
    _assert_interrupted_success(tmp_path, temp_root, receipt_dir, tag_state, image_id)


def test_build_helper_stdout_failure_cannot_reverse_success(tmp_path: Path) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "d" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id

    with open("/dev/full", "w") as full_output:
        result = subprocess.run(
            ["bash", str(BUILD_SCRIPT)],
            cwd=ROOT,
            env=environment,
            check=False,
            stdout=full_output,
            stderr=subprocess.PIPE,
            text=True,
        )

    assert result.returncode != 0
    assert "No space left on device" in result.stderr
    _assert_interrupted_success(tmp_path, temp_root, receipt_dir, tag_state, image_id)


def test_build_helper_refuses_stale_shared_tag_without_retagging(
    tmp_path: Path,
) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    stale_id = "sha256:" + "2" * 64
    tag_state.write_text(stale_id + "\n", encoding="utf-8")
    image_id = "sha256:" + "3" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id

    result = _run_fake_build(environment)

    assert result.returncode != 0
    assert "already names different bytes" in result.stderr
    assert tag_state.read_text(encoding="utf-8").strip() == stale_id
    assert _docker_actions(tmp_path) == []
    _assert_failure_receipt(
        receipt_dir, image_id, reason="full-SHA tag already names different bytes"
    )
    assert len(list(temp_root.glob("npa-robomimic-context.*"))) == 1


def test_build_helper_refuses_ambiguous_inspection_without_retagging(
    tmp_path: Path,
) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    stale_id = "sha256:" + "2" * 64
    tag_state.write_text(stale_id + "\n", encoding="utf-8")
    image_id = "sha256:" + "3" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id
    environment["FAKE_DOCKER_MODE"] = "tag-inspect-error"

    result = _run_fake_build(environment)

    assert result.returncode != 0
    assert "tag absence could not be proven" in result.stderr
    assert tag_state.read_text(encoding="utf-8").strip() == stale_id
    assert _docker_actions(tmp_path) == []
    _assert_failure_receipt(
        receipt_dir,
        image_id,
        reason="existing full-SHA tag absence could not be proven",
    )
    assert len(list(temp_root.glob("npa-robomimic-context.*"))) == 1


def test_build_helper_accepts_identical_shared_tag_idempotently(tmp_path: Path) -> None:
    environment, _, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "3" * 64
    tag_state.write_text(image_id + "\n", encoding="utf-8")
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id

    result = _run_fake_build(environment)

    assert result.returncode == 0, result.stderr
    receipt = _records_with_schema(
        receipt_dir, "npa.robomimic.neutral-build-receipt.v1"
    )[0][1]
    assert receipt["tag_compare_and_set"] == "existing-identical"
    assert tag_state.read_text(encoding="utf-8").strip() == image_id
    assert _docker_actions(tmp_path) == []


def test_build_helper_serializes_distinct_uid_namespaces_in_one_daemon(
    tmp_path: Path,
) -> None:
    environment, _, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    processes = []
    for index, digit in enumerate(("4", "5")):
        candidate = dict(environment)
        candidate_tmp = tmp_path / f"tmp-{index}"
        candidate_tmp.mkdir(mode=0o700)
        candidate["TMPDIR"] = str(candidate_tmp)
        candidate["NPA_BYOF_ROBOMIMIC_LOCK_DIR"] = str(
            tmp_path / f"attempted-lock-root-{index}"
        )
        candidate["FAKE_CALLER_UID"] = str(1000 + index)
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
    records = _receipt_records(receipt_dir)
    assert len(records) == 5
    fences = _records_with_schema(receipt_dir, "npa.robomimic.tag-terminal.v1")
    assert len(fences) == 1 and fences[0][1]["image"] == winner
    assert sorted(record["status"] for _, record in records if "status" in record) == [
        "completed",
        "failure",
    ]
    assert sum("consumer_image_ref" in record for _, record in records) == 1
    assert any(
        "already names different bytes" in stderr
        or "daemon-wide shared-tag coordination" in stderr
        for _, stderr in outcomes
    )
    assert _docker_actions(tmp_path) == ["tag"]
    lock_actions = (tmp_path / "daemon-lock-actions").read_text().splitlines()
    assert {line.split(":", 1)[0] for line in lock_actions} == {"1000", "1001"}
    assert sum(line.endswith(":create-acquired") for line in lock_actions) >= 1
    assert sum(line.endswith(":create-refused") for line in lock_actions) <= 1
    assert not (tmp_path / "daemon-lock").exists()
    assert not (tmp_path / "attempted-lock-root-0").exists()
    assert not (tmp_path / "attempted-lock-root-1").exists()


def test_build_helper_serializes_equivalent_registry_references(
    tmp_path: Path,
) -> None:
    environment, _, _, tag_state = _fake_build_environment(tmp_path)
    processes = []
    for index, (registry, digit) in enumerate(
        (("example/team", "4"), ("docker.io/example/team", "5"))
    ):
        candidate = dict(environment)
        candidate_tmp = tmp_path / f"alias-tmp-{index}"
        candidate_tmp.mkdir(mode=0o700)
        candidate["TMPDIR"] = str(candidate_tmp)
        candidate["NPA_BYOF_ROBOMIMIC_REGISTRY"] = registry
        candidate["FAKE_CALLER_UID"] = f"alias-{index}"
        candidate["FAKE_DOCKER_IMAGE_ID"] = "sha256:" + digit * 64
        candidate["FAKE_DOCKER_IID_RECORD"] = str(tmp_path / f"alias-iid-{index}")
        candidate["FAKE_DOCKER_CONTEXT_RECORD"] = str(
            tmp_path / f"alias-context-{index}"
        )
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
    assert sorted(process.returncode for process in processes) == [0, 1]
    lock_actions = (tmp_path / "daemon-lock-actions").read_text().splitlines()
    lock_names = set()
    for line in lock_actions:
        fields = line.split(":")
        if len(fields) == 3 and fields[2] in {"create-acquired", "create-refused"}:
            lock_names.add(fields[1])
    assert len(lock_names) == 1
    assert sum(line.endswith(":create-acquired") for line in lock_actions) == 1
    assert sum(line.endswith(":create-refused") for line in lock_actions) == 1
    assert _docker_actions(tmp_path) == ["tag"]
    assert tag_state.read_text(encoding="utf-8").strip() in {
        "sha256:" + "4" * 64,
        "sha256:" + "5" * 64,
    }
    assert any(
        "already names different bytes" in stderr
        or "daemon-wide shared-tag coordination" in stderr
        for _, stderr in outcomes
    )


def test_build_helper_serializes_docker_hub_default_namespace_aliases(
    tmp_path: Path,
) -> None:
    environment, _, _, _ = _fake_build_environment(tmp_path)
    processes = []
    for index, registry in enumerate(
        ("docker.io/npa-robomimic", "docker.io/library/npa-robomimic")
    ):
        candidate = dict(environment)
        candidate_tmp = tmp_path / f"dockerhub-alias-tmp-{index}"
        candidate_tmp.mkdir(mode=0o700)
        candidate["TMPDIR"] = str(candidate_tmp)
        candidate["NPA_BYOF_ROBOMIMIC_REGISTRY"] = registry
        candidate["FAKE_CALLER_UID"] = f"dockerhub-alias-{index}"
        candidate["FAKE_DOCKER_IMAGE_ID"] = "sha256:" + str(6 + index) * 64
        candidate["FAKE_DOCKER_IID_RECORD"] = str(tmp_path / f"dockerhub-iid-{index}")
        candidate["FAKE_DOCKER_CONTEXT_RECORD"] = str(
            tmp_path / f"dockerhub-context-{index}"
        )
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
    assert sorted(process.returncode for process in processes) == [0, 1]
    lock_actions = (tmp_path / "daemon-lock-actions").read_text().splitlines()
    lock_names = {
        line.split(":", 2)[1]
        for line in lock_actions
        if line.endswith(":create-acquired") or line.endswith(":create-refused")
    }
    assert len(lock_names) == 1
    assert any(
        "daemon-wide shared-tag coordination" in stderr for _, stderr in outcomes
    )


def test_build_helper_refuses_shared_tag_without_daemon_lock(tmp_path: Path) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "5" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id
    environment["FAKE_DOCKER_MODE"] = "lock-unavailable"

    result = _run_fake_build(environment)

    assert result.returncode != 0
    assert "daemon-wide shared-tag coordination" in result.stderr
    assert not tag_state.exists()
    assert _docker_actions(tmp_path) == []
    _assert_failure_receipt(
        receipt_dir,
        image_id,
        reason="daemon-wide shared-tag coordination could not be established",
    )
    assert len(list(temp_root.glob("npa-robomimic-context.*"))) == 1


@pytest.mark.parametrize(
    ("mode", "has_trusted_id"),
    (
        ("missing-iid", False),
        ("malformed-iid", False),
        ("malformed-inspect", True),
        ("changed-inspect", True),
        ("changed-iid", True),
        ("changed-tag-inspect", True),
        ("malformed-tag-inspect", True),
    ),
)
def test_build_helper_refuses_malformed_or_changed_identity(
    tmp_path: Path, mode: str, has_trusted_id: bool
) -> None:
    environment, temp_root, receipt_dir, _ = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "6" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id
    environment["FAKE_DOCKER_MODE"] = mode

    result = _run_fake_build(environment)

    assert result.returncode != 0
    if has_trusted_id:
        _assert_failure_receipt(receipt_dir, image_id)
        assert len(list(temp_root.glob("npa-robomimic-context.*"))) == 1
    else:
        assert list(receipt_dir.glob("*.json")) == []
        assert list(temp_root.glob("npa-robomimic-context.*")) == []
    assert "rm" not in _docker_actions(tmp_path)


@pytest.mark.parametrize("mode", ("signal-tag", "signal-after-build-id"))
def test_build_helper_records_id_and_retains_context_on_signal(
    tmp_path: Path, mode: str
) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "6" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id
    environment["FAKE_DOCKER_MODE"] = mode

    result = _run_fake_build(environment)

    assert result.returncode == 143
    assert len(list(temp_root.glob("npa-robomimic-context.*"))) == 1
    _assert_failure_receipt(receipt_dir, image_id, reason="signal-TERM")
    assert not tag_state.exists()
    assert not (tmp_path / "daemon-volume").exists()
    assert "rm" not in _docker_actions(tmp_path)


def test_build_helper_receipt_generation_failure_retains_staged_evidence(
    tmp_path: Path,
) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "a" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id
    environment["FAKE_RECEIPT_FAILURE"] = "1"

    result = _run_fake_build(environment)

    assert result.returncode == 97
    assert tag_state.read_text(encoding="utf-8").strip() == image_id
    fences = _records_with_schema(receipt_dir, "npa.robomimic.tag-terminal.v1")
    assert len(fences) == 1 and fences[0][1]["image"] == image_id
    assert _receipt_records(receipt_dir) == fences
    assert len(list(receipt_dir.glob(".*.receipt.*"))) == 1
    assert len(list(temp_root.glob("npa-robomimic-context.*"))) == 1
    assert _docker_actions(tmp_path) == ["tag"]


def test_build_helper_post_assignment_failure_retains_tag_and_receipt(
    tmp_path: Path,
) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "b" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id
    environment["FAKE_DOCKER_MODE"] = "post-assignment-built-inspect-failure"

    result = _run_fake_build(environment)

    assert result.returncode != 0
    assert tag_state.read_text(encoding="utf-8").strip() == image_id
    assert _docker_actions(tmp_path) == ["tag"]
    _assert_failure_receipt(
        receipt_dir, image_id, reason="built immutable image ID could not be reverified"
    )
    assert len(list(temp_root.glob("npa-robomimic-context.*"))) == 1


@pytest.mark.parametrize("mode", ("replace-after-tag", "remove-after-tag"))
def test_build_helper_never_removes_competing_shared_tag(
    tmp_path: Path, mode: str
) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "c" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id
    environment["FAKE_DOCKER_MODE"] = mode

    result = _run_fake_build(environment)

    assert result.returncode != 0
    if mode == "replace-after-tag":
        assert tag_state.read_text(encoding="utf-8").strip() == (
            "sha256:" + f"{7:064d}"
        )
    else:
        assert not tag_state.exists()
    assert _docker_actions(tmp_path) == ["tag"]
    _assert_failure_receipt(receipt_dir, image_id)
    assert len(list(temp_root.glob("npa-robomimic-context.*"))) == 1


@pytest.mark.parametrize("failure_kind", ("regular", "directory"))
def test_build_helper_refuses_receipt_fsync_failure_without_deleting_evidence(
    tmp_path: Path, failure_kind: str
) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "e" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id
    environment["FAKE_FSYNC_FAILURE_KIND"] = failure_kind
    environment["FAKE_FSYNC_FAILURE_ORDINAL"] = "2"

    result = _run_fake_build(environment)

    assert result.returncode != 0
    assert (tmp_path / "fsync-failure-marker").is_file()
    assert tag_state.read_text(encoding="utf-8").strip() == image_id
    records = _receipt_records(receipt_dir)
    assert (
        _records_with_schema(receipt_dir, "npa.robomimic.neutral-build-failure.v1")
        == []
    )
    expected_builds = 1 if failure_kind == "directory" else 0
    assert (
        sum("consumer_image_ref" in record for _, record in records) == expected_builds
    )
    fsync_actions = (tmp_path / "fsync-actions").read_text().splitlines()
    expected_fsync = ["regular:ok", "directory:ok"]
    if failure_kind == "directory":
        expected_fsync.append("regular:ok")
    expected_fsync.append(f"{failure_kind}:failed")
    if failure_kind == "directory":
        # A visible success outcome is never reversed by failed durability
        # acknowledgement. The nonzero exit retains context for reconciliation.
        expected_fsync.extend(["regular:ok", "directory:ok"] * 2)
        _assert_cleanup_receipt(
            receipt_dir,
            image_id,
            status="unresolved",
            disposition="retained-owner-private-for-reconciliation",
        )
    assert fsync_actions == expected_fsync
    assert list(receipt_dir.glob(".*.receipt.*")) or expected_builds == 1
    assert len(list(temp_root.glob("npa-robomimic-context.*"))) == 1
    assert _docker_actions(tmp_path) == ["tag"]


def test_build_helper_receipt_link_failure_retains_staged_evidence(
    tmp_path: Path,
) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "f" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id
    environment["FAKE_RECEIPT_LINK_FAILURE"] = "1"

    result = _run_fake_build(environment)

    assert result.returncode != 0
    assert (tmp_path / "link-failure-marker").is_file()
    assert tag_state.read_text(encoding="utf-8").strip() == image_id
    fences = _records_with_schema(receipt_dir, "npa.robomimic.tag-terminal.v1")
    assert len(fences) == 1 and fences[0][1]["image"] == image_id
    assert _receipt_records(receipt_dir) == fences
    assert len(list(receipt_dir.glob(".*.receipt.*"))) == 1
    assert len(list(temp_root.glob("npa-robomimic-context.*"))) == 1
    assert _docker_actions(tmp_path) == ["tag"]


def test_build_helper_records_unresolved_context_cleanup(tmp_path: Path) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "9" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id
    environment["FAKE_RM_CONTEXT_FAILURE"] = "1"

    result = _run_fake_build(environment)

    assert result.returncode != 0
    assert (tmp_path / "rm-context-marker").is_file()
    assert tag_state.read_text(encoding="utf-8").strip() == image_id
    records = _receipt_records(receipt_dir)
    assert sum("consumer_image_ref" in record for _, record in records) == 1
    _assert_cleanup_receipt(
        receipt_dir,
        image_id,
        status="unresolved",
        disposition="retained-owner-private-for-reconciliation",
    )
    _assert_cleanup_journal(receipt_dir, image_id)
    assert (
        _records_with_schema(receipt_dir, "npa.robomimic.neutral-build-failure.v1")
        == []
    )
    assert len(list(temp_root.glob("npa-robomimic-context.*"))) == 1
    assert _docker_actions(tmp_path) == ["tag"]


@pytest.mark.parametrize(
    ("failure", "failure_kind"),
    (
        ("creation", None),
        ("identity", None),
        ("link", None),
        ("file-fsync", "regular"),
        ("directory-fsync", "directory"),
    ),
)
def test_build_helper_preserves_cleanup_journal_on_terminal_publication_failure(
    tmp_path: Path, failure: str, failure_kind: str | None
) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "8" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id
    if failure == "creation":
        environment["FAKE_TERMINAL_RECEIPT_FAILURE"] = "1"
    elif failure == "identity":
        environment["FAKE_TERMINAL_RECEIPT_IDENTITY_FAILURE"] = "1"
    elif failure == "link":
        environment["FAKE_TERMINAL_RECEIPT_LINK_FAILURE"] = "1"
    else:
        assert failure_kind is not None
        environment["FAKE_FSYNC_FAILURE_KIND"] = failure_kind
        environment["FAKE_FSYNC_FAILURE_ORDINAL"] = "4"

    result = _run_fake_build(environment)

    assert result.returncode != 0
    journal = _assert_cleanup_journal(receipt_dir, image_id)
    assert journal["terminal_outcome_record"].endswith(".cleanup.json")
    terminal_records = _records_with_schema(
        receipt_dir, "npa.robomimic.neutral-build-cleanup.v1"
    )
    terminal_path = receipt_dir / str(journal["terminal_outcome_record"])
    if failure == "identity":
        assert terminal_records == []
        assert json.loads(terminal_path.read_text(encoding="utf-8")) == {
            "replacement": True
        }
        retained = Path(f"{terminal_path}.retained")
        assert json.loads(retained.read_text(encoding="utf-8"))["status"] == (
            "completed"
        )
    elif failure == "directory-fsync":
        assert len(terminal_records) == 1
        assert terminal_records[0][0] == terminal_path
        assert "cleanup or failure-receipt publication was incomplete" in result.stderr
    else:
        assert terminal_records == []
    assert tag_state.read_text(encoding="utf-8").strip() == image_id
    assert list(temp_root.glob("npa-robomimic-context.*")) == []
    lifecycle = (tmp_path / "lifecycle-actions").read_text().splitlines()
    journal_event = f"published:{journal['transaction_id']}.cleanup-journal.json"
    assert lifecycle.index(journal_event) < lifecycle.index("context-delete")


@pytest.mark.parametrize("replacement", ("scratch-parent", "context"))
def test_build_helper_refuses_replaced_prebuild_directory(
    tmp_path: Path, replacement: str
) -> None:
    environment, temp_root, receipt_dir, _ = _fake_build_environment(tmp_path)
    environment["FAKE_DOCKER_IMAGE_ID"] = "sha256:" + "1" * 64
    environment["FAKE_PREBUILD_REPLACEMENT"] = replacement

    result = _run_fake_build(environment)

    if replacement == "scratch-parent":
        replacement_path = temp_root
        retained = Path(f"{temp_root}.retained")
    else:
        replacement_path = next(
            path
            for path in temp_root.glob("npa-robomimic-context.*")
            if (path / "replacement-marker").is_file()
        )
        retained = Path(f"{replacement_path}.retained")
    assert result.returncode != 0
    assert "directory identity changed" in result.stderr
    assert (replacement_path / "replacement-marker").read_text() == "replacement\n"
    assert retained.is_dir()
    assert list(receipt_dir.glob("*.json")) == []
    assert not (tmp_path / "docker-actions").exists()
    assert not (tmp_path / "iid-path").exists()


def test_build_helper_refuses_replaced_receipt_directory(tmp_path: Path) -> None:
    environment, temp_root, receipt_dir, _ = _fake_build_environment(tmp_path)
    environment["FAKE_DOCKER_IMAGE_ID"] = "sha256:" + "2" * 64
    environment["FAKE_DOCKER_MODE"] = "replace-receipt-directory"

    result = _run_fake_build(environment)

    retained = Path(f"{receipt_dir}.retained")
    assert result.returncode != 0
    assert "directory identity changed" in result.stderr
    assert (receipt_dir / "replacement-marker").read_text() == "replacement\n"
    assert retained.is_dir()
    assert list(receipt_dir.glob("*.json")) == []
    assert list(retained.glob("*.json")) == []
    assert _docker_actions(tmp_path) == []
    assert len(list(temp_root.glob("npa-robomimic-context.*"))) == 1


def test_build_helper_refuses_replaced_final_receipt_target(tmp_path: Path) -> None:
    environment, temp_root, receipt_dir, tag_state = _fake_build_environment(tmp_path)
    image_id = "sha256:" + "3" * 64
    environment["FAKE_DOCKER_IMAGE_ID"] = image_id
    environment["FAKE_REPLACE_FINAL_RECEIPT_TARGET"] = "1"

    result = _run_fake_build(environment)

    replacement = next(
        path
        for path in receipt_dir.glob("*.json")
        if json.loads(path.read_text(encoding="utf-8")) == {"replacement": True}
    )
    assert result.returncode != 0
    assert "receipt publication refused" in result.stderr
    assert replacement.read_text(encoding="utf-8") == '{"replacement":true}\n'
    retained = Path(f"{replacement}.retained")
    assert retained.is_file()
    retained_record = json.loads(retained.read_text(encoding="utf-8"))
    assert retained_record["consumer_image_ref"] == image_id
    assert retained.stat().st_ino != replacement.stat().st_ino
    _assert_failure_receipt(
        receipt_dir, image_id, reason="transaction receipt publication refused"
    )
    assert tag_state.read_text(encoding="utf-8").strip() == image_id
    assert len(list(temp_root.glob("npa-robomimic-context.*"))) == 1
    assert _docker_actions(tmp_path) == ["tag"]


def test_build_helper_shell_functions_remain_reviewable() -> None:
    lines = BUILD_SCRIPT.read_text(encoding="utf-8").splitlines()
    function_lengths: dict[str, int] = {}
    function_name: str | None = None
    function_start = 0
    for line_number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([a-z][a-z0-9_]*)\(\) \{", line)
        if match:
            function_name = match.group(1)
            function_start = line_number
        elif line == "}" and function_name is not None:
            function_lengths[function_name] = line_number - function_start + 1
            function_name = None
    assert function_name is None
    assert function_lengths
    assert max(function_lengths.values()) < 40, function_lengths


def test_changed_trust_boundary_functions_remain_reviewable() -> None:
    for relative_path in ("verify_image.py", "smoke.py"):
        source = IMAGE_ROOT / relative_path
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        lengths = {node.name: node.end_lineno - node.lineno + 1 for node in functions}
        assert functions, f"{relative_path} must expose reviewable functions"
        assert max(lengths.values()) < 40, {relative_path: lengths}


def test_smoke_result_builder_call_matches_signature() -> None:
    tree = ast.parse((IMAGE_ROOT / "smoke.py").read_text(encoding="utf-8"))
    definitions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_build_result"
    ]
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_build_result"
    ]
    assert len(definitions) == 1
    assert len(definitions[0].args.args) == 6
    assert len(calls) == 1
    assert len(calls[0].args) == 6
    assert not calls[0].keywords


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


def _inert_wheel_members() -> dict[str, bytes]:
    """Harmless non-executable content; not production wheel evidence."""
    members = {
        "inertpkg/__init__.py": b"inert fixture bytes, never import\n",
        "inertpkg-1.0.dist-info/METADATA": b"Name: inertpkg\nVersion: 1.0\n",
        "inertpkg-1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        "inertpkg-1.0.dist-info/RECORD": b"",
        "inertpkg-1.0.dist-info/entry_points.txt": b"[console_scripts]\ninert-script = inertpkg:unused\n",
        "inertpkg-1.0.data/purelib/inert_data.txt": b"inert relocated data\n",
        "inertpkg-1.0.data/platlib/inert_platform.txt": b"inert platform data\n",
    }
    rows = []
    for name, raw in members.items():
        if name.endswith("/RECORD"):
            rows.append((name, "", ""))
        else:
            digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
            rows.append((name, "sha256=" + digest.decode().rstrip("="), str(len(raw))))
    output = io.StringIO(newline="")
    csv.writer(output).writerows(sorted(rows))
    members["inertpkg-1.0.dist-info/RECORD"] = output.getvalue().encode()
    return members


def _write_inert_wheel(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in members.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, raw)


def _inert_install(root: Path, members: dict[str, bytes]) -> None:
    record = "inertpkg-1.0.dist-info/RECORD"
    installed = {}
    record_paths = {}
    for name, raw in members.items():
        if name == record:
            continue
        if name.startswith("inertpkg-1.0.data/purelib/"):
            target = name.removeprefix("inertpkg-1.0.data/purelib/")
            record_path = target
        elif name.startswith("inertpkg-1.0.data/platlib/"):
            target = name.removeprefix("inertpkg-1.0.data/platlib/")
            record_path = target
        else:
            target = name
            record_path = name
        installed[target] = raw
        record_paths[target] = record_path
    installed["inertpkg-1.0.dist-info/INSTALLER"] = b"pip\n"
    installed["inertpkg-1.0.dist-info/REQUESTED"] = b""
    installed["bin/inert-script"] = (
        b"#!/usr/local/bin/python3\nimport sys\n"
        b"from inertpkg import unused\nif __name__ == '__main__':\n"
        b"    sys.argv[0] = sys.argv[0].removesuffix('.exe')\n"
        b"    sys.exit(unused())\n"
    )
    record_paths["inertpkg-1.0.dist-info/INSTALLER"] = (
        "inertpkg-1.0.dist-info/INSTALLER"
    )
    record_paths["inertpkg-1.0.dist-info/REQUESTED"] = (
        "inertpkg-1.0.dist-info/REQUESTED"
    )
    rows = []
    for name, raw in installed.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o755 if name.startswith("bin/") else 0o644)
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        )
        rows.append(
            (
                "../../" + name if name.startswith("bin/") else record_paths[name],
                "sha256=" + digest,
                str(len(raw)),
            )
        )
    with (root / record).open("w", newline="") as stream:
        csv.writer(stream).writerows(sorted([*rows, (record, "", "")]))
    (root / record).chmod(0o644)
    for directory in (root, *(p for p in root.rglob("*") if p.is_dir())):
        directory.chmod(0o755)


def _inert_installed_source(tmp_path: Path) -> tuple[Path, Path, dict]:
    source_members = {
        "LICENSE": b"MIT\n",
        "robomimic/__init__.py": b"inert source bytes\n",
    }
    archive, manifest = _source_archive(tmp_path, source_members)
    manifest["git_tree_sha1"] = "0" * 40
    source = tmp_path / "source"
    for name, raw in source_members.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o644)
        path.parent.chmod(0o755)
    return source, archive, manifest


@pytest.fixture
def inert_installed_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Synthetic trust roots exercise interfaces, never production acceptance."""
    source, archive, manifest = _inert_installed_source(tmp_path)
    deps, wheels = (tmp_path / name for name in ("deps", "wheels"))
    wheels.mkdir()
    wheel = wheels / "inertpkg-1.0-py3-none-any.whl"
    members = _inert_wheel_members()
    _write_inert_wheel(wheel, members)
    _inert_install(deps, members)
    lock = tmp_path / "baked.lock"
    lock.write_text(
        "inertpkg==1.0 --hash=sha256:"
        + hashlib.sha256(wheel.read_bytes()).hexdigest()
        + "\n"
    )
    monkeypatch.setattr(
        VERIFIER, "BAKED_LOCK_SHA256", hashlib.sha256(lock.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(VERIFIER, "BAKED_DISTRIBUTION_COUNT", 1)
    monkeypatch.setattr(VERIFIER, "SOURCE_REVISION", "0" * 40)
    monkeypatch.setattr(VERIFIER, "_source_manifest", lambda _: manifest)
    monkeypatch.setattr(
        VERIFIER, "verify_debian_install", lambda **_: {"package_count": 0}
    )
    monkeypatch.setattr(VERIFIER, "_verified_installer", _inert_installer_identity)
    return dict(
        source_root=source,
        metadata_path=tmp_path / "inert-manifest",
        debian_lock_path=tmp_path / "inert-debian",
        baked_lock_path=lock,
        baked_deps_path=deps,
        source_archive_path=archive,
        selected_wheel_root=wheels,
        empty_boundary_paths=(),
        installer_input_root=tmp_path / "inert-build-input",
    )


def _inert_installer_identity(*_) -> dict:
    return {
        "sha256": "0" * 64,
        "members": {"inventory_sha256": "0" * 64},
        "source_revision": "0" * 40,
        "publisher": {"verified_statement_sha256": "0" * 64},
    }


def test_installed_byte_proof_binds_archive_wheels_and_inventories(
    inert_installed_image: dict,
) -> None:
    proof = VERIFIER.verify_neutral_image(**inert_installed_image)
    assert proof["schema"] == "npa.robomimic.neutral-image-verification.v2"
    assert proof["source_revision"] == "0" * 40  # deliberately synthetic
    assert proof["installed_source"]["file_count"] == 2
    selected = proof["installed_dependencies"]["selected_wheels"]
    wheel = (
        inert_installed_image["selected_wheel_root"] / selected["inertpkg"]["filename"]
    )
    assert (
        selected["inertpkg"]["sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    )
    assert (
        proof["source_archive_sha256"]
        == hashlib.sha256(
            inert_installed_image["source_archive_path"].read_bytes()
        ).hexdigest()
    )
    assert (
        proof["installed_dependencies"]["installation_policy"]
        == "pip-26.2.1-posix-home-target-no-compile-v2"
    )
    record = (
        inert_installed_image["baked_deps_path"] / "inertpkg-1.0.dist-info/RECORD"
    ).read_text()
    assert "inert_data.txt," in record
    assert "inert_platform.txt," in record
    assert "../../inert_data.txt," not in record
    assert "../../inert_platform.txt," not in record


def test_extracted_source_tree_is_authenticated_before_dpkg(
    inert_installed_image: dict,
) -> None:
    proof = VERIFIER.verify_installed_source_tree(
        source_root=inert_installed_image["source_root"],
        source_archive_path=inert_installed_image["source_archive_path"],
        metadata_path=inert_installed_image["metadata_path"],
    )
    assert proof["file_count"] == 2


def test_dockerfile_authenticates_extracted_source_before_debian_install() -> None:
    text = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    source_check = text.index("verify_image.py source-tree")
    dpkg_install = text.index("dpkg --unpack")
    assert source_check < dpkg_install
    assert "--source-archive /mnt/robomimic-build-inputs/source/robomimic.tar" in text


def test_source_tree_cli_mode_uses_build_dispatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_image.py",
            "source-tree",
            "--source-root",
            str(tmp_path / "source"),
            "--source-archive",
            str(tmp_path / "source.tar"),
            "--source-manifest",
            str(tmp_path / "manifest.json"),
        ],
    )
    monkeypatch.setattr(
        VERIFIER, "_dispatch_build", lambda _args: {"dispatch": "build"}
    )
    monkeypatch.setattr(
        VERIFIER,
        "_dispatch_runtime",
        lambda _args: pytest.fail("source-tree must not use runtime dispatch"),
    )

    assert VERIFIER.main() == 0
    assert json.loads(capsys.readouterr().out) == {"dispatch": "build"}


def test_production_wheel_proof_refuses_without_complete_authenticated_closure(
    tmp_path: Path,
) -> None:
    lock = IMAGE_ROOT / "baked-requirements.lock"
    locked = VERIFIER._locked_baked_artifacts(lock)
    assert len(locked) == 40
    wheel_root = tmp_path / "wheels"
    wheel_root.mkdir()
    with pytest.raises(VERIFIER.VerificationError, match="selected wheel closure"):
        VERIFIER._baked_dependency_proof(wheel_root, tmp_path / "installed", lock)


@pytest.mark.parametrize(
    "root_key,relative",
    [
        ("source_root", "robomimic/__init__.py"),
        ("baked_deps_path", "inertpkg/__init__.py"),
        ("baked_deps_path", "inertpkg-1.0.dist-info/METADATA"),
        ("baked_deps_path", "bin/inert-script"),
        ("baked_deps_path", "inert_data.txt"),
    ],
)
def test_installed_byte_proof_rejects_changed_bytes(
    inert_installed_image: dict, root_key: str, relative: str
) -> None:
    (inert_installed_image[root_key] / relative).write_bytes(
        b"changed harmless fixture\n"
    )
    with pytest.raises(VERIFIER.VerificationError, match="inventory mismatch"):
        VERIFIER.verify_neutral_image(**inert_installed_image)


@pytest.mark.parametrize("root_key", ["source_root", "baked_deps_path"])
@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "extra-directory", "duplicate-distribution"]
)
def test_installed_byte_proof_rejects_closure_mutations(
    inert_installed_image: dict, root_key: str, mutation: str
) -> None:
    root = inert_installed_image[root_key]
    if mutation == "missing":
        # Move, do not delete, the inert fixture; nothing is executed.
        path = next(p for p in root.rglob("*") if p.is_file() and p.name != "RECORD")
        path.rename(root.parent / "retained-missing-member")
    elif mutation == "extra-directory":
        (root / "extra-directory").mkdir()
    elif mutation == "duplicate-distribution":
        directory = root / "inertpkg-duplicate.dist-info"
        directory.mkdir()
        (directory / "METADATA").write_bytes(b"Name: inertpkg\nVersion: 1.0\n")
    else:
        (root / "extra-member").write_bytes(b"inert extra\n")
    with pytest.raises(VERIFIER.VerificationError):
        VERIFIER.verify_neutral_image(**inert_installed_image)


@pytest.mark.parametrize("key", ["source_archive_path", "selected_wheel_root"])
def test_installed_byte_proof_requires_independent_inputs(
    inert_installed_image: dict, key: str
) -> None:
    inert_installed_image[key] = inert_installed_image[key].parent / "absent"
    with pytest.raises(VERIFIER.VerificationError):
        VERIFIER.verify_neutral_image(**inert_installed_image)


@pytest.mark.parametrize(
    "mutation", ["wheel", "archive", "lock", "record", "duplicate-wheel"]
)
def test_installed_byte_proof_rejects_trust_root_mutation(
    inert_installed_image: dict, mutation: str
) -> None:
    paths = {
        "wheel": next(inert_installed_image["selected_wheel_root"].iterdir()),
        "archive": inert_installed_image["source_archive_path"],
        "lock": inert_installed_image["baked_lock_path"],
        "record": inert_installed_image["baked_deps_path"]
        / "inertpkg-1.0.dist-info/RECORD",
    }
    if mutation == "duplicate-wheel":
        (paths["wheel"].parent / "inertpkg-1.0-1-py3-none-any.whl").write_bytes(
            paths["wheel"].read_bytes()
        )
    else:
        paths[mutation].write_bytes(
            paths[mutation].read_bytes() + b"inert changed bytes\n"
        )
    with pytest.raises(VERIFIER.VerificationError):
        VERIFIER.verify_neutral_image(**inert_installed_image)


def test_installed_byte_proof_rejects_symlink_type_without_following(
    inert_installed_image: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = inert_installed_image["source_root"] / "robomimic/__init__.py"
    original = Path.lstat

    def observed(path: Path):
        return (
            SimpleNamespace(st_mode=stat.S_IFLNK | 0o777)
            if path == target
            else original(path)
        )

    monkeypatch.setattr(Path, "lstat", observed)
    with pytest.raises(VERIFIER.VerificationError, match="non-regular"):
        VERIFIER.verify_neutral_image(**inert_installed_image)


def test_dockerfile_is_bootstrap_only_and_does_not_install_python_wheels() -> None:
    text = DOCKERFILE.read_text()
    python_commands = re.findall(r"python3 [^\n]+", text)
    assert python_commands and all(
        command.startswith("python3 -I -S -B ") for command in python_commands
    )
    assert "-m pip" not in text
    assert "installer download" not in text
    assert "installer install" not in text
    assert "verify_image.py image" not in text
    assert "runtime-requirements.lock" in text
    assert "/opt/robomimic-deps" not in text


def test_installed_byte_proof_does_not_trust_self_consistent_installed_record(
    inert_installed_image: dict,
) -> None:
    root = inert_installed_image["baked_deps_path"]
    changed = b"changed inert bytes with a matching installed record\n"
    (root / "inertpkg/__init__.py").write_bytes(changed)
    record = root / "inertpkg-1.0.dist-info/RECORD"
    rows = list(csv.reader(io.StringIO(record.read_text())))
    for row in rows:
        if row[0] == "inertpkg/__init__.py":
            digest = base64.urlsafe_b64encode(hashlib.sha256(changed).digest())
            row[1:] = ["sha256=" + digest.decode().rstrip("="), str(len(changed))]
    with record.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(VERIFIER.VerificationError, match="RECORD transformation"):
        VERIFIER.verify_neutral_image(**inert_installed_image)


@pytest.mark.parametrize("object_kind", [stat.S_IFLNK, stat.S_IFIFO])
def test_wheel_inventory_refuses_non_regular_metadata_without_extraction(
    object_kind: int,
) -> None:
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as archive:
        member = zipfile.ZipInfo("inert-member")
        member.external_attr = (object_kind | 0o644) << 16
        archive.writestr(member, b"harmless inert bytes")
    with pytest.raises(VERIFIER.VerificationError, match="non-regular"):
        VERIFIER._wheel_members(raw.getvalue())


def test_wheel_inventory_refuses_duplicate_members_without_extraction() -> None:
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as archive:
        archive.writestr("inert-member", b"harmless")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("inert-member", b"harmless duplicate")
    with pytest.raises(VERIFIER.VerificationError, match="duplicate"):
        VERIFIER._wheel_members(raw.getvalue())


def test_wheel_inventory_refuses_unknown_installation_transformation() -> None:
    with pytest.raises(VERIFIER.VerificationError, match="installation scheme"):
        VERIFIER._wheel_target(
            "inertpkg-1.0.data/scripts/inert", "inertpkg-1.0.dist-info"
        )


def test_wheel_inventory_accepts_authenticated_data_scheme_target() -> None:
    assert (
        VERIFIER._wheel_target(
            "fonttools-4.64.0.data/data/share/man/man1/ttx.1",
            "fonttools-4.64.0.dist-info",
        )
        == "share/man/man1/ttx.1"
    )


def test_wheel_inventory_accepts_authenticated_script_scheme_transform() -> None:
    target, entry, record_path = VERIFIER._wheel_script_member(
        "jmespath-1.0.1.data/scripts/jp.py",
        "jmespath-1.0.1.dist-info",
        b"#!python\nprint('inert')\n",
    )
    assert target == "bin/jp.py"
    assert record_path == "../../bin/jp.py"
    assert entry == {
        "type": "file",
        "size": 40,
        "sha256": "eec29f22848f719f67c3d3e32f35d14b7c901359ab7cba525e30d7b6b376a6f2",
        "executable": True,
        "mode": 0o755,
    }
    assert VERIFIER._installed_record_bytes(
        {target: entry},
        "jmespath-1.0.1.dist-info/RECORD",
        {target: record_path},
    ) == (
        b"../../bin/jp.py,sha256=7sKfIoSPcZ9nw9PjLzXRS3yQE1mrfLpSXjDXtrN2pvI,40\r\n"
        b"jmespath-1.0.1.dist-info/RECORD,,\r\n"
    )


@pytest.mark.parametrize(
    "raw", [b"#!/usr/bin/python\nprint('inert')\n", b"#!python -x\nprint('inert')\n"]
)
def test_wheel_inventory_refuses_unproven_script_transform(raw: bytes) -> None:
    with pytest.raises(VERIFIER.VerificationError, match="script transformation"):
        VERIFIER._wheel_script_member(
            "jmespath-1.0.1.data/scripts/jp.py",
            "jmespath-1.0.1.dist-info",
            raw,
        )


def test_wheel_distribution_accepts_pep427_name_case_and_separator_normalization() -> (
    None
):
    members = {
        "PyYAML-6.0.2.dist-info/METADATA": (
            b"Name: PyYAML\nVersion: 6.0.2\n",
            False,
        ),
        "PyYAML-6.0.2.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\n",
            False,
        ),
    }
    assert (
        VERIFIER._wheel_distribution(members, "pyyaml", "6.0.2")
        == "PyYAML-6.0.2.dist-info"
    )


@pytest.mark.parametrize(
    "directory",
    ["PyYAML-6.0.3.dist-info", "Other-6.0.2.dist-info"],
)
def test_wheel_distribution_refuses_version_or_name_mismatch(directory: str) -> None:
    members = {
        f"{directory}/METADATA": (b"Name: PyYAML\nVersion: 6.0.2\n", False),
        f"{directory}/WHEEL": (b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\n", False),
    }
    with pytest.raises(VERIFIER.VerificationError, match="directory mismatch"):
        VERIFIER._wheel_distribution(members, "pyyaml", "6.0.2")


def test_wheel_inventory_refuses_relocated_duplicate_distribution() -> None:
    with pytest.raises(VERIFIER.VerificationError, match="undeclared distribution"):
        VERIFIER._wheel_target(
            "inertpkg-1.0.data/purelib/other-1.0.dist-info/METADATA",
            "inertpkg-1.0.dist-info",
        )


def test_deterministic_installer_script_matches_independent_pip_source() -> None:
    # Literal from the independently authenticated PipScriptMaker source,
    # never produced by invoking/importing the candidate installer.
    expected = (
        b"#!/usr/local/bin/python3\nimport sys\n"
        b"from example.cli import Runner\n"
        b"if __name__ == '__main__':\n"
        b"    sys.argv[0] = sys.argv[0].removesuffix('.exe')\n"
        b"    sys.exit(Runner.main())\n"
    )
    assert VERIFIER._console_script("example.cli", "Runner.main") == expected


@pytest.mark.parametrize(
    "declaration",
    [
        "tool = example:run [extra]",
        "tool = example",
        "pip = example:run",
        "easy_install = example:run",
        "tool = example:run --option",
        "tool = class:run",
        "tool = example:while",
    ],
)
def test_deterministic_installer_refuses_unmodeled_entrypoints(declaration) -> None:
    members = {
        "inert.dist-info/entry_points.txt": (
            ("[console_scripts]\n" + declaration + "\n").encode(),
            False,
        )
    }
    with pytest.raises(VERIFIER.VerificationError, match="unsupported"):
        VERIFIER._wheel_scripts(members, "inert.dist-info")


def test_deterministic_installer_requires_exact_metadata_and_build_input() -> None:
    manifest = VERIFIER._source_manifest(IMAGE_ROOT / "source-manifest.json")
    installer = manifest["build_installer"]
    assert installer["version"] == "26.2.1"
    assert installer["members"]["count"] == 476
    assert len(installer["members"]["pe_launchers"]) == 6
    assert installer["members"]["notice_files"] == 42
    expected = VERIFIER._expected_build_input_objects({"packages": []}, manifest)
    assert expected == {
        "debian",
        "source",
        "source/robomimic.tar",
        "tools",
        "tools/pip-26.2.1-py3-none-any.whl",
    }
    for key, value in (
        ("version", "unknown"),
        ("sha256", "0" * 64),
        ("source_revision", "0" * 40),
        ("parent_qualification", "PASS"),
    ):
        with pytest.raises(VERIFIER.VerificationError):
            VERIFIER._checked_installer({**installer, key: value})


@pytest.mark.parametrize("raw", [b"", b"harmless incomplete archive"])
def test_deterministic_installer_refuses_bytes_before_archive_parsing(raw) -> None:
    with pytest.raises(VERIFIER.VerificationError, match="bytes mismatch"):
        VERIFIER._checked_installer_bytes(raw)


def test_deterministic_installer_record_order_and_modes_are_exact() -> None:
    digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    entry = {"sha256": digest, "size": 3}
    assert VERIFIER._installed_record_bytes(
        {"bin/tool": entry, "inert": entry}, "inert.dist-info/RECORD"
    ) == (
        b"../../bin/tool,sha256=ungWv48Bz-pBQUDeXa4iI7ADYaOWF3qctBD_YfIAFa0,3\r\n"
        b"inert,sha256=ungWv48Bz-pBQUDeXa4iI7ADYaOWF3qctBD_YfIAFa0,3\r\n"
        b"inert.dist-info/RECORD,,\r\n"
    )
    assert VERIFIER._file_identity(b"abc")["mode"] == 0o644
    assert VERIFIER._file_identity(b"abc", True)["mode"] == 0o755


def test_deterministic_installer_records_data_scheme_relative_path() -> None:
    digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    entry = {"sha256": digest, "size": 3}
    assert VERIFIER._installed_record_bytes(
        {"share/man/ttx.1": entry},
        "inert.dist-info/RECORD",
        {"share/man/ttx.1": "../../share/man/ttx.1"},
    ) == (
        b"../../share/man/ttx.1,sha256=ungWv48Bz-pBQUDeXa4iI7ADYaOWF3qctBD_YfIAFa0,3\r\n"
        b"inert.dist-info/RECORD,,\r\n"
    )


def test_deterministic_installer_arguments_do_not_inherit_configuration() -> None:
    wheels = Path("/opt/npa/robomimic/installer-wheels")
    lock = Path("/opt/npa/robomimic/baked-requirements.lock")
    common = [
        "--isolated",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--only-binary=:all:",
        "--no-deps",
        "--require-hashes",
        "--requirement",
        str(lock),
    ]
    assert VERIFIER._installer_arguments("install", wheels, lock) == common + [
        "--no-compile",
        "--no-index",
        "--find-links",
        str(wheels),
        "--target",
        "/opt/robomimic-deps",
    ]
    assert VERIFIER._installer_environment(Path("/inert-owned-scratch")) == {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": "/inert-owned-scratch",
        "TMPDIR": "/inert-owned-scratch",
        "PIP_CONFIG_FILE": "/dev/null",
        "XDG_CACHE_HOME": "/inert-owned-scratch",
    }
    with pytest.raises(VERIFIER.VerificationError, match="unsupported"):
        VERIFIER._installer_arguments("arbitrary", wheels, lock)


def test_deterministic_installer_unknown_platform_has_no_side_effects(
    monkeypatch,
) -> None:
    def refused():
        raise VERIFIER.VerificationError("unsupported inert platform")

    def unexpected(*args, **kwargs):
        pytest.fail("side effect before interpreter refusal")

    monkeypatch.setattr(VERIFIER, "_require_installer_platform", refused)
    monkeypatch.setattr(VERIFIER, "_source_manifest", unexpected)
    monkeypatch.setattr(VERIFIER, "_execute_build_installer", unexpected)
    with pytest.raises(VERIFIER.VerificationError, match="unsupported inert"):
        VERIFIER.run_build_installer(
            action="install",
            input_root=Path("/inert"),
            wheels=Path("/opt/npa/robomimic/installer-wheels"),
        )


def _inert_installer_call_arguments() -> tuple:
    return (
        Path("/inert"),
        {"path": "tools/inert.whl", "sha256": "0" * 64},
        "install",
        ["--isolated", "install"],
    )


@pytest.mark.parametrize("returncode", [0, 1])
def test_deterministic_installer_mocked_execution_owns_scratch(
    monkeypatch, returncode
) -> None:
    events = []

    class Scratch:
        def __init__(self, **kwargs):
            assert kwargs == {"prefix": "installer-scratch-", "dir": "/inert-parent"}

        def __enter__(self):
            events.append("enter")
            return "/inert-owned-scratch"

        def __exit__(self, *args):
            events.append("exit")

    def execute(command, **kwargs):
        events.append("mock-call")
        assert command[:5] == ["/usr/local/bin/python3", "-I", "-S", "-B", "-c"]
        assert "sys.path.insert(0,sys.argv.pop(1))" in command[5]
        assert kwargs["umask"] == 0o022
        assert kwargs["env"]["TMPDIR"] == str(kwargs["cwd"]) == "/inert-owned-scratch"
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(VERIFIER.tempfile, "TemporaryDirectory", Scratch)
    monkeypatch.setattr(
        VERIFIER, "_installer_workspace_parent", lambda: Path("/inert-parent")
    )
    monkeypatch.setattr(VERIFIER.subprocess, "run", execute)
    args = _inert_installer_call_arguments()
    if returncode:
        with pytest.raises(VERIFIER.VerificationError, match="installer refused"):
            VERIFIER._execute_build_installer(*args)
    else:
        assert not VERIFIER._execute_build_installer(*args)["image_qualified"]
    assert events == ["enter", "mock-call", "exit"]


@pytest.mark.parametrize("fault", ["preferred", "override", "layout"])
def test_deterministic_installer_unknown_scheme_refuses(monkeypatch, fault) -> None:
    monkeypatch.setattr(
        VERIFIER.sysconfig,
        "get_preferred_scheme",
        lambda _: "unknown" if fault == "preferred" else "posix_home",
    )
    monkeypatch.setattr(
        VERIFIER.sysconfig, "_PIP_USE_SYSCONFIG", fault != "override", raising=False
    )
    paths = {
        "purelib": "/npa-scheme-probe/lib/python",
        "platlib": "/npa-scheme-probe/lib/python",
        "scripts": "/npa-scheme-probe/bin",
        "data": "/npa-scheme-probe",
    }
    if fault == "layout":
        paths["scripts"] = "/inert-unsupported-bin"
    monkeypatch.setattr(VERIFIER.sysconfig, "get_paths", lambda **_: paths)
    with pytest.raises(VERIFIER.VerificationError, match="unsupported installer"):
        VERIFIER._require_installer_scheme()


@pytest.mark.parametrize(
    ("mode", "owner", "euid"),
    [
        (stat.S_IFLNK | 0o755, 0, 0),
        (stat.S_IFDIR | 0o777, 0, 0),
        (stat.S_IFDIR | 0o755, 1000, 0),
        (stat.S_IFDIR | 0o755, 0, 1000),
    ],
)
def test_deterministic_installer_workspace_refuses_unowned_metadata(
    monkeypatch, mode, owner, euid
) -> None:
    # Inert metadata only: no filesystem object, ownership change or race.
    monkeypatch.setattr(
        Path, "lstat", lambda _: SimpleNamespace(st_mode=mode, st_uid=owner)
    )
    monkeypatch.setattr(VERIFIER.os, "geteuid", lambda: euid)
    with pytest.raises(VERIFIER.VerificationError, match="root-owned"):
        VERIFIER._installer_workspace_parent()
