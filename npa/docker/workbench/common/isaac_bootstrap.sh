#!/usr/bin/env bash
#
# isaac_bootstrap.sh - fetch NVIDIA Isaac Sim / Isaac Lab at FIRST RUN, never at build.
#
# WHY THIS EXISTS
#   The npa Isaac workbench images (npa-isaac-lab, npa-sonic, npa-sonic-mujoco,
#   npa-groot) contain no NVIDIA Isaac bytes at all. Isaac Sim's Omniverse Kit SDK and
#   the isaacsim/isaaclab wheels are NVIDIA-proprietary - both wheels literally declare
#   `License: NVIDIA Proprietary Software` - so an image that baked them could not be
#   published to a public registry without making us the third-party redistributor.
#
#   A download token cannot fix a baked image: a token gates a download, and the bytes
#   are already in the layers. So instead of arguing about the statement, this makes the
#   statement true. NVIDIA delivers Isaac to the operator's own machine, on first run,
#   under the operator's OWN EULA acceptance. That is the same pattern the workbench
#   already uses for runtime model weights, which is why
#   those images are already public.
#
# DEFAULT ACCEPTANCE
#   NPA defaults NVIDIA's documented ACCEPT_EULA to Y for Isaac-backed workloads so
#   non-interactive workflows do not stop at the vendor prompt. Operators can explicitly
#   opt out with an empty value or N/NO/0/FALSE. Legacy affirmative spellings
#   Y/YES/1/TRUE migrate case-insensitively. Other values are rejected as invalid.
#   (https://pypi.nvidia.com serves these wheels anonymously, so the credential was
#   never the gate. Acceptance is.)
#
# MODES
#   ensure   (default) make the pinned Isaac install available, installing if needed
#   warm     pre-populate the cache and exit; for a per-node/PVC init Job so ordinary
#            pods can then run with NPA_ISAAC_CACHE_READONLY=1
#   verify   ensure, then additionally launch Isaac Sim headless (needs a GPU)
#   status   report what is cached without installing (no EULA required, no network)
#
# ENVIRONMENT
#   ACCEPT_EULA                   defaults to Y; empty/N/NO/0/FALSE opt out
#   NPA_ISAAC_CACHE_DIR           cache volume root                (/opt/isaac-cache)
#   NPA_ISAAC_INDEX_URL           NVIDIA wheel index               (https://pypi.nvidia.com)
#   NPA_ISAAC_BASE_PYTHON         image python3.11 that has torch  (per image)
#   NPA_ISAAC_WHEELS_FILE         hash-pinned wheel manifest       (next to this script)
#   NPA_ISAAC_OSS_DEPS_FILE        image-baked OSS dependency lock  (beside wheel manifest)
#   ISAAC_SIM_VERSION             pin                              (5.1.0.0)
#   ISAAC_LAB_VERSION             pin                              (2.3.2.post1)
#   NPA_ISAAC_LAB_SRC_COMMIT      BSD-3 IsaacLab commit for scripts/ (pinned below)
#   NPA_ISAAC_LAB_SRC_URL         IsaacLab git remote
#   NPA_ISAAC_BOOTSTRAP_OFFLINE   1 => never touch the network; require a ready cache
#   NPA_ISAAC_CACHE_READONLY      1 => never write; require a ready cache
#   NPA_ISAAC_BOOTSTRAP_TIMEOUT   seconds to wait for a concurrent installer   (3600)
#
# All diagnostic output goes to STDERR. Callers parse the interpreter's stdout - the
# SkyPilot Isaac task templates read hydra overrides out of it in a `while read` loop -
# so a single stray line on stdout would corrupt a real workflow.
#
set -uo pipefail

readonly EX_CONFIG=78          # sysexits.h EX_CONFIG: the operator must act
readonly EX_UNAVAILABLE=69     # sysexits.h EX_UNAVAILABLE: cache missing in offline mode
readonly EX_SOFTWARE=70        # sysexits.h EX_SOFTWARE: install produced a broken tree

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CACHE_DIR="${NPA_ISAAC_CACHE_DIR:-/opt/isaac-cache}"
INDEX_URL="${NPA_ISAAC_INDEX_URL:-https://pypi.nvidia.com}"
WHEELS_FILE="${NPA_ISAAC_WHEELS_FILE:-$SCRIPT_DIR/isaac-nvidia-wheels.txt}"
OSS_DEPS_FILE="${NPA_ISAAC_OSS_DEPS_FILE:-$(dirname "$WHEELS_FILE")/isaac-oss-deps.txt}"
BASE_PYTHON="${NPA_ISAAC_BASE_PYTHON:-}"
ISAAC_SIM_VERSION="${ISAAC_SIM_VERSION:-5.1.0.0}"
ISAAC_LAB_VERSION="${ISAAC_LAB_VERSION:-2.3.2.post1}"
# Isaac Lab v2.3.2. The GitHub repo is BSD-3-Clause (unlike the wheel) and is the only
# source of scripts/reinforcement_learning/, which every SkyPilot Isaac task invokes -
# the wheel ships the library but no scripts/. Pinned by COMMIT, not tag: git tags are
# mutable, so the commit is the load-bearing selector (same rule as lichtblick/groot).
ISAAC_LAB_SRC_URL="${NPA_ISAAC_LAB_SRC_URL:-https://github.com/isaac-sim/IsaacLab.git}"
ISAAC_LAB_SRC_COMMIT="${NPA_ISAAC_LAB_SRC_COMMIT:-37ddf626871758333d6ed89cf64ad702aef127d0}"
OFFLINE="${NPA_ISAAC_BOOTSTRAP_OFFLINE:-0}"
READONLY="${NPA_ISAAC_CACHE_READONLY:-0}"
LOCK_TIMEOUT="${NPA_ISAAC_BOOTSTRAP_TIMEOUT:-3600}"
# Default only when the variable is absent. An explicitly empty/negative value remains an
# opt-out and is rejected by require_eula_acceptance below.
ACCEPT_EULA="${ACCEPT_EULA-Y}"

log()  { printf 'isaac-bootstrap: %s\n' "$*" >&2; }
die()  { local code="$1"; shift; printf 'isaac-bootstrap: %s\n' "$*" >&2; exit "$code"; }

# A half-written cache tree is the worst outcome here: the next pod would find it and
# fail in a confusing place. `die` exits the shell outright, so `install_isaac ... || rm`
# would never run - the cleanup has to be an EXIT trap. Set on entry to install_isaac
# and cleared once the tree has been published.
TMP_TREE=""
cleanup_tmp_tree() {
  [ -n "$TMP_TREE" ] && rm -rf "$TMP_TREE"
  return 0
}
trap cleanup_tmp_tree EXIT

# ---------------------------------------------------------------------------------
# EULA acceptance. This is the whole legal mechanism; keep it first and keep it strict.
# ---------------------------------------------------------------------------------
_acceptance_state() {
  case "$(printf '%s' "${1-}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | tr '[:lower:]' '[:upper:]')" in
    YES|Y|1|TRUE) printf 'accepted' ;;
    ''|N|NO|0|FALSE) printf 'opt-out' ;;
    *) printf 'invalid' ;;
  esac
}

require_eula_acceptance() {
  local state
  state="$(_acceptance_state "$ACCEPT_EULA")"
  [ "$state" = accepted ] && {
    ACCEPT_EULA=Y
    export ACCEPT_EULA
    # The Python-wheel launcher uses this vendor variable internally. It is
    # derived only inside the already-authorized operation, never user-facing
    # plumbing or a repository default.
    export OMNI_KIT_ACCEPT_EULA=YES
    return 0
  }

  if [ "$state" = invalid ]; then
    cat >&2 <<EOF
isaac-bootstrap: invalid ACCEPT_EULA value '${ACCEPT_EULA}'.

  Expected Y, YES, 1, TRUE, N, NO, 0, FALSE, or an empty string
  (case-insensitive). Nothing has been downloaded.
EOF
    exit "$EX_CONFIG"
  fi

  cat >&2 <<EOF
isaac-bootstrap: refusing to download NVIDIA Isaac Sim / Isaac Lab.

  This image deliberately ships NO NVIDIA Isaac Sim or Isaac Lab code. Continuing
  would download them from NVIDIA (${INDEX_URL}) onto this machine under NVIDIA's
  licence terms. NPA normally enables acceptance for Isaac workloads, but this run
  explicitly disabled it. Acceptance and proprietary Isaac bytes are never baked
  into the image.

  Acceptance was explicitly disabled (exact accepted value: ACCEPT_EULA=Y).

  To accept and continue for this operation, set exactly:

      ACCEPT_EULA=Y

  e.g.  docker run -e ACCEPT_EULA=Y ...

  Terms you are accepting:
    NVIDIA Omniverse Licence Agreement
      https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html
    NVIDIA Isaac Sim Additional Software and Materials Licence
      https://docs.isaacsim.omniverse.nvidia.com/latest/common/licenses.html
    NVIDIA Software Licence Agreement
      https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/

  Nothing has been downloaded. See docs/workbench/container-packaging.md.
EOF
  exit "$EX_CONFIG"
}

# ---------------------------------------------------------------------------------
# Cache identity. Any pin change produces a new stamp, so a running pod's cache is
# never mutated underneath it - a new tree is built alongside and swapped in.
# ---------------------------------------------------------------------------------
resolve_base_python() {
  if [ -n "$BASE_PYTHON" ] && [ -x "$BASE_PYTHON" ]; then
    printf '%s' "$BASE_PYTHON"
    return 0
  fi
  local candidate
  for candidate in /opt/npa/sim/venv/bin/python /opt/npa/venv/bin/python \
                   /opt/isaac-lab/venv/bin/python "$(command -v python3.11 || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

cache_stamp() {
  local base_python="$1" abi wheels_sha bootstrap_sha oss_deps_sha
  abi="$("$base_python" -c 'import sys,sysconfig; print("%d.%d-%s" % (sys.version_info[0], sys.version_info[1], sysconfig.get_platform()))' 2>/dev/null || echo unknown)"
  wheels_sha="$(sha256sum "$WHEELS_FILE" 2>/dev/null | cut -d' ' -f1 || echo nowheels)"
  bootstrap_sha="$(sha256sum "${BASH_SOURCE[0]}" 2>/dev/null | cut -d' ' -f1 || echo nobootstrap)"
  oss_deps_sha="$(sha256sum "$OSS_DEPS_FILE" 2>/dev/null | cut -d' ' -f1 || echo noossdeps)"
  printf '%s|%s|%s|%s|%s|%s|%s|%s' \
    "$ISAAC_SIM_VERSION" "$ISAAC_LAB_VERSION" "$ISAAC_LAB_SRC_COMMIT" \
    "$INDEX_URL" "$wheels_sha" "$bootstrap_sha" "$oss_deps_sha" "$abi" \
    | sha256sum | cut -d' ' -f1
}

# ---------------------------------------------------------------------------------
# Install. Atomic: build in <stamp>.tmp.$$, rename into place, mark .complete LAST,
# then swap the `current` symlink. A reader either sees a complete tree or no tree.
# ---------------------------------------------------------------------------------
install_isaac() {
  local base_python="$1" target="$2" tmp="$3"
  local started; started="$(date +%s)"

  # Any failure from here on must leave the cache exactly as it was.
  TMP_TREE="$tmp"
  rm -rf "$tmp"
  mkdir -p "$tmp"

  log "creating python environment (layered over ${base_python})"
  "$base_python" -m venv "$tmp/venv" \
    || die "$EX_SOFTWARE" "failed to create the cache virtualenv at $tmp/venv"

  # Layer the IMAGE's site-packages (torch, numpy, gear_sonic, the OSS isaaclab deps)
  # into the cache venv with a .pth. `venv --system-site-packages` cannot do this: a
  # venv created from a venv resolves to the BASE interpreter's site-packages, not the
  # image venv's, so torch would be invisible and pip would try to download 3 GB of it.
  # .pth dirs are appended to sys.path, so the cache venv still shadows the image for
  # anything it installs itself.
  local base_site cache_site
  base_site="$("$base_python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  cache_site="$("$tmp/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  printf '%s\n' "$base_site" > "$cache_site/_npa_image_site.pth"

  "$tmp/venv/bin/python" - <<'PY' >&2 || die "$EX_SOFTWARE" "the cache venv cannot see the image's torch; the .pth layering is broken"
import torch
print(f"isaac-bootstrap: image torch {torch.__version__} visible from the cache venv")
PY

  log "installing pinned NVIDIA Isaac wheels from ${INDEX_URL} (hash-verified)"
  # --index-url (not --extra-index-url): the NVIDIA set must not be shadowable from
  # PyPI. --no-deps + --require-hashes: every wheel is pinned to a reviewed sha256, and
  # all OSS transitive deps are already baked in the image (install_isaac_runtime_base.sh).
  "$tmp/venv/bin/python" -m pip install \
      --no-cache-dir --no-deps --require-hashes --disable-pip-version-check \
      --index-url "$INDEX_URL" -r "$WHEELS_FILE" >&2 \
    || die "$EX_SOFTWARE" "NVIDIA Isaac wheel install failed (hash mismatch, or the index is unreachable)"

  log "fetching the BSD-3 Isaac Lab source tree (the wheel ships no scripts/)"
  git clone -q --filter=blob:none --no-checkout "$ISAAC_LAB_SRC_URL" "$tmp/isaaclab-src" >&2 \
    || die "$EX_SOFTWARE" "could not clone $ISAAC_LAB_SRC_URL"
  git -C "$tmp/isaaclab-src" checkout -q "$ISAAC_LAB_SRC_COMMIT" >&2 \
    || die "$EX_SOFTWARE" "could not check out pinned Isaac Lab commit $ISAAC_LAB_SRC_COMMIT"
  local head
  head="$(git -C "$tmp/isaaclab-src" rev-parse HEAD)"
  [ "$head" = "$ISAAC_LAB_SRC_COMMIT" ] \
    || die "$EX_SOFTWARE" "Isaac Lab source is $head, expected $ISAAC_LAB_SRC_COMMIT"
  rm -rf "$tmp/isaaclab-src/.git"

  verify_tree "$tmp" || die "$EX_SOFTWARE" "post-install verification failed; not publishing this cache"

  local bytes elapsed
  bytes="$(du -sb "$tmp" | cut -f1)"
  elapsed=$(( $(date +%s) - started ))
  cat > "$tmp/MANIFEST.json" <<EOF
{
  "format": "npa_isaac_runtime_cache_v1",
  "cache_stamp": "$(basename "$target")",
  "isaacsim_version": "${ISAAC_SIM_VERSION}",
  "isaaclab_version": "${ISAAC_LAB_VERSION}",
  "isaaclab_src_commit": "${ISAAC_LAB_SRC_COMMIT}",
  "index_url": "${INDEX_URL}",
  "wheels_file_sha256": "$(sha256sum "$WHEELS_FILE" | cut -d' ' -f1)",
  "bootstrap_sha256": "$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)",
  "oss_dependencies_sha256": "$(sha256sum "$OSS_DEPS_FILE" | cut -d' ' -f1)",
  "base_python": "${base_python}",
  "bytes": ${bytes},
  "install_seconds": ${elapsed},
  "installed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "installed_by_host": "$(hostname)"
}
EOF

  # Publish atomically. mkdir -p the parent first so the rename stays same-filesystem.
  mkdir -p "$(dirname "$target")"
  rm -rf "$target"
  mv "$tmp" "$target" || die "$EX_SOFTWARE" "could not publish the cache tree to $target"
  TMP_TREE=""   # published; the trap must not delete it now
  : > "$target/.complete"

  # Swap `current` atomically: create a temp symlink then rename over the old one.
  ln -sfn "$target" "${CACHE_DIR}/.current.tmp.$$"
  mv -T "${CACHE_DIR}/.current.tmp.$$" "${CACHE_DIR}/current"

  log "ready in ${elapsed}s ($(( bytes / 1024 / 1024 )) MiB) at ${target}"
}

verify_tree() {
  local root="$1"
  ISAAC_SIM_VERSION="$ISAAC_SIM_VERSION" ISAAC_LAB_VERSION="$ISAAC_LAB_VERSION" \
  NPA_ISAAC_TREE="$root" "$root/venv/bin/python" - >&2 <<'PY'
import importlib.util
import os
import sys
from importlib import metadata
from pathlib import Path

root = Path(os.environ["NPA_ISAAC_TREE"])
problems = []
for package, expected in (
    ("isaacsim", os.environ["ISAAC_SIM_VERSION"]),
    ("isaaclab", os.environ["ISAAC_LAB_VERSION"]),
):
    try:
        found = metadata.version(package)
    except metadata.PackageNotFoundError:
        problems.append(f"{package} is not installed")
        continue
    if found != expected:
        problems.append(f"{package} is {found}, expected {expected}")
    if importlib.util.find_spec(package) is None:
        problems.append(f"{package} has no importable module")

train = root / "isaaclab-src" / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
if not train.is_file():
    problems.append(f"missing Isaac Lab source entrypoint {train}")

if problems:
    for problem in problems:
        print(f"isaac-bootstrap: VERIFY FAILED: {problem}")
    sys.exit(1)
print("isaac-bootstrap: verify ok (isaacsim, isaaclab, and the Isaac Lab source tree)")
PY
}

deep_verify() {
  local venv="$1"
  log "launching Isaac Sim headless (deep verify; needs a GPU with RT cores)"
  "$venv/bin/python" - >&2 <<'PY'
import os

from isaaclab.app import AppLauncher

launcher = AppLauncher(
    headless=True,
    enable_cameras=True,
    kit_args=os.environ.get(
        "NPA_ISAAC_KIT_ARGS", "--portable-root /tmp/npa-isaac-kit"
    ),
)
app = launcher.app
for _ in range(8):
    app.update()
print("isaac-bootstrap: NPA_ISAAC_DEEP_VERIFY_OK headless app launched and stepped")
app.close()
PY
}

# ---------------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------------
tree_is_ready() { [ -f "$1/.complete" ] && [ -x "$1/venv/bin/python" ]; }

ensure() {
  local deep="${1:-0}" base_python target
  [ -f "$WHEELS_FILE" ] \
    || die "$EX_CONFIG" "Isaac wheel manifest is missing: ${WHEELS_FILE}"
  [ -f "$OSS_DEPS_FILE" ] \
    || die "$EX_CONFIG" "Isaac OSS dependency lock is missing: ${OSS_DEPS_FILE}"
  base_python="$(resolve_base_python)" \
    || die "$EX_SOFTWARE" "no python3.11 interpreter found; set NPA_ISAAC_BASE_PYTHON"
  target="${CACHE_DIR}/v/$(cache_stamp "$base_python")"

  # Consent is required before first acquisition and every use, including a
  # warm cache. It must never be manufactured from cache presence.
  require_eula_acceptance

  # Fast path: no lock or network for an already-warm cache.
  if tree_is_ready "$target"; then
    [ "$(readlink -f "${CACHE_DIR}/current" 2>/dev/null || true)" = "$(readlink -f "$target")" ] \
      || { ln -sfn "$target" "${CACHE_DIR}/.current.tmp.$$" 2>/dev/null \
             && mv -T "${CACHE_DIR}/.current.tmp.$$" "${CACHE_DIR}/current" 2>/dev/null; } || true
    [ "$deep" = "1" ] && { deep_verify "$target/venv" || exit "$EX_SOFTWARE"; }
    printf '%s\n' "$target"
    return 0
  fi

  if [ "$OFFLINE" = "1" ]; then
    die "$EX_UNAVAILABLE" "NPA_ISAAC_BOOTSTRAP_OFFLINE=1 but no ready cache at ${target}. Pre-populate it with 'isaac-bootstrap warm' (see the warm-cache Job in npa/docker/workbench/common/warm-isaac-cache.yaml)."
  fi
  if [ "$READONLY" = "1" ]; then
    die "$EX_UNAVAILABLE" "NPA_ISAAC_CACHE_READONLY=1 but no ready cache at ${target}. Run the warm-cache Job against this volume first."
  fi

  mkdir -p "${CACHE_DIR}/v" 2>/dev/null || true
  [ -w "${CACHE_DIR}" ] || [ -w "${CACHE_DIR}/v" ] \
    || die "$EX_CONFIG" "cache dir ${CACHE_DIR} is not writable by $(id -un); mount a writable volume or pre-warm it and set NPA_ISAAC_CACHE_READONLY=1"

  # Serialise installers. Up to 8 pods per GPU node race the same cache, so this is a
  # real contention path, not a theoretical one. flock is fd-based: a killed pod
  # releases automatically, so there is no stale-lock recovery to get wrong.
  exec 9>"${CACHE_DIR}/.lock" || die "$EX_CONFIG" "cannot open ${CACHE_DIR}/.lock"
  if ! flock -w 5 9; then
    log "another process is installing Isaac into ${CACHE_DIR}; waiting (up to ${LOCK_TIMEOUT}s)"
    local waited=0
    while ! flock -w 30 9; do
      waited=$(( waited + 30 ))
      if tree_is_ready "$target"; then
        log "cache became ready while waiting (${waited}s)"
        printf '%s\n' "$target"
        return 0
      fi
      [ "$waited" -ge "$LOCK_TIMEOUT" ] \
        && die "$EX_UNAVAILABLE" "timed out after ${waited}s waiting for a concurrent Isaac install"
      log "still waiting for the Isaac install lock (${waited}s elapsed)"
    done
  fi

  # Double-checked: the winner of the race may have finished while we blocked.
  if tree_is_ready "$target"; then
    log "cache was completed by another process"
  else
    install_isaac "$base_python" "$target" "${target}.tmp.$$"
  fi
  flock -u 9

  [ "$deep" = "1" ] && { deep_verify "$target/venv" || exit "$EX_SOFTWARE"; }
  printf '%s\n' "$target"
}

status() {
  local base_python target acceptance_state
  if base_python="$(resolve_base_python)"; then
    target="${CACHE_DIR}/v/$(cache_stamp "$base_python")"
  else
    base_python="(none found)"; target="(unknown)"
  fi
  printf 'cache_dir=%s\n' "$CACHE_DIR"
  printf 'base_python=%s\n' "$base_python"
  printf 'isaacsim=%s\nisaaclab=%s\nisaaclab_src_commit=%s\n' \
    "$ISAAC_SIM_VERSION" "$ISAAC_LAB_VERSION" "$ISAAC_LAB_SRC_COMMIT"
  printf 'expected_tree=%s\n' "$target"
  if [ "$target" != "(unknown)" ] && tree_is_ready "$target"; then
    printf 'ready=yes\n'
    [ -f "$target/MANIFEST.json" ] && cat "$target/MANIFEST.json"
  else
    printf 'ready=no\n'
  fi
  acceptance_state="$(_acceptance_state "$ACCEPT_EULA")"
  printf 'eula_state=%s\n' "$acceptance_state"
  printf 'eula_accepted=%s\n' \
    "$([ "$acceptance_state" = accepted ] && echo yes || echo no)"
}

usage() {
  sed -n '3,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

main() {
  case "${1:-ensure}" in
    ensure) ensure 0 ;;
    warm)   NPA_ISAAC_CACHE_READONLY=0 NPA_ISAAC_BOOTSTRAP_OFFLINE=0 ensure 0 ;;
    verify) ensure 1 ;;
    status) status ;;
    -h|--help|help) usage ;;
    *) die 64 "unknown mode ${1@Q}; expected ensure, warm, verify, or status" ;;
  esac
}

main "$@"
