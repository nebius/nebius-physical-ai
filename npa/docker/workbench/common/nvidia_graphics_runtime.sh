#!/usr/bin/env bash
# Materialize driver-matched NVIDIA graphics userspace at run time.
#
# Managed GPU nodes may expose CUDA compute without the Vulkan/GL libraries that
# Isaac Sim needs.  This helper keeps those third-party bytes out of the public
# image: NVIDIA delivers the exact running-driver payload directly to an
# operator-owned ephemeral/PVC volume after runtime EULA acceptance.  The
# upstream SHA-256 file is verified before extraction and publication is atomic.
set -euo pipefail

readonly EX_CONFIG=78
readonly EX_UNAVAILABLE=69
readonly EX_SOFTWARE=70

ROOT="${NPA_NVIDIA_GRAPHICS_DIR:-/opt/nvidia-graphics}"
LOCK_TIMEOUT="${NPA_NVIDIA_GRAPHICS_LOCK_TIMEOUT:-900}"
DOWNLOAD_ORIGIN="${NPA_NVIDIA_DRIVER_ORIGIN:-https://us.download.nvidia.com/XFree86/Linux-x86_64}"

log() { printf 'nvidia-graphics-runtime: %s\n' "$*" >&2; }
die() { local code="$1"; shift; log "$*"; exit "$code"; }

acceptance_state="$({
  printf '%s' "${ACCEPT_EULA-}" \
    | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' \
    | tr '[:lower:]' '[:upper:]'
} 2>/dev/null)"
case "$acceptance_state" in
  Y|YES|1|TRUE) ;;
  ''|N|NO|0|FALSE)
    die "$EX_CONFIG" "refusing NVIDIA graphics download: runtime acceptance is disabled"
    ;;
  *) die "$EX_CONFIG" "refusing NVIDIA graphics download: runtime acceptance is invalid" ;;
esac

command -v nvidia-smi >/dev/null \
  || die "$EX_UNAVAILABLE" "nvidia-smi is unavailable; a GPU must be assigned"
version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \
  | head -1 | tr -d '[:space:]')"
case "$version" in
  ''|*[!0-9.]*) die "$EX_SOFTWARE" "running NVIDIA driver version is invalid" ;;
esac

artifact="NVIDIA-Linux-x86_64-${version}-no-compat32.run"
identity="$(printf 'nvidia|%s|linux-x86_64|%s|runfile-v1' "$DOWNLOAD_ORIGIN" "$version" \
  | sha256sum | cut -d' ' -f1)"
target="${ROOT}/${identity}"
ready="${target}/.ready"

verify_tree() {
  local tree="$1"
  test -s "${tree}/.ready" \
    && test -e "${tree}/libGLX_nvidia.so.0" \
    && test -e "${tree}/libEGL_nvidia.so.0" \
    && test -e "${tree}/libGL.so.1" \
    && test -e "${tree}/libGLdispatch.so.0" \
    && test -e "${tree}/libnvidia-glcore.so.${version}" \
    && test -e "${tree}/libnvidia-eglcore.so.${version}" \
    && grep -qxF "driver=${version}" "${tree}/.ready"
}

mkdir -p "$ROOT"
exec 9>"${ROOT}/.populate.lock"
flock -w "$LOCK_TIMEOUT" 9 \
  || die "$EX_UNAVAILABLE" "timed out waiting for the graphics cache writer"

if ! verify_tree "$target"; then
  tmp="$(mktemp -d "${ROOT}/.${identity}.tmp.XXXXXX")"
  cleanup() { rm -rf "$tmp"; }
  trap cleanup EXIT

  base="${DOWNLOAD_ORIGIN}/${version}/${artifact}"
  log "downloading driver-matched graphics userspace from NVIDIA"
  curl -fsSL --retry 3 --connect-timeout 15 -o "${tmp}/${artifact}" "$base" \
    || die "$EX_UNAVAILABLE" "NVIDIA graphics payload download failed"
  curl -fsSL --retry 3 --connect-timeout 15 \
    -o "${tmp}/${artifact}.sha256sum" "${base}.sha256sum" \
    || die "$EX_UNAVAILABLE" "NVIDIA graphics checksum download failed"
  expected="$(awk 'NR == 1 {print $1}' "${tmp}/${artifact}.sha256sum")"
  case "$expected" in
    [0-9a-fA-F][0-9a-fA-F]*) ;;
    *) die "$EX_SOFTWARE" "NVIDIA graphics checksum file is malformed" ;;
  esac
  test "${#expected}" -eq 64 \
    || die "$EX_SOFTWARE" "NVIDIA graphics checksum has the wrong length"
  printf '%s  %s\n' "$expected" "${tmp}/${artifact}" | sha256sum -c - >&2 \
    || die "$EX_SOFTWARE" "NVIDIA graphics payload checksum mismatch"

  sh "${tmp}/${artifact}" --extract-only --target "${tmp}/extracted" >/dev/null \
    || die "$EX_SOFTWARE" "NVIDIA graphics payload extraction failed"
  library_root="$(dirname "$(find "${tmp}/extracted" -type f \
    -name "libnvidia-glcore.so.${version}" -print -quit)")"
  test -n "$library_root" \
    || die "$EX_SOFTWARE" "extracted payload has no graphics library root"
  test -f "${library_root}/libGLX_nvidia.so.${version}" \
    || die "$EX_SOFTWARE" "extracted payload has no NVIDIA GLX library"
  test -f "${library_root}/libnvidia-eglcore.so.${version}" \
    || die "$EX_SOFTWARE" "extracted payload has no NVIDIA EGL library"

  ln -sf "libGLX_nvidia.so.${version}" "${library_root}/libGLX_nvidia.so.0"
  # `--extract-only` does not create the GLVND links the interactive installer
  # normally publishes. Without the matching libGL.so.1, the loader combines
  # the distro GL library with this driver's libGLdispatch and Isaac/Iray fails
  # on `_glapi_tls_Current` before camera rendering can advance.
  test ! -f "${library_root}/libGL.so.1.7.0" \
    || ln -sf "libGL.so.1.7.0" "${library_root}/libGL.so.1"
  test ! -f "${library_root}/libGLESv1_CM.so.1.2.0" \
    || ln -sf "libGLESv1_CM.so.1.2.0" "${library_root}/libGLESv1_CM.so.1"
  test ! -f "${library_root}/libGLESv2.so.2.1.0" \
    || ln -sf "libGLESv2.so.2.1.0" "${library_root}/libGLESv2.so.2"
  test ! -f "${library_root}/libEGL_nvidia.so.${version}" \
    || ln -sf "libEGL_nvidia.so.${version}" "${library_root}/libEGL_nvidia.so.0"
  test ! -f "${library_root}/libnvidia-egl-gbm.so.${version}" \
    || ln -sf "libnvidia-egl-gbm.so.${version}" \
      "${library_root}/libnvidia-egl-gbm.so.1"
  printf 'driver=%s\nsha256=%s\n' "$version" "$expected" \
    > "${library_root}/.ready"

  rm -rf "$target"
  mv "$library_root" "$target"
  trap - EXIT
  rm -rf "$tmp"
fi

ln -sfn "$identity" "${ROOT}/current"
printf '%s\n' \
  '{"file_format_version":"1.0.0","ICD":{"library_path":"/opt/nvidia-graphics/current/libEGL_nvidia.so.0","api_version":"1.3.277"}}' \
  > "${ROOT}/nvidia_icd.json"
cat > "${ROOT}/runtime.env" <<'EOF'
export LD_LIBRARY_PATH=/opt/nvidia-graphics/current:${LD_LIBRARY_PATH-}
export VK_DRIVER_FILES=/opt/nvidia-graphics/nvidia_icd.json
export VK_ICD_FILENAMES=/opt/nvidia-graphics/nvidia_icd.json
EOF
log "driver-matched graphics userspace is ready"
