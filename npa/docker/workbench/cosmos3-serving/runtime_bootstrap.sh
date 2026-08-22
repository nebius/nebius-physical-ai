#!/usr/bin/env bash
# Materialize the pinned serving closure at runtime under the operator's terms.
set -euo pipefail

ROOT=/opt/npa-cosmos3-serving/runtime
VENV="${ROOT}/venv"
MARKER="${ROOT}/ready-${NPA_COSMOS3_CLOSURE_SHA256:-missing}"
LOCK=/opt/npa-cosmos3-serving/requirements.lock

fail() { echo "[npa-cosmos3-serving] ERROR: $*" >&2; exit 78; }

case "${NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE:-}" in
  YES) ;;
  *)
    fail "the CUDA Python serving closure is delivered at runtime under the NVIDIA \
Software License (v. May 12, 2021). Review the license shipped by the upstream \
cuda-bindings distribution, then set NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE=YES \
for this run. This value is never baked or persisted."
    ;;
esac

REVISION="${NPA_VLLM_OMNI_REVISION:?missing pinned vLLM-Omni revision}"
SOURCE_SHA="${NPA_VLLM_OMNI_SOURCE_SHA256:?missing pinned source checksum}"

mkdir -p "${ROOT}"
[ -w "${ROOT}" ] || fail "runtime root ${ROOT} is not writable by uid $(id -u)"
if [ -f "${MARKER}" ] && [ -x "${VENV}/bin/vllm" ]; then
  export PATH="${VENV}/bin:${PATH}"
  exec "$@"
fi

work="${ROOT}/.install-$$"
trap 'rm -rf "${work}"' EXIT
mkdir -p "${work}"
echo "${NPA_COSMOS3_CLOSURE_SHA256}  ${LOCK}" | sha256sum -c -
rm -rf "${VENV}"
# Virtualenv console scripts embed their interpreter as an absolute shebang.
# Build at the final path: moving a completed venv from ${work} would leave
# every entry point referring to a deleted temporary interpreter.
python -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --no-cache-dir --require-hashes --only-binary=:all: \
  --no-binary=antlr4-python3-runtime,openai-whisper -r "${LOCK}"
site_packages="$("${VENV}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cp /opt/npa-cosmos3-serving/hf_snapshot_pin.py "${site_packages}/sitecustomize.py"
"${VENV}/bin/python" /opt/npa-cosmos3-serving/prepare_guardrail_runtime.py
curl -fL --retry 3 --proto '=https' --tlsv1.2 \
  -o "${work}/vllm-omni.tar.gz" \
  "https://github.com/vllm-project/vllm-omni/archive/${REVISION}.tar.gz"
echo "${SOURCE_SHA}  ${work}/vllm-omni.tar.gz" | sha256sum -c -
mkdir "${work}/source"
tar -xzf "${work}/vllm-omni.tar.gz" --strip-components=1 -C "${work}/source"
VLLM_OMNI_TARGET_DEVICE=cuda VLLM_OMNI_VERSION_OVERRIDE=0.26.0 \
  "${VENV}/bin/python" -m pip install --no-cache-dir --no-deps "${work}/source"
"${VENV}/bin/python" -m pip check
touch "${MARKER}"
export PATH="${VENV}/bin:${PATH}"
exec "$@"
