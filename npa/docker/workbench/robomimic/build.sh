#!/usr/bin/env bash
set -euo pipefail
umask 077

fail() {
  echo "robomimic neutral build refused: $1" >&2
  exit 1
}

require_owner_private_directory() {
  local path="$1"
  [[ -d "${path}" && ! -L "${path}" ]] \
    || fail "owner directory is absent or unsafe"
  [[ "$(stat -c '%u' -- "${path}")" == "$(id -u)" ]] \
    || fail "owner directory has the wrong owner"
  [[ "$(stat -c '%a' -- "${path}")" == "700" ]] \
    || fail "owner directory must have mode 0700"
}

read_image_id_file() {
  local path="$1"
  local -a lines=()
  [[ -f "${path}" && ! -L "${path}" ]] \
    || fail "Docker iidfile is absent or unsafe"
  [[ "$(stat -c '%u:%h' -- "${path}")" == "$(id -u):1" ]] \
    || fail "Docker iidfile ownership is unsafe"
  mapfile -t lines < "${path}"
  [[ "${#lines[@]}" -eq 1 && "${lines[0]}" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || fail "Docker iidfile is malformed"
  printf '%s\n' "${lines[0]}"
}

inspect_image_id() {
  local reference="$1"
  local output_path="$2"
  local -a lines=()
  if ! docker image inspect --format '{{.Id}}' "${reference}" > "${output_path}"; then
    return 1
  fi
  mapfile -t lines < "${output_path}"
  if [[ "${#lines[@]}" -ne 1 || ! "${lines[0]}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    return 2
  fi
  printf '%s\n' "${lines[0]}"
}

repo_root="$(git rev-parse --show-toplevel)"
revision="$(git -C "${repo_root}" rev-parse HEAD)"
[[ "${revision}" =~ ^[0-9a-f]{40}$ ]] || fail "expected a full Git revision"
registry="${NPA_BYOF_ROBOMIMIC_REGISTRY:-local.invalid}"
[[ "${registry}" =~ ^[A-Za-z0-9][A-Za-z0-9./:_-]*$ ]] \
  || fail "local registry name is malformed"
image="${registry}/npa-robomimic:dev-${revision}"

context="$(mktemp -d "${TMPDIR:-/tmp}/npa-robomimic-context.XXXXXXXX")"
transaction_id="$(basename "${context}")"
receipt_tmp=""
cleanup() {
  if [[ -n "${receipt_tmp}" ]]; then
    rm -f -- "${receipt_tmp}"
  fi
  rm -rf -- "${context}"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

receipt_dir="${NPA_BYOF_ROBOMIMIC_RECEIPT_DIR:-${TMPDIR:-/tmp}/npa-robomimic-build-receipts}"
lock_dir="${NPA_BYOF_ROBOMIMIC_LOCK_DIR:-${TMPDIR:-/tmp}/npa-robomimic-build-locks}"
mkdir -p -m 700 -- "${receipt_dir}" "${lock_dir}"
require_owner_private_directory "${receipt_dir}"
require_owner_private_directory "${lock_dir}"
receipt="${receipt_dir}/${transaction_id}.json"
[[ ! -e "${receipt}" && ! -L "${receipt}" ]] \
  || fail "transaction receipt already exists"

# Build only the exact committed image context named by the immutable tag.
git -C "${repo_root}" archive "${revision}:npa/docker/workbench/robomimic" \
  | tar -x --same-permissions -C "${context}"
python3 "${context}/verify_image.py" prepare-build-inputs \
  --output-root "${context}/build-inputs" \
  --debian-lock "${context}/debian-packages.lock" \
  --source-manifest "${context}/source-manifest.json"

iid_file="${context}/image.iid"
docker build --platform linux/amd64 --pull=false --iidfile "${iid_file}" "${context}"
image_id="$(read_image_id_file "${iid_file}")"
observed_id="$(inspect_image_id "${image_id}" "${context}/inspect-built.txt")" \
  || fail "built immutable image ID could not be verified"
[[ "${observed_id}" == "${image_id}" ]] \
  || fail "built immutable image ID changed during inspection"

tag_lock_name="$(printf '%s' "${image}" | sha256sum | cut -d ' ' -f 1)"
lock_path="${lock_dir}/${tag_lock_name}.lock"
[[ ! -e "${lock_path}" || ( -f "${lock_path}" && ! -L "${lock_path}" ) ]] \
  || fail "shared-tag lock is unsafe"
exec {tag_lock_fd}> "${lock_path}"
flock -x "${tag_lock_fd}"

cas_outcome=""
if existing_id="$(inspect_image_id "${image}" "${context}/inspect-tag-before.txt")"; then
  [[ "${existing_id}" == "${image_id}" ]] \
    || fail "full-SHA tag already names different bytes"
  cas_outcome="existing-identical"
else
  inspect_status=$?
  [[ "${inspect_status}" -eq 1 ]] \
    || fail "existing full-SHA tag identity is malformed"
  docker image tag "${image_id}" "${image}"
  cas_outcome="assigned"
fi

tagged_id="$(inspect_image_id "${image}" "${context}/inspect-tag-after.txt")" \
  || fail "full-SHA tag could not be verified after compare-and-set"
[[ "${tagged_id}" == "${image_id}" ]] \
  || fail "full-SHA tag changed during compare-and-set"
final_iid="$(read_image_id_file "${iid_file}")"
final_built_id="$(inspect_image_id "${image_id}" "${context}/inspect-built-final.txt")" \
  || fail "built immutable image ID could not be reverified"
final_tagged_id="$(inspect_image_id "${image}" "${context}/inspect-tag-final.txt")" \
  || fail "full-SHA tag could not be reverified"
[[ "${final_iid}" == "${image_id}" \
  && "${final_built_id}" == "${image_id}" \
  && "${final_tagged_id}" == "${image_id}" ]] \
  || fail "image identity changed before receipt publication"

receipt_tmp="$(mktemp "${receipt_dir}/.${transaction_id}.receipt.XXXXXXXX")"
python3 - "${revision}" "${transaction_id}" "${image_id}" "${image}" "${cas_outcome}" \
  > "${receipt_tmp}" <<'PY'
import json
import sys

revision, transaction_id, image_id, intended_tag, cas_outcome = sys.argv[1:]
print(
    json.dumps(
        {
            "schema": "npa.robomimic.neutral-build-receipt.v1",
            "revision": revision,
            "transaction_id": transaction_id,
            "immutable_image_id": image_id,
            "consumer_image_ref": image_id,
            "intended_full_sha_tag": intended_tag,
            "tag_compare_and_set": cas_outcome,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
chmod 600 -- "${receipt_tmp}"
ln -- "${receipt_tmp}" "${receipt}" \
  || fail "transaction receipt publication refused"
rm -f -- "${receipt_tmp}"
receipt_tmp=""
[[ "$(stat -c '%u:%a:%h' -- "${receipt}")" == "$(id -u):600:1" ]] \
  || fail "transaction receipt is not owner-only"

echo "built local candidate ${image_id}; receipt ${receipt}; this helper does not push or publish"
