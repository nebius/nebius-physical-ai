#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/dev-loop.sh --name <unique-name> --session <session> --job "<job>" --success "<success-cmd>"

Required environment variables:
  DEV_VM_SSH_PRIVATE_KEY
  DEV_VM_SSH_USER
  DEV_VM_SSH_HOST

Example:
  scripts/dev-loop.sh \
    --name sim2real-fix-a1 \
    --session sim2real-fix-a1 \
    --job "Fix failing sim2real validation in npa/workflows/..." \
    --success "cd ~/work/sim2real-fix-a1 && npa/.venv/bin/python -m pytest -q npa/tests/..."
EOF
}

die() {
  echo "dev-loop: $*" >&2
  exit 1
}

for required in DEV_VM_SSH_PRIVATE_KEY DEV_VM_SSH_USER DEV_VM_SSH_HOST; do
  [[ -n "${!required:-}" ]] || die "missing required env var: ${required}"
done

NAME=""
SESSION=""
JOB=""
SUCCESS_CMD=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      NAME="${2:-}"
      shift 2
      ;;
    --session)
      SESSION="${2:-}"
      shift 2
      ;;
    --job)
      JOB="${2:-}"
      shift 2
      ;;
    --success)
      SUCCESS_CMD="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1 (use --help)"
      ;;
  esac
done

[[ -n "${NAME}" ]] || die "--name is required"
[[ -n "${SESSION}" ]] || die "--session is required"
[[ -n "${JOB}" ]] || die "--job is required"
[[ -n "${SUCCESS_CMD}" ]] || die "--success is required"

[[ "${NAME}" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || die "--name must match ^[a-z0-9][a-z0-9._-]*$"
[[ "${SESSION}" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || die "--session must match ^[a-z0-9][a-z0-9._-]*$"

mkdir -p "${HOME}/.ssh"
KEY_PATH="${HOME}/.ssh/k"
printf '%s\n' "${DEV_VM_SSH_PRIVATE_KEY}" > "${KEY_PATH}"
chmod 600 "${KEY_PATH}"

SSH_TARGET="${DEV_VM_SSH_USER}@${DEV_VM_SSH_HOST}"
SSH_OPTS=(-i "${KEY_PATH}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
SSH_LOG="$(mktemp)"

set +e
REMOTE_OUTPUT="$(
  ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" \
    "bash -s" -- "${NAME}" "${SESSION}" "${JOB}" "${SUCCESS_CMD}" 2>&1 <<'EOF'
set -euo pipefail

NAME="$1"
SESSION="$2"
JOB="$3"
SUCCESS_CMD="$4"

REPO="${HOME}/nebius-physical-ai"
WORK_ROOT="${HOME}/work"
BR="agent/${NAME}"
WT="${WORK_ROOT}/${NAME}"

if ! command -v cursor-loop >/dev/null 2>&1; then
  echo "cursor-loop is not installed on the dev VM."
  exit 12
fi

cd "${REPO}"
git fetch origin

if tmux has-session -t "=${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  exit 13
fi

if [[ -e "${WT}" ]]; then
  echo "worktree already exists: ${WT}"
  exit 14
fi

if git show-ref --verify --quiet "refs/heads/${BR}"; then
  echo "branch already exists locally: ${BR}"
  exit 15
fi

if git ls-remote --exit-code origin "${BR}" >/dev/null 2>&1; then
  echo "branch already exists on origin: ${BR}"
  exit 16
fi

mkdir -p "${WORK_ROOT}"
git worktree add -b "${BR}" "${WT}" origin/main

PROMPT="$(cat <<PROMPT_EOF
${JOB}

Commit each logical change with clear commit messages. When done and SUCCESS_CMD passes, push the branch and open a PR:
git push -u origin ${BR} && gh pr create --fill --base main --head ${BR}
(if gh is unavailable, push and print the PR compare URL)

Use a unique run ID / S3 prefix / workbench namespace, and do not touch other agents' sessions, worktrees, or infrastructure.
PROMPT_EOF
)"

cursor-loop "${SESSION}" "${PROMPT}" "${SUCCESS_CMD}" --workspace "${WT}"

echo "branch=${BR}"
echo "worktree=${WT}"
echo "session=${SESSION}"
echo "log=/tmp/cursor-loop-${SESSION}.log"
EOF
)"
SSH_EXIT=$?
set -e

printf '%s\n' "${REMOTE_OUTPUT}" | tee "${SSH_LOG}" >/dev/null

if [[ ${SSH_EXIT} -ne 0 ]]; then
  if [[ "${REMOTE_OUTPUT}" == *"Connection reset by peer"* ]] || \
     [[ "${REMOTE_OUTPUT}" == *"kex_exchange_identification"* ]] || \
     [[ "${REMOTE_OUTPUT}" == *"Connection closed by remote host"* ]]; then
    die "SSH reset before banner; this is likely a dev VM ingress issue."
  fi
  cat "${SSH_LOG}" >&2
  exit "${SSH_EXIT}"
fi

cat "${SSH_LOG}"
echo "monitor: ssh ${SSH_OPTS[*]} ${SSH_TARGET} \"tail -f /tmp/cursor-loop-${SESSION}.log\""
