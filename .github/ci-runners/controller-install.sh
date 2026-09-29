#!/usr/bin/env bash
# Install trusted controller code and metadata authentication without operator tokens.
set -euo pipefail
umask 077
test "$(id -u)" -eq 0
export HOME=/var/lib/npa-ci-runners
export PATH=/usr/local/bin:/usr/bin:/bin
unset NEBIUS_IAM_TOKEN NEBIUS_IAM_TOKEN_FILE GH_TOKEN GITHUB_TOKEN
controller_source=/opt/npa-ci-controller
install -d -m 0700 "$HOME"
exec 9>"$HOME/controller.lock"
flock --nonblock 9
test -s /mnt/cloud-metadata/token
test -d "$controller_source/state"
rm -rf "$HOME/workers"
cp -a "$controller_source/state/." "$HOME/"
rm -rf "$controller_source/state"
python3 -m venv "$controller_source/npa/.venv"

download=$(mktemp)
trap 'rm -f "$download"' EXIT
curl --fail --location --retry 5 --proto '=https' --tlsv1.2 \
  https://storage.eu-north1.nebius.cloud/cli/release/0.12.254/linux/x86_64/nebius \
  --output "$download"
printf '%s  %s\n' 928c943ae517123e284a39a3695589c7621b2fd032cdbeb2e3169a9eb22cda5c \
  "$download" | sha256sum --check
install -m 0755 "$download" /usr/local/bin/nebius
project_id=$(/opt/npa-ci-controller/npa/.venv/bin/python -c \
  'import json; print(json.load(open("/var/lib/npa-ci-runners/config.json"))["project_id"])')
profile_command=create
if [ -f "$HOME/.nebius/config.yaml" ]; then
  profile_command=update
fi
nebius profile "$profile_command" ci-controller --endpoint api.eu.nebius.cloud \
  --token-file /mnt/cloud-metadata/token --parent-id "$project_id" --skip-auth >/dev/null
# Prove the attached identity directly, without the profile wizard's prompts.
nebius --profile ci-controller iam project get --id "$project_id" >/dev/null
install -m 0755 "$controller_source/.github/ci-runners/controller-command.sh" \
  /usr/local/sbin/npa-ci-controller
install -m 0644 "$controller_source/.github/ci-runners/npa-ci-runners.service" \
  /etc/systemd/system/npa-ci-runners.service
systemctl daemon-reload
systemctl enable npa-ci-runners.service
echo 'Controller installed. Start it only after the previous controller is stopped and App access is verified.'
