#!/usr/bin/env bash
#
# Promote the validated runtime-fetch Isaac images onto their canonical tags.
#
# WHY THIS IS A SCRIPT AND NOT A COMMAND IN A PR BODY
# ---------------------------------------------------
# Promotion is the single irreversible-in-practice step of the runtime-fetch
# re-architecture, and it has an ordering constraint that is easy to get wrong. Encoding it
# here keeps the validated digests, the rollback path, and the sequencing rule in the repo
# rather than in a comment thread.
#
# THE ORDERING CONSTRAINT -- READ THIS BEFORE RUNNING
# ---------------------------------------------------
# Do NOT promote before the branch that carries the EULA plumbing has merged.
#
# The runtime-fetch images refuse to run (exit 78) unless the caller sets
# OMNI_KIT_ACCEPT_EULA and ISAACSIM_ACCEPT_EULA. Four separate layers had to be taught to
# forward that acceptance: the golden-eval runner, the shared serverless job-env builder,
# the SkyPilot templates, and the K8s sim2real Isaac sibling jobs.
#
# Until those land on the default branch, anyone running from it who pulls a canonical tag
# gets an image that their code does not know how to consent to. Promoting first would
# break every in-flight Isaac run in the fleet -- not subtly, but immediately and totally.
#
# Correct order:
#   1. merge the re-architecture (image changes AND the EULA plumbing) to the default branch
#   2. run this script
#   3. only then consider `npa.deploy.publish_public` with dry_run=false
#
# Step 3 matters because publish_public resolves CANONICAL tags. Publishing before step 2
# would push the old Omniverse-baked images to a public registry -- precisely the outcome
# this whole change exists to prevent.
#
# EVIDENCE BEHIND THE DIGESTS BELOW
# ---------------------------------
# Each NEW digest was: built from a non-proprietary base; scanned clean by
# npa/scripts/scan_image_omniverse_payload.py (627,823 filesystem entries across the five,
# zero Isaac/Omniverse payload); exercised on RTX PRO 6000 through the serverless
# golden-eval path; and, for isaac-lab, through the SkyPilot RTX PRO training smoke.
#
# Each OLD digest is recorded so rollback is one `crane copy` per image. Rollback is always
# safe: the old images bake Isaac, so they run without any acceptance plumbing.
#
# Usage:
#   ./npa/scripts/promote_isaac_rtfetch_tags.sh --dry-run          # default; prints only
#   ./npa/scripts/promote_isaac_rtfetch_tags.sh --i-have-sign-off  # actually promotes
#   ./npa/scripts/promote_isaac_rtfetch_tags.sh --rollback --i-have-sign-off
#
set -euo pipefail

EU_REGISTRY="${NPA_EU_REGISTRY:-cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw}"
US_REGISTRY="${NPA_US_REGISTRY:-cr.us-central1.nebius.cloud/u00j7q4jjkahvsx0jy}"

# repo | canonical tag | OLD digest (rollback target) | validated RC tag | NEW digest
IMAGES=(
  "npa-isaac-lab|2.3.2.post1|sha256:5aaa0a85b47dbaa6f1fb431fb747e7862c09edcff5dd260b951798518ec135cc|2.3.2.post1-rtfetch-rc3|sha256:c2e142c5312c903fd2a165028ca17e34335813d034ce036b936c6dc3a886bea8"
  "npa-sonic|0.1.2|sha256:47196e98951bd9fb840967718fb7be4f8730fc599195d329930cb1bbadb7de74|0.1.2-rtfetch-rc6|sha256:c00f25fb589ac8893ab64f4139b32fc0a48951f769dbdf8074109417e3bb449e"
  "npa-sonic|0.1.2-k8s-runtime|sha256:2c753ac476fc00688dcbc759e8ea3f066b1e0dc4efaaadac473a5a33a879c184|0.1.2-k8s-runtime-rtfetch-rc2|sha256:7bef8863cd6f8d66667b4d7cd89d261a5bc5479940f1077710c19759b2117339"
  "npa-sonic-mujoco|0.1.3-mvp|sha256:a3e79c99f01f6cd79ec2cdf4d55bf57b9a398b44833160886a58ffca9bbdd122|0.1.3-mvp-rtfetch-rc3|sha256:ab70bc167ad3665964470bf81ed9f23a1d3353426a0c355e8598097736e3d77b"
  "npa-groot|0.1.0|sha256:ce74758f5dd7becfa1cb17e45ae495b38f52c4eb026b6bb3e6b320225be20cd1|0.1.0-rtfetch-rc2|sha256:47fd6b727f249fbdb0ec237dc748c8bdc7cbf38474dc12c1cffe82f17fdde37b"
)

MODE="dry-run"
DIRECTION="promote"
for arg in "$@"; do
  case "$arg" in
    --dry-run) MODE="dry-run" ;;
    --i-have-sign-off) MODE="live" ;;
    --rollback) DIRECTION="rollback" ;;
    -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

command -v crane >/dev/null || { echo "crane is required" >&2; exit 1; }

if [[ "$MODE" == "live" ]]; then
  echo "!! LIVE MODE -- canonical tags will be rewritten in BOTH registries."
  echo "!! Confirm the EULA plumbing has merged to the default branch (see the header)."
  echo
fi

echo "direction: $DIRECTION    mode: $MODE"
echo

failures=0
for entry in "${IMAGES[@]}"; do
  IFS='|' read -r repo canonical old_digest rc_tag new_digest <<<"$entry"

  if [[ "$DIRECTION" == "promote" ]]; then
    source_ref="${EU_REGISTRY}/${repo}@${new_digest}"
    expect="$new_digest"
    label="$rc_tag"
  else
    source_ref="${EU_REGISTRY}/${repo}@${old_digest}"
    expect="$old_digest"
    label="pre-re-architecture"
  fi

  echo "=== ${repo}:${canonical}  <-  ${label}"

  # Verify the source digest still exists before touching anything. A promotion that
  # half-succeeds across two registries is worse than one that refuses to start.
  if ! crane manifest "$source_ref" >/dev/null 2>&1; then
    echo "    SKIP: source digest not found: $source_ref"
    failures=$((failures + 1))
    continue
  fi

  for registry in "$EU_REGISTRY" "$US_REGISTRY"; do
    target="${registry}/${repo}:${canonical}"
    if [[ "$MODE" == "dry-run" ]]; then
      echo "    would: crane copy ${source_ref} ${target}"
      continue
    fi
    echo "    crane copy -> ${target}"
    if ! crane copy "$source_ref" "$target"; then
      echo "    FAILED: $target"
      failures=$((failures + 1))
      continue
    fi
    actual="$(crane digest --platform linux/amd64 "$target" 2>/dev/null || echo MISSING)"
    if [[ "$actual" == "$expect" ]]; then
      echo "    verified: $actual"
    else
      echo "    MISMATCH: expected $expect, got $actual"
      failures=$((failures + 1))
    fi
  done
  echo
done

if [[ "$MODE" == "dry-run" ]]; then
  echo "Dry run. Re-run with --i-have-sign-off to apply."
  echo "Rollback at any time: $0 --rollback --i-have-sign-off"
  exit 0
fi

if (( failures > 0 )); then
  echo "COMPLETED WITH ${failures} FAILURE(S) -- registries may be inconsistent."
  echo "Roll back with: $0 --rollback --i-have-sign-off"
  exit 1
fi

echo "All canonical tags updated and verified at digest parity across both registries."
[[ "$DIRECTION" == "promote" ]] && echo "Rollback: $0 --rollback --i-have-sign-off"
exit 0
