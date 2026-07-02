#!/usr/bin/env bash
# Rewrite main so each merged PR is one commit (no merge bubbles / branch noise).
# Tree at HEAD is unchanged; only history is linearized.
#
# Usage:
#   ./scripts/squash-main-merge-commits.sh [BASE_COMMIT] [TARGET_COMMIT]
#
# Defaults:
#   BASE   = merge commit for PR #151 (0e3694b)
#   TARGET = current origin/main
#
# After running, verify then force-push (main is protected — needs admin/temporary unprotect):
#   git diff origin/main HEAD   # must be empty
#   git push --force-with-lease origin main

set -euo pipefail

BASE="${1:-0e3694b}"
TARGET="${2:-origin/main}"

git fetch origin main

BACKUP="backup/main-pre-squash-$(date +%Y%m%d-%H%M%S)"
WORK="history-squash-work-$$"

cleanup() {
  git checkout main >/dev/null 2>&1 || true
  git branch -D "$WORK" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Base:   $BASE"
echo "Target: $TARGET"
echo "Backup branch will be: $BACKUP"

git branch "$BACKUP" main
git checkout -b "$WORK" "$BASE"

# First-parent merge commits on main since BASE (oldest → newest).
MERGES=$(git log --reverse --first-parent --merges --format=%H "$BASE".."$TARGET")

for merge in $MERGES; do
  echo "Squashing merge $merge ..."
  git cherry-pick -m 1 --no-commit "$merge"
  git commit -C "$merge"
done

# Non-merge commits at the tip (e.g. squash-merged PRs).
NON_MERGES=$(git log --reverse --first-parent --no-merges --format=%H "$BASE".."$TARGET")
for commit in $NON_MERGES; do
  # Skip commits already absorbed by a merge replay above.
  if git merge-base --is-ancestor "$commit" HEAD 2>/dev/null; then
    continue
  fi
  echo "Cherry-picking $commit ..."
  git cherry-pick --no-commit "$commit"
  git commit -C "$commit"
done

if ! git diff --quiet "$TARGET" HEAD; then
  echo "ERROR: tree differs from $TARGET — aborting." >&2
  git diff --stat "$TARGET" HEAD
  exit 1
fi

OLD_COUNT=$(git rev-list --count "$BASE".."$TARGET")
NEW_COUNT=$(git rev-list --count "$BASE"..HEAD)
echo "OK: $OLD_COUNT commits → $NEW_COUNT commits (tree identical)"

git checkout main
git reset --hard "$WORK"
git branch -D "$WORK"

echo ""
echo "main updated locally. Review:"
echo "  git log --oneline ${BASE}..main"
echo ""
echo "Force-push when ready:"
echo "  git push --force-with-lease origin main"
