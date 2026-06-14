# Safe Branch Cleanup Candidates

Date: 2026-06-14

Scope: remote branches already fully merged into `origin/main`.

## Safe-to-delete candidates (pending maintainer approval)

- `origin/dev-refactor-move-npa-workbench-npa`
- `origin/dev-refactor-move-npa-workbench-npa-c226`

## Safety checks used

- Included only branches reported by:
  - `git for-each-ref --format='%(refname:short)' refs/remotes/origin --merged origin/main`
- Excluded protected/default refs:
  - `origin/main`, `origin/HEAD`

## Important

No remote branches were deleted as part of this change. Deletion should happen only after PR review and explicit approval.
