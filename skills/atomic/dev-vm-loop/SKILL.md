---
name: dev-vm-loop
description: Bootstrap isolated dev-VM worktrees and launch cursor-loop sessions with push/PR instructions.
---

# Dev VM Loop

Use this when an operator asks you to run a job on a shared dev VM (not in your own VM) with an isolated branch/worktree and continuous retry loop.

Run this command from the repository root:

```bash
scripts/dev-loop.sh \
  --name <unique-name> \
  --session <unique-session> \
  --job "<job description>" \
  --success "<success command>"
```

Environment variables required in the caller environment:

- `DEV_VM_SSH_PRIVATE_KEY`
- `DEV_VM_SSH_USER`
- `DEV_VM_SSH_HOST`

What this command enforces on the dev VM:

1. SSH using the provided key and host/user secrets.
2. `cd ~/nebius-physical-ai && git fetch origin`
3. Creates isolated branch/worktree:
   - branch: `agent/<unique-name>`
   - worktree: `~/work/<unique-name>`
   - base: `origin/main`
4. Launches `cursor-loop` with:
   - job prompt text
   - explicit commit-as-you-go instruction
   - explicit final push + PR instruction:
     `git push -u origin $BR && gh pr create --fill --base main --head $BR`
   - gh fallback: print compare URL if `gh` is unavailable
5. Prints branch/worktree/session/log paths for monitoring.

If SSH resets before banner, treat it as a dev VM ingress issue and stop.

Monitor progress:

```bash
ssh -i ~/.ssh/k -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new "$DEV_VM_SSH_USER@$DEV_VM_SSH_HOST" \
  "tail -f /tmp/cursor-loop-<session>.log"
```
