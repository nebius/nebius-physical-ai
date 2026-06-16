# Private customer / operator pack

Customer runbooks, operator scripts, and live credentials do **not** belong in the public
`nebius/nebius-physical-ai` repository.

Use the private operator handoff repo instead:

- **GitHub:** https://github.com/timothy-le7/npa-sim2real-demo-walkthrough
- **Clone:** `git clone git@github.com:timothy-le7/npa-sim2real-demo-walkthrough.git ~/npa-sim2real-demo`

After clone: `cd ~/npa-sim2real-demo && ./setup.sh && ./run.sh`

Do not re-add `ops/private/sim2real-rtxpro/` to this public tree. Local-only files
(`env.local`, `*.local.md`) are gitignored.
