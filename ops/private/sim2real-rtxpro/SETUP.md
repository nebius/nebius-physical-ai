# Sim2Real Setup — Start Here

**Quick start:** [QUICKSTART.md](./QUICKSTART.md) (Mac + Linux, copy-paste)

**Customer handoff:** [CUSTOMER-HANDOFF.md](./CUSTOMER-HANDOFF.md)

**Agent / maintainer:** [AGENT-HANDOFF.md](./AGENT-HANDOFF.md)

---

## Quick reference

| Phase | Command |
| --- | --- |
| First time | `bash ops/private/sim2real-rtxpro/first-time-setup.sh` |
| Edit secrets | `~/npa-sim2real-demo/private/` |
| Seed demo data | `cd ~/npa-sim2real-demo && ./run.sh seed-trigger` |
| Run pipeline | `cd ~/npa-sim2real-demo && ./run.sh demo` |
| Monitor | `./run.sh status <RUN_ID>` |
| Results | `./run.sh sync <RUN_ID>` |

Python / npa:

```text
~/npa-sim2real-demo/nebius-physical-ai/npa/.venv/bin/npa
```

Create venv manually:

```bash
bash ops/private/sim2real-rtxpro/bootstrap-npa-venv.sh ~/npa-sim2real-demo/nebius-physical-ai
```

Verify scripts (no cluster):

```bash
bash ops/private/sim2real-rtxpro/verify-paste-demo.sh
```
