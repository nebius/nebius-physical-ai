# Sim2Real — Customer Handoff

**Start here:** **[CUSTOMER-RUNBOOK.md](./CUSTOMER-RUNBOOK.md)** (full pipeline: install → 800-env industrial demo → customize)

Quick paste path: **[QUICKSTART.md](./QUICKSTART.md)**

---

## Public vs private

| In public git | On your machine only |
| --- | --- |
| Scripts under `ops/private/sim2real-rtxpro/` | `~/npa-sim2real-demo/private/config.yaml` |
| Templates `private/*.example` | `~/npa-sim2real-demo/private/credentials.yaml` |
| Documentation | `~/npa-sim2real-demo/private/clusters/*/kubeconfig` |

Never commit credentials or infra IDs to the public repo.

---

## Linux manual scaffold

```bash
git clone --branch feat/sim2real-mandatory-stages \
  https://github.com/nebius/nebius-physical-ai.git ~/npa-sim2real-demo/nebius-physical-ai
bash ~/npa-sim2real-demo/nebius-physical-ai/ops/private/sim2real-rtxpro/first-time-setup.sh
```

Then follow QUICKSTART.md for credentials and daily `./run.sh` commands.
