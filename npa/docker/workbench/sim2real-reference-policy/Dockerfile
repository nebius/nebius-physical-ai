ARG BASE_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-envgen@sha256:08eb75118f5a04194d33a60308212db7706dd9c339d74afc5471a58608bf0422
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="npa-reference-policy"
LABEL org.opencontainers.image.description="Reference BYO-compatible Sim2Real action policy image"
LABEL npa.base.image="npa-envgen"

ENV POLICY_IMAGE_KIND=npa-reference-policy
ENV NPA_SIM2REAL_POLICY_VARIANT=reference

COPY --chown=ubuntu:ubuntu src/npa/smoke/test_sim2real_policy_functional.py /opt/npa/src/npa/smoke/test_sim2real_policy_functional.py

USER ubuntu

CMD ["python", "-m", "npa.workflows.sim2real_envgen", "policy-contract", "--help"]
