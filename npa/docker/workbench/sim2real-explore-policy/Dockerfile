ARG BASE_IMAGE=npa-sim2real-envgen:0.1.1
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="npa-sim2real-explore-policy"
LABEL org.opencontainers.image.description="Alternate BYO-compatible Sim2Real policy image (wider action exploration)"

ENV POLICY_IMAGE_KIND=npa-sim2real-explore-policy
ENV NPA_SIM2REAL_POLICY_VARIANT=explore

USER ubuntu

CMD ["python", "-m", "npa.workflows.sim2real_envgen", "policy-contract", "--help"]
