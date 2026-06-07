ARG BASE_IMAGE=npa-sim2real-envgen:0.1.1
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="npa-sim2real-reference-policy"
LABEL org.opencontainers.image.description="Reference BYO-compatible Sim2Real action policy image"

ENV POLICY_IMAGE_KIND=npa-sim2real-reference-policy

CMD ["python", "-m", "npa.workflows.sim2real_envgen", "policy-contract", "--help"]
