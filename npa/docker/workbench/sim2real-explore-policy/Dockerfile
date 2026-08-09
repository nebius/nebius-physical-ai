ARG BASE_IMAGE=npa-envgen:0.1.1
FROM ${BASE_IMAGE}

ARG NPA_SOURCE_SHA

LABEL org.opencontainers.image.title="npa-explore-policy"
LABEL org.opencontainers.image.description="Alternate BYO-compatible Sim2Real policy image (wider action exploration)"
LABEL org.opencontainers.image.revision="${NPA_SOURCE_SHA}"

RUN test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40

ENV POLICY_IMAGE_KIND=npa-explore-policy
ENV NPA_SIM2REAL_POLICY_VARIANT=explore
ENV NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}

USER ubuntu

CMD ["python", "-m", "npa.workflows.sim2real_envgen", "policy-contract", "--help"]
