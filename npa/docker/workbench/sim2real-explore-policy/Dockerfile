ARG BASE_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-envgen@sha256:55f1541c1a86e9d865963753cfa38fa1dbf88928c8d28d9ee30e2851a5604b88
FROM ${BASE_IMAGE}

ARG NPA_SOURCE_SHA

LABEL org.opencontainers.image.title="npa-explore-policy"
LABEL org.opencontainers.image.description="Alternate BYO-compatible Sim2Real policy image (wider action exploration)"
LABEL org.opencontainers.image.revision="${NPA_SOURCE_SHA}"

ENV POLICY_IMAGE_KIND=npa-explore-policy
ENV NPA_SIM2REAL_POLICY_VARIANT=explore
ENV NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}

USER root

# Preserve the policy variant while binding the complete running SDK to the
# child source revision, rather than only relabeling the inherited parent.
COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows

RUN test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && find /opt/npa/src /opt/npa/workflows -type d -exec chmod a+rx {} + \
    && find /opt/npa/src /opt/npa/workflows -type f -exec chmod a+r {} + \
    && chmod a+r /opt/npa/pyproject.toml \
    && python -m pip check \
    && env -u PYTHONPATH python -c \
      'from npa.workflows.sim2real_envgen import main; assert callable(main)'

USER ubuntu

CMD ["python", "-m", "npa.workflows.sim2real_envgen", "policy-contract", "--help"]
