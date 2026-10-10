"""Validate operator inputs for a persistent Workbench control-plane endpoint."""

from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from .models import AbsolutePath, Contract, Name


class SetupRequest(Contract):
    """Select one server installation without requiring a personal client.

    Args:
        **data: Exact provider, Kubernetes, server, and HTTPS configuration.
    Returns:
        Validated operator setup request.
    Raises:
        ValueError: Endpoint, image, or cluster selection is invalid.
    """

    project_id: str = Field(min_length=1)
    cluster_id: str = Field(min_length=1)
    kubeconfig: AbsolutePath
    context: str = Field(min_length=1)
    namespace: Name
    service_name: Name = "npa-team"
    image: str = Field(pattern=r"^.+@sha256:[a-f0-9]{64}$")
    configuration_secret: Name
    state_claim: Name
    tls_secret: Name
    endpoint: str
    ca_file: AbsolutePath | None = None
    node_selector: dict[str, str] = Field(min_length=1)
    image_pull_secrets: tuple[Name, ...] = ()
    existing_service_uid: str = ""
    existing_deployment_uid: str = ""
    proxy_image: str = Field(
        default="caddy@sha256:f2a1290d0463aad60660d4ec134943f183ee2a5f6c3eb7bf32dd984f2f020772",
        pattern=r"^.+@sha256:[a-f0-9]{64}$",
    )

    @model_validator(mode="after")
    def validate_endpoint(self):
        """Require a plain HTTPS origin and immutable proxy image.

        Args:
            None.
        Returns:
            The validated request.
        Raises:
            ValueError: An endpoint or proxy image is unsafe or ambiguous.
        """
        endpoint = urlsplit(self.endpoint)
        if (
            endpoint.scheme != "https"
            or not endpoint.hostname
            or endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
            or endpoint.path not in ("", "/")
            or endpoint.port not in (None, 443)
        ):
            raise ValueError("setup endpoint must be an HTTPS origin on port 443")
        if "@sha256:" not in self.proxy_image:
            raise ValueError("the HTTPS proxy image must use an immutable digest")
        return self


def read_setup_request(path: Path) -> SetupRequest:
    """Read the operator's private installation settings.

    Args:
        path: Local YAML input; contains references, not personal bearer tokens.
    Returns:
        Validated setup request.
    Raises:
        TeamError: Input cannot be read or validated.
    """
    import yaml
    from .errors import TeamError

    try:
        return SetupRequest.model_validate(yaml.safe_load(path.read_text()))
    except (OSError, ValueError, yaml.YAMLError):
        raise TeamError("control-plane setup input is unreadable or invalid") from None
