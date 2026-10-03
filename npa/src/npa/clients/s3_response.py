"""Keep S3 error status parsing separate from successful payload checksums."""

from botocore.httpchecksum import StreamingChecksumBody


def _normalize_checksum_error_body(response_dict, **_kwargs):
    body = response_dict.get("body")
    if int(response_dict.get("status_code", 0)) < 300:
        return
    if not isinstance(body, StreamingChecksumBody):
        return
    # botocore's streaming checksum adapter can wrap an error after the HTTP
    # adapter has already consumed its XML bytes. RestXMLParser requires bytes,
    # not StreamingChecksumBody. Preserve the trusted HTTP status and standard
    # retry/error parsing, without treating a stored object's checksum as an
    # error-body checksum or changing successful response validation.
    response_dict["body"] = b""
    try:
        body.close()
    except (OSError, RuntimeError):
        pass


def register_s3_error_body_compat(client):
    """Normalize only SDK-wrapped S3 GetObject errors, not successful bodies.

    Args:
        client: A boto3 S3 client, or a test transport without event support.
    Returns:
        The unchanged client with an idempotently registered response adapter.
    Raises:
        None for ordinary clients.
    """
    events = getattr(getattr(client, "meta", None), "events", None)
    if events is not None:
        events.register(
            "before-parse.s3.GetObject",
            _normalize_checksum_error_body,
            unique_id="npa-s3-getobject-error-body-compat",
        )
    return client
