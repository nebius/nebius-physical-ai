"""Check customer PAIDF media through the submission selectors without writing S3."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
from urllib.parse import urlparse

from botocore.exceptions import BotoCoreError, ClientError

from npa.errors import ScopedCredentialError
from npa.workflows import data_factory_input as inputs
from npa.workflows import paidf_cosmos3_media as media

_INPUT_ERRORS = (
    inputs.PaidfInputError,
    media.VideoAlignmentError,
    OSError,
    ValueError,
    BotoCoreError,
    ClientError,
    ScopedCredentialError,
)


def _storage(project: str, endpoint: str):
    from npa.clients.project_credentials import s3_client_for_project

    return SimpleNamespace(
        s3=s3_client_for_project(project or None, endpoint_url=endpoint)
    )


def _source(directory, client, input_video, input_uri, lerobot_uri, camera, episode):
    if input_video is not None:
        return input_video.expanduser().resolve()
    destination = directory / "selected-input.mp4"
    if lerobot_uri:
        inputs._materialize_lerobot_episode(
            client,
            lerobot_uri=lerobot_uri,
            camera=camera,
            episode=episode,
            explicit_selection=True,
            destination=destination,
        )
    else:
        location = urlparse(input_uri)
        client.s3.download_file(
            location.netloc, location.path.lstrip("/"), str(destination)
        )
    return destination


def _hint(message: str) -> str:
    text = message.lower()
    if "camera" in text or "video feature" in text:
        return "Use the exact video feature key from meta/info.json, such as observation.images.front."
    if "episode" in text or "timestamp" in text:
        return "Check the episode index and v3 per-camera chunk/file indices and from/to timestamps."
    if "codec" in text or "container" in text:
        return "Transcode a copy with ffmpeg -i INPUT -c:v libx264 -pix_fmt yuv420p OUTPUT.mp4."
    if "meta" in text or "lerobot" in text:
        return "Point to the dataset root containing meta/info.json; preserve meta/ and videos/ layout."
    if "ffmpeg" in text or "ffprobe" in text:
        return "Install FFmpeg and verify that ffmpeg and ffprobe are on PATH."
    return "Check input selection, storage read/list access, and full local video decoding."


def _inspect(directory, client, input_video, input_uri, lerobot_uri, camera, episode):
    source = _source(
        directory, client, input_video, input_uri, lerobot_uri, camera, episode
    )
    summary = inputs.probe_video(source)
    timeline = media.probe_video(source)
    return {
        "schema": "npa.paidf.input-check.v1",
        "status": "passed",
        "checks": ["selection", "metadata-and-codec", "full-decode-and-timeline"],
        "media": summary
        | {
            "decoded_frames": timeline["decoded_frames"],
            "full_decode_passed": timeline["full_decode_passed"],
        },
        "scope": "selected episode/camera or MP4; no generation or quality acceptance",
    }


def check_paidf_input(
    *,
    input_video: Path | None = None,
    input_uri: str = "",
    lerobot_uri: str = "",
    camera: str = "",
    episode: int | None = None,
    project: str = "",
    endpoint: str = "",
    storage_client=None,
) -> dict:
    """Validate one explicit customer input without upload, model access, or GPU use.

    Args:
        input_video: Local MP4, mutually exclusive with the S3 selectors.
        input_uri: One S3 MP4 object.
        lerobot_uri: S3 LeRobot dataset root.
        camera: Explicit full LeRobot video feature name.
        episode: Explicit non-negative LeRobot episode index.
        project: Saved project used to resolve storage credentials.
        endpoint: Optional S3 endpoint override.
        storage_client: Optional compatible client for controlled callers.
    Returns:
        Structured pass/fail evidence and a repair hint, excluding source identifiers.
    Raises:
        None: Expected input, storage, and media failures are reported in the result.
    """
    try:
        selection = (input_video, input_uri, lerobot_uri, camera, episode)
        return _check_selection(
            *selection,
            project,
            endpoint,
            storage_client,
        )
    except _INPUT_ERRORS as exc:
        return _failure(exc, input_video, input_uri, lerobot_uri)


def _check_selection(
    input_video, input_uri, lerobot_uri, camera, episode, project, endpoint, client
):
    _validate_selection(input_video, input_uri, lerobot_uri, camera, episode)
    if input_video is None and client is None:
        client = _storage(project, endpoint)
    with tempfile.TemporaryDirectory(prefix="npa-paidf-check-") as temporary:
        return _inspect(
            Path(temporary),
            client,
            input_video,
            input_uri,
            lerobot_uri,
            camera,
            episode,
        )


def _validate_selection(input_video, input_uri, lerobot_uri, camera, episode):
    if episode is not None and (type(episode) is not int or episode < 0):
        raise inputs.PaidfInputError("episode must be a non-negative integer")
    selection = inputs.select_paidf_input(
        input_video=input_video,
        input_uri=input_uri,
        lerobot_uri=lerobot_uri,
    )
    if selection == "starter":
        raise inputs.PaidfInputError("choose an explicit MP4 or LeRobot input to check")
    inputs.validate_lerobot_selector(
        selection=selection,
        camera=camera,
        episode=episode or 0,
        require_explicit_selection=bool(lerobot_uri),
        episode_was_explicit=episode is not None,
    )


def _failure(exc, input_video, input_uri, lerobot_uri):
    from npa.verification import sanitize_failure_reason

    message = _failure_message(exc)
    secrets = tuple(
        str(value) for value in (input_video, input_uri, lerobot_uri) if value
    )
    message = sanitize_failure_reason(message, secrets=secrets)
    return {
        "schema": "npa.paidf.input-check.v1",
        "status": "failed",
        "error": message,
        "hint": _hint(message),
    }


def _failure_message(exc):
    if isinstance(exc, ScopedCredentialError):
        return "Storage credentials are missing for the selected project; configure its own credential pair."
    if isinstance(exc, ClientError):
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {
            "AccessDenied",
            "403",
            "InvalidAccessKeyId",
            "SignatureDoesNotMatch",
        }:
            return "Storage access was denied; check project credentials and object-read/list permissions."
        if code in {"NoSuchKey", "NoSuchBucket", "404", "NotFound"}:
            return "The selected input or referenced dataset metadata/video is missing in storage."
        if code in {"ExpiredToken", "InvalidToken", "TokenRefreshRequired"}:
            return "Storage credentials expired or are invalid; refresh the selected project's credentials."
    if isinstance(exc, (inputs.PaidfInputError, media.VideoAlignmentError)):
        return str(exc)
    return "Input could not be read or decoded; check storage credentials, layout, and FFmpeg."
