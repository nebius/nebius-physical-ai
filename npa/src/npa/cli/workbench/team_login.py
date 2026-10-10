"""Guide personal-key and Nebius sign-in through the shared team session API."""

from pathlib import Path
import sys
import uuid

import typer

from npa.lifecycle_intent import json_stdout_contract
from npa.workbench.team.client import load_bearer_token
from npa.workbench.team.errors import TeamError


@json_stdout_contract
def login_cmd(
    endpoint: str | None = typer.Option(None, envvar="NPA_TEAM_ENDPOINT"),
    token_file: Path | None = typer.Option(
        None, help="Personal key supplied by your administrator."
    ),
    nebius: bool = typer.Option(
        False, "--nebius", help="Sign in with your human Nebius account."
    ),
    nebius_profile: str | None = typer.Option(
        None, help="Human Nebius CLI profile; preserves its global default."
    ),
    nebius_config: Path | None = typer.Option(
        None, help="Optional isolated Nebius CLI configuration."
    ),
    profile: str | None = typer.Option(
        None, help="Saved connection name; defaults to the active connection."
    ),
    workspace: str | None = typer.Option(None, help="Preferred authorized workspace."),
    cluster: str | None = typer.Option(None, help="Preferred allocated cluster."),
    ca_file: Path | None = typer.Option(
        None, help="Administrator-provided CA for a private HTTPS service."
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Print the official sign-in link instead of opening a browser.",
    ),
    ssh_host: str = typer.Option(
        "", help="When logging in on a VM, print its loopback callback forward."
    ),
    output_format: str = typer.Option(
        "text", help="text or json; credentials are never printed."
    ),
):
    """Verify access and remember a Workbench connection for subsequent commands.

    Args:
        endpoint, profile: Team HTTPS endpoint and saved connection name.
        token_file, nebius, nebius_profile, nebius_config: Personal key or Nebius sign-in.
        workspace, cluster: Optional placement defaults, verified against access.
        ca_file: Optional trusted private CA; TLS verification remains enabled.
        no_browser, ssh_host: Local-browser or remote callback instructions.
        output_format: Human text or credential-free JSON.
    Returns:
        None; prints verified access after saving the session.
    Raises:
        TeamError: Login, placement, configuration, or persistence fails.
    """
    from npa.workbench.team.sessions import SavedSession, SessionStore, login_session

    _output_format(output_format)
    profile = SessionStore().selected_profile(profile)
    endpoint, token_file, nebius, nebius_profile, nebius_config, ca_file = (
        _login_options(
            endpoint,
            token_file,
            nebius,
            nebius_profile,
            nebius_config,
            profile,
            ca_file,
        )
    )
    token = _credential(
        token_file, nebius_profile, nebius_config, no_browser, ssh_host, profile
    )
    session = SavedSession(
        endpoint=endpoint,
        auth_mode="nebius" if nebius else "local",
        nebius_profile=nebius_profile,
        nebius_config=str(nebius_config) if nebius_config else None,
        workspace=workspace,
        cluster=cluster,
        ca_file=str(Path(ca_file).expanduser().resolve()) if ca_file else None,
    )
    saved, access = login_session(session, token, profile=profile)
    _show_access(profile, saved, access, output_format)


def _login_options(
    endpoint, token_file, nebius, nebius_profile, nebius_config, profile, ca_file
):
    from npa.workbench.team.sessions import SessionStore

    if token_file and (nebius or nebius_profile or nebius_config):
        raise TeamError("Choose either --token-file or --nebius.")
    store = SessionStore()
    saved = store.load(profile) if store.exists(profile) else None
    endpoint = endpoint or (saved.endpoint if saved else None)
    if not endpoint:
        endpoint = _ask("Workbench HTTPS endpoint", "--endpoint")
    nebius = nebius or bool(nebius_profile) or bool(nebius_config)
    if not token_file and not nebius and saved and endpoint == saved.endpoint:
        if saved.auth_mode == "nebius":
            nebius, nebius_profile = True, saved.nebius_profile
            nebius_config = saved.nebius_config
        else:
            return endpoint, None, False, None, None, ca_file or saved.ca_file
    if not token_file and not nebius:
        choice = _ask("Sign-in method (key or nebius)", "--token-file or --nebius")
        if choice not in {"key", "nebius"}:
            raise TeamError("Choose key or nebius.")
        nebius = choice == "nebius"
        if not nebius:
            token_file = Path(_ask("Personal key file", "--token-file")).expanduser()
    if nebius and not nebius_profile:
        nebius_profile, nebius_config = _choose_nebius_profile(nebius_config)
    if saved and endpoint == saved.endpoint:
        ca_file = ca_file or saved.ca_file
    return endpoint, token_file, nebius, nebius_profile, nebius_config, ca_file


def _ask(label, flag):
    if not sys.stdin.isatty():
        raise TeamError(f"Supply {flag}, or run npa login in a terminal.")
    return typer.prompt(label, err=True).strip()


def _choose_nebius_profile(config_file):
    from npa.workbench.team.nebius_login import create_human_profile, list_profiles
    from npa.workbench.team.sessions import SessionStore

    profiles = [
        item.name
        for item in list_profiles(config_file=config_file)
        if item.auth_type == "federation"
    ]
    if len(profiles) == 1:
        return profiles[0], config_file
    if not profiles:
        config_file = config_file or (
            SessionStore().root / ("nebius-" + uuid.uuid4().hex + ".yaml")
        )
        create_human_profile("workbench", config_file)
        return "workbench", config_file
    typer.echo("Human Nebius profiles: " + ", ".join(profiles), err=True)
    selected = _ask("Nebius profile", "--nebius-profile")
    if selected not in profiles:
        raise TeamError("Choose one of the listed human Nebius profiles.")
    return selected, config_file


def _credential(
    token_file, nebius_profile, nebius_config, no_browser, ssh_host, profile
):
    if nebius_profile:
        from npa.workbench.team.nebius_login import login_token

        return login_token(
            nebius_profile,
            no_browser=no_browser,
            ssh_host=ssh_host,
            output=sys.stderr,
            config_file=nebius_config,
        )
    if token_file is None:
        from npa.workbench.team.sessions import SessionStore

        return SessionStore().token(profile)
    return load_bearer_token("NPA_TEAM_TOKEN", token_file)


def _output_format(value):
    if value not in {"text", "json"}:
        raise TeamError("--output-format must be text or json")


def _show_access(profile, session, access, output_format):
    import json

    result = {
        "profile": profile,
        "authentication": session.auth_mode,
        "workspace": session.workspace,
        "cluster": session.cluster,
        "access": access,
    }
    if output_format == "json":
        typer.echo(json.dumps(result))
        return
    typer.echo(f"Signed in. Saved Workbench connection: {profile}")
    for entry in access.get("workspaces", []):
        clusters = ", ".join(entry.get("clusters", {})) or "no execution allocation"
        typer.echo(
            f"  {entry['name']}: {entry['role']}; {clusters}; GPU cap {entry.get('gpu_limit')}"
        )
    if session.workspace and session.cluster:
        typer.echo("Next: npa workbench team submit --spec workflow.yaml")
    else:
        typer.echo(
            "Use npa workbench team whoami to inspect access and choose placement."
        )


@json_stdout_contract
def logout_cmd(
    profile: str | None = typer.Option(
        None, help="Connection to forget; defaults to the active connection."
    ),
    output_format: str = typer.Option("text", help="text or json."),
):
    """Forget a local Workbench connection without revoking accounts or cloud login.

    Args:
        profile: Optional saved connection name.
        output_format: Human text or JSON.
    Returns:
        None; prints whether a saved connection was removed.
    Raises:
        TeamError: Session storage cannot be safely updated.
    """
    import json
    from npa.workbench.team.sessions import SessionStore

    _output_format(output_format)
    removed = SessionStore().remove(profile)
    typer.echo(
        json.dumps({"logged_out": removed})
        if output_format == "json"
        else "Workbench connection forgotten. Existing jobs continue."
    )
