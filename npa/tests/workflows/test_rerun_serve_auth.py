from npa.workflows.rerun_serve import build_rerun_nginx_config, RERUN_HTPASSWD_PATH
import bcrypt
import pytest


def test_nginx_config_no_auth_by_default():
    cfg = build_rerun_nginx_config()
    assert "auth_basic" not in cfg


def test_nginx_config_basic_auth_when_required():
    cfg = build_rerun_nginx_config(auth_required=True)
    assert 'auth_basic "NPA Rerun";' in cfg
    assert f"auth_basic_user_file {RERUN_HTPASSWD_PATH};" in cfg


def _items(manifest):
    return manifest["items"]


def test_manifest_no_auth_when_no_password():
    from npa.workflows.rerun_serve import RerunServeConfig, build_rerun_serve_manifest
    cfg = RerunServeConfig(run_id="sim2real-staged-20260620t010101z", s3_bucket="b", name="npa-rerun")
    m = build_rerun_serve_manifest(cfg)
    kinds = [(i["kind"], i["metadata"]["name"]) for i in _items(m)]
    assert ("Secret", "npa-rerun-auth") not in kinds
    cm = next(i for i in _items(m) if i["kind"] == "ConfigMap")
    assert "auth_basic" not in cm["data"]["nginx.conf"]


def test_manifest_basic_auth_secret_volume_mount_when_enabled():
    from npa.workflows.rerun_serve import RerunServeConfig, build_rerun_serve_manifest
    cfg = RerunServeConfig(run_id="sim2real-staged-20260620t010101z", s3_bucket="b",
                           name="npa-rerun", auth_user="demo", auth_password="s3cret-pw")
    assert cfg.auth_enabled
    assert cfg.htpasswd_line.startswith("demo:$2b$12$")
    m = build_rerun_serve_manifest(cfg)
    names = [(i["kind"], i["metadata"]["name"]) for i in _items(m)]
    assert ("Secret", "npa-rerun-auth") in names
    cm = next(i for i in _items(m) if i["kind"] == "ConfigMap")
    assert 'auth_basic "NPA Rerun";' in cm["data"]["nginx.conf"]
    dep = next(i for i in _items(m) if i["kind"] == "Deployment")
    spec = dep["spec"]["template"]["spec"]
    vols = [v["name"] for v in spec["volumes"]]
    assert "nginx-auth" in vols
    nginx = next(c for c in spec["containers"] if c["name"] == "nginx")
    mounts = [vm["mountPath"] for vm in nginx["volumeMounts"]]
    assert "/etc/nginx/auth" in mounts


def test_nginx_healthz_unauthed_and_probe_uses_it():
    from npa.workflows.rerun_serve import RerunServeConfig, build_rerun_serve_manifest, build_rerun_nginx_config
    cfg = build_rerun_nginx_config(auth_required=True)
    assert "location = /healthz" in cfg and "auth_basic off;" in cfg
    c = RerunServeConfig(run_id="sim2real-staged-20260620t010101z", s3_bucket="b", name="npa-rerun",
                         auth_user="demo", auth_password="pw")
    m = build_rerun_serve_manifest(c)
    dep = next(i for i in m["items"] if i["kind"] == "Deployment")
    nginx = next(x for x in dep["spec"]["template"]["spec"]["containers"] if x["name"] == "nginx")
    assert nginx["readinessProbe"]["httpGet"]["path"] == "/healthz"


def test_password_hash_uses_unique_salts_and_rejects_wrong_password():
    from npa.workflows.rerun_serve import RerunServeConfig

    cfg = RerunServeConfig(run_id="recording", s3_bucket="bucket",
                          auth_user="viewer", auth_password="correct password")
    first = cfg.htpasswd_line.partition(":")[2].strip().encode()
    second = cfg.htpasswd_line.partition(":")[2].strip().encode()
    assert first != second
    assert bcrypt.checkpw(b"correct password", first)
    assert not bcrypt.checkpw(b"incorrect password", first)


@pytest.mark.parametrize("user,password", [
    ("viewer", ""), ("", "password"), ("viewer\nother", "password"),
    ("viewer:injected", "password"), ("viewer", "x" * 73),
    ("viewer", "nul\x00suffix"), ("viewer", "\N{SNOWMAN}" * 25),
])
def test_invalid_authentication_cannot_disable_or_inject_password_file(user, password):
    from npa.workflows.rerun_serve import RerunServeConfig, RerunServeError

    with pytest.raises(RerunServeError):
        RerunServeConfig(run_id="recording", s3_bucket="bucket",
                        auth_user=user, auth_password=password)


@pytest.mark.parametrize("service_type", ["LoadBalancer", "NodePort", "lb"])
def test_direct_public_http_exposure_is_rejected_before_manifest(service_type):
    from npa.workflows.rerun_serve import RerunServeConfig, RerunServeError, build_rerun_serve_manifest

    config = RerunServeConfig(run_id="recording", s3_bucket="bucket",
                             service_type=service_type, auth_user="viewer", auth_password="password")
    with pytest.raises(RerunServeError, match="private ClusterIP"):
        build_rerun_serve_manifest(config)


def test_recording_has_same_auth_boundary_and_no_raw_service_port():
    from npa.workflows.rerun_serve import RerunServeConfig, build_rerun_serve_manifest

    config = RerunServeConfig(run_id="recording", s3_bucket="bucket",
                             auth_user="viewer", auth_password="password")
    items = build_rerun_serve_manifest(config)["items"]
    service = next(item for item in items if item["kind"] == "Service")
    assert service["spec"]["ports"] == [{"name": "http", "port": config.port, "targetPort": "http"}]
    nginx_config = next(item for item in items if item["kind"] == "ConfigMap")["data"]["nginx.conf"]
    recording_location = nginx_config.split("location = /recording.rrd {", 1)[1].split("}", 1)[0]
    assert "alias /data/sim2real.rrd;" in recording_location
    assert 'auth_basic "NPA Rerun";' in recording_location
    pod = next(item for item in items if item["kind"] == "Deployment")["spec"]["template"]["spec"]
    rerun = next(container for container in pod["containers"] if container["name"] == "rerun")
    assert "--bind 127.0.0.1" in rerun["command"][-1]
    assert "ports" not in rerun
    assert "exec" in rerun["readinessProbe"]
    nginx = next(container for container in pod["containers"] if container["name"] == "nginx")
    assert {"name": "rrd-data", "mountPath": "/data", "readOnly": True} in nginx["volumeMounts"]
