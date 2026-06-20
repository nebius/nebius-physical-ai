from npa.workflows.sim2real_rerun_serve import build_rerun_nginx_config, RERUN_HTPASSWD_PATH


def test_nginx_config_no_auth_by_default():
    cfg = build_rerun_nginx_config()
    assert "auth_basic" not in cfg


def test_nginx_config_basic_auth_when_required():
    cfg = build_rerun_nginx_config(auth_required=True)
    assert 'auth_basic "NPA Sim2Real Rerun";' in cfg
    assert f"auth_basic_user_file {RERUN_HTPASSWD_PATH};" in cfg


def _items(manifest):
    return manifest["items"]


def test_manifest_no_auth_when_no_password():
    from npa.workflows.sim2real_rerun_serve import RerunServeConfig, build_rerun_serve_manifest
    cfg = RerunServeConfig(run_id="sim2real-staged-20260620t010101z", s3_bucket="b", name="npa-rerun")
    m = build_rerun_serve_manifest(cfg)
    kinds = [(i["kind"], i["metadata"]["name"]) for i in _items(m)]
    assert ("Secret", "npa-rerun-auth") not in kinds
    cm = next(i for i in _items(m) if i["kind"] == "ConfigMap")
    assert "auth_basic" not in cm["data"]["nginx.conf"]


def test_manifest_basic_auth_secret_volume_mount_when_enabled():
    from npa.workflows.sim2real_rerun_serve import RerunServeConfig, build_rerun_serve_manifest
    cfg = RerunServeConfig(run_id="sim2real-staged-20260620t010101z", s3_bucket="b",
                           name="npa-rerun", auth_user="demo", auth_password="s3cret-pw")
    assert cfg.auth_enabled
    assert cfg.htpasswd_line.startswith("demo:{SHA}")
    m = build_rerun_serve_manifest(cfg)
    names = [(i["kind"], i["metadata"]["name"]) for i in _items(m)]
    assert ("Secret", "npa-rerun-auth") in names
    cm = next(i for i in _items(m) if i["kind"] == "ConfigMap")
    assert 'auth_basic "NPA Sim2Real Rerun";' in cm["data"]["nginx.conf"]
    dep = next(i for i in _items(m) if i["kind"] == "Deployment")
    spec = dep["spec"]["template"]["spec"]
    vols = [v["name"] for v in spec["volumes"]]
    assert "nginx-auth" in vols
    nginx = next(c for c in spec["containers"] if c["name"] == "nginx")
    mounts = [vm["mountPath"] for vm in nginx["volumeMounts"]]
    assert "/etc/nginx/auth" in mounts


def test_nginx_healthz_unauthed_and_probe_uses_it():
    from npa.workflows.sim2real_rerun_serve import RerunServeConfig, build_rerun_serve_manifest, build_rerun_nginx_config
    cfg = build_rerun_nginx_config(auth_required=True)
    assert "location = /healthz" in cfg and "auth_basic off;" in cfg
    c = RerunServeConfig(run_id="sim2real-staged-20260620t010101z", s3_bucket="b", name="npa-rerun",
                         auth_user="demo", auth_password="pw")
    m = build_rerun_serve_manifest(c)
    dep = next(i for i in m["items"] if i["kind"] == "Deployment")
    nginx = next(x for x in dep["spec"]["template"]["spec"]["containers"] if x["name"] == "nginx")
    assert nginx["readinessProbe"]["httpGet"]["path"] == "/healthz"
