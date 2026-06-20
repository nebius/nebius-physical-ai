from npa.workflows.sim2real_rerun_serve import build_rerun_nginx_config, RERUN_HTPASSWD_PATH


def test_nginx_config_no_auth_by_default():
    cfg = build_rerun_nginx_config()
    assert "auth_basic" not in cfg


def test_nginx_config_basic_auth_when_required():
    cfg = build_rerun_nginx_config(auth_required=True)
    assert 'auth_basic "NPA Sim2Real Rerun";' in cfg
    assert f"auth_basic_user_file {RERUN_HTPASSWD_PATH};" in cfg
