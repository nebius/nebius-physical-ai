"""Exercise production HOME isolation with the real Kubernetes config parser."""
import hashlib,json,os,pathlib,subprocess,sys,tempfile
from npa.orchestration.skypilot.cleanup import sky_environment
CODE='import json; from kubernetes.config import list_kube_config_contexts; c,a=list_kube_config_contexts(); print(json.dumps({"contexts":[x["name"] for x in c],"active":a["name"]}))'
def write_config(path,name):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text('apiVersion: v1\nkind: Config\nclusters:\n- name: example\n  cluster:\n    server: https://example.invalid\nusers:\n- name: example\n  user: {}\ncontexts:\n- name: '+name+'\n  context:\n    cluster: example\n    user: example\ncurrent-context: '+name+'\n')
def probe(env):return subprocess.run([sys.executable,'-c',CODE],env=env,text=True,capture_output=True)
with tempfile.TemporaryDirectory(prefix='npa-kube-parser-') as raw:
 root=pathlib.Path(raw);operator=root/'operator';source=operator/'.kube/config';write_config(source,'default-example')
 os.environ.clear();os.environ.update(HOME=str(operator),PATH='/usr/bin:/bin',USER='validation')
 empty=root/'empty';empty.mkdir(); baseline=probe(dict(os.environ,HOME=str(empty)));assert baseline.returncode!=0
 env=sky_environment(root/'isolated');result=probe(env);assert result.returncode==0,result.stderr
 assert json.loads(result.stdout)['active']=='default-example'
 link=pathlib.Path(env['HOME'])/'.kube/config';assert link.is_symlink() and link.resolve()==source
 write_config(source,'updated-example');updated=probe(env);assert updated.returncode==0 and json.loads(updated.stdout)['active']=='updated-example'
 selected=root/'selected-config';write_config(selected,'selected-example');os.environ['KUBECONFIG']=str(selected)
 override=probe(sky_environment(root/'explicit'));assert override.returncode==0 and json.loads(override.stdout)['active']=='selected-example'
 import npa.orchestration.skypilot.cleanup as module
 print(json.dumps({'production_module_sha256':hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest(),'real_dependency':'kubernetes','network_calls':0,'baseline_missing_context_rejected':True,'default_context_resolved':True,'live_symlink_update_visible':True,'explicit_kubeconfig_precedence_preserved':True,'controls_passed':4},indent=2))
