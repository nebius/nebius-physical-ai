"""Require behavioral regression tests to reject targeted reversed fixes."""
import ast
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import shutil

P=Path(__file__).parent; SOURCE=P/'combined/source';PYTHON=SOURCE/'npa/.venv/bin/python';HEAD=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
RUN_STATE='npa/src/npa/orchestration/npa_workflow/run_state.py'
SUPERVISOR='npa/src/npa/orchestration/npa_workflow/supervisor.py'
RESOLVER='npa/src/npa/orchestration/npa_workflow/run_resolution.py'
RECOVERY='npa/src/npa/orchestration/npa_workflow/launch_recovery.py'
TESTS='npa/tests/orchestration/npa_workflow/'

def replace_once(s, old, new):
    assert s.count(old)==1,(old,s.count(old));return s.replace(old,new,1)

def delayed_reuse(s):
    function=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='decide_recovery')
    node=next(n for n in function.body if isinstance(n,ast.If) and ast.unparse(n.test)=='context.outputs.all_valid')
    lines=s.splitlines(keepends=True);block=''.join(lines[node.lineno-1:node.end_lineno]);s=''.join(lines[:node.lineno-1]+lines[node.end_lineno:])
    anchor='    if not context.outputs.all_absent:';assert anchor in s
    return s.replace(anchor,block+anchor,1)

def skip_identity(s):
    function=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='decide_recovery')
    node=next(n for n in function.body if isinstance(n,ast.If) and 'context.outputs.all_valid' in ast.unparse(n.test) and '_immutable_identity_matches' in ast.unparse(n.test))
    lines=s.splitlines(keepends=True);return ''.join(lines[:node.lineno-1]+lines[node.end_lineno:])

CASES=[
 ('first-page-only',RUN_STATE,lambda s:replace_once(s,'        continuation_token = _prefix_next_token(response, seen_tokens)','        return False'), 'test_run_state.py','prefix_output_looks_past_zero_byte_marker'),
 ('marker-counts-present',RUN_STATE,lambda s:replace_once(s,'        if size > 0:','        if size >= 0:'),'test_run_state.py','prefix_output_requires_nonempty_content'),
 ('missing-flag-is-absence',RUN_STATE,lambda s:replace_once(s,'    truncated = response.get("IsTruncated")','    truncated = response.get("IsTruncated", False)'),'test_run_state.py','prefix_output_rejects_malformed_pagination'),
 ('falsey-records-are-absence',RUN_STATE,lambda s:replace_once(s,'    contents = response.get("Contents", [])','    contents = response.get("Contents") or []'),'test_run_state.py','prefix_listing_rejects_falsey_malformed_contents'),
 ('budget-before-reuse',SUPERVISOR,delayed_reuse,'test_supervisor.py','valid_outputs_are_reused_at_infrastructure_recovery_limit'),
 ('skip-live-cancellation',SUPERVISOR,lambda s:replace_once(s,'            decision.action is RecoveryAction.REUSE_COMPLETED_WAVE\n','            False and decision.action is RecoveryAction.REUSE_COMPLETED_WAVE\n'),'test_supervisor.py','live_valid_outputs_cancel_exact_attempt_before_reuse'),
 ('metadata-widens-artifacts',RESOLVER,lambda s:replace_once(s,'    uses_run_root_layout = (\n','    uses_run_root_layout = (\n        resolution.manifest_pending or resolution.workflow_name == PAIDF_WORKFLOW_NAME or\n'),'test_run_resolution.py','ambiguous_manifest_pending_prefix or terminal_ledger_workflow_name'),
 ('succeeded-identity-bypass',SUPERVISOR,skip_identity,'test_supervisor.py','succeeded_valid_outputs'),
 ('reserved-provider-overwrite',RECOVERY,lambda s:replace_once(s,'            attempt.sky_status = poll_result.provider_status','            attempt.sky_status = poll_result.workflow_status'),'test_partial_launch_recovery.py','reserved and reuse'),
]

def run_case(case):
    name,path,mutate,test,selector=case;root=P/'mutation-runs'/name;root.mkdir(parents=True,exist_ok=True)
    receipt=root/'result.json'
    if receipt.exists():
        result=json.loads(receipt.read_text());assert result['log_sha256']==hashlib.sha256((root/'pytest.log').read_bytes()).hexdigest();return result
    clone=root/'source'
    if clone.exists():
        subprocess.run(['git','-C',str(clone),'reset','--hard',HEAD],check=True,stdout=subprocess.DEVNULL)
    else: subprocess.run(['git','clone','--shared','--quiet','--no-checkout',str(SOURCE),str(clone)],check=True)
    subprocess.run(['git','-C',str(clone),'checkout','--quiet','--detach',HEAD],check=True)
    f=clone/path;original=f.read_text();changed=mutate(original);assert changed!=original;compile(changed,path,'exec');f.write_text(changed)
    env=dict(os.environ,PYTHONPATH=str(clone/'npa/src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');env['PATH']=str(P/'tools/git-tools/bin')+':'+str(PYTHON.parent)+':'+env['PATH']
    cmd=[str(PYTHON),'-m','pytest',TESTS+test,'-k',selector,'-q','--tb=short']
    proc=subprocess.run(cmd,cwd=clone,env=env,text=True,capture_output=True);data=proc.stdout+proc.stderr;(root/'pytest.log').write_text(data)
    assert proc.returncode==1 and re.search(r'\d+ failed',data),(name,proc.returncode,data[-3000:])
    summaries=[line for line in data.splitlines() if re.search(r'\d+ failed',line)]
    result=dict(name=name,status='killed',returncode=proc.returncode,source_file=path,source_sha256=hashlib.sha256(original.encode()).hexdigest(),mutation_sha256=hashlib.sha256(changed.encode()).hexdigest(),test=test,selector=selector,summary=summaries[-1],log_sha256=hashlib.sha256(data.encode()).hexdigest())
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results=list(pool.map(run_case,CASES))
(P/'mutation-proof.json').write_text(json.dumps(dict(source_head=HEAD,mutations=results,killed=len(results)),indent=2)+'\n')
print(json.dumps(dict(killed=len(results))))
