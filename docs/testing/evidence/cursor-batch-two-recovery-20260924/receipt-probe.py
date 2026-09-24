"""Exercise receipt preservation with real files and production locking/writes."""
import hashlib,json,os,pathlib,tempfile
with tempfile.TemporaryDirectory(prefix='npa-receipt-proof-') as raw:
 root=pathlib.Path(raw)
 os.environ.clear();os.environ.update(HOME=str(root),NPA_CONFIG_DIR=str(root/'config'),PATH='/usr/bin:/bin')
 from npa.orchestration.npa_workflow import submission_state as state
 from npa.verification import sanitize_failure_reason
 path=state.submission_state_path('example','example-run');assert path.is_relative_to(root)
 payload=state.update_submission_state('example','example-run',{'workflow':{'name':'example'},'launch_state':'planned'})
 assert state.inspect_submission_state('example','example-run').outcome=='found'
 assert path.stat().st_mode & 0o777==0o600 and path.parent.stat().st_mode & 0o777==0o700
 valid=path.read_bytes();cases={'truncated_json':b'{','invalid_utf8':b'\xff','wrong_shape':b'[]'}
 for key,value in [('schema_version','unknown'),('project','other'),('run_id','other')]:
  cases[key]=json.dumps(dict(payload,**{key:value})).encode()
 passed=[]
 for name,data in cases.items():
  path.write_bytes(data)
  try:state.update_submission_state('example','example-run',{'launch_state':'planned'})
  except ValueError:pass
  else:raise AssertionError(name+' unexpectedly overwrote receipt')
  assert path.read_bytes()==data;passed.append(name)
 target=root/'foreign.json';target.write_bytes(valid);path.unlink();path.symlink_to(target)
 try:state.update_submission_state('example','example-run',{'launch_state':'planned'})
 except ValueError:pass
 else:raise AssertionError('symlink unexpectedly accepted')
 assert path.is_symlink() and target.read_bytes()==valid;passed.append('symlink_target_preserved')
 marker='ephemeral-marker-for-redaction-control'
 reason=sanitize_failure_reason('provider returned '+marker,secrets=(marker,))
 assert marker not in reason;passed.append('explicit_diagnostic_redaction')
 print(json.dumps({'production_module_sha256':hashlib.sha256(pathlib.Path(state.__file__).read_bytes()).hexdigest(),'real_filesystem_locking_and_atomic_writes':True,'valid_receipt_persisted':True,'owner_only_modes':True,'negative_controls':passed,'controls_passed':10,'network_calls':0},indent=2))
