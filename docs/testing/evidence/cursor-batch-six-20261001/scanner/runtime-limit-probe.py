import sys
"""Exercise the exact memory.go policy in task-owned nested kernel cgroups."""
import hashlib,json,os,signal,subprocess,time,uuid
from pathlib import Path
os.umask(0o077)
r=Path(sys.argv[1]).resolve();out=r/'analysis/final-memory-probe';uid=gid=1000
root=Path('/sys/fs/cgroup')/('npa-scanner-probe-'+uuid.uuid4().hex);root.mkdir();root.chmod(0o755);(root/'cgroup.subtree_control').write_text('+memory')
owned=[];result={'scope':'Standard-library probe compiles byte-identical candidate memory.go and reads debug.SetMemoryLimit after configureCgroupMemory; exact production helper remains covered separately.','source_sha256':hashlib.sha256((out/'memory.go').read_bytes()).hexdigest(),'cases':{}}
def child(group):
 os.setsid();(group/'cgroup.procs').write_text(str(os.getpid()));os.setgid(gid);os.setuid(uid)
try:
 for name,parent_limit,leaf_limit,explicit in [('ancestor',512,'max',None),('leaf',512,128,None),('unlimited','max','max',None),('stricter-direct',512,'max','64MiB')]:
  group=root/name;group.mkdir();group.chmod(0o755);(group/'memory.max').write_text('max' if parent_limit=='max' else str(parent_limit*1024*1024));(group/'memory.swap.max').write_text('0');(group/'cgroup.subtree_control').write_text('+memory')
  leaf=group/'helper';leaf.mkdir();leaf.chmod(0o755);(leaf/'memory.max').write_text('max' if leaf_limit=='max' else str(leaf_limit*1024*1024));sibling=group/'sibling';sibling.mkdir()
  ballast=subprocess.Popen(['/usr/bin/python3','-c','import sys; held=bytearray(64*1024*1024); print("ready",flush=True); sys.stdin.buffer.read()'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={'PATH':os.defpath},preexec_fn=lambda:child(sibling));owned.append(ballast);assert ballast.stdout.readline()==b'ready\n'
  current=int((group/'memory.current').read_text());env={'PATH':os.defpath}
  if explicit:env['GOMEMLIMIT']=explicit
  probe=subprocess.Popen([str(out/'probe')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,preexec_fn=lambda:child(leaf));owned.append(probe);stdout,stderr=probe.communicate();assert probe.returncode==0 and not stderr,(probe.returncode,stderr)
  entry=json.loads(stdout);print('raw',name,json.dumps(entry),flush=True);assert entry['failure_code']==''
  if name=='ancestor':assert 300*1024*1024<entry['configured_limit']<512*1024*1024
  if name=='leaf':assert 96*1024*1024<entry['configured_limit']<128*1024*1024
  if name=='unlimited':assert entry['configured_limit']==2**63-1
  if name=='stricter-direct':assert entry['configured_limit']==64*1024*1024
  entry.update(parent_limit=parent_limit,leaf_limit=leaf_limit,sibling_bytes=64*1024*1024,parent_current_before_probe=current,explicit_environment_limit=explicit)
  result['cases'][name]=entry;print(name,json.dumps(entry),flush=True)
  ballast.stdin.close();ballast.wait();assert ballast.returncode==0
  leaf.rmdir();sibling.rmdir();group.rmdir()
finally:
 for proc in owned:
  if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 if root.exists():
  for folder in sorted((p for p in root.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
   for pid in (folder/'cgroup.procs').read_text().split():
    try:os.kill(int(pid),signal.SIGKILL)
    except ProcessLookupError:pass
   while (folder/'cgroup.procs').read_text().strip():time.sleep(0.01)
   folder.rmdir()
  root.rmdir()
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');os.chown(out/'summary.json',uid,gid)
