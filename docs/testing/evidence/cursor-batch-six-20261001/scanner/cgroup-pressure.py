import sys
"""Measure only task-owned nested cgroups; never alter shared cgroups."""
import hashlib,json,os,re,signal,struct,subprocess,sys,time,uuid,threading
from pathlib import Path
os.umask(0o077)
r=Path(sys.argv[1]).resolve();out=r/'analysis/cgroup-final-pressure';out.mkdir(mode=0o700)
uid=1000;gid=1000;os.chown(out,uid,gid)
root=Path('/sys/fs/cgroup')/('npa-scanner-review-'+uuid.uuid4().hex)
root.mkdir();root.chmod(0o755);(root/'cgroup.subtree_control').write_text('+memory')
owned=[]
result={'scope':'Actual helper bytes under owned ancestor memory.max with leaf max, sibling64MiB allocation, PATH-only environment; no archive traversal claim in this subexperiment.','cases':{}}
body=b'neutral public scanner corpus\n';unit=(body*((8*1024*1024)//len(body)+1))[:8*1024*1024]
secret=b'gh'+b'p_'+b'aB3dE6gH9jK2mN5pQ8sT1vW4yZ7bC0eF3hI6'
records=[unit if n%3 else unit[:-len(secret)-1]+b' '+secret for n in range(12)]
framed=b''.join(struct.pack('>Q',len(data))+data for data in records)
result['corpus']={'records':len(records),'bytes':sum(map(len,records)),'framed_sha256':hashlib.sha256(framed).hexdigest()}
def child(group):
 os.setsid();(group/'cgroup.procs').write_text(str(os.getpid()));os.setgid(gid);os.setuid(uid)
def counters(path):
 return {k:int(v) for k,v in (line.split() for line in path.read_text().splitlines())}
try:
 for budget in (384,):
  for version in ('base','candidate'):
   name=version+'-'+str(budget);group=root/name;group.mkdir();group.chmod(0o755);(group/'memory.max').write_text(str(budget*1024*1024));(group/'memory.swap.max').write_text('0');(group/'cgroup.subtree_control').write_text('+memory')
   leaf=group/'helper';leaf.mkdir();leaf.chmod(0o755);ballast_group=group/'sibling';ballast_group.mkdir()
   ballast_code='import sys; held=bytearray(64*1024*1024); print("ready",flush=True); sys.stdin.buffer.read()'
   ballast=subprocess.Popen(['/usr/bin/python3','-c',ballast_code],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={'PATH':'/usr/bin:/bin'},preexec_fn=lambda:child(ballast_group))
   owned.append(ballast);assert ballast.stdout.readline()==b'ready\n'
   binary=r/'analysis'/(('base' if version=='base' else 'final-candidate')+'-tools')/'whole-file-scanner';config=binary.parent/'gitleaks-config.toml';timer=out/(name+'-time.txt')
   before=counters(group/'memory.events');current=int((group/'memory.current').read_text());started=time.monotonic()
   proc=subprocess.Popen(['/usr/bin/time','-v','-o',str(timer),str(binary),'--config',str(config)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={'PATH':os.defpath},preexec_fn=lambda:child(leaf))
   owned.append(proc);sampled=[current];stop=threading.Event()
   def sample():
    while not stop.wait(0.005):sampled.append(int((group/'memory.current').read_text()))
   observer=threading.Thread(target=sample);observer.start()
   stdout,stderr=proc.communicate(framed);elapsed=time.monotonic()-started;stop.set();observer.join()
   (out/(name+'-stdout.jsonl')).write_bytes(stdout);(out/(name+'-stderr.txt')).write_bytes(stderr)
   rows=[json.loads(line) for line in stdout.splitlines()];summary=rows[-1] if rows and rows[-1].get('type')=='summary' else None
   after=counters(group/'memory.events');peak=int((group/'memory.peak').read_text()) if (group/'memory.peak').exists() else None;rss=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',timer.read_text()).group(1))
   result['cases'][name]={'exit':proc.returncode,'seconds':round(elapsed,3),'hard_limit_bytes':budget*1024*1024,'startup_group_current':current,'group_peak_bytes':peak,'sampled_group_peak_bytes':max(sampled),'sample_interval_seconds':0.005,'helper_max_rss_kib':rss,'oom_events_delta':{k:after[k]-before[k] for k in before},'complete':summary is not None,'summary':summary,'response_sha256':hashlib.sha256(stdout).hexdigest(),'response_bytes':len(stdout),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}
   ballast.stdin.close();ballast.wait();assert ballast.returncode==0
   assert not (leaf/'cgroup.procs').read_text().strip() and not (ballast_group/'cgroup.procs').read_text().strip()
   leaf.rmdir();ballast_group.rmdir();group.rmdir()
   (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(name,json.dumps(result['cases'][name]),flush=True)
finally:
 for process in owned:
  if process.poll() is None:
   os.killpg(process.pid,signal.SIGKILL);process.wait()
 if root.exists():
  # Only this freshly created namespace is eligible for cleanup.
  for folder in sorted(root.rglob('*'),key=lambda p:len(p.parts),reverse=True):
   if folder.is_dir():
    for pid in (folder/'cgroup.procs').read_text().split():
     try:os.kill(int(pid),signal.SIGKILL)
     except ProcessLookupError:pass
    while (folder/'cgroup.procs').read_text().strip():time.sleep(0.01)
    folder.rmdir()
  root.rmdir()
for item in out.iterdir():os.chown(item,uid,gid)
