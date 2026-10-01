import sys
import importlib.util,json,shutil,subprocess
from pathlib import Path
r=Path(sys.argv[1]).resolve();source=r/'final-source';analysis=r/'analysis';out=analysis/'final-memory-mutations';out.mkdir(mode=0o700)
receipt=json.loads((analysis/'final-candidate-tools/dependency-receipt.json').read_text());work=Path(receipt['validation']['logs']).parent
spec=importlib.util.spec_from_file_location('builder',source/'npa/scripts/image_byte_scan/go_helper/build.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
env=builder.isolated_environment(work,analysis/'final-candidate-tools/gitleaks-config.toml');go=work/'toolchain/go/bin/go';results={}
for name,old,new in [('skip-ancestors','directory = filepath.Dir(directory)','directory = ""'),('ignore-current','remaining := limit - current','remaining := limit'),('omit-runtime-set','setLimit(limit)','setLimit(-1)'),('fatal-pressure','if current >= limit {\n\t\treturn 0, \"\"','if current >= limit {\n\t\treturn 0, \"cgroup_memory_headroom\"'),('vanished-usage-fatal','if os.IsNotExist(err) || errors.Is(err, syscall.ENODEV) {\n\t\treturn 0, \"\"','if os.IsNotExist(err) || errors.Is(err, syscall.ENODEV) {\n\t\treturn 0, \"cgroup_memory_usage_unavailable\"')]:
 scratch=out/name;shutil.copytree(work/'source',scratch);path=scratch/'memory.go';text=path.read_text();assert text.count(old)==1;path.write_text(text.replace(old,new))
 run=subprocess.run([str(go),'test','-count=1','-json','-run','TestCgroupMemory','./...'],cwd=scratch,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/(name+'.jsonl')).write_bytes(run.stdout)
 rows=[json.loads(line) for line in run.stdout.splitlines() if line.startswith(b'{')];failed=[row['Test'] for row in rows if row.get('Action')=='fail' and 'Test' in row]
 assert run.returncode!=0 and failed,(name,run.stdout[-2000:]);results[name]={'exit':run.returncode,'failed_tests':failed};print(name,results[name],flush=True)
(out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
