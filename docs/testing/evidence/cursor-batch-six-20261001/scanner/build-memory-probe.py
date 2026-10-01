import sys
import hashlib,json,os,subprocess,uuid
from pathlib import Path
os.umask(0o077)
r=Path(sys.argv[1]).resolve();w=r/'analysis/final-memory-probe';w.mkdir(mode=0o700,exist_ok=True)
source=r/'final-source/npa/scripts/image_byte_scan/go_helper/memory.go';(w/'memory.go').write_bytes(source.read_bytes());(w/'go.mod').write_text('module memoryprobe\n\ngo 1.26.0\n')
(w/'main.go').write_text('package main\nimport("encoding/json";"os";"runtime/debug")\nfunc main(){before:=debug.SetMemoryLimit(-1);code:=configureCgroupMemory();json.NewEncoder(os.Stdout).Encode(map[string]any{"initial_limit":before,"configured_limit":debug.SetMemoryLimit(-1),"failure_code":code})}\n')
receipt=json.loads((r/'analysis/final-candidate-tools/dependency-receipt.json').read_text());prior=Path(receipt['validation']['logs']).parent
env={'PATH':'/usr/bin:/bin','GOENV':'off','GOWORK':'off','GOTOOLCHAIN':'local','GOCACHE':str(prior/'cache'),'CGO_ENABLED':'0'}
subprocess.run([str(prior/'toolchain/go/bin/go'),'build','-o',str(w/'probe'),'.'],cwd=w,env=env,check=True)
print('probe-built',flush=True)
