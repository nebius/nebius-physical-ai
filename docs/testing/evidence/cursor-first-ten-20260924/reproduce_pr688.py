import hashlib,importlib.util,io,json,subprocess,sys
from pathlib import Path
from cryptography.hazmat.primitives.serialization import load_pem_private_key
path=Path('npa/scripts/image_payload_credentials.py')
assert hashlib.sha256(path.read_bytes()).hexdigest() == 'f0b00e4792cbd571754ba69b355868a8109fbf5e7b286111ed7bb3e95fe33236'
spec=importlib.util.spec_from_file_location('credential_review',path)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
pem=subprocess.check_output(['openssl','genpkey','-algorithm','ED25519'],stderr=subprocess.DEVNULL)
header,body=pem.split(b'\n',1)
cases=[('standard',b'\n'),('nine-newlines',b'\n'*9),('nine-spaces',b' '*9+b'\n'),('two-MiB-newlines',b'\n'*(2*1024*1024))]
rows=[]
for label,gap in cases:
 candidate=header+gap+body
 try:load_pem_private_key(candidate,password=None);valid=True
 except ValueError:valid=False
 rows.append({'case':label,'parse_valid':valid,'detection':m.content_credential(io.BytesIO(candidate)),'input_sha256':hashlib.sha256(candidate).hexdigest()})
print(json.dumps({'pr':688,'head':'cf141d7a67affda8089b6fc35befcf03190e9dd4','helper_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'key_origin':'A fresh disposable Ed25519 PKCS8 key generated in memory for this control; no customer key or private-key bytes are recorded.','cases':rows},indent=2))
