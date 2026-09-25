"""Exercise production S3 wire parsing and compare the recovery decision refactor."""
import ast
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import threading
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse
from xml.sax.saxutils import escape

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from npa.orchestration.npa_workflow import supervisor as sup
from npa.orchestration.npa_workflow.run_state import RunStateStore
from npa.orchestration.npa_workflow.runtime import s3_artifact_exists

P=Path(__file__).parent
requests=[]
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_GET(self):
        query=parse_qs(urlparse(self.path).query)
        prefix=query['prefix'][0]; scenario=prefix.split('/')[1]
        token=query.get('continuation-token',[''])[0]
        requests.append((scenario,token))
        if scenario=='denied':
            body=b'<Error><Code>AccessDenied</Code><Message>denied</Message></Error>'
            status=403
        else:
            status=200
            if scenario=='paged' and not token:
                rows=''.join(f'<Contents><Key>{escape(prefix)}marker-{i:04d}</Key><Size>0</Size></Contents>' for i in range(1000))
                fields='<IsTruncated>true</IsTruncated><NextContinuationToken>page-2</NextContinuationToken>'
            elif scenario=='paged':
                assert token=='page-2'
                rows=f'<Contents><Key>{escape(prefix)}result.json</Key><Size>17</Size></Contents>'
                fields='<IsTruncated>false</IsTruncated>'
            else:
                rows=f'<Contents><Key>{escape(prefix)}</Key><Size>0</Size></Contents>'
                fields={
                    'empty':'<IsTruncated>false</IsTruncated>',
                    'missing-flag':'<NextContinuationToken>page-2</NextContinuationToken>',
                    'missing-token':'<IsTruncated>true</IsTruncated>',
                    'contradictory':'<IsTruncated>false</IsTruncated><NextContinuationToken>page-2</NextContinuationToken>',
                    'cycle':'<IsTruncated>true</IsTruncated><NextContinuationToken>repeat</NextContinuationToken>',
                }[scenario]
            body=('<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'+rows+fields+'</ListBucketResult>').encode()
        self.send_response(status);self.send_header('Content-Type','application/xml');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)

server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
client=boto3.client('s3',endpoint_url=f'http://127.0.0.1:{server.server_port}',aws_access_key_id='unit',aws_secret_access_key='unit',region_name='us-east-1',config=Config(retries={'max_attempts':0},s3={'addressing_style':'path'}))
wire=[]
try:
    from types import SimpleNamespace
    with patch('npa.clients.storage.StorageClient.from_environment',return_value=SimpleNamespace(_s3=client,s3=client)):
        store=RunStateStore(bucket='unit-bucket',prefix='runs/unit')
        for caller in (store.artifact_exists,s3_artifact_exists):
            for scenario in ('paged','empty','missing-flag','missing-token','contradictory','cycle','denied'):
                start=len(requests)
                try:
                    value=caller(f's3://unit-bucket/runs/{scenario}/')
                    assert scenario in {'paged','empty'}
                    assert value is (scenario=='paged')
                    outcome='present' if value else 'absent'
                except RuntimeError:
                    assert scenario in {'missing-flag','missing-token','contradictory','cycle'}
                    outcome='indeterminate'
                except ClientError as error:
                    assert scenario=='denied' and error.response['Error']['Code']=='AccessDenied'
                    outcome='denied'
                count=len(requests)-start
                assert count==(2 if scenario in {'paged','cycle'} else 1)
                wire.append(dict(caller=caller.__qualname__,scenario=scenario,outcome=outcome,requests=count))
finally:
    server.shutdown();server.server_close();thread.join()

old_source=subprocess.check_output(['git','show','281fd30bf38ed84c61a94213b2b3177248365828:npa/src/npa/orchestration/npa_workflow/supervisor.py'],text=True)
module=ast.parse(old_source);old=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='decide_recovery')
namespace=dict(vars(sup));exec(compile(ast.Module(body=[old],type_ignores=[]),'original-decision','exec'),namespace)
prior=namespace['decide_recovery']
base_identity=sup.AttemptIdentity('skypilot','unit-run',1,'unit-wave',provider_job_id='unit-job',workflow_sha256='workflow',source_sha256='source',image_digest='image')
outputs=[sup.ArtifactValidation(s,declared=('output',),valid=('output',) if s=='valid' else (),missing=('output',) if s in {'absent','partial'} else ()) for s in ('valid','absent','partial','indeterminate')]
outputs.append(sup.ArtifactValidation('valid'))
ready=sup.PreflightEvidence(checks={k:'pass' for k in sup.PreflightEvidence.REQUIRED_RELAUNCH_CHECKS})
identity_cases=[(base_identity,dict()),(replace(base_identity,provider_job_id=''),dict())]
for field in ('workflow_sha256','source_sha256','image_digest'):
    identity_cases.extend([(replace(base_identity,**{field:'drift'}),dict()),(replace(base_identity,**{field:''}),dict()),(replace(base_identity,**{field:''}),{f'expected_{field}':''})])
codes=sorted({'' ,'UNRECOGNIZED'}|sup.CONFIGURATION_REASON_CODES|sup.TRANSIENT_REASON_CODES|sup.PAYLOAD_REASON_CODES)
count=0
for state,code,exact,observable,output,identity_case,recoveries,preflight in itertools.product(sup.BackendState,codes,(True,False),(True,False),outputs,identity_cases,(0,1,2),(sup.PreflightEvidence(),ready)):
    identity,overrides=identity_case
    observation=sup.BackendObservation(state,reason_code=code,exact_identity=exact,workload_observable=observable,observed_at='fixed')
    context=replace(sup.RecoveryContext('workflow','source','image',output,preflight=preflight,infrastructure_recoveries=recoveries),**overrides)
    assert prior(identity,observation,context).to_dict()==sup.decide_recovery(identity,observation,context).to_dict(),(state,code,identity_case,recoveries)
    count+=1
result=dict(source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip(),wire_cases=wire,wire_passed=len(wire),decision_equivalence_cases=count,original_source_sha256=hashlib.sha256(old_source.encode()).hexdigest(),scope='Real boto3 HTTP/XML requests against loopback fault server; production recovery decisions. No cloud job or GPU workload.')
(P/'behavior-proof.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(wire_passed=len(wire),decision_equivalence_cases=count)))
