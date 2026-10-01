"""Real installed clients against an isolated TLS reference provider; synthetic credentials only."""
import base64, hashlib, hmac, importlib.metadata as metadata, json, os, ssl, subprocess, sys, threading, time, unittest
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, parse_qsl, quote, unquote, urlencode, urlsplit
import requests
from requests_oauthlib import OAuth1Session, OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient, InvalidClientError, InvalidGrantError, MissingTokenError, MismatchingStateError, InsecureTransportError

ROOT=Path(__file__).parent/'oauth-evidence';ROOT.mkdir(mode=0o700,exist_ok=True)
CLIENT='synthetic-client';SECRET='synthetic-client-secret';AUTHCODE='synthetic-authorization-code'
REQUEST='synthetic-request-token';REQUEST_SECRET='synthetic-request-secret';ACCESS='synthetic-access-token';ACCESS_SECRET='synthetic-access-secret'
STATE={'requests':Counter(),'challenges':{},'token_number':0,'bearers':set(),'server_errors':[],'hmac_verified':0,'refreshes':0}
def pct(value):return quote(str(value),safe='~-._')
def signed(request,body):
    header=request.headers.get('Authorization','')
    if not header.startswith('OAuth '):return False
    fields={unquote(k):unquote(v.strip('"')) for k,v in (pair.strip().split('=',1) for pair in header[6:].split(','))}
    token=fields.get('oauth_token','');secret={REQUEST:REQUEST_SECRET,ACCESS:ACCESS_SECRET,'':''}.get(token)
    if secret is None or fields.get('oauth_consumer_key')!=CLIENT or fields.get('oauth_signature_method')!='HMAC-SHA1':return False
    parameters=[(k,v) for k,v in fields.items() if k not in ('oauth_signature','realm')]+parse_qsl(urlsplit(request.path).query,keep_blank_values=True)
    if body:parameters+=parse_qsl(body,keep_blank_values=True)
    normalized='&'.join(k+'='+v for k,v in sorted((pct(k),pct(v)) for k,v in parameters))
    target=BASE+urlsplit(request.path).path
    signature_base='&'.join((request.command,pct(target),pct(normalized)))
    expected=base64.b64encode(hmac.new((pct(SECRET)+'&'+pct(secret)).encode(),signature_base.encode(),hashlib.sha1).digest()).decode()
    valid=hmac.compare_digest(expected,fields.get('oauth_signature',''))
    if valid:STATE['hmac_verified']+=1
    return valid
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def respond(self,status,body,content='application/json',headers=None):
        raw=body.encode() if isinstance(body,str) else json.dumps(body).encode()
        self.send_response(status);self.send_header('Content-Type',content);self.send_header('Content-Length',str(len(raw)))
        for k,v in (headers or {}).items():self.send_header(k,v)
        self.end_headers();self.wfile.write(raw)
    def do_GET(self):self.dispatch('')
    def do_POST(self):self.dispatch(self.rfile.read(int(self.headers.get('Content-Length','0'))).decode())
    def dispatch(self,body):
        path=urlsplit(self.path).path;STATE['requests'][self.command+' '+path]+=1
        try:
            if path=='/authorize':
                params=parse_qs(urlsplit(self.path).query);assert params['client_id']==[CLIENT] and params['response_type']==['code']
                assert params['code_challenge_method']==['S256'];STATE['challenges'][AUTHCODE]=params['code_challenge'][0]
                return self.respond(302,'',headers={'Location':params['redirect_uri'][0]+'?'+urlencode({'code':AUTHCODE,'state':params['state'][0]})})
            if path=='/token-error':return self.respond(401,{'error':'invalid_client'})
            if path=='/token-missing':return self.respond(200,{'token_type':'Bearer','expires_in':3600})
            if path=='/token':
                params=parse_qs(body);basic='Basic '+base64.b64encode((CLIENT+':'+SECRET).encode()).decode()
                authenticated=self.headers.get('Authorization')==basic or (params.get('client_id')==[CLIENT] and params.get('client_secret')==[SECRET])
                if not authenticated:return self.respond(401,{'error':'invalid_client'})
                grant=params.get('grant_type',[''])[0]
                if grant=='authorization_code':
                    verifier=params.get('code_verifier',[''])[0];challenge=base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
                    if params.get('code')!=[AUTHCODE] or not hmac.compare_digest(challenge,STATE['challenges'].get(AUTHCODE,'')):
                        return self.respond(400,{'error':'invalid_grant'})
                elif grant=='refresh_token':
                    if params.get('refresh_token')!=['synthetic-refresh']:return self.respond(400,{'error':'invalid_grant'})
                    STATE['refreshes']+=1
                else:assert grant=='client_credentials'
                STATE['token_number']+=1;token='synthetic-bearer-'+str(STATE['token_number']);STATE['bearers'].add(token)
                return self.respond(200,{'access_token':token,'token_type':'Bearer','refresh_token':'synthetic-refresh','expires_in':3600,'scope':'read'})
            if path=='/protected':
                authorization=self.headers.get('Authorization','')
                valid=authorization.startswith('Bearer ') and authorization[7:] in STATE['bearers']
                return self.respond(200 if valid else 401,{'allowed':valid})
            if path.startswith('/oauth1/'):
                if not signed(self,body):return self.respond(401,{'error':'invalid_signature'})
                if path=='/oauth1/request':return self.respond(200,urlencode({'oauth_token':REQUEST,'oauth_token_secret':REQUEST_SECRET,'oauth_callback_confirmed':'true'}),'application/x-www-form-urlencoded')
                if path=='/oauth1/access':
                    assert 'oauth_verifier="synthetic-verifier"' in self.headers['Authorization']
                    return self.respond(200,urlencode({'oauth_token':ACCESS,'oauth_token_secret':ACCESS_SECRET}),'application/x-www-form-urlencoded')
                if path=='/oauth1/resource':return self.respond(200,{'allowed':True})
            raise AssertionError('unexpected route')
        except Exception as exc:
            STATE['server_errors'].append(type(exc).__name__);self.respond(500,{'error':'reference_provider_assertion'})

CERT=ROOT/'localhost-cert.pem';KEY=ROOT/'localhost-key.pem'
if not CERT.exists():
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-noenc','-keyout',str(KEY),'-out',str(CERT),'-days','1','-subj','/CN=localhost','-addext','subjectAltName=DNS:localhost'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    KEY.chmod(0o600);CERT.chmod(0o600)
SERVER=ThreadingHTTPServer(('127.0.0.1',0),Handler);BASE='https://localhost:'+str(SERVER.server_port)
context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.load_cert_chain(CERT,KEY);SERVER.socket=context.wrap_socket(SERVER.socket,server_side=True)
thread=threading.Thread(target=SERVER.serve_forever,daemon=True);thread.start()
def session(*args,**kwargs):
    client=OAuth2Session(*args,**kwargs);client.trust_env=False;client.verify=str(CERT);return client
def backend():return session(client=BackendApplicationClient(CLIENT),scope=['read'])
def obtain(client):return client.fetch_token(BASE+'/token',client_id=CLIENT,client_secret=SECRET)
def authcode():
    client=session(CLIENT,redirect_uri=BASE+'/callback',scope=['read'],pkce='S256')
    uri,state=client.authorization_url(BASE+'/authorize');response=requests.get(uri,verify=str(CERT),allow_redirects=False,proxies={});assert response.status_code==302
    return client,response.headers['Location']
class Flows(unittest.TestCase):
    def tearDown(self):self.assertEqual(STATE['server_errors'],[])
    def test_authorization_code_pkce_and_bearer(self):
        client,callback=authcode();token=client.fetch_token(BASE+'/token',authorization_response=callback,client_secret=SECRET)
        self.assertIn('access_token',token);self.assertEqual(client.get(BASE+'/protected').json(),{'allowed':True})
    def test_state_mismatch_refuses_before_http(self):
        client,callback=authcode();before=STATE['requests']['POST /token']
        with self.assertRaises(MismatchingStateError):client.fetch_token(BASE+'/token',authorization_response=callback.replace('state=','state=incorrect-'),client_secret=SECRET)
        self.assertEqual(STATE['requests']['POST /token'],before)
    def test_client_credentials_basic_auth_and_bearer(self):
        client=backend();self.assertTrue(obtain(client)['access_token']);self.assertEqual(client.get(BASE+'/protected').status_code,200)
    def test_manual_refresh_and_new_bearer(self):
        client=backend();first=obtain(client);second=client.refresh_token(BASE+'/token',client_id=CLIENT,client_secret=SECRET)
        self.assertNotEqual(first['access_token'],second['access_token']);self.assertEqual(client.get(BASE+'/protected').status_code,200)
    def test_automatic_refresh_persists_new_token(self):
        saved=[];before=STATE['refreshes'];token={'access_token':'synthetic-expired','token_type':'Bearer','refresh_token':'synthetic-refresh','expires_at':time.time()-60}
        client=session(CLIENT,token=token,auto_refresh_url=BASE+'/token',auto_refresh_kwargs={'client_id':CLIENT,'client_secret':SECRET},token_updater=saved.append)
        self.assertEqual(client.get(BASE+'/protected').status_code,200);self.assertEqual(STATE['refreshes'],before+1);self.assertEqual(len(saved),1);self.assertEqual(saved[0],client.token)
    def test_missing_access_token_is_typed_failure(self):
        with self.assertRaises(MissingTokenError):backend().fetch_token(BASE+'/token-missing',client_id=CLIENT,client_secret=SECRET)
    def test_invalid_client_is_typed_failure(self):
        with self.assertRaises(InvalidClientError):backend().fetch_token(BASE+'/token-error',client_id=CLIENT,client_secret=SECRET)
    def test_bad_pkce_is_rejected(self):
        client,callback=authcode();client._code_verifier='incorrect-synthetic-verifier'
        with self.assertRaises(InvalidGrantError):client.fetch_token(BASE+'/token',authorization_response=callback,client_secret=SECRET)
    def test_oauth1_full_token_flow_independent_hmac(self):
        before=STATE['hmac_verified'];client=OAuth1Session(CLIENT,client_secret=SECRET,callback_uri=BASE+'/oauth1/callback');client.trust_env=False;client.verify=str(CERT)
        first=client.fetch_request_token(BASE+'/oauth1/request');self.assertEqual(first['oauth_token'],REQUEST)
        parsed=client.parse_authorization_response(BASE+'/oauth1/callback?'+urlencode({'oauth_token':REQUEST,'oauth_verifier':'synthetic-verifier'}));self.assertEqual(parsed['oauth_verifier'],'synthetic-verifier')
        final=client.fetch_access_token(BASE+'/oauth1/access');self.assertEqual(final['oauth_token'],ACCESS)
        self.assertEqual(client.post(BASE+'/oauth1/resource?duplicate=one&duplicate=two',data={'space':'value + unicode é'}).status_code,200)
        self.assertEqual(STATE['hmac_verified'],before+3)
    def test_oauth1_bad_signature_refused(self):
        client=OAuth1Session(CLIENT,client_secret='incorrect-synthetic-secret',resource_owner_key=ACCESS,resource_owner_secret=ACCESS_SECRET);client.trust_env=False;client.verify=str(CERT)
        self.assertEqual(client.get(BASE+'/oauth1/resource').status_code,401)
    def test_untrusted_certificate_refused(self):
        client=backend();client.verify=True
        with self.assertRaises(requests.exceptions.SSLError):obtain(client)
    def test_insecure_transport_refused(self):
        self.assertNotIn('OAUTHLIB_INSECURE_TRANSPORT',os.environ)
        with self.assertRaises(InsecureTransportError):backend().fetch_token(BASE.replace('https:','http:')+'/token',client_id=CLIENT,client_secret=SECRET)

try:
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Flows);names=[test.id().split('.')[-1] for test in suite]
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    dist={name:metadata.version(name) for name in ['oauthlib','requests-oauthlib','requests','urllib3','certifi','charset-normalizer','idna']}
    bindings={name:hashlib.sha256(Path(__import__(module).__file__).read_bytes()).hexdigest() for name,module in [('oauthlib','oauthlib'),('requests-oauthlib','requests_oauthlib'),('requests','requests')]}
    report={'schema':'npa.oauth4.offline-compatibility.v1','passed':result.wasSuccessful(),'tests_run':result.testsRun,'failed':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'test_cases':names,'distribution_versions':dist,'package_init_sha256':bindings,'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'tls':'Loopback HTTPS with explicitly trusted generated localhost certificate; certificate verification enabled. No insecure-transport environment override.','requests_by_method_path':dict(sorted(STATE['requests'].items())),'independently_verified_oauth1_signatures':STATE['hmac_verified'],'successful_refreshes':STATE['refreshes'],'limitations':['Synthetic local reference provider; no live vendor credentials, browser consent, or vendor-specific extensions.','Checks real installed client interoperability with OAuthlib 4.0.0 and requests-oauthlib 2.0.0; does not certify every upstream server extension.','The reference provider verifies OAuth1 HMAC signatures independently with the standard library; OAuth2 token payloads and PKCE are checked server-side.'],'references':['https://github.com/oauthlib/oauthlib/releases/tag/v4.0.0','https://requests-oauthlib.readthedocs.io/en/latest/oauth2_workflow.html','https://requests-oauthlib.readthedocs.io/en/latest/oauth1_workflow.html']}
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
finally:SERVER.shutdown();SERVER.server_close();thread.join()
