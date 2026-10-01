# OAuthlib compatibility proof

The included, unchanged harness ran 12 checks against a local HTTPS reference
provider using OAuthlib 4.0.0, requests-oauthlib 2.0.0, and Requests 2.34.2.
`summary.json` binds the executed harness and installed library source manifests;
the summary records all twelve tested cases. Raw logs are excluded. All credentials in the source are deliberately
named synthetic test fixtures and grant access only to this process-local test
server. No vendor login or external API is contacted by the tests.

The server independently verifies OAuth1 HMAC signatures and OAuth2 request
parameters, including PKCE. Certificate verification stays enabled, and the
plain-HTTP and untrusted-certificate controls must fail. The harness creates a
short-lived localhost certificate and private key in its own `oauth-evidence/`
directory at runtime. Those generated files are private local test material and
are not included in this proof packet.

To reproduce from a repository checkout with Python 3.12 and OpenSSL installed:

```sh
npa/.venv/bin/python -m venv .oauth-proof-venv
.oauth-proof-venv/bin/python -m pip install oauthlib==4.0.0 requests-oauthlib==2.0.0 requests==2.34.2
.oauth-proof-venv/bin/python /path/to/oauth-compatibility.py
```

Run the harness from a writable evidence directory; it writes its generated
certificate and summary beside the script. The dependencies install from the
package index, while all flow tests use loopback HTTPS. This demonstrates the
installed client libraries' tested interfaces; vendor-specific extensions and
live browser consent are outside the scope.
