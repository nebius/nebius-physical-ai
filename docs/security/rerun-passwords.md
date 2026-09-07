# Rerun viewer passwords

New viewer manifests use salted bcrypt password hashes with work factor 12.
Redeploy an existing authenticated viewer to replace its previous password hash.
The nginx image continues to accept the same username and password.

Supply both authentication fields together. A partial configuration now fails
instead of creating a viewer with authentication disabled. Usernames cannot
contain colons or control characters. Passwords must contain no NUL and fit
within bcrypt's 72-byte UTF-8 input boundary; longer passwords are rejected
instead of truncated. A configuration with both fields empty retains the
existing unauthenticated viewer behavior.

Viewer deployments now use private ClusterIP Services. Open them through the
generated Kubernetes port-forward command, which authenticates the connection
to the cluster. Direct LoadBalancer and NodePort deployment is refused; wider
access requires an operator-managed authenticated TLS ingress.

The web page and exact `/recording.rrd` download share the nginx authentication
boundary. Raw Rerun web and gRPC listeners bind only to pod loopback and are
absent from the Service. The viewer opens the recording over the same HTTP
origin, so only the web port needs forwarding. Destroy removes both storage
and authentication secrets, even when the caller no longer supplies a password.
