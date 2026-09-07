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
