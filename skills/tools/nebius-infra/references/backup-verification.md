<!-- register: agent reference | reader: Workbench operator agent | consumed: retrieved for a backup verification task -->

# Verify a retained S3 backup

Use this reference when a backup check reports missing or mismatched checksum
metadata, or when reviewing retained restore evidence. Use the selected NPA
project's storage configuration and preserve the exact bucket, case-sensitive
object key and recorded version. Review supplied receipts before requesting
another read; distinguish recorded verification from a new live check.

Compare custom metadata names case-insensitively: `Sha256` and `sha256` are
equivalent names. Preserve values exactly and report conflicting values after
name normalization. A missing metadata field does not establish upload failure
or authorize replacing the object.

Verify the recorded version's content against a trusted digest. When using an
S3 get-object request through boto3, pass the recorded `VersionId`; a read of
the current version cannot establish the contents of an earlier version. Match
the digest to the representation it describes: encrypted or compressed backup
bytes and restored source bytes are different checks. Follow the recorded
restore procedure before comparing a source-content digest. Report a missing
version or trusted digest as a verification limit.

Report metadata discrepancies separately from content verification. Matching
custom metadata or an ETag alone does not prove content integrity. Custom
`sha256` metadata is distinct from native `ChecksumSHA256`; preserve the native
field's documented encoding and checksum semantics. Keep the retained object
while resolving an ambiguous result.

Related guidance: [PS Services storage reference](https://github.com/nebius/nebius-ps-services/blob/feffd29ee1436c4eb8fd19729e098181f586f00d/skills/nebius/references/storage.md#custom-checksum-metadata)
([contribution 193](https://github.com/nebius/nebius-ps-services/pull/193)),
[S3 metadata](https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingMetadata.html),
[Nebius object integrity](https://docs.nebius.com/object-storage/objects/manage).
