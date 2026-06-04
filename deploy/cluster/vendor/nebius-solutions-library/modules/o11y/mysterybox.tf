ephemeral "nebius_mysterybox_v1_secret_payload_entry" "loki_s3_secret" {
  count     = var.o11y.loki.enabled ? 1 : 0
  secret_id = nebius_iam_v2_access_key.loki_s3_key[0].status.secret_reference_id
  key       = "secret"
}