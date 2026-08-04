# Temporal Forward Evidence contract

Read this reference only when a caller wants to prove that a frozen temporal model made
predictions before future labels were available.

## Required sequence

1. Freeze one `TemporalForwardProtocol` after all development data, features, models,
   thresholds, and selection rules have been observed.
2. Set `forward_start_session` strictly after `development_label_end_session`.
3. Run `purged-cv-forward init` once for the protocol and private local store.
4. Before `label_available_at`, submit prediction rows containing sample/asset identity,
   Decision Session, label end, prediction, and feature-snapshot digest. Never include a
   target or raw features in this batch.
5. Preserve the returned prediction digest.
6. After the label is available, submit exactly one settlement for that prediction digest,
   with a finite target and target-source digest.
7. Read `status`; do not interpret metrics before all pre-registered maturity gates pass.

## State meanings

- `WAITING_FOR_FUTURE_DATA`: no eligible prediction exists.
- `COLLECTING`: at least one prediction exists, but matured evidence is insufficient.
- `READY_FOR_REVIEW`: sufficiency and registered metric checks pass; human/governed review
  may begin.
- `FAIL`: evidence is sufficient and at least one registered metric check fails.

Every state returns `production_authorization=NOT_AUTHORIZED`. Local exclusive-create
files are not an external timestamp or WORM attestation; always disclose
`attestation_scope=LOCAL_APPEND_ONLY_NOT_EXTERNALLY_NOTARIZED`.

## Fail-closed conditions

Reject a prediction recorded at or after label availability, a Decision Session before
the frozen start, a duplicate sample, a settlement without a prior prediction, an early
or duplicate settlement, unknown fields, non-finite values, invalid digests, and any
protocol/content mismatch. Historical replay remains development evidence even when its
split geometry is leakage-safe.
