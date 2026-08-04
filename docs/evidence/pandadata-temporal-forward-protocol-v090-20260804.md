# PandaData temporal forward protocol v0.9.0 — 2026-08-04

## Outcome

The forward-evidence infrastructure is implemented and the first PandaData protocol is frozen. Its current evidence state is honestly `WAITING_FOR_FUTURE_DATA`; no forecast or target was fabricated to force a result, and production remains `NOT_AUTHORIZED`.

## Frozen development identity

- development report digest: `c3e8f962b950e307958ff622694b8595afd9d581bc73b9316f71f948a3f9cd38`
- development data digest: `9748565065b1a49036bd4a30481157d6e959f0cd02e7e3c317ac683edcad3746`
- selected model: fixed LightGBM lag-sequence from v0.8.0
- model spec digest: `4d6e1e7967184b9c7c2bcf327d2009098a54ecb7f92559077f114f31a494dc8f`
- temporal dataset spec digest: `ce9f2a7813e82d7711d0da2cae69b9e61cb2e39e4dbb28b1f36afdd68e24ce7a`
- selection rule: minimum causal Walk-Forward MSE among the three frozen v0.8 cases
- development label boundary: 2026-08-03
- forward start: 2026-08-04
- protocol digest: `478d3a749462d9cca022b19c6b676783679a941be8d8b5ec1b3fe9f727553b70`

## Pre-registered evidence gate

- label horizon: T+5 Trading Sessions
- minimum matured Decision Sessions: 252
- minimum settled observations: 3,000
- minimum assets: 8
- model MSE must not exceed the zero-return baseline MSE
- mean per-session cross-sectional Spearman IC must be at least 0

## Initial receipt

- predictions: 0
- settlements: 0
- status: `WAITING_FOR_FUTURE_DATA`
- report digest: `49fdfba4b11ea9f4a86bdc669e5b06c5dde6a0b2e875b71428b20baf644c0898`
- production authorization: `NOT_AUTHORIZED`
- attestation scope: `LOCAL_APPEND_ONLY_NOT_EXTERNALLY_NOTARIZED`

This receipt is not a failed model result. It states that independent future evidence does not yet exist. Historical replay cannot satisfy the gate because the development comparison has already consumed those observations.

The local exclusive-create ledger detects violations that occur through the maintained interface, but it is not an external timestamp or WORM attestation. A machine administrator can bypass the interface or alter the system clock. Production-grade non-repudiation therefore requires independently controlled receipt storage.

## Operational sequence

For every eligible Decision Session, write predictions before the declared label-availability instant. After T+5 labels mature, append settlements referencing the returned prediction digests. Review the aggregate status only after all sample-sufficiency gates pass. Runtime prediction and settlement files are private local state and are excluded from version control.
