from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from purged_kfold_validation.errors import (
    DuplicateForwardEvidenceError,
    ForwardEvidenceError,
)
from purged_kfold_validation.temporal_forward import (
    ForwardEvidenceStatus,
    ForwardPredictionReceipt,
    LocalTemporalForwardStore,
    TemporalForwardProtocol,
)


class MutableClock:
    def __init__(self, value: str) -> None:
        self.value = datetime.fromisoformat(value).astimezone(timezone.utc)

    def __call__(self) -> datetime:
        return self.value


def _protocol(
    *, sessions: int = 2, observations: int = 4, assets: int = 2
) -> TemporalForwardProtocol:
    return TemporalForwardProtocol(
        protocol_id="forward-v1",
        development_report_digest="a" * 64,
        development_dataset_digest="b" * 64,
        model_spec_digest="c" * 64,
        temporal_dataset_spec_digest="d" * 64,
        development_label_end_session="2026-08-03",
        forward_start_session="2026-08-04",
        label_horizon_sessions=5,
        minimum_matured_sessions=sessions,
        minimum_matured_observations=observations,
        minimum_assets=assets,
        minimum_mean_session_spearman_ic=0.0,
        require_model_mse_not_worse_than_zero=True,
        selection_policy={"kind": "frozen", "models": ["lightgbm"]},
    )


def _record(
    store: LocalTemporalForwardStore,
    protocol: TemporalForwardProtocol,
    *,
    sample: str,
    asset: str,
    session: str,
    prediction: float,
) -> ForwardPredictionReceipt:
    return store.record_prediction(
        protocol,
        sample_id=sample,
        asset_id=asset,
        decision_session=session,
        label_end_session="2026-08-12",
        label_available_at="2026-08-12T16:00:00+00:00",
        prediction=prediction,
        feature_snapshot_digest="e" * 64,
    )


def test_protocol_requires_strictly_future_start_and_deep_freezes_policy() -> None:
    with pytest.raises(ForwardEvidenceError, match="strictly after"):
        TemporalForwardProtocol(
            protocol_id="invalid",
            development_report_digest="a" * 64,
            development_dataset_digest="b" * 64,
            model_spec_digest="c" * 64,
            temporal_dataset_spec_digest="d" * 64,
            development_label_end_session="2026-08-03",
            forward_start_session="2026-08-03",
            label_horizon_sessions=5,
            minimum_matured_sessions=1,
            minimum_matured_observations=1,
            minimum_assets=1,
            minimum_mean_session_spearman_ic=0.0,
            require_model_mse_not_worse_than_zero=True,
            selection_policy={"kind": "frozen"},
        )
    policy = {"grid": [1, 2]}
    other = TemporalForwardProtocol(
        protocol_id="forward-v1",
        development_report_digest="a" * 64,
        development_dataset_digest="b" * 64,
        model_spec_digest="c" * 64,
        temporal_dataset_spec_digest="d" * 64,
        development_label_end_session="2026-08-03",
        forward_start_session="2026-08-04",
        label_horizon_sessions=5,
        minimum_matured_sessions=2,
        minimum_matured_observations=4,
        minimum_assets=2,
        minimum_mean_session_spearman_ic=0.0,
        require_model_mse_not_worse_than_zero=True,
        selection_policy=policy,
    )
    policy["grid"].append(3)
    assert other.selection_policy["grid"] == (1, 2)


def test_prediction_is_recorded_without_target_and_duplicate_sample_fails(
    tmp_path: Path,
) -> None:
    clock = MutableClock("2026-08-04T08:00:00+00:00")
    protocol = _protocol()
    store = LocalTemporalForwardStore(tmp_path / "store", clock=clock)
    store.register(protocol)
    receipt = _record(
        store,
        protocol,
        sample="2026-08-04|A",
        asset="A",
        session="2026-08-04",
        prediction=0.1,
    )

    persisted = (store.root / "predictions" / f"{receipt.digest}.json").read_text()
    assert '"target"' not in persisted
    assert '"features"' not in persisted
    with pytest.raises(DuplicateForwardEvidenceError, match="sample prediction"):
        _record(
            store,
            protocol,
            sample="2026-08-04|A",
            asset="A",
            session="2026-08-04",
            prediction=0.2,
        )


def test_prediction_cannot_be_backfilled_or_recorded_after_label_availability(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    store = LocalTemporalForwardStore(
        tmp_path / "store",
        clock=MutableClock("2026-08-13T08:00:00+00:00"),
    )
    store.register(protocol)
    with pytest.raises(ForwardEvidenceError, match="forward_start"):
        _record(
            store,
            protocol,
            sample="old",
            asset="A",
            session="2026-08-03",
            prediction=0.0,
        )
    with pytest.raises(ForwardEvidenceError, match="before label_available_at"):
        _record(
            store,
            protocol,
            sample="late",
            asset="A",
            session="2026-08-04",
            prediction=0.0,
        )


def test_settlement_requires_prior_prediction_and_label_maturity(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    clock = MutableClock("2026-08-04T08:00:00+00:00")
    store = LocalTemporalForwardStore(tmp_path / "store", clock=clock)
    store.register(protocol)
    with pytest.raises(ForwardEvidenceError, match="does not exist"):
        store.settle(
            protocol,
            prediction_digest="f" * 64,
            target=0.1,
            target_source_digest="1" * 64,
        )
    receipt = _record(
        store,
        protocol,
        sample="future",
        asset="A",
        session="2026-08-04",
        prediction=0.1,
    )
    with pytest.raises(ForwardEvidenceError, match="not yet available"):
        store.settle(
            protocol,
            prediction_digest=receipt.digest,
            target=0.1,
            target_source_digest="1" * 64,
        )
    clock.value = datetime.fromisoformat("2026-08-12T17:00:00+00:00").astimezone(
        timezone.utc
    )
    store.settle(
        protocol,
        prediction_digest=receipt.digest,
        target=0.1,
        target_source_digest="1" * 64,
    )
    with pytest.raises(DuplicateForwardEvidenceError, match="settlement"):
        store.settle(
            protocol,
            prediction_digest=receipt.digest,
            target=0.2,
            target_source_digest="2" * 64,
        )


def test_report_moves_waiting_collecting_ready_and_remains_redacted(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    clock = MutableClock("2026-08-04T08:00:00+00:00")
    store = LocalTemporalForwardStore(tmp_path / "store", clock=clock)
    store.register(protocol)
    waiting = store.report(protocol)
    assert waiting.status is ForwardEvidenceStatus.WAITING_FOR_FUTURE_DATA

    receipts = []
    for session in ("2026-08-04", "2026-08-05"):
        for asset, prediction in (("A", -0.1), ("B", 0.1)):
            receipts.append(
                _record(
                    store,
                    protocol,
                    sample=f"{session}|{asset}",
                    asset=asset,
                    session=session,
                    prediction=prediction,
                )
            )
    assert store.report(protocol).status is ForwardEvidenceStatus.COLLECTING
    clock.value = datetime.fromisoformat("2026-08-12T17:00:00+00:00").astimezone(
        timezone.utc
    )
    for receipt in receipts:
        target = -0.2 if receipt.asset_id == "A" else 0.2
        store.settle(
            protocol,
            prediction_digest=receipt.digest,
            target=target,
            target_source_digest="1" * 64,
        )
    report = store.report(protocol)
    assert report.status is ForwardEvidenceStatus.READY_FOR_REVIEW
    assert report.model_mse == pytest.approx(0.01)
    assert report.zero_baseline_mse == pytest.approx(0.04)
    assert report.mean_session_spearman_ic == pytest.approx(1.0)
    assert report.production_authorization == "NOT_AUTHORIZED"
    assert report.attestation_scope == "LOCAL_APPEND_ONLY_NOT_EXTERNALLY_NOTARIZED"
    public = str(report.canonical())
    assert "sample_id" not in public
    assert "prediction_digest" not in public


def test_sufficient_bad_forecasts_fail(tmp_path: Path) -> None:
    protocol = _protocol(sessions=1, observations=2, assets=2)
    clock = MutableClock("2026-08-04T08:00:00+00:00")
    store = LocalTemporalForwardStore(tmp_path / "store", clock=clock)
    store.register(protocol)
    receipts = [
        _record(
            store,
            protocol,
            sample="A",
            asset="A",
            session="2026-08-04",
            prediction=1.0,
        ),
        _record(
            store,
            protocol,
            sample="B",
            asset="B",
            session="2026-08-04",
            prediction=-1.0,
        ),
    ]
    clock.value = datetime.fromisoformat("2026-08-12T17:00:00+00:00").astimezone(
        timezone.utc
    )
    for receipt, target in zip(receipts, (-0.1, 0.1), strict=True):
        store.settle(
            protocol,
            prediction_digest=receipt.digest,
            target=target,
            target_source_digest="1" * 64,
        )
    assert store.report(protocol).status is ForwardEvidenceStatus.FAIL
