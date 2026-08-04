from __future__ import annotations

from dataclasses import replace
from importlib.util import find_spec

import numpy as np
import pytest

from purged_kfold_validation import DensePricePanel
from purged_kfold_validation.temporal_model_benchmark import (
    TemporalDatasetSpec,
    TemporalModelBenchmarkConfig,
    TemporalModelCase,
    build_temporal_supervised_dataset,
    run_temporal_model_benchmark,
)
from purged_kfold_validation.temporal_models import (
    LightGBMTemporalEstimator,
    NumpyRidgeTemporalEstimator,
    TorchLSTMTemporalEstimator,
)
from purged_kfold_validation.domain import ModelSpec


def _panel(rows: int = 90, assets: int = 3) -> DensePricePanel:
    rng = np.random.default_rng(20260804)
    returns = rng.normal(0.0001, 0.01, size=(rows, assets))
    common = 0.003 * np.sin(np.arange(rows, dtype=float) / 8.0)
    returns += common[:, None]
    returns[0] = 0.0
    prices = np.linspace(80.0, 120.0, assets) * np.cumprod(1.0 + returns, axis=0)
    return DensePricePanel(
        sessions=np.datetime64("2024-01-01", "D")
        + np.arange(rows).astype("timedelta64[D]"),
        asset_ids=tuple(f"A{index}" for index in range(assets)),
        signal_prices=prices,
        tradable_returns=returns,
        source_digest="temporal-test-panel",
    )


def test_temporal_dataset_uses_only_pre_decision_features() -> None:
    panel = _panel()
    spec = TemporalDatasetSpec(lookback_sessions=5, label_horizon_sessions=2)

    dataset = build_temporal_supervised_dataset(panel, spec)

    np.testing.assert_allclose(dataset.features[0], panel.tradable_returns[0:5, 0])
    assert dataset.sessions[0] == panel.sessions[5]
    assert dataset.decision_times is not None
    assert dataset.feature_availability is not None
    assert bool(np.all(dataset.feature_availability < dataset.decision_times[:, None]))
    assert dataset.information_intervals[0].start == panel.sessions[0]
    assert dataset.information_intervals[0].end == panel.sessions[7]
    assert set(dataset.sessions[:3]) == {panel.sessions[5]}


def test_label_horizon_changes_dataset_identity_and_interval() -> None:
    panel = _panel()
    short = build_temporal_supervised_dataset(panel, TemporalDatasetSpec(5, 1))
    long = build_temporal_supervised_dataset(panel, TemporalDatasetSpec(5, 3))

    assert short.digest != long.digest
    assert short.information_intervals[0].end == panel.sessions[6]
    assert long.information_intervals[0].end == panel.sessions[8]


def test_future_price_change_does_not_change_earlier_lag_features() -> None:
    panel = _panel()
    changed_prices = panel.signal_prices.copy()
    changed_prices[30:] *= 1.5
    changed = replace(panel, signal_prices=changed_prices)
    spec = TemporalDatasetSpec(5, 2)

    baseline_dataset = build_temporal_supervised_dataset(panel, spec)
    changed_dataset = build_temporal_supervised_dataset(changed, spec)
    earlier = np.asarray(
        [session < panel.sessions[30] for session in baseline_dataset.sessions]
    )

    np.testing.assert_allclose(
        baseline_dataset.features[earlier], changed_dataset.features[earlier]
    )


def test_ridge_temporal_estimator_returns_finite_predictions() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(40, 5))
    targets = features[:, 0] * 0.1 + rng.normal(scale=0.01, size=40)

    estimator = NumpyRidgeTemporalEstimator().fit(features, targets)
    predictions = estimator.predict(features[:7])

    assert predictions.shape == (7,)
    assert np.all(np.isfinite(predictions))


@pytest.mark.skipif(find_spec("lightgbm") is None, reason="optional LightGBM absent")
def test_lightgbm_temporal_estimator_returns_finite_predictions() -> None:
    rng = np.random.default_rng(8)
    features = rng.normal(size=(60, 5))
    targets = features[:, 0] * features[:, 1]

    estimator = LightGBMTemporalEstimator(n_estimators=5).fit(features, targets)

    assert np.all(np.isfinite(estimator.predict(features[:6])))


@pytest.mark.skipif(find_spec("torch") is None, reason="optional Torch absent")
def test_lstm_temporal_estimator_returns_finite_predictions() -> None:
    rng = np.random.default_rng(9)
    features = rng.normal(size=(48, 5))
    targets = np.mean(features, axis=1)

    estimator = TorchLSTMTemporalEstimator(hidden_size=4, epochs=1, batch_size=24).fit(
        features, targets
    )
    predictions = estimator.predict(features[:6])

    assert predictions.shape == (6,)
    assert np.all(np.isfinite(predictions))


def test_temporal_benchmark_proves_safe_channels_have_zero_overlap() -> None:
    dataset = build_temporal_supervised_dataset(_panel(), TemporalDatasetSpec(5, 2))
    model = TemporalModelCase(
        estimator_factory=NumpyRidgeTemporalEstimator,
        model_spec=ModelSpec("test-ridge", "1", {"alpha": 1.0}),
    )
    config = TemporalModelBenchmarkConfig(
        n_splits=3,
        embargo_sessions=5,
        pre_test_gap_sessions=2,
        walk_forward_test_sessions=10,
        cpcv_groups=4,
        cpcv_test_groups=2,
        minimum_train_observations=30,
        minimum_train_sessions=10,
    )

    report = run_temporal_model_benchmark(dataset, (model,), config)
    canonical = report.canonical()
    channels = {item.name: item for item in report.models[0].channels}

    assert canonical["decision"]["leakage_control_status"] == "PASS"  # type: ignore[index]
    assert channels["unsafe-shuffled-kfold"].overlap_count > 0
    for name in (
        "purged-kfold-no-embargo",
        "purged-kfold-embargo",
        "cpcv",
        "causal-walk-forward",
    ):
        assert channels[name].overlap_count == 0
    embargo_minimum = channels["purged-kfold-embargo"].minimum_train_observations
    no_embargo_minimum = channels["purged-kfold-no-embargo"].minimum_train_observations
    assert embargo_minimum is not None
    assert no_embargo_minimum is not None
    assert embargo_minimum <= no_embargo_minimum
    assert len(channels["cpcv"].per_path_mse) == 3
    assert canonical["decision"]["production_authorization"] == "NOT_AUTHORIZED"  # type: ignore[index]
    assert canonical["decision"]["embargo_incremental_status"] in {  # type: ignore[index]
        "ACTIVE",
        "NO_INCREMENTAL_EXCLUSION_AFTER_FULL_INTERVAL_PURGE",
    }


def test_temporal_benchmark_is_deterministic() -> None:
    dataset = build_temporal_supervised_dataset(_panel(), TemporalDatasetSpec(5, 2))
    model = TemporalModelCase(
        estimator_factory=NumpyRidgeTemporalEstimator,
        model_spec=ModelSpec("test-ridge", "1", {"alpha": 1.0}),
    )
    config = TemporalModelBenchmarkConfig(
        n_splits=3,
        embargo_sessions=5,
        pre_test_gap_sessions=2,
        walk_forward_test_sessions=10,
        cpcv_groups=4,
        cpcv_test_groups=2,
        minimum_train_observations=30,
        minimum_train_sessions=10,
    )

    first = run_temporal_model_benchmark(dataset, (model,), config)
    second = run_temporal_model_benchmark(dataset, (model,), config)

    assert first.digest == second.digest
    assert first.canonical() == second.canonical()
