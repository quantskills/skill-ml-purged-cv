from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from purged_kfold_validation import (
    DensePricePanel,
    StrategyBenchmarkConfig,
    StrategyReturnMatrix,
    TSMOMCandidate,
    TSMOMExecutionSpec,
    TimeSeriesBenchmarkConfig,
    ValidationError,
    analyze_strategy_return_matrix,
    build_tsmom_return_matrix,
    default_tsmom_candidates,
    run_time_series_strategy_benchmark,
)


def _sessions(count: int) -> np.ndarray:
    start = np.datetime64("2024-01-01", "D")
    return start + np.arange(count).astype("timedelta64[D]")


def _panel(count: int = 80) -> DensePricePanel:
    x = np.arange(count, dtype=np.float64)
    first = 100.0 * np.exp(0.002 * x + 0.01 * np.sin(x / 3.0))
    second = 120.0 * np.exp(-0.001 * x + 0.012 * np.cos(x / 4.0))
    prices = np.column_stack((first, second))
    returns = np.zeros_like(prices)
    returns[1:] = prices[1:] / prices[:-1] - 1.0
    return DensePricePanel(
        sessions=_sessions(count),
        asset_ids=("A", "B"),
        signal_prices=prices,
        tradable_returns=returns,
        source_digest="a" * 64,
    )


def _candidate(candidate_id: str = "short") -> TSMOMCandidate:
    return TSMOMCandidate(
        candidate_id=candidate_id,
        lookback_sessions=5,
        volatility_window=5,
        rebalance_sessions=2,
        leverage_cap=2.0,
    )


def test_default_tsmom_family_is_the_registered_32_candidate_grid() -> None:
    candidates = default_tsmom_candidates()

    assert len(candidates) == 32
    assert len({candidate.candidate_id for candidate in candidates}) == 32
    assert {candidate.lookback_sessions for candidate in candidates} == {
        21,
        63,
        126,
        252,
    }


def test_dense_panel_rejects_non_increasing_sessions() -> None:
    panel = _panel(20)
    sessions = panel.sessions.copy()
    sessions[4] = sessions[3]

    with pytest.raises(ValidationError, match="strictly increasing"):
        DensePricePanel(
            sessions=sessions,
            asset_ids=panel.asset_ids,
            signal_prices=panel.signal_prices,
            tradable_returns=panel.tradable_returns,
            source_digest=panel.source_digest,
        )


def test_tsmom_return_at_t_does_not_use_price_at_t_or_later() -> None:
    panel = _panel()
    execution = TSMOMExecutionSpec(
        annualization_sessions=252,
        target_annual_volatility=0.10,
        cost_bps=0.0,
    )
    candidates = (_candidate(), replace(_candidate("second"), rebalance_sessions=3))
    baseline = build_tsmom_return_matrix(panel, candidates, execution)
    changed_prices = panel.signal_prices.copy()
    changed_prices[40:, 0] *= 0.01
    changed = build_tsmom_return_matrix(
        replace(panel, signal_prices=changed_prices),
        candidates,
        execution,
    )

    np.testing.assert_allclose(
        baseline.gross_returns[:41],
        changed.gross_returns[:41],
    )
    assert not np.allclose(
        baseline.gross_returns[41:],
        changed.gross_returns[41:],
    )


def test_cost_is_deducted_from_the_same_frozen_weight_track() -> None:
    panel = _panel()
    candidate = (_candidate(), replace(_candidate("second"), rebalance_sessions=3))
    zero = build_tsmom_return_matrix(
        panel,
        candidate,
        TSMOMExecutionSpec(cost_bps=0.0),
    )
    costly = build_tsmom_return_matrix(
        panel,
        candidate,
        TSMOMExecutionSpec(cost_bps=10.0),
    )

    np.testing.assert_allclose(zero.gross_returns, costly.gross_returns)
    np.testing.assert_allclose(zero.turnover, costly.turnover)
    assert float(np.sum(costly.net_returns)) <= float(np.sum(zero.net_returns))
    assert np.all(costly.net_returns <= costly.gross_returns + 1e-15)


def _candidate_matrix(
    *, sessions: int = 240, candidates: int = 6
) -> StrategyReturnMatrix:
    rng = np.random.default_rng(20260803)
    returns = rng.normal(0.0002, 0.01, size=(sessions, candidates))
    returns[:, 0] += 0.00025
    return StrategyReturnMatrix(
        sessions=_sessions(sessions),
        candidate_ids=tuple(f"candidate-{index}" for index in range(candidates)),
        gross_returns=returns,
        net_returns=returns,
        turnover=np.zeros_like(returns),
        source_digest="b" * 64,
        cost_bps=0.0,
    )


def test_generic_strategy_matrix_produces_pbo_dsr_cpcv_and_walk_forward() -> None:
    matrix = _candidate_matrix()
    config = StrategyBenchmarkConfig(
        annualization_sessions=252,
        cscv_groups=8,
        cpcv_groups=6,
        cpcv_test_groups=2,
        cpcv_embargo_sessions=1,
        walk_forward_windows=4,
        minimum_train_sessions=60,
    )

    first = analyze_strategy_return_matrix(matrix, config)
    second = analyze_strategy_return_matrix(matrix, config)

    assert first.digest == second.digest
    assert first.canonical() == second.canonical()
    assert first.pbo.trial_count == 6
    assert 0.0 <= first.pbo.probability <= 1.0
    assert 0.0 <= first.deflated_sharpe.probability <= 1.0
    assert first.cpcv.path_count == 5
    assert len(first.cpcv.paths) == 5
    assert {path.observations for path in first.cpcv.paths} == {240}
    assert len(first.walk_forward.windows) == 4


def test_cscv_detects_deliberate_selection_rank_reversal() -> None:
    group_count = 8
    group_size = 24
    trial_count = 8
    rows = group_count * group_size
    returns = np.full((rows, trial_count), -0.004, dtype=np.float64)
    wave = np.tile(np.asarray((-0.001, 0.001)), rows // 2)
    for group_index in range(group_count):
        start = group_index * group_size
        end = start + group_size
        returns[start:end] += wave[start:end, None]
        returns[start:end, group_index] = 0.012 + wave[start:end]
    matrix = StrategyReturnMatrix(
        sessions=_sessions(rows),
        candidate_ids=tuple(f"regime-{index}" for index in range(trial_count)),
        gross_returns=returns,
        net_returns=returns,
        turnover=np.zeros_like(returns),
        source_digest="c" * 64,
        cost_bps=0.0,
    )

    report = analyze_strategy_return_matrix(
        matrix,
        StrategyBenchmarkConfig(
            cscv_groups=8,
            cpcv_groups=4,
            cpcv_test_groups=2,
            walk_forward_windows=3,
            minimum_train_sessions=24,
        ),
    )

    assert report.pbo.probability >= 0.75
    assert report.pbo.negative_logit_count > 0


def test_walk_forward_selection_cannot_read_later_test_returns() -> None:
    matrix = _candidate_matrix(sessions=180, candidates=4)
    config = StrategyBenchmarkConfig(
        cscv_groups=6,
        cpcv_groups=4,
        cpcv_test_groups=2,
        walk_forward_windows=3,
        minimum_train_sessions=45,
    )
    baseline = analyze_strategy_return_matrix(matrix, config)
    changed_returns = matrix.net_returns.copy()
    first_test_start = baseline.walk_forward.windows[0].test_start_index
    changed_returns[first_test_start:, 3] += 0.20
    changed = analyze_strategy_return_matrix(
        replace(
            matrix,
            gross_returns=changed_returns,
            net_returns=changed_returns,
        ),
        config,
    )

    assert (
        baseline.walk_forward.windows[0].selected_candidate_id
        == changed.walk_forward.windows[0].selected_candidate_id
    )


def test_high_level_benchmark_keeps_cost_scenarios_separate_from_trials() -> None:
    panel = _panel(120)
    config = TimeSeriesBenchmarkConfig(
        candidates=(_candidate("a"), _candidate("b")),
        cost_scenarios_bps=(0.0, 5.0),
        analysis=StrategyBenchmarkConfig(
            cscv_groups=6,
            cpcv_groups=4,
            cpcv_test_groups=2,
            walk_forward_windows=3,
            minimum_train_sessions=30,
        ),
    )

    report = run_time_series_strategy_benchmark(panel, config)

    assert [scenario.cost_bps for scenario in report.scenarios] == [0.0, 5.0]
    assert {scenario.analysis.pbo.trial_count for scenario in report.scenarios} == {2}
    zero_metrics = report.scenarios[0].analysis.candidates
    costly_metrics = report.scenarios[1].analysis.candidates
    for zero, costly in zip(zero_metrics, costly_metrics, strict=True):
        assert costly.total_return <= zero.total_return + 1e-15
