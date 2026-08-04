from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from purged_kfold_validation import (
    DensePricePanel,
    GateStatus,
    PathSharpeDistribution,
    StrategyAcceptancePolicy,
    StrategyBenchmarkConfig,
    StrategyHoldoutEvidence,
    StrategyHoldoutStatus,
    TSMOMCandidate,
    TimeSeriesBenchmarkConfig,
    TimeSeriesBenchmarkReport,
    ValidationError,
    assess_time_series_benchmark,
    run_time_series_strategy_benchmark,
)


def _report() -> TimeSeriesBenchmarkReport:
    rows = 120
    x = np.arange(rows, dtype=np.float64)
    returns = np.column_stack(
        (
            0.0002 + 0.004 * np.sin(x / 7.0),
            0.0001 + 0.005 * np.cos(x / 9.0),
        )
    )
    prices = np.asarray((100.0, 120.0)) * np.cumprod(1.0 + returns, axis=0)
    panel = DensePricePanel(
        sessions=np.datetime64("2025-01-01", "D")
        + np.arange(rows).astype("timedelta64[D]"),
        asset_ids=("A", "B"),
        signal_prices=prices,
        tradable_returns=returns,
        source_digest="acceptance-test",
    )
    candidates = (
        TSMOMCandidate("a", 5, 5, 2, 2.0),
        TSMOMCandidate("b", 7, 6, 3, 2.0),
    )
    return run_time_series_strategy_benchmark(
        panel,
        TimeSeriesBenchmarkConfig(
            candidates=candidates,
            cost_scenarios_bps=(3.0, 5.0),
            analysis=StrategyBenchmarkConfig(
                cscv_groups=6,
                cpcv_groups=4,
                cpcv_test_groups=2,
                walk_forward_windows=3,
                minimum_train_sessions=30,
            ),
        ),
    )


def _with_gate_metrics(
    report: TimeSeriesBenchmarkReport, *, dsr_probability: float
) -> TimeSeriesBenchmarkReport:
    scenarios = []
    for scenario in report.scenarios:
        analysis = scenario.analysis
        distribution = PathSharpeDistribution(
            median=0.8,
            p10=0.5,
            worst=0.3,
            interquartile_range=0.2,
        )
        analysis = replace(
            analysis,
            pbo=replace(analysis.pbo, probability=0.10),
            deflated_sharpe=replace(
                analysis.deflated_sharpe, probability=dsr_probability
            ),
            cpcv=replace(analysis.cpcv, sharpe_distribution=distribution),
            walk_forward=replace(analysis.walk_forward, annualized_sharpe=0.4),
        )
        scenarios.append(replace(scenario, analysis=analysis))
    return TimeSeriesBenchmarkReport(
        panel=report.panel,
        config_digest=report.config_digest,
        scenarios=tuple(scenarios),
    )


def test_failed_dsr_separates_tool_pass_from_strategy_failure() -> None:
    report = _with_gate_metrics(_report(), dsr_probability=0.70)

    decision = assess_time_series_benchmark(report)

    assert decision.validation_tool_status is GateStatus.PASS
    assert decision.research_gate_status is GateStatus.FAIL
    assert decision.production_gate_status is GateStatus.FAIL
    assert "DSR_BELOW_THRESHOLD" in decision.reason_codes
    assert "UNTOUCHED_HOLDOUT_NOT_RUN" in decision.reason_codes
    assert "RESEARCH_GATE_FAILED" in decision.reason_codes
    assert decision.dsr_track_record_gap.assumption == (
        "constant-distribution approximation"
    )


def test_passing_research_without_holdout_is_inconclusive_for_production() -> None:
    report = _with_gate_metrics(_report(), dsr_probability=0.99)

    decision = assess_time_series_benchmark(report)

    assert decision.research_gate_status is GateStatus.PASS
    assert decision.production_gate_status is GateStatus.INCONCLUSIVE
    assert decision.evidence_gaps[0].code == "UNTOUCHED_HOLDOUT_NOT_RUN"


def test_untouched_holdout_pass_allows_production_pass() -> None:
    report = _with_gate_metrics(_report(), dsr_probability=0.99)
    holdout = StrategyHoldoutEvidence(
        StrategyHoldoutStatus.UNTOUCHED_PASS, "receipt-digest"
    )

    decision = assess_time_series_benchmark(report, holdout=holdout)

    assert decision.research_gate_status is GateStatus.PASS
    assert decision.production_gate_status is GateStatus.PASS
    assert decision.reason_codes == ()


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (StrategyHoldoutStatus.REUSED, "HOLDOUT_REUSED"),
        (StrategyHoldoutStatus.UNTOUCHED_FAIL, "UNTOUCHED_HOLDOUT_FAILED"),
    ],
)
def test_ineligible_or_failed_holdout_fails_production(
    status: StrategyHoldoutStatus, reason: str
) -> None:
    report = _with_gate_metrics(_report(), dsr_probability=0.99)

    decision = assess_time_series_benchmark(
        report,
        holdout=StrategyHoldoutEvidence(status, "receipt-digest"),
    )

    assert decision.research_gate_status is GateStatus.PASS
    assert decision.production_gate_status is GateStatus.FAIL
    assert reason in decision.reason_codes


def test_missing_registered_cost_scenario_fails_closed() -> None:
    report = _with_gate_metrics(_report(), dsr_probability=0.99)
    incomplete = TimeSeriesBenchmarkReport(
        panel=report.panel,
        config_digest=report.config_digest,
        scenarios=(report.scenarios[0],),
    )

    with pytest.raises(ValidationError, match="exactly one 5 bps scenario"):
        assess_time_series_benchmark(incomplete)


def test_decision_and_track_record_gap_are_deterministic() -> None:
    report = _with_gate_metrics(_report(), dsr_probability=0.70)

    first = assess_time_series_benchmark(report)
    second = assess_time_series_benchmark(report)

    assert first.digest == second.digest
    assert first.canonical() == second.canonical()
    gap = first.dsr_track_record_gap
    assert gap.required_sessions is not None
    assert gap.required_sessions >= 2
    assert gap.additional_sessions == max(0, gap.required_sessions - 120)


def test_non_positive_dsr_edge_has_no_finite_sample_only_solution() -> None:
    report = _with_gate_metrics(_report(), dsr_probability=0.10)
    primary = report.scenarios[0]
    deflated = replace(
        primary.analysis.deflated_sharpe,
        selected_annualized_sharpe=0.1,
        benchmark_annualized_sharpe=0.2,
    )
    changed_primary = replace(
        primary, analysis=replace(primary.analysis, deflated_sharpe=deflated)
    )
    changed = TimeSeriesBenchmarkReport(
        panel=report.panel,
        config_digest=report.config_digest,
        scenarios=(changed_primary, report.scenarios[1]),
    )

    gap = assess_time_series_benchmark(changed).dsr_track_record_gap

    assert gap.required_sessions is None
    assert gap.additional_sessions is None


def test_policy_digest_changes_when_preregistered_threshold_changes() -> None:
    baseline = StrategyAcceptancePolicy()
    changed = replace(baseline, maximum_pbo=0.10)

    assert baseline.digest != changed.digest
