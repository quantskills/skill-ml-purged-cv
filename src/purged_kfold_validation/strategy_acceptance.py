"""Pre-registered decisions for time-series strategy benchmark evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil, isfinite, sqrt
from statistics import NormalDist

from .domain import canonical_digest
from .errors import StrategyBenchmarkError
from .strategy_benchmark import (
    DeflatedSharpeEvidence,
    StrategyScenarioReport,
    TimeSeriesBenchmarkReport,
)


class GateStatus(str, Enum):
    """Closed vocabulary for validation, research, and production decisions."""

    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class StrategyHoldoutStatus(str, Enum):
    """Whether independent strategy holdout evidence exists and is eligible."""

    NOT_RUN = "NOT_RUN"
    UNTOUCHED_PASS = "UNTOUCHED_PASS"
    UNTOUCHED_FAIL = "UNTOUCHED_FAIL"
    REUSED = "REUSED"


def _finite(value: float, *, name: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise StrategyBenchmarkError(f"{name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class StrategyAcceptancePolicy:
    """Immutable thresholds registered before benchmark evidence is observed."""

    primary_cost_bps: float = 3.0
    stress_cost_bps: float = 5.0
    maximum_pbo: float = 0.20
    minimum_dsr_probability: float = 0.95
    minimum_cpcv_median_sharpe: float = 0.0
    minimum_cpcv_p10_sharpe: float = 0.0
    minimum_cpcv_worst_sharpe: float = 0.0
    minimum_walk_forward_sharpe: float = 0.0
    minimum_stress_cpcv_worst_sharpe: float = 0.0
    minimum_stress_walk_forward_sharpe: float = 0.0
    require_untouched_holdout: bool = True

    def __post_init__(self) -> None:
        primary = _finite(self.primary_cost_bps, name="primary_cost_bps")
        stress = _finite(self.stress_cost_bps, name="stress_cost_bps")
        maximum_pbo = _finite(self.maximum_pbo, name="maximum_pbo")
        probability = _finite(
            self.minimum_dsr_probability, name="minimum_dsr_probability"
        )
        if primary < 0.0 or stress < 0.0:
            raise StrategyBenchmarkError("acceptance cost scenarios cannot be negative")
        if not 0.0 <= maximum_pbo <= 1.0:
            raise StrategyBenchmarkError("maximum_pbo must be between 0 and 1")
        if not 0.0 < probability < 1.0:
            raise StrategyBenchmarkError(
                "minimum_dsr_probability must be strictly between 0 and 1"
            )
        for name in (
            "minimum_cpcv_median_sharpe",
            "minimum_cpcv_p10_sharpe",
            "minimum_cpcv_worst_sharpe",
            "minimum_walk_forward_sharpe",
            "minimum_stress_cpcv_worst_sharpe",
            "minimum_stress_walk_forward_sharpe",
        ):
            _finite(float(getattr(self, name)), name=name)
        if not isinstance(self.require_untouched_holdout, bool):
            raise StrategyBenchmarkError("require_untouched_holdout must be boolean")

    @property
    def digest(self) -> str:
        return canonical_digest(self.canonical())

    def canonical(self) -> dict[str, object]:
        return {
            "primary_cost_bps": self.primary_cost_bps,
            "stress_cost_bps": self.stress_cost_bps,
            "maximum_pbo": self.maximum_pbo,
            "minimum_dsr_probability": self.minimum_dsr_probability,
            "minimum_cpcv_median_sharpe": self.minimum_cpcv_median_sharpe,
            "minimum_cpcv_p10_sharpe": self.minimum_cpcv_p10_sharpe,
            "minimum_cpcv_worst_sharpe": self.minimum_cpcv_worst_sharpe,
            "minimum_walk_forward_sharpe": self.minimum_walk_forward_sharpe,
            "minimum_stress_cpcv_worst_sharpe": (self.minimum_stress_cpcv_worst_sharpe),
            "minimum_stress_walk_forward_sharpe": (
                self.minimum_stress_walk_forward_sharpe
            ),
            "require_untouched_holdout": self.require_untouched_holdout,
        }


@dataclass(frozen=True, slots=True)
class StrategyHoldoutEvidence:
    """Redacted eligibility evidence for one frozen strategy holdout."""

    status: StrategyHoldoutStatus = StrategyHoldoutStatus.NOT_RUN
    receipt_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, StrategyHoldoutStatus):
            raise StrategyBenchmarkError("holdout status is invalid")
        if self.status is StrategyHoldoutStatus.NOT_RUN:
            if self.receipt_digest is not None:
                raise StrategyBenchmarkError(
                    "NOT_RUN holdout cannot carry a receipt digest"
                )
        elif (
            not isinstance(self.receipt_digest, str) or not self.receipt_digest.strip()
        ):
            raise StrategyBenchmarkError(
                "observed holdout status requires a non-empty receipt digest"
            )

    def canonical(self) -> dict[str, object]:
        return {"status": self.status.value, "receipt_digest": self.receipt_digest}


@dataclass(frozen=True, slots=True)
class GateCheck:
    code: str
    status: GateStatus
    observed: float
    threshold: float
    comparison: str
    cost_bps: float

    def canonical(self) -> dict[str, object]:
        return {
            "code": self.code,
            "status": self.status.value,
            "observed": self.observed,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "cost_bps": self.cost_bps,
        }


@dataclass(frozen=True, slots=True)
class EvidenceGap:
    code: str
    message: str

    def canonical(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class DSRTrackRecordGap:
    current_sessions: int
    required_sessions: int | None
    additional_sessions: int | None
    target_probability: float
    current_probability: float
    assumption: str = "constant-distribution approximation"

    def canonical(self) -> dict[str, object]:
        return {
            "current_sessions": self.current_sessions,
            "required_sessions": self.required_sessions,
            "additional_sessions": self.additional_sessions,
            "target_probability": self.target_probability,
            "current_probability": self.current_probability,
            "assumption": self.assumption,
            "warning": "This estimate is not a guarantee and cannot replace holdout evidence.",
        }


@dataclass(frozen=True, slots=True)
class StrategyAcceptanceDecision:
    validation_tool_status: GateStatus
    research_gate_status: GateStatus
    production_gate_status: GateStatus
    checks: tuple[GateCheck, ...]
    reason_codes: tuple[str, ...]
    evidence_gaps: tuple[EvidenceGap, ...]
    dsr_track_record_gap: DSRTrackRecordGap
    policy_digest: str
    report_digest: str
    holdout: StrategyHoldoutEvidence
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "digest", canonical_digest(self.canonical(include_digest=False))
        )

    def canonical(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "1",
            "validation_tool_status": self.validation_tool_status.value,
            "research_gate_status": self.research_gate_status.value,
            "production_gate_status": self.production_gate_status.value,
            "checks": [check.canonical() for check in self.checks],
            "reason_codes": list(self.reason_codes),
            "evidence_gaps": [gap.canonical() for gap in self.evidence_gaps],
            "dsr_track_record_gap": self.dsr_track_record_gap.canonical(),
            "policy_digest": self.policy_digest,
            "report_digest": self.report_digest,
            "holdout": self.holdout.canonical(),
            "warnings": [
                "A validation-tool PASS is not a profitability or deployment claim.",
                "Thresholds must be registered before observing benchmark results.",
            ],
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


def _scenario(
    report: TimeSeriesBenchmarkReport, cost_bps: float
) -> StrategyScenarioReport:
    matches = [
        scenario
        for scenario in report.scenarios
        if abs(scenario.cost_bps - cost_bps) <= 1e-12
    ]
    if len(matches) != 1:
        raise StrategyBenchmarkError(
            f"acceptance requires exactly one {cost_bps:g} bps scenario"
        )
    return matches[0]


def _maximum_check(
    code: str, observed: float, threshold: float, cost_bps: float
) -> GateCheck:
    return GateCheck(
        code=code,
        status=GateStatus.PASS if observed <= threshold else GateStatus.FAIL,
        observed=observed,
        threshold=threshold,
        comparison="<=",
        cost_bps=cost_bps,
    )


def _minimum_check(
    code: str, observed: float, threshold: float, cost_bps: float
) -> GateCheck:
    return GateCheck(
        code=code,
        status=GateStatus.PASS if observed >= threshold else GateStatus.FAIL,
        observed=observed,
        threshold=threshold,
        comparison=">=",
        cost_bps=cost_bps,
    )


def _track_record_gap(
    evidence: DeflatedSharpeEvidence, target_probability: float
) -> DSRTrackRecordGap:
    annualizer = sqrt(evidence.annualization_sessions)
    selected = evidence.selected_annualized_sharpe / annualizer
    benchmark = evidence.benchmark_annualized_sharpe / annualizer
    delta = selected - benchmark
    required: int | None
    additional: int | None
    if delta <= 0.0:
        required = None
        additional = None
    else:
        denominator = sqrt(
            max(
                1.0
                - evidence.skewness * selected
                + ((evidence.kurtosis - 1.0) / 4.0) * selected * selected,
                1e-12,
            )
        )
        z_score = NormalDist().inv_cdf(target_probability)
        required = ceil(1.0 + (z_score * denominator / delta) ** 2)
        additional = max(0, required - evidence.observations)
    return DSRTrackRecordGap(
        current_sessions=evidence.observations,
        required_sessions=required,
        additional_sessions=additional,
        target_probability=target_probability,
        current_probability=evidence.probability,
    )


def assess_time_series_benchmark(
    report: TimeSeriesBenchmarkReport,
    policy: StrategyAcceptancePolicy = StrategyAcceptancePolicy(),
    holdout: StrategyHoldoutEvidence = StrategyHoldoutEvidence(),
) -> StrategyAcceptanceDecision:
    """Assess pre-registered research and production gates without retuning."""

    primary = _scenario(report, policy.primary_cost_bps)
    stress = _scenario(report, policy.stress_cost_bps)
    analysis = primary.analysis
    distribution = analysis.cpcv.sharpe_distribution
    checks = (
        _maximum_check(
            "PBO_ABOVE_THRESHOLD",
            analysis.pbo.probability,
            policy.maximum_pbo,
            primary.cost_bps,
        ),
        _minimum_check(
            "DSR_BELOW_THRESHOLD",
            analysis.deflated_sharpe.probability,
            policy.minimum_dsr_probability,
            primary.cost_bps,
        ),
        _minimum_check(
            "CPCV_MEDIAN_BELOW_THRESHOLD",
            distribution.median,
            policy.minimum_cpcv_median_sharpe,
            primary.cost_bps,
        ),
        _minimum_check(
            "CPCV_P10_BELOW_THRESHOLD",
            distribution.p10,
            policy.minimum_cpcv_p10_sharpe,
            primary.cost_bps,
        ),
        _minimum_check(
            "CPCV_WORST_BELOW_THRESHOLD",
            distribution.worst,
            policy.minimum_cpcv_worst_sharpe,
            primary.cost_bps,
        ),
        _minimum_check(
            "WALK_FORWARD_BELOW_THRESHOLD",
            analysis.walk_forward.annualized_sharpe,
            policy.minimum_walk_forward_sharpe,
            primary.cost_bps,
        ),
        _minimum_check(
            "STRESS_CPCV_WORST_BELOW_THRESHOLD",
            stress.analysis.cpcv.sharpe_distribution.worst,
            policy.minimum_stress_cpcv_worst_sharpe,
            stress.cost_bps,
        ),
        _minimum_check(
            "STRESS_WALK_FORWARD_BELOW_THRESHOLD",
            stress.analysis.walk_forward.annualized_sharpe,
            policy.minimum_stress_walk_forward_sharpe,
            stress.cost_bps,
        ),
    )
    failed = tuple(check.code for check in checks if check.status is GateStatus.FAIL)
    research_status = GateStatus.FAIL if failed else GateStatus.PASS
    gaps: list[EvidenceGap] = []
    reasons = list(failed)
    if policy.require_untouched_holdout:
        if holdout.status is StrategyHoldoutStatus.NOT_RUN:
            reasons.append("UNTOUCHED_HOLDOUT_NOT_RUN")
            gaps.append(
                EvidenceGap(
                    code="UNTOUCHED_HOLDOUT_NOT_RUN",
                    message=(
                        "Freeze the strategy and thresholds, then run one previously "
                        "unseen strategy holdout exactly once."
                    ),
                )
            )
        elif holdout.status is StrategyHoldoutStatus.REUSED:
            reasons.append("HOLDOUT_REUSED")
            gaps.append(
                EvidenceGap(
                    code="HOLDOUT_REUSED",
                    message="Reused data cannot provide untouched production evidence.",
                )
            )
        elif holdout.status is StrategyHoldoutStatus.UNTOUCHED_FAIL:
            reasons.append("UNTOUCHED_HOLDOUT_FAILED")

    if research_status is GateStatus.FAIL:
        production_status = GateStatus.FAIL
        reasons.append("RESEARCH_GATE_FAILED")
    elif not policy.require_untouched_holdout:
        production_status = GateStatus.PASS
    elif holdout.status is StrategyHoldoutStatus.NOT_RUN:
        production_status = GateStatus.INCONCLUSIVE
    elif holdout.status is StrategyHoldoutStatus.UNTOUCHED_PASS:
        production_status = GateStatus.PASS
    else:
        production_status = GateStatus.FAIL

    return StrategyAcceptanceDecision(
        validation_tool_status=GateStatus.PASS,
        research_gate_status=research_status,
        production_gate_status=production_status,
        checks=checks,
        reason_codes=tuple(dict.fromkeys(reasons)),
        evidence_gaps=tuple(gaps),
        dsr_track_record_gap=_track_record_gap(
            analysis.deflated_sharpe, policy.minimum_dsr_probability
        ),
        policy_digest=policy.digest,
        report_digest=report.digest,
        holdout=holdout,
    )
