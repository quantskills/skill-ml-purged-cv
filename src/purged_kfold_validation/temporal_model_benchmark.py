"""Trainable temporal-model comparison across unsafe and leakage-safe channels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .benchmark import BenchmarkChannelResult, run_validation_benchmark
from .domain import (
    FoldAssignment,
    InformationInterval,
    MetricSpec,
    ModelSpec,
    PITSnapshot,
    ValidationDataset,
    canonical_digest,
)
from .errors import StrategyBenchmarkError
from .evaluation import EstimatorFactory, LeakageSafeEvaluator
from .leakage import overlapping_candidate_positions
from .splitters import CausalWalkForward, CombinatorialPurgedCV, PurgedKFold
from .strategy_benchmark import DensePricePanel


def _positive_integer(value: int, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StrategyBenchmarkError(f"{name} must be an integer of at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class TemporalDatasetSpec:
    """Frozen causal lag/label geometry for one supervised temporal dataset."""

    lookback_sessions: int = 20
    label_horizon_sessions: int = 5

    def __post_init__(self) -> None:
        _positive_integer(self.lookback_sessions, name="lookback_sessions", minimum=2)
        _positive_integer(
            self.label_horizon_sessions,
            name="label_horizon_sessions",
            minimum=1,
        )

    @property
    def digest(self) -> str:
        return canonical_digest(self.canonical())

    def canonical(self) -> dict[str, int]:
        return {
            "lookback_sessions": self.lookback_sessions,
            "label_horizon_sessions": self.label_horizon_sessions,
        }


def build_temporal_supervised_dataset(
    panel: DensePricePanel,
    spec: TemporalDatasetSpec = TemporalDatasetSpec(),
) -> ValidationDataset:
    """Build Asset x Session lag sequences with explicit PIT and label intervals."""

    rows = len(panel.sessions)
    if spec.lookback_sessions + spec.label_horizon_sessions + 2 >= rows:
        raise StrategyBenchmarkError(
            "panel has insufficient sessions for temporal dataset geometry"
        )
    sample_ids: list[tuple[str, str]] = []
    sessions: list[np.datetime64] = []
    intervals: list[InformationInterval] = []
    features: list[np.ndarray] = []
    targets: list[float] = []
    asset_ids: list[str] = []
    decisions: list[np.datetime64] = []
    availability: list[np.ndarray] = []
    for decision_index in range(
        spec.lookback_sessions, rows - spec.label_horizon_sessions
    ):
        decision_session = panel.sessions[decision_index]
        feature_sessions = panel.sessions[
            decision_index - spec.lookback_sessions : decision_index
        ]
        label_end_index = decision_index + spec.label_horizon_sessions
        for asset_index, asset_id in enumerate(panel.asset_ids):
            sample_ids.append((asset_id, str(decision_session)))
            sessions.append(decision_session)
            intervals.append(
                InformationInterval(
                    feature_sessions[0], panel.sessions[label_end_index]
                )
            )
            features.append(
                np.asarray(
                    panel.tradable_returns[
                        decision_index - spec.lookback_sessions : decision_index,
                        asset_index,
                    ],
                    dtype=np.float64,
                )
            )
            targets.append(
                float(
                    panel.signal_prices[label_end_index, asset_index]
                    / panel.signal_prices[decision_index, asset_index]
                    - 1.0
                )
            )
            asset_ids.append(asset_id)
            decisions.append(decision_session)
            availability.append(feature_sessions)
    manifest_digest = canonical_digest(
        {
            "kind": "own-return-lag-sequence-v1",
            "panel_digest": panel.digest,
            "dataset_spec": spec.canonical(),
        }
    )
    return ValidationDataset(
        sample_ids=sample_ids,
        session_axis=panel.sessions,
        sessions=sessions,
        information_intervals=intervals,
        features=np.asarray(features, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.float64),
        asset_ids=asset_ids,
        decision_times=decisions,
        feature_availability=np.asarray(availability, dtype="datetime64[ns]"),
        pit_snapshot=PITSnapshot(
            snapshot_id=f"temporal-panel-{panel.digest[:16]}",
            source_digest=panel.source_digest,
            revision_policy="point-in-time",
        ),
        feature_manifest_digest=manifest_digest,
    )


@dataclass(frozen=True, slots=True)
class TemporalModelCase:
    """One fixed estimator factory and its immutable model identity."""

    estimator_factory: EstimatorFactory
    model_spec: ModelSpec

    def __post_init__(self) -> None:
        if not callable(self.estimator_factory):
            raise StrategyBenchmarkError("temporal model factory must be callable")


@dataclass(frozen=True, slots=True)
class TemporalModelBenchmarkConfig:
    """Frozen six-channel comparison policy."""

    n_splits: int = 5
    embargo_sessions: int = 20
    pre_test_gap_sessions: int = 5
    walk_forward_test_sessions: int = 120
    cpcv_groups: int = 6
    cpcv_test_groups: int = 2
    minimum_train_observations: int = 3_000
    minimum_train_sessions: int = 252
    random_seed: int = 20260804

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("n_splits", self.n_splits, 2),
            ("cpcv_groups", self.cpcv_groups, 3),
            ("cpcv_test_groups", self.cpcv_test_groups, 2),
            ("minimum_train_observations", self.minimum_train_observations, 1),
            ("minimum_train_sessions", self.minimum_train_sessions, 1),
            ("walk_forward_test_sessions", self.walk_forward_test_sessions, 1),
        ):
            _positive_integer(value, name=name, minimum=minimum)
        if self.cpcv_test_groups >= self.cpcv_groups:
            raise StrategyBenchmarkError(
                "cpcv_test_groups must be less than cpcv_groups"
            )
        for name, value in (
            ("embargo_sessions", self.embargo_sessions),
            ("pre_test_gap_sessions", self.pre_test_gap_sessions),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise StrategyBenchmarkError(f"{name} must be non-negative")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise StrategyBenchmarkError("random_seed must be an integer")

    @property
    def digest(self) -> str:
        return canonical_digest(self.canonical())

    def canonical(self) -> dict[str, int]:
        return {
            "n_splits": self.n_splits,
            "embargo_sessions": self.embargo_sessions,
            "pre_test_gap_sessions": self.pre_test_gap_sessions,
            "walk_forward_test_sessions": self.walk_forward_test_sessions,
            "cpcv_groups": self.cpcv_groups,
            "cpcv_test_groups": self.cpcv_test_groups,
            "minimum_train_observations": self.minimum_train_observations,
            "minimum_train_sessions": self.minimum_train_sessions,
            "random_seed": self.random_seed,
        }


@dataclass(frozen=True, slots=True)
class TemporalChannelEvidence:
    name: str
    evidence_channel: str
    mse: float
    per_fold_mse: tuple[float, ...]
    per_path_mse: tuple[float, ...]
    observation_count: int
    observation_coverage: float
    overlap_count: int
    minimum_train_observations: int | None
    minimum_train_sessions: int | None
    excluded_purged: int
    excluded_embargoed: int
    excluded_pre_test_gapped: int

    def canonical(self) -> dict[str, object]:
        paths = np.asarray(self.per_path_mse, dtype=np.float64)
        path_distribution: dict[str, float] | None = None
        if len(paths):
            path_distribution = {
                "median": float(np.median(paths)),
                "p10": float(np.quantile(paths, 0.10)),
                "p90": float(np.quantile(paths, 0.90)),
                "worst": float(np.max(paths)),
                "standard_deviation": float(np.std(paths)),
            }
        return {
            "name": self.name,
            "evidence_channel": self.evidence_channel,
            "mse": self.mse,
            "per_fold_mse": list(self.per_fold_mse),
            "per_path_mse": list(self.per_path_mse),
            "path_mse_distribution": path_distribution,
            "observation_count": self.observation_count,
            "observation_coverage": self.observation_coverage,
            "overlap_count": self.overlap_count,
            "training_minimum": {
                "observations": self.minimum_train_observations,
                "sessions": self.minimum_train_sessions,
            },
            "exclusions": {
                "purged": self.excluded_purged,
                "embargoed": self.excluded_embargoed,
                "pre_test_gapped": self.excluded_pre_test_gapped,
            },
        }


@dataclass(frozen=True, slots=True)
class TemporalModelEvidence:
    model_name: str
    model_version: str
    model_parameters: dict[str, Any]
    model_digest: str
    channels: tuple[TemporalChannelEvidence, ...]

    def canonical(self) -> dict[str, object]:
        by_name = {channel.name: channel for channel in self.channels}
        unsafe = by_name["unsafe-shuffled-kfold"].mse
        purged_embargo = by_name["purged-kfold-embargo"].mse
        walk_forward = by_name["causal-walk-forward"].mse
        return {
            "model": {
                "name": self.model_name,
                "version": self.model_version,
                "parameters": self.model_parameters,
                "digest": self.model_digest,
            },
            "channels": [channel.canonical() for channel in self.channels],
            "optimism_gap": {
                "purged_embargo_mse_minus_unsafe_mse": purged_embargo - unsafe,
                "walk_forward_mse_minus_unsafe_mse": walk_forward - unsafe,
                "interpretation": (
                    "Positive values mean the unsafe split reported lower error; "
                    "the difference is diagnostic and not wholly attributable to leakage."
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class TemporalModelBenchmarkReport:
    dataset: dict[str, object]
    config: TemporalModelBenchmarkConfig
    models: tuple[TemporalModelEvidence, ...]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "digest", canonical_digest(self.canonical(include_digest=False))
        )

    def canonical(self, *, include_digest: bool = True) -> dict[str, object]:
        safe_names = {
            "purged-kfold-no-embargo",
            "purged-kfold-embargo",
            "cpcv",
            "causal-walk-forward",
        }
        safe_overlaps = [
            channel.overlap_count
            for model in self.models
            for channel in model.channels
            if channel.name in safe_names
        ]
        unsafe_detected = all(
            next(
                channel.overlap_count
                for channel in model.channels
                if channel.name == "unsafe-shuffled-kfold"
            )
            > 0
            for model in self.models
        )
        embargo_exclusions = [
            channel.excluded_embargoed
            for model in self.models
            for channel in model.channels
            if channel.name == "purged-kfold-embargo"
        ]
        payload: dict[str, object] = {
            "schema_version": "1",
            "benchmark_kind": "trainable-temporal-model-leakage-comparison",
            "dataset": dict(self.dataset),
            "config": self.config.canonical(),
            "config_digest": self.config.digest,
            "models": [model.canonical() for model in self.models],
            "decision": {
                "leakage_control_status": (
                    "PASS" if safe_overlaps and max(safe_overlaps) == 0 else "FAIL"
                ),
                "unsafe_overlap_canary_detected": unsafe_detected,
                "embargo_incremental_status": (
                    "ACTIVE"
                    if embargo_exclusions and max(embargo_exclusions) > 0
                    else "NO_INCREMENTAL_EXCLUSION_AFTER_FULL_INTERVAL_PURGE"
                ),
                "production_authorization": "NOT_AUTHORIZED",
            },
            "warnings": [
                "Leakage-control PASS means retained safe folds have zero interval overlap; it does not mean model performance improved.",
                "Continuous-price and external point-in-time source claims remain caller-governed limitations.",
                "No model tuning, final holdout, trading costs, or deployment authorization is part of this report.",
            ],
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


def _mse(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean((targets - predictions) ** 2))


def _geometry(
    dataset: ValidationDataset, assignments: tuple[FoldAssignment, ...]
) -> tuple[int, int, int, int, int]:
    minimum_observations = min(len(item.train_positions) for item in assignments)
    minimum_sessions = min(
        len({dataset.sessions[int(position)] for position in item.train_positions})
        for item in assignments
    )
    return (
        minimum_observations,
        minimum_sessions,
        sum(item.exclusion_summary.purged for item in assignments),
        sum(item.exclusion_summary.embargoed for item in assignments),
        sum(item.exclusion_summary.pre_test_gapped for item in assignments),
    )


def _safe_evidence(
    name: str,
    dataset: ValidationDataset,
    splitter: PurgedKFold | CombinatorialPurgedCV,
    model: TemporalModelCase,
    metric: MetricSpec,
) -> TemporalChannelEvidence:
    plan = splitter.plan(dataset)
    assignments = plan.require_assignments()
    result = LeakageSafeEvaluator(
        splitter=splitter,
        estimator_factory=model.estimator_factory,
        model_spec=model.model_spec,
        metrics=(metric,),
    ).evaluate(dataset)
    derived = result.metrics[0]
    overlap = sum(
        len(
            overlapping_candidate_positions(
                dataset,
                candidate_positions=item.train_positions,
                protected_positions=item.test_positions,
            )
        )
        for item in assignments
    )
    if overlap:
        raise StrategyBenchmarkError(f"safe channel {name!r} retained interval overlap")
    minimum_observations, minimum_sessions, purged, embargoed, pre_gap = _geometry(
        dataset, assignments
    )
    return TemporalChannelEvidence(
        name=name,
        evidence_channel=result.evidence_channel.value,
        mse=derived.overall,
        per_fold_mse=derived.per_fold,
        per_path_mse=derived.per_path,
        observation_count=derived.observation_count,
        observation_coverage=derived.observation_coverage,
        overlap_count=overlap,
        minimum_train_observations=minimum_observations,
        minimum_train_sessions=minimum_sessions,
        excluded_purged=purged,
        excluded_embargoed=embargoed,
        excluded_pre_test_gapped=pre_gap,
    )


def _benchmark_evidence(
    source: BenchmarkChannelResult,
    *,
    name: str | None = None,
    geometry: tuple[int, int, int, int, int] | None = None,
) -> TemporalChannelEvidence:
    minimum_observations: int | None = None
    minimum_sessions: int | None = None
    purged = embargoed = pre_gap = 0
    if geometry is not None:
        minimum_observations, minimum_sessions, purged, embargoed, pre_gap = geometry
    return TemporalChannelEvidence(
        name=source.name if name is None else name,
        evidence_channel=source.evidence_channel,
        mse=source.metric_value,
        per_fold_mse=source.per_fold,
        per_path_mse=(),
        observation_count=source.observation_count,
        observation_coverage=source.observation_coverage,
        overlap_count=source.overlap_count,
        minimum_train_observations=minimum_observations,
        minimum_train_sessions=minimum_sessions,
        excluded_purged=purged,
        excluded_embargoed=embargoed,
        excluded_pre_test_gapped=pre_gap,
    )


def run_temporal_model_benchmark(
    dataset: ValidationDataset,
    model_cases: tuple[TemporalModelCase, ...],
    config: TemporalModelBenchmarkConfig = TemporalModelBenchmarkConfig(),
) -> TemporalModelBenchmarkReport:
    """Run fixed trainable models through the same six validation channels."""

    dataset.require_formal_scoring()
    models = tuple(model_cases)
    if not models:
        raise StrategyBenchmarkError("temporal benchmark requires at least one model")
    if len({model.model_spec.digest for model in models}) != len(models):
        raise StrategyBenchmarkError(
            "temporal benchmark model identities must be unique"
        )
    metric = MetricSpec(name="mean-squared-error", version="1", function=_mse)
    evidence: list[TemporalModelEvidence] = []
    for model in models:
        baseline = run_validation_benchmark(
            dataset,
            estimator_factory=model.estimator_factory,
            model_spec=model.model_spec,
            metric=metric,
            n_splits=config.n_splits,
            embargo_sessions=config.embargo_sessions,
            pre_test_gap_sessions=config.pre_test_gap_sessions,
            random_seed=config.random_seed,
            test_sessions=config.walk_forward_test_sessions,
            min_train_observations=config.minimum_train_observations,
            min_train_sessions=config.minimum_train_sessions,
        )
        by_name = {channel.name: channel for channel in baseline.channels}
        purged_embargo = PurgedKFold(
            n_splits=config.n_splits,
            embargo_sessions=config.embargo_sessions,
            min_train_samples=config.minimum_train_observations,
            min_train_sessions=config.minimum_train_sessions,
        )
        purged_geometry = _geometry(
            dataset, purged_embargo.plan(dataset).require_assignments()
        )
        walk_forward = CausalWalkForward(
            n_splits=config.n_splits,
            test_sessions=config.walk_forward_test_sessions,
            pre_test_gap_sessions=config.pre_test_gap_sessions,
            min_train_samples=config.minimum_train_observations,
            min_train_sessions=config.minimum_train_sessions,
        )
        walk_geometry = _geometry(
            dataset, walk_forward.plan(dataset).require_assignments()
        )
        purged_no_embargo = _safe_evidence(
            "purged-kfold-no-embargo",
            dataset,
            PurgedKFold(
                n_splits=config.n_splits,
                embargo_sessions=0,
                min_train_samples=config.minimum_train_observations,
                min_train_sessions=config.minimum_train_sessions,
            ),
            model,
            metric,
        )
        cpcv = _safe_evidence(
            "cpcv",
            dataset,
            CombinatorialPurgedCV(
                n_groups=config.cpcv_groups,
                n_test_groups=config.cpcv_test_groups,
                embargo_sessions=config.embargo_sessions,
                min_train_samples=config.minimum_train_observations,
                min_train_sessions=config.minimum_train_sessions,
            ),
            model,
            metric,
        )
        channels = (
            _benchmark_evidence(by_name["unsafe-shuffled-kfold"]),
            _benchmark_evidence(by_name["chronological-no-purge"]),
            purged_no_embargo,
            _benchmark_evidence(
                by_name["purged-kfold"],
                name="purged-kfold-embargo",
                geometry=purged_geometry,
            ),
            cpcv,
            _benchmark_evidence(by_name["causal-walk-forward"], geometry=walk_geometry),
        )
        evidence.append(
            TemporalModelEvidence(
                model_name=model.model_spec.name,
                model_version=model.model_spec.version,
                model_parameters=dict(model.model_spec.parameters),
                model_digest=model.model_spec.digest,
                channels=channels,
            )
        )
    observed_sessions = tuple(sorted(set(dataset.sessions)))
    dataset_summary: dict[str, object] = {
        "digest": dataset.digest,
        "observations": len(dataset.sample_ids),
        "sessions": len(observed_sessions),
        "assets": 0 if dataset.asset_ids is None else len(set(dataset.asset_ids)),
        "features": dataset.features.shape[1],
        "first_session": str(observed_sessions[0]),
        "last_session": str(observed_sessions[-1]),
        "feature_manifest_digest": dataset.feature_manifest_digest,
        "pit_snapshot_digest": (
            None
            if dataset.pit_snapshot is None
            else dataset.pit_snapshot.provenance_digest
        ),
    }
    return TemporalModelBenchmarkReport(
        dataset=dataset_summary,
        config=config,
        models=tuple(evidence),
    )
