"""Governed financial effectiveness comparison across safe validation channels."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Callable, Hashable

import numpy as np

from .domain import (
    FoldAssignment,
    EvaluationResult,
    ModelSpec,
    OOSObservation,
    TransformerSpec,
    ValidationDataset,
    canonical_digest,
)
from .errors import EvaluationError, MetricEvaluationError, SplitPlanError
from .evaluation import LeakageSafeEvaluator, Splitter, TransformerFactory
from .leakage import overlapping_candidate_positions
from .splitters import CausalWalkForward, CombinatorialPurgedCV, PurgedKFold


EstimatorFactory = Callable[[], Any]


@dataclass(frozen=True, slots=True)
class EffectivenessComparisonConfig:
    """Frozen split and sufficiency policy for one comparison."""

    n_groups: int = 6
    n_test_groups: int = 2
    walk_forward_splits: int = 5
    embargo_sessions: int = 20
    pre_test_gap_sessions: int = 20
    min_train_observations: int = 10_000
    min_train_sessions: int = 252
    min_train_assets: int = 20
    max_combinations: int = 10_000

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("n_groups", self.n_groups, 3),
            ("n_test_groups", self.n_test_groups, 2),
            ("walk_forward_splits", self.walk_forward_splits, 1),
            ("min_train_observations", self.min_train_observations, 1),
            ("min_train_sessions", self.min_train_sessions, 1),
            ("min_train_assets", self.min_train_assets, 1),
            ("max_combinations", self.max_combinations, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise SplitPlanError(f"{name} must be an integer of at least {minimum}")
        if self.n_test_groups >= self.n_groups:
            raise SplitPlanError("n_test_groups must be less than n_groups")
        for name, value in (
            ("embargo_sessions", self.embargo_sessions),
            ("pre_test_gap_sessions", self.pre_test_gap_sessions),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SplitPlanError(f"{name} must be a non-negative integer")

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "kind": "effectiveness-comparison-config",
                "n_groups": self.n_groups,
                "n_test_groups": self.n_test_groups,
                "walk_forward_splits": self.walk_forward_splits,
                "embargo_sessions": self.embargo_sessions,
                "pre_test_gap_sessions": self.pre_test_gap_sessions,
                "min_train_observations": self.min_train_observations,
                "min_train_sessions": self.min_train_sessions,
                "min_train_assets": self.min_train_assets,
                "max_combinations": self.max_combinations,
            }
        )


@dataclass(frozen=True, slots=True)
class FinancialMetrics:
    """Financial diagnostics for one complete OOS track."""

    mse: float
    cross_sectional_spearman_ic: float
    diagnostic_sharpe: float
    observations: int
    sessions: int
    ic_sessions: int
    sharpe_sessions: int

    def canonical(self) -> dict[str, float | int]:
        return {
            "mse": self.mse,
            "cross_sectional_spearman_ic": self.cross_sectional_spearman_ic,
            "diagnostic_sharpe": self.diagnostic_sharpe,
            "observations": self.observations,
            "sessions": self.sessions,
            "ic_sessions": self.ic_sessions,
            "sharpe_sessions": self.sharpe_sessions,
        }


@dataclass(frozen=True, slots=True)
class PathFinancialMetrics:
    path_index: int | None
    metrics: FinancialMetrics

    def canonical(self) -> dict[str, object]:
        return {"path_index": self.path_index, "metrics": self.metrics.canonical()}


@dataclass(frozen=True, slots=True)
class MetricDistribution:
    median: float
    worst: float
    standard_deviation: float
    p10: float
    p90: float

    def canonical(self) -> dict[str, float]:
        return {
            "median": self.median,
            "worst": self.worst,
            "standard_deviation": self.standard_deviation,
            "p10": self.p10,
            "p90": self.p90,
        }


@dataclass(frozen=True, slots=True)
class FinancialMetricDistributions:
    mse: MetricDistribution
    ic: MetricDistribution
    diagnostic_sharpe: MetricDistribution

    def canonical(self) -> dict[str, object]:
        return {
            "mse": self.mse.canonical(),
            "cross_sectional_spearman_ic": self.ic.canonical(),
            "diagnostic_sharpe": self.diagnostic_sharpe.canonical(),
        }


@dataclass(frozen=True, slots=True)
class EvaluationView:
    name: str
    unique_observations: int
    paths: tuple[PathFinancialMetrics, ...]
    distributions: FinancialMetricDistributions

    def canonical(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unique_observations": self.unique_observations,
            "paths": [path.canonical() for path in self.paths],
            "distributions": self.distributions.canonical(),
        }


@dataclass(frozen=True, slots=True)
class FoldTrainingGeometry:
    fold_index: int
    combination_index: int | None
    observations: int
    sessions: int
    assets: int
    sufficient: bool

    def canonical(self) -> dict[str, object]:
        return {
            "fold_index": self.fold_index,
            "combination_index": self.combination_index,
            "observations": self.observations,
            "sessions": self.sessions,
            "assets": self.assets,
            "sufficient": self.sufficient,
        }


@dataclass(frozen=True, slots=True)
class TrainingSufficiency:
    minimum_observations: int
    minimum_sessions: int
    minimum_assets: int
    folds: tuple[FoldTrainingGeometry, ...]

    @property
    def all_sufficient(self) -> bool:
        return all(fold.sufficient for fold in self.folds)

    def canonical(self) -> dict[str, object]:
        observations = np.asarray([fold.observations for fold in self.folds])
        sessions = np.asarray([fold.sessions for fold in self.folds])
        assets = np.asarray([fold.assets for fold in self.folds])
        return {
            "thresholds": {
                "observations": self.minimum_observations,
                "sessions": self.minimum_sessions,
                "assets": self.minimum_assets,
            },
            "all_sufficient": self.all_sufficient,
            "summary": {
                "observations": _min_median_max(observations),
                "sessions": _min_median_max(sessions),
                "assets": _min_median_max(assets),
            },
            "folds": [fold.canonical() for fold in self.folds],
        }


@dataclass(frozen=True, slots=True)
class GroupBreadth:
    group_index: int
    sessions: int
    observations: int
    assets: int

    def canonical(self) -> dict[str, int]:
        return {
            "group_index": self.group_index,
            "sessions": self.sessions,
            "observations": self.observations,
            "assets": self.assets,
        }


@dataclass(frozen=True, slots=True)
class EffectivenessChannelResult:
    name: str
    evidence_channel: str
    split_spec_digest: str
    observation_coverage: float
    overlap_count: int
    training_sufficiency: TrainingSufficiency
    native: EvaluationView
    common: EvaluationView
    roll_clean: EvaluationView

    def canonical(self) -> dict[str, object]:
        return {
            "name": self.name,
            "evidence_channel": self.evidence_channel,
            "split_spec_digest": self.split_spec_digest,
            "observation_coverage": self.observation_coverage,
            "overlap_count": self.overlap_count,
            "training_sufficiency": self.training_sufficiency.canonical(),
            "views": {
                "native": self.native.canonical(),
                "common": self.common.canonical(),
                "roll_clean": self.roll_clean.canonical(),
            },
        }


@dataclass(frozen=True, slots=True)
class EffectivenessComparisonReport:
    dataset_digest: str
    model_digest: str
    config_digest: str
    common_sample_count: int
    group_breadth: tuple[GroupBreadth, ...]
    channels: tuple[EffectivenessChannelResult, ...]
    transformer_spec_digests: tuple[str, ...] = ()
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_breadth", tuple(self.group_breadth))
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(
            self, "transformer_spec_digests", tuple(self.transformer_spec_digests)
        )
        object.__setattr__(
            self,
            "digest",
            canonical_digest(self.canonical(include_digest=False)),
        )

    def canonical(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "dataset_digest": self.dataset_digest,
            "model_digest": self.model_digest,
            "config_digest": self.config_digest,
            "transformer_spec_digests": list(self.transformer_spec_digests),
            "common_sample_count": self.common_sample_count,
            "metric_definitions": {
                "mse": "raw-forward-simple-return-mean-squared-error",
                "ic": "mean-session-cross-sectional-spearman-min-3-assets",
                "diagnostic_sharpe": (
                    "sqrt-252-zero-cost-demeaned-score-unit-gross-no-costs"
                ),
            },
            "group_breadth": [item.canonical() for item in self.group_breadth],
            "channels": [channel.canonical() for channel in self.channels],
            "claim_boundary": (
                "diagnostic validation evidence; not executable strategy, costs, "
                "holdout, profitability, or deployment evidence"
            ),
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


def run_cpcv_effectiveness_comparison(
    dataset: ValidationDataset,
    *,
    label_roll_clean: np.ndarray,
    estimator_factory: EstimatorFactory,
    model_spec: ModelSpec,
    transformer_factories: tuple[TransformerFactory, ...] = (),
    transformer_specs: tuple[TransformerSpec, ...] = (),
    config: EffectivenessComparisonConfig = EffectivenessComparisonConfig(),
) -> EffectivenessComparisonReport:
    """Compare safe channels and retain deterministic financial path evidence."""

    dataset.require_formal_scoring()
    if dataset.asset_ids is None:
        raise EvaluationError("effectiveness comparison requires explicit asset_ids")
    roll_mask = np.asarray(label_roll_clean, dtype=bool)
    if roll_mask.shape != (len(dataset.sample_ids),):
        raise EvaluationError("label_roll_clean must align with the dataset")
    observed_sessions = set(dataset.sessions)
    active_sessions = tuple(
        session for session in dataset.session_axis if session in observed_sessions
    )
    test_sessions = max(1, len(active_sessions) // (config.walk_forward_splits + 1))
    purged_splitter = PurgedKFold(
        n_splits=config.n_groups,
        embargo_sessions=config.embargo_sessions,
        min_train_sessions=config.min_train_sessions,
        min_train_samples=config.min_train_observations,
    )
    cpcv_splitter = CombinatorialPurgedCV(
        n_groups=config.n_groups,
        n_test_groups=config.n_test_groups,
        embargo_sessions=config.embargo_sessions,
        min_train_sessions=config.min_train_sessions,
        min_train_samples=config.min_train_observations,
        max_combinations=config.max_combinations,
    )
    walk_forward_splitter = CausalWalkForward(
        n_splits=config.walk_forward_splits,
        test_sessions=test_sessions,
        pre_test_gap_sessions=config.pre_test_gap_sessions,
        min_train_sessions=config.min_train_sessions,
        min_train_samples=config.min_train_observations,
    )
    clean_ids = {
        sample_id
        for sample_id, is_clean in zip(dataset.sample_ids, roll_mask)
        if bool(is_clean)
    }

    causal_result, causal_assignments, causal_sufficiency = _evaluate_channel(
        dataset,
        splitter=walk_forward_splitter,
        estimator_factory=estimator_factory,
        model_spec=model_spec,
        transformer_factories=transformer_factories,
        transformer_specs=transformer_specs,
        config=config,
    )
    common_ids = {item.sample_id for item in causal_result.ledger.observations}
    causal_channel = _channel_result(
        "causal-walk-forward",
        dataset,
        causal_result.ledger.observations,
        causal_assignments,
        causal_sufficiency,
        common_ids=common_ids,
        clean_ids=clean_ids,
        split_spec_digest=walk_forward_splitter.digest,
        evidence_channel=causal_result.evidence_channel.value,
    )

    channels_by_name: dict[str, EffectivenessChannelResult] = {
        "causal-walk-forward": causal_channel
    }
    del causal_result
    for name, splitter in (
        ("purged-kfold", purged_splitter),
        ("cpcv", cpcv_splitter),
    ):
        result, assignments, sufficiency = _evaluate_channel(
            dataset,
            splitter=splitter,
            estimator_factory=estimator_factory,
            model_spec=model_spec,
            transformer_factories=transformer_factories,
            transformer_specs=transformer_specs,
            config=config,
        )
        channels_by_name[name] = _channel_result(
            name,
            dataset,
            result.ledger.observations,
            assignments,
            sufficiency,
            common_ids=common_ids,
            clean_ids=clean_ids,
            split_spec_digest=splitter.digest,
            evidence_channel=result.evidence_channel.value,
        )
        del result

    cpcv_plan = cpcv_splitter.plan(dataset)
    assert cpcv_plan.path_decomposition is not None
    group_breadth = _group_breadth(dataset, cpcv_plan.path_decomposition.groups)
    return EffectivenessComparisonReport(
        dataset_digest=dataset.digest,
        model_digest=model_spec.digest,
        config_digest=config.digest,
        transformer_spec_digests=tuple(spec.digest for spec in transformer_specs),
        common_sample_count=len(common_ids),
        group_breadth=group_breadth,
        channels=tuple(
            channels_by_name[name]
            for name in ("purged-kfold", "cpcv", "causal-walk-forward")
        ),
    )


def _evaluate_channel(
    dataset: ValidationDataset,
    *,
    splitter: Splitter,
    estimator_factory: EstimatorFactory,
    model_spec: ModelSpec,
    transformer_factories: tuple[TransformerFactory, ...],
    transformer_specs: tuple[TransformerSpec, ...],
    config: EffectivenessComparisonConfig,
) -> tuple[EvaluationResult, tuple[FoldAssignment, ...], TrainingSufficiency]:
    plan = splitter.plan(dataset)
    assignments = plan.require_assignments()
    sufficiency = _training_sufficiency(dataset, assignments, config)
    if not sufficiency.all_sufficient:
        raise EvaluationError("validation channel failed training sufficiency gates")
    overlap_count = _overlap_count(dataset, assignments)
    if overlap_count:
        raise EvaluationError("safe validation channel retained interval overlap")
    result = LeakageSafeEvaluator(
        splitter=splitter,
        estimator_factory=estimator_factory,
        model_spec=model_spec,
        transformer_factories=transformer_factories,
        transformer_specs=transformer_specs,
    ).evaluate(dataset)
    return result, assignments, sufficiency


def _training_sufficiency(
    dataset: ValidationDataset,
    assignments: tuple[FoldAssignment, ...],
    config: EffectivenessComparisonConfig,
) -> TrainingSufficiency:
    assert dataset.asset_ids is not None
    folds = []
    for assignment in assignments:
        observations = len(assignment.train_positions)
        sessions = len(
            {dataset.sessions[int(position)] for position in assignment.train_positions}
        )
        assets = len(
            {
                dataset.asset_ids[int(position)]
                for position in assignment.train_positions
            }
        )
        folds.append(
            FoldTrainingGeometry(
                fold_index=assignment.fold_index,
                combination_index=assignment.combination_index,
                observations=observations,
                sessions=sessions,
                assets=assets,
                sufficient=(
                    observations >= config.min_train_observations
                    and sessions >= config.min_train_sessions
                    and assets >= config.min_train_assets
                ),
            )
        )
    return TrainingSufficiency(
        minimum_observations=config.min_train_observations,
        minimum_sessions=config.min_train_sessions,
        minimum_assets=config.min_train_assets,
        folds=tuple(folds),
    )


def _channel_result(
    name: str,
    dataset: ValidationDataset,
    observations: tuple[OOSObservation, ...],
    assignments: tuple[FoldAssignment, ...],
    sufficiency: TrainingSufficiency,
    *,
    common_ids: set[Hashable],
    clean_ids: set[Hashable],
    split_spec_digest: str,
    evidence_channel: str,
) -> EffectivenessChannelResult:
    native_ids = {item.sample_id for item in observations}
    return EffectivenessChannelResult(
        name=name,
        evidence_channel=evidence_channel,
        split_spec_digest=split_spec_digest,
        observation_coverage=len(native_ids) / len(dataset.sample_ids),
        overlap_count=_overlap_count(dataset, assignments),
        training_sufficiency=sufficiency,
        native=_evaluation_view("native", observations, allowed_ids=native_ids),
        common=_evaluation_view("common", observations, allowed_ids=common_ids),
        roll_clean=_evaluation_view(
            "roll-clean", observations, allowed_ids=native_ids & clean_ids
        ),
    )


def _evaluation_view(
    name: str,
    observations: tuple[OOSObservation, ...],
    *,
    allowed_ids: set[Hashable],
) -> EvaluationView:
    selected = tuple(item for item in observations if item.sample_id in allowed_ids)
    path_values = sorted(
        {item.path_index for item in selected if item.path_index is not None}
    )
    groups: tuple[tuple[int | None, tuple[OOSObservation, ...]], ...]
    if path_values:
        groups = tuple(
            (
                path_index,
                tuple(item for item in selected if item.path_index == path_index),
            )
            for path_index in path_values
        )
    else:
        groups = ((None, selected),)
    paths = tuple(
        PathFinancialMetrics(path_index, _financial_metrics(path_observations))
        for path_index, path_observations in groups
    )
    return EvaluationView(
        name=name,
        unique_observations=len({item.sample_id for item in selected}),
        paths=paths,
        distributions=_metric_distributions(paths),
    )


def _financial_metrics(
    observations: tuple[OOSObservation, ...],
) -> FinancialMetrics:
    if not observations:
        raise MetricEvaluationError("financial metric view has no observations")
    targets = np.asarray([item.target for item in observations], dtype=float)
    predictions = np.asarray([item.prediction for item in observations], dtype=float)
    mse = float(np.mean((targets - predictions) ** 2))
    by_session: dict[np.datetime64, list[OOSObservation]] = {}
    for item in observations:
        by_session.setdefault(item.session, []).append(item)
    ic_values: list[float] = []
    diagnostic_returns: list[float] = []
    for session_observations in by_session.values():
        session_targets = np.asarray(
            [item.target for item in session_observations], dtype=float
        )
        session_predictions = np.asarray(
            [item.prediction for item in session_observations], dtype=float
        )
        if len(session_observations) >= 3:
            target_ranks = _average_ranks(session_targets)
            prediction_ranks = _average_ranks(session_predictions)
            if target_ranks.std() > 0.0 and prediction_ranks.std() > 0.0:
                ic_values.append(
                    float(np.corrcoef(target_ranks, prediction_ranks)[0, 1])
                )
        if len(session_observations) >= 2:
            centered = session_predictions - session_predictions.mean()
            gross = float(np.abs(centered).sum())
            if gross > 0.0:
                diagnostic_returns.append(
                    float(np.sum((centered / gross) * session_targets))
                )
    if not ic_values:
        raise MetricEvaluationError("cross-sectional IC has no valid sessions")
    if len(diagnostic_returns) < 2:
        raise MetricEvaluationError("diagnostic Sharpe has fewer than two sessions")
    return_std = float(np.std(diagnostic_returns, ddof=1))
    if return_std == 0.0:
        raise MetricEvaluationError("diagnostic Sharpe has zero return dispersion")
    sharpe = float(np.mean(diagnostic_returns) / return_std * sqrt(252.0))
    values = (mse, float(np.mean(ic_values)), sharpe)
    if not all(np.isfinite(value) for value in values):
        raise MetricEvaluationError("financial metrics must be finite")
    return FinancialMetrics(
        mse=mse,
        cross_sectional_spearman_ic=values[1],
        diagnostic_sharpe=sharpe,
        observations=len(observations),
        sessions=len(by_session),
        ic_sessions=len(ic_values),
        sharpe_sessions=len(diagnostic_returns),
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _metric_distributions(
    paths: tuple[PathFinancialMetrics, ...],
) -> FinancialMetricDistributions:
    return FinancialMetricDistributions(
        mse=_distribution([path.metrics.mse for path in paths], higher_is_worse=True),
        ic=_distribution(
            [path.metrics.cross_sectional_spearman_ic for path in paths],
            higher_is_worse=False,
        ),
        diagnostic_sharpe=_distribution(
            [path.metrics.diagnostic_sharpe for path in paths],
            higher_is_worse=False,
        ),
    )


def _distribution(values: list[float], *, higher_is_worse: bool) -> MetricDistribution:
    array = np.asarray(values, dtype=float)
    return MetricDistribution(
        median=float(np.median(array)),
        worst=float(np.max(array) if higher_is_worse else np.min(array)),
        standard_deviation=float(np.std(array)),
        p10=float(np.quantile(array, 0.1)),
        p90=float(np.quantile(array, 0.9)),
    )


def _group_breadth(
    dataset: ValidationDataset, groups: tuple[Any, ...]
) -> tuple[GroupBreadth, ...]:
    assert dataset.asset_ids is not None
    result = []
    for group_index, group in enumerate(groups):
        positions = tuple(
            position
            for position, session in enumerate(dataset.sessions)
            if group.start_session <= session <= group.end_session
        )
        result.append(
            GroupBreadth(
                group_index=group_index,
                sessions=len({dataset.sessions[position] for position in positions}),
                observations=len(positions),
                assets=len({dataset.asset_ids[position] for position in positions}),
            )
        )
    return tuple(result)


def _overlap_count(
    dataset: ValidationDataset, assignments: tuple[FoldAssignment, ...]
) -> int:
    return sum(
        len(
            overlapping_candidate_positions(
                dataset,
                candidate_positions=assignment.train_positions,
                protected_positions=assignment.test_positions,
            )
        )
        for assignment in assignments
    )


def _min_median_max(values: np.ndarray) -> dict[str, float | int]:
    return {
        "minimum": int(np.min(values)),
        "median": float(np.median(values)),
        "maximum": int(np.max(values)),
    }
