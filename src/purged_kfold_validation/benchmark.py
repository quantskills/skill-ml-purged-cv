"""Deterministic structural benchmark across unsafe and leakage-safe channels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .domain import (
    FoldAssignment,
    MetricSpec,
    ModelSpec,
    ValidationDataset,
    canonical_digest,
)
from .errors import EvaluationError, SplitPlanError
from .evaluation import LeakageSafeEvaluator
from .leakage import overlapping_candidate_positions
from .splitters import CausalWalkForward, PurgedKFold


EstimatorFactory = Callable[[], Any]


@dataclass(frozen=True, slots=True)
class BenchmarkChannelResult:
    """One validation channel's metric, coverage, and structural leakage audit."""

    name: str
    evidence_channel: str
    fold_count: int
    observation_count: int
    observation_coverage: float
    metric_name: str
    metric_version: str
    metric_value: float
    per_fold: tuple[float, ...]
    overlap_count: int
    dataset_digest: str
    split_spec_digest: str
    model_digest: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "digest",
            canonical_digest(
                {
                    "name": self.name,
                    "evidence_channel": self.evidence_channel,
                    "fold_count": self.fold_count,
                    "observation_count": self.observation_count,
                    "observation_coverage": self.observation_coverage,
                    "metric_name": self.metric_name,
                    "metric_version": self.metric_version,
                    "metric_value": self.metric_value,
                    "per_fold": list(self.per_fold),
                    "overlap_count": self.overlap_count,
                    "dataset_digest": self.dataset_digest,
                    "split_spec_digest": self.split_spec_digest,
                    "model_digest": self.model_digest,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Immutable four-channel validation comparison."""

    dataset_digest: str
    model_digest: str
    metric_digest: str
    channels: tuple[BenchmarkChannelResult, ...]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        channels = tuple(self.channels)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(
            self,
            "digest",
            canonical_digest(
                {
                    "dataset_digest": self.dataset_digest,
                    "model_digest": self.model_digest,
                    "metric_digest": self.metric_digest,
                    "channel_digests": [channel.digest for channel in channels],
                }
            ),
        )

    def canonical(self) -> dict[str, Any]:
        """Return deterministic redacted JSON-compatible benchmark evidence."""

        return {
            "dataset_digest": self.dataset_digest,
            "model_digest": self.model_digest,
            "metric_digest": self.metric_digest,
            "digest": self.digest,
            "channels": [
                {
                    "name": channel.name,
                    "evidence_channel": channel.evidence_channel,
                    "fold_count": channel.fold_count,
                    "observation_count": channel.observation_count,
                    "observation_coverage": channel.observation_coverage,
                    "metric": {
                        "name": channel.metric_name,
                        "version": channel.metric_version,
                        "value": channel.metric_value,
                        "per_fold": list(channel.per_fold),
                    },
                    "overlap_count": channel.overlap_count,
                    "split_spec_digest": channel.split_spec_digest,
                    "digest": channel.digest,
                }
                for channel in self.channels
            ],
        }


def run_validation_benchmark(
    dataset: ValidationDataset,
    *,
    estimator_factory: EstimatorFactory,
    model_spec: ModelSpec,
    metric: MetricSpec,
    n_splits: int = 5,
    embargo_sessions: int = 0,
    pre_test_gap_sessions: int = 0,
    random_seed: int = 0,
) -> BenchmarkReport:
    """Compare two diagnostic baselines with Purged and causal evidence."""

    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise SplitPlanError("benchmark n_splits must be an integer of at least 2")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise SplitPlanError("random_seed must be an integer")
    dataset.require_formal_scoring()
    active_sessions = _active_sessions(dataset)
    if n_splits >= len(active_sessions):
        raise SplitPlanError(
            "benchmark n_splits must be smaller than active session count"
        )
    test_sessions = max(1, len(active_sessions) // (n_splits + 1))

    shuffled = _shuffled_session_assignments(
        dataset,
        active_sessions=active_sessions,
        n_splits=n_splits,
        random_seed=random_seed,
    )
    chronological = _chronological_assignments(
        dataset,
        active_sessions=active_sessions,
        n_splits=n_splits,
        test_sessions=test_sessions,
    )
    purged = PurgedKFold(
        n_splits=n_splits,
        embargo_sessions=embargo_sessions,
    )
    causal = CausalWalkForward(
        n_splits=n_splits,
        test_sessions=test_sessions,
        pre_test_gap_sessions=pre_test_gap_sessions,
    )
    channels = (
        _evaluate_baseline(
            "unsafe-shuffled-kfold",
            "diagnostic-unsafe-baseline",
            dataset,
            shuffled,
            estimator_factory,
            model_spec,
            metric,
            canonical_digest(
                {
                    "kind": "unsafe-shuffled-session-kfold",
                    "n_splits": n_splits,
                    "random_seed": random_seed,
                }
            ),
        ),
        _evaluate_baseline(
            "chronological-no-purge",
            "diagnostic-chronological-baseline",
            dataset,
            chronological,
            estimator_factory,
            model_spec,
            metric,
            canonical_digest(
                {
                    "kind": "chronological-no-purge",
                    "n_splits": n_splits,
                    "test_sessions": test_sessions,
                }
            ),
        ),
        _evaluate_safe(
            "purged-kfold",
            dataset,
            purged,
            estimator_factory,
            model_spec,
            metric,
        ),
        _evaluate_safe(
            "causal-walk-forward",
            dataset,
            causal,
            estimator_factory,
            model_spec,
            metric,
        ),
    )
    return BenchmarkReport(
        dataset_digest=dataset.digest,
        model_digest=model_spec.digest,
        metric_digest=metric.digest,
        channels=channels,
    )


def _evaluate_safe(
    name: str,
    dataset: ValidationDataset,
    splitter: PurgedKFold | CausalWalkForward,
    estimator_factory: EstimatorFactory,
    model_spec: ModelSpec,
    metric: MetricSpec,
) -> BenchmarkChannelResult:
    plan = splitter.plan(dataset)
    assignments = plan.require_assignments()
    result = LeakageSafeEvaluator(
        splitter=splitter,
        estimator_factory=estimator_factory,
        model_spec=model_spec,
        metrics=(metric,),
    ).evaluate(dataset)
    derived = result.metrics[0]
    overlap_count = _overlap_count(dataset, assignments)
    if overlap_count:
        raise EvaluationError(f"safe channel {name!r} retained interval overlap")
    return BenchmarkChannelResult(
        name=name,
        evidence_channel=result.evidence_channel.value,
        fold_count=derived.fold_count,
        observation_count=derived.observation_count,
        observation_coverage=derived.observation_coverage,
        metric_name=derived.name,
        metric_version=derived.version,
        metric_value=derived.overall,
        per_fold=derived.per_fold,
        overlap_count=overlap_count,
        dataset_digest=dataset.digest,
        split_spec_digest=splitter.digest,
        model_digest=model_spec.digest,
    )


def _evaluate_baseline(
    name: str,
    evidence_channel: str,
    dataset: ValidationDataset,
    assignments: tuple[tuple[np.ndarray, np.ndarray], ...],
    estimator_factory: EstimatorFactory,
    model_spec: ModelSpec,
    metric: MetricSpec,
    split_spec_digest: str,
) -> BenchmarkChannelResult:
    rows: list[tuple[int, int, float]] = []
    per_fold: list[float] = []
    created: list[Any] = []
    for fold_index, (train, test) in enumerate(assignments):
        try:
            estimator = estimator_factory()
        except Exception as exc:
            raise EvaluationError(
                f"benchmark {name!r} estimator factory failed"
            ) from exc
        if any(estimator is previous for previous in created):
            raise EvaluationError(
                f"benchmark {name!r} estimator factory reused an object"
            )
        created.append(estimator)
        if not callable(getattr(estimator, "fit", None)) or not callable(
            getattr(estimator, "predict", None)
        ):
            raise EvaluationError(
                f"benchmark {name!r} estimator must support fit and predict"
            )
        try:
            estimator.fit(dataset.features[train], dataset.targets[train])
            predictions = np.asarray(estimator.predict(dataset.features[test]))
        except Exception as exc:
            raise EvaluationError(
                f"benchmark {name!r} fold {fold_index} execution failed"
            ) from exc
        if predictions.shape != (len(test),) or not bool(
            np.issubdtype(predictions.dtype, np.number)
            and np.isfinite(predictions).all()
        ):
            raise EvaluationError(
                f"benchmark {name!r} fold {fold_index} returned invalid predictions"
            )
        fold_value = float(metric.function(dataset.targets[test], predictions))
        if not np.isfinite(fold_value):
            raise EvaluationError(f"benchmark metric {metric.name!r} was not finite")
        per_fold.append(fold_value)
        rows.extend(
            (int(position), fold_index, float(prediction))
            for position, prediction in zip(test, predictions)
        )
    rows.sort(key=lambda item: item[0])
    positions = np.asarray([row[0] for row in rows], dtype=np.int64)
    predictions = np.asarray([row[2] for row in rows], dtype=float)
    overall = float(metric.function(dataset.targets[positions], predictions))
    if not np.isfinite(overall):
        raise EvaluationError(f"benchmark metric {metric.name!r} was not finite")
    return BenchmarkChannelResult(
        name=name,
        evidence_channel=evidence_channel,
        fold_count=len(assignments),
        observation_count=len(rows),
        observation_coverage=len(rows) / len(dataset.sample_ids),
        metric_name=metric.name,
        metric_version=metric.version,
        metric_value=overall,
        per_fold=tuple(per_fold),
        overlap_count=_baseline_overlap_count(dataset, assignments),
        dataset_digest=dataset.digest,
        split_spec_digest=split_spec_digest,
        model_digest=model_spec.digest,
    )


def _active_sessions(dataset: ValidationDataset) -> tuple[np.datetime64, ...]:
    values = set(dataset.sessions)
    return tuple(session for session in dataset.session_axis if session in values)


def _positions_for_sessions(
    dataset: ValidationDataset, sessions: set[np.datetime64]
) -> np.ndarray:
    return np.asarray(
        [
            position
            for position, session in enumerate(dataset.sessions)
            if session in sessions
        ],
        dtype=np.int64,
    )


def _shuffled_session_assignments(
    dataset: ValidationDataset,
    *,
    active_sessions: tuple[np.datetime64, ...],
    n_splits: int,
    random_seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    rng = np.random.default_rng(random_seed)
    shuffled = rng.permutation(len(active_sessions))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_positions = np.arange(len(dataset.sample_ids), dtype=np.int64)
    for chunk in np.array_split(shuffled, n_splits):
        test_sessions = {active_sessions[int(position)] for position in chunk}
        test = _positions_for_sessions(dataset, test_sessions)
        train = np.setdiff1d(all_positions, test, assume_unique=True)
        folds.append((train, test))
    return tuple(folds)


def _chronological_assignments(
    dataset: ValidationDataset,
    *,
    active_sessions: tuple[np.datetime64, ...],
    n_splits: int,
    test_sessions: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    required = n_splits * test_sessions
    first_test = len(active_sessions) - required
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_index in range(n_splits):
        start = first_test + fold_index * test_sessions
        stop = start + test_sessions
        train_values = set(active_sessions[:start])
        test_values = set(active_sessions[start:stop])
        folds.append(
            (
                _positions_for_sessions(dataset, train_values),
                _positions_for_sessions(dataset, test_values),
            )
        )
    return tuple(folds)


def _baseline_overlap_count(
    dataset: ValidationDataset,
    assignments: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> int:
    return sum(
        _positions_overlap_count(dataset, train, test) for train, test in assignments
    )


def _overlap_count(
    dataset: ValidationDataset, assignments: tuple[FoldAssignment, ...]
) -> int:
    return sum(
        _positions_overlap_count(
            dataset, assignment.train_positions, assignment.test_positions
        )
        for assignment in assignments
    )


def _positions_overlap_count(
    dataset: ValidationDataset, train: np.ndarray, test: np.ndarray
) -> int:
    return len(
        overlapping_candidate_positions(
            dataset,
            candidate_positions=train,
            protected_positions=test,
        )
    )
