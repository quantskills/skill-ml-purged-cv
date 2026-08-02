"""Canonical fold-local leakage-safe evaluation boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from .domain import (
    DerivedMetric,
    EvaluationResult,
    FoldAssignment,
    MetricSpec,
    ModelSpec,
    OOSLedger,
    OOSObservation,
    SplitPlan,
    TransformerSpec,
    ValidationDataset,
    canonical_digest,
)
from .errors import (
    EvaluationError,
    FactoryLifecycleError,
    MetricEvaluationError,
    PredictionShapeError,
)


class Splitter(Protocol):
    """Evidence-bearing split planner accepted by the canonical evaluator."""

    def plan(self, dataset: ValidationDataset) -> SplitPlan: ...


class Transformer(Protocol):
    """Fold-local learned transformation protocol."""

    def fit(self, features: np.ndarray, targets: np.ndarray) -> Any: ...

    def transform(self, features: np.ndarray) -> np.ndarray: ...


class Estimator(Protocol):
    """Fold-local estimator protocol."""

    def fit(self, features: np.ndarray, targets: np.ndarray) -> Any: ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...


EstimatorFactory = Callable[[], Estimator]
TransformerFactory = Callable[[], Transformer]


@dataclass(frozen=True, slots=True)
class LeakageSafeEvaluator:
    """Execute a complete split plan without allowing learned state to cross folds."""

    splitter: Splitter
    estimator_factory: EstimatorFactory
    model_spec: ModelSpec
    transformer_factories: tuple[TransformerFactory, ...] = ()
    transformer_specs: tuple[TransformerSpec, ...] = ()
    metrics: tuple[MetricSpec, ...] = ()

    def __post_init__(self) -> None:
        if not callable(self.estimator_factory):
            raise FactoryLifecycleError("estimator_factory must be callable")
        if any(not callable(factory) for factory in self.transformer_factories):
            raise FactoryLifecycleError("every transformer factory must be callable")
        object.__setattr__(
            self, "transformer_factories", tuple(self.transformer_factories)
        )
        specs = tuple(self.transformer_specs)
        if len(specs) != len(self.transformer_factories) or any(
            not isinstance(spec, TransformerSpec) for spec in specs
        ):
            raise FactoryLifecycleError(
                "transformer factories require one ordered TransformerSpec each"
            )
        object.__setattr__(self, "transformer_specs", specs)
        object.__setattr__(self, "metrics", tuple(self.metrics))

    def evaluate(self, dataset: ValidationDataset) -> EvaluationResult:
        """Return complete OOS evidence or fail without a partial result."""

        dataset.require_formal_scoring()
        plan = self.splitter.plan(dataset)
        assignments = plan.require_assignments()
        channels = {assignment.evidence_channel for assignment in assignments}
        if len(channels) != 1:
            raise EvaluationError(
                "formal evaluation requires one consistent Evidence Channel"
            )
        evidence_channel = next(iter(channels))
        snapshot = dataset.pit_snapshot
        assert snapshot is not None

        run_id = canonical_digest(
            {
                "kind": "evaluation-run",
                "dataset_digest": dataset.digest,
                "plan_digest": plan.digest,
                "model_digest": self.model_spec.digest,
                "metric_digests": [metric.digest for metric in self.metrics],
                "transformer_spec_digests": [
                    spec.digest for spec in self.transformer_specs
                ],
                "evidence_channel": evidence_channel.value,
            }
        )
        observations: list[OOSObservation] = []
        created_estimators: list[Estimator] = []
        created_transformers: list[Transformer] = []

        for assignment in assignments:
            train_features = np.array(
                dataset.features[assignment.train_positions], copy=True
            )
            test_features = np.array(
                dataset.features[assignment.test_positions], copy=True
            )
            train_targets = np.array(
                dataset.targets[assignment.train_positions], copy=True
            )

            for factory in self.transformer_factories:
                transformer = self._create_transformer(factory)
                if any(transformer is previous for previous in created_transformers):
                    raise FactoryLifecycleError(
                        "transformer factories reused an object across stages or folds"
                    )
                created_transformers.append(transformer)
                try:
                    transformer.fit(train_features, train_targets)
                    train_features = np.asarray(transformer.transform(train_features))
                    test_features = np.asarray(transformer.transform(test_features))
                except Exception as exc:
                    raise EvaluationError(
                        f"fold {assignment.fold_index} transformation failed"
                    ) from exc
                self._validate_feature_rows(
                    train_features,
                    expected=len(assignment.train_positions),
                    fold_index=assignment.fold_index,
                )
                self._validate_feature_rows(
                    test_features,
                    expected=len(assignment.test_positions),
                    fold_index=assignment.fold_index,
                )

            estimator = self._create_estimator()
            if any(estimator is previous for previous in created_estimators):
                raise FactoryLifecycleError(
                    "estimator factory reused an object across folds"
                )
            created_estimators.append(estimator)
            try:
                estimator.fit(train_features, train_targets)
                predictions = np.asarray(estimator.predict(test_features))
            except Exception as exc:
                raise EvaluationError(
                    f"fold {assignment.fold_index} estimator execution failed"
                ) from exc
            if predictions.ndim != 1 or predictions.shape[0] != len(
                assignment.test_positions
            ):
                raise PredictionShapeError(
                    f"fold {assignment.fold_index} predictions must be one-dimensional "
                    f"with {len(assignment.test_positions)} observations"
                )
            self._require_finite_numeric(
                predictions,
                message=f"fold {assignment.fold_index} predictions must be finite numeric values",
            )

            for position, prediction in zip(assignment.test_positions, predictions):
                combination_index: int | None = None
                group_index: int | None = None
                path_index: int | None = None
                if plan.path_decomposition is not None:
                    combination_index = assignment.combination_index
                    if combination_index is None:
                        raise EvaluationError(
                            "CPCV assignment is missing combination identity"
                        )
                    group_index = plan.path_decomposition.group_index_for(
                        dataset.sessions[int(position)]
                    )
                    if group_index not in assignment.test_group_indices:
                        raise EvaluationError(
                            "CPCV test observation is outside its selected groups"
                        )
                    path_index = plan.path_decomposition.path_index_for(
                        combination_index, group_index
                    )
                observations.append(
                    OOSObservation(
                        run_id=run_id,
                        sample_id=dataset.sample_ids[int(position)],
                        session=dataset.sessions[int(position)],
                        asset_id=(
                            None
                            if dataset.asset_ids is None
                            else dataset.asset_ids[int(position)]
                        ),
                        fold_index=assignment.fold_index,
                        split_id=assignment.split_id,
                        target=float(dataset.targets[int(position)]),
                        prediction=float(prediction),
                        dataset_digest=dataset.digest,
                        split_spec_digest=assignment.split_spec_digest,
                        model_digest=self.model_spec.digest,
                        pit_snapshot_digest=snapshot.provenance_digest,
                        feature_manifest_digest=dataset.feature_manifest_digest,
                        transformer_spec_digests=tuple(
                            spec.digest for spec in self.transformer_specs
                        ),
                        evidence_channel=evidence_channel,
                        combination_index=combination_index,
                        group_index=group_index,
                        path_index=path_index,
                    )
                )

        position_by_id = {
            sample_id: index for index, sample_id in enumerate(dataset.sample_ids)
        }
        observations.sort(
            key=lambda item: (
                position_by_id[item.sample_id],
                -1 if item.path_index is None else item.path_index,
                -1 if item.combination_index is None else item.combination_index,
            )
        )
        if plan.path_decomposition is not None:
            self._require_complete_cpcv_paths(
                observations,
                expected_sample_ids=dataset.sample_ids,
                expected_path_count=plan.path_decomposition.path_count,
            )
        ledger = OOSLedger(tuple(observations))
        derived = tuple(
            self._derive_metric(
                metric,
                ledger,
                assignments,
                len(dataset.sample_ids),
                expected_path_count=(
                    0
                    if plan.path_decomposition is None
                    else plan.path_decomposition.path_count
                ),
            )
            for metric in self.metrics
        )
        return EvaluationResult(
            run_id=run_id,
            plan_digest=plan.digest,
            ledger=ledger,
            metrics=derived,
            evidence_channel=evidence_channel,
        )

    def _create_transformer(self, factory: TransformerFactory) -> Transformer:
        try:
            transformer = factory()
        except Exception as exc:
            raise FactoryLifecycleError("transformer factory failed") from exc
        if not callable(getattr(transformer, "fit", None)) or not callable(
            getattr(transformer, "transform", None)
        ):
            raise FactoryLifecycleError(
                "transformer factory must create an unfitted fit/transform object"
            )
        return transformer

    def _create_estimator(self) -> Estimator:
        try:
            estimator = self.estimator_factory()
        except Exception as exc:
            raise FactoryLifecycleError("estimator factory failed") from exc
        if not callable(getattr(estimator, "fit", None)) or not callable(
            getattr(estimator, "predict", None)
        ):
            raise FactoryLifecycleError(
                "estimator factory must create an unfitted fit/predict object"
            )
        return estimator

    @staticmethod
    def _validate_feature_rows(
        features: np.ndarray, *, expected: int, fold_index: int
    ) -> None:
        if features.ndim != 2 or features.shape[0] != expected:
            raise PredictionShapeError(
                f"fold {fold_index} transformed features must have {expected} rows"
            )
        LeakageSafeEvaluator._require_finite_numeric(
            features,
            message=f"fold {fold_index} transformed features must be finite numeric values",
        )

    @staticmethod
    def _require_finite_numeric(values: np.ndarray, *, message: str) -> None:
        if not np.issubdtype(values.dtype, np.number) or not bool(
            np.isfinite(values).all()
        ):
            raise PredictionShapeError(message)

    @staticmethod
    def _require_complete_cpcv_paths(
        observations: list[OOSObservation],
        *,
        expected_sample_ids: tuple[Any, ...],
        expected_path_count: int,
    ) -> None:
        expected = set(expected_sample_ids)
        for path_index in range(expected_path_count):
            path_sample_ids = [
                item.sample_id for item in observations if item.path_index == path_index
            ]
            if (
                len(path_sample_ids) != len(expected_sample_ids)
                or set(path_sample_ids) != expected
            ):
                raise EvaluationError(
                    f"CPCV path {path_index} must cover every sample exactly once"
                )

    @staticmethod
    def _derive_metric(
        metric: MetricSpec,
        ledger: OOSLedger,
        assignments: tuple[FoldAssignment, ...],
        dataset_size: int,
        *,
        expected_path_count: int,
    ) -> DerivedMetric:
        try:
            overall = float(metric.function(ledger.targets, ledger.predictions))
            observations_by_fold = {
                assignment.fold_index: tuple(
                    item
                    for item in ledger.observations
                    if item.fold_index == assignment.fold_index
                )
                for assignment in assignments
            }
            per_fold = tuple(
                float(
                    metric.function(
                        np.asarray(
                            [item.target for item in observations_by_fold[index]]
                        ),
                        np.asarray(
                            [item.prediction for item in observations_by_fold[index]]
                        ),
                    )
                )
                for index in sorted(observations_by_fold)
            )
            observations_by_path = {
                path_index: tuple(
                    item
                    for item in ledger.observations
                    if item.path_index == path_index
                )
                for path_index in range(expected_path_count)
            }
            per_path = tuple(
                float(
                    metric.function(
                        np.asarray(
                            [item.target for item in observations_by_path[index]]
                        ),
                        np.asarray(
                            [item.prediction for item in observations_by_path[index]]
                        ),
                    )
                )
                for index in range(expected_path_count)
            )
            if not np.isfinite(overall) or not all(
                np.isfinite(value) for value in (*per_fold, *per_path)
            ):
                raise MetricEvaluationError(
                    f"metric {metric.name!r} must return finite values"
                )
        except MetricEvaluationError:
            raise
        except Exception as exc:
            raise MetricEvaluationError(f"metric {metric.name!r} failed") from exc
        fold_count = sum(bool(values) for values in observations_by_fold.values())
        path_count = sum(bool(values) for values in observations_by_path.values())
        unique_sample_count = len({item.sample_id for item in ledger.observations})
        return DerivedMetric(
            name=metric.name,
            version=metric.version,
            overall=overall,
            per_fold=per_fold,
            observation_count=len(ledger.observations),
            observation_coverage=(
                unique_sample_count / dataset_size if dataset_size else 0.0
            ),
            fold_count=fold_count,
            fold_coverage=(fold_count / len(assignments) if assignments else 0.0),
            ledger_digest=ledger.digest,
            metric_digest=metric.digest,
            per_path=per_path,
            path_count=path_count,
            path_coverage=(
                path_count / expected_path_count if expected_path_count else 0.0
            ),
        )
