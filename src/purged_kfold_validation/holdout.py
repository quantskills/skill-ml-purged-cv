"""Frozen, one-time, locally persisted holdout confirmation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, cast

import numpy as np

from .domain import (
    DerivedMetric,
    EvidenceChannel,
    EvaluationResult,
    MetricSpec,
    ModelSpec,
    OOSLedger,
    OOSObservation,
    TransformerSpec,
    ValidationDataset,
    canonical_digest,
)
from .errors import (
    HoldoutEvaluationError,
    HoldoutProtocolError,
    ReusedHoldoutError,
)
from .evaluation import EstimatorFactory, TransformerFactory


def _canonical_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    try:
        normalized = json.loads(
            json.dumps(dict(value), ensure_ascii=False, allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise HoldoutProtocolError(
            f"{name} must contain canonical JSON values"
        ) from exc
    return cast(Mapping[str, Any], _deep_freeze(normalized))


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _digest_text(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HoldoutProtocolError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    """Frozen identities required before a final holdout may be consumed."""

    protocol_id: str
    training_dataset_digest: str
    holdout_dataset_digest: str
    model_spec_digest: str
    transformer_spec_digests: tuple[str, ...]
    metric_digests: tuple[str, ...]
    search_policy: Mapping[str, Any]
    split_spec_digest: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.protocol_id, str) or not self.protocol_id.strip():
            raise HoldoutProtocolError("protocol_id must not be empty")
        object.__setattr__(self, "protocol_id", self.protocol_id.strip())
        for name in (
            "training_dataset_digest",
            "holdout_dataset_digest",
            "model_spec_digest",
            "split_spec_digest",
        ):
            object.__setattr__(self, name, _digest_text(getattr(self, name), name=name))
        transformer_digests = tuple(
            _digest_text(value, name="transformer_spec_digest")
            for value in self.transformer_spec_digests
        )
        metric_digests = tuple(
            _digest_text(value, name="metric_digest") for value in self.metric_digests
        )
        if not metric_digests:
            raise HoldoutProtocolError("at least one frozen metric is required")
        policy = _canonical_mapping(self.search_policy, name="search_policy")
        object.__setattr__(self, "transformer_spec_digests", transformer_digests)
        object.__setattr__(self, "metric_digests", metric_digests)
        object.__setattr__(self, "search_policy", policy)
        object.__setattr__(
            self,
            "digest",
            canonical_digest(
                {
                    "kind": "evaluation-protocol",
                    "protocol_id": self.protocol_id,
                    "training_dataset_digest": self.training_dataset_digest,
                    "holdout_dataset_digest": self.holdout_dataset_digest,
                    "model_spec_digest": self.model_spec_digest,
                    "transformer_spec_digests": list(transformer_digests),
                    "metric_digests": list(metric_digests),
                    "search_policy": _plain_json(policy),
                    "split_spec_digest": self.split_spec_digest,
                }
            ),
        )

    @classmethod
    def freeze(
        cls,
        *,
        protocol_id: str,
        training_dataset: ValidationDataset,
        holdout_dataset: ValidationDataset,
        model_spec: ModelSpec,
        metrics: tuple[MetricSpec, ...],
        search_policy: Mapping[str, Any],
        split_spec_digest: str,
        transformer_specs: tuple[TransformerSpec, ...] = (),
    ) -> EvaluationProtocol:
        """Freeze final-evaluation identities after enforcing a causal boundary."""

        training_dataset.require_formal_scoring()
        holdout_dataset.require_formal_scoring()
        if max(training_dataset.sessions) >= min(holdout_dataset.sessions):
            raise HoldoutProtocolError(
                "holdout sessions must be strictly after every training session"
            )
        if max(
            interval.end for interval in training_dataset.information_intervals
        ) >= min(interval.start for interval in holdout_dataset.information_intervals):
            raise HoldoutProtocolError(
                "training information intervals must end strictly before the holdout"
            )
        if set(training_dataset.sample_ids).intersection(holdout_dataset.sample_ids):
            raise HoldoutProtocolError(
                "training and holdout sample identities must not overlap"
            )
        return cls(
            protocol_id=protocol_id,
            training_dataset_digest=training_dataset.digest,
            holdout_dataset_digest=holdout_dataset.digest,
            model_spec_digest=model_spec.digest,
            transformer_spec_digests=tuple(spec.digest for spec in transformer_specs),
            metric_digests=tuple(metric.digest for metric in metrics),
            search_policy=search_policy,
            split_spec_digest=split_spec_digest,
        )


@dataclass(frozen=True, slots=True)
class HoldoutReceipt:
    """Redacted append-only identity and metric summary for one final evaluation."""

    protocol_digest: str
    holdout_dataset_digest: str
    result_run_id: str
    ledger_digest: str
    metric_values: Mapping[str, float]
    evaluated_at: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        values = {str(name): float(value) for name, value in self.metric_values.items()}
        if not values or any(not np.isfinite(value) for value in values.values()):
            raise HoldoutEvaluationError("holdout receipt metrics must be finite")
        frozen_values = MappingProxyType(dict(sorted(values.items())))
        canonical = {
            "schema_version": "1",
            "protocol_digest": self.protocol_digest,
            "holdout_dataset_digest": self.holdout_dataset_digest,
            "result_run_id": self.result_run_id,
            "ledger_digest": self.ledger_digest,
            "metric_values": dict(frozen_values),
            "evaluated_at": self.evaluated_at,
        }
        object.__setattr__(self, "metric_values", frozen_values)
        object.__setattr__(self, "digest", canonical_digest(canonical))

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": "1",
            "protocol_digest": self.protocol_digest,
            "holdout_dataset_digest": self.holdout_dataset_digest,
            "result_run_id": self.result_run_id,
            "ledger_digest": self.ledger_digest,
            "metric_values": dict(self.metric_values),
            "evaluated_at": self.evaluated_at,
            "digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class HoldoutEvaluation:
    """In-memory final result plus its redacted durable receipt."""

    result: EvaluationResult
    receipt: HoldoutReceipt


@dataclass(frozen=True, slots=True)
class LocalHoldoutStore:
    """Filesystem store that atomically consumes each holdout digest once."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    def evaluate_once(
        self,
        protocol: EvaluationProtocol,
        *,
        training_dataset: ValidationDataset,
        holdout_dataset: ValidationDataset,
        estimator_factory: EstimatorFactory,
        model_spec: ModelSpec,
        metrics: tuple[MetricSpec, ...],
        transformer_factories: tuple[TransformerFactory, ...] = (),
        transformer_specs: tuple[TransformerSpec, ...] = (),
    ) -> HoldoutEvaluation:
        """Consume the bound holdout before fitting and return final evidence once."""

        self._require_binding(
            protocol,
            training_dataset=training_dataset,
            holdout_dataset=holdout_dataset,
            model_spec=model_spec,
            metrics=metrics,
            transformer_specs=transformer_specs,
        )
        if len(transformer_factories) != len(transformer_specs):
            raise HoldoutProtocolError(
                "transformer factories require one ordered TransformerSpec each"
            )
        self._consume(protocol)
        try:
            result = self._evaluate(
                protocol,
                training_dataset=training_dataset,
                holdout_dataset=holdout_dataset,
                estimator_factory=estimator_factory,
                model_spec=model_spec,
                metrics=metrics,
                transformer_factories=transformer_factories,
                transformer_specs=transformer_specs,
            )
            receipt = HoldoutReceipt(
                protocol_digest=protocol.digest,
                holdout_dataset_digest=holdout_dataset.digest,
                result_run_id=result.run_id,
                ledger_digest=result.ledger.digest,
                metric_values={
                    f"{metric.name}@{metric.version}": metric.overall
                    for metric in result.metrics
                },
                evaluated_at=datetime.now(timezone.utc).isoformat(),
            )
            self._persist_receipt(receipt)
            return HoldoutEvaluation(result=result, receipt=receipt)
        except HoldoutEvaluationError:
            raise
        except Exception as exc:
            raise HoldoutEvaluationError(
                "holdout evaluation failed after consuming the holdout identity"
            ) from exc

    @staticmethod
    def _require_binding(
        protocol: EvaluationProtocol,
        *,
        training_dataset: ValidationDataset,
        holdout_dataset: ValidationDataset,
        model_spec: ModelSpec,
        metrics: tuple[MetricSpec, ...],
        transformer_specs: tuple[TransformerSpec, ...],
    ) -> None:
        observed = (
            training_dataset.digest,
            holdout_dataset.digest,
            model_spec.digest,
            tuple(spec.digest for spec in transformer_specs),
            tuple(metric.digest for metric in metrics),
        )
        expected = (
            protocol.training_dataset_digest,
            protocol.holdout_dataset_digest,
            protocol.model_spec_digest,
            protocol.transformer_spec_digests,
            protocol.metric_digests,
        )
        if observed != expected:
            raise HoldoutProtocolError(
                "supplied holdout evaluation components do not match the frozen protocol"
            )

    def _consume(self, protocol: EvaluationProtocol) -> None:
        claims = self.root / "claims"
        receipts = self.root / "receipts"
        claims.mkdir(parents=True, exist_ok=True)
        receipts.mkdir(parents=True, exist_ok=True)
        claim = claims / f"{protocol.holdout_dataset_digest}.json"
        payload = {
            "schema_version": "1",
            "holdout_dataset_digest": protocol.holdout_dataset_digest,
            "protocol_digest": protocol.digest,
            "status": "consumed",
        }
        try:
            with claim.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
        except FileExistsError as exc:
            raise ReusedHoldoutError(
                "holdout identity has already been consumed by an evaluation attempt"
            ) from exc
        except OSError as exc:
            raise HoldoutProtocolError(
                "holdout claim could not be persisted atomically"
            ) from exc

    @staticmethod
    def _evaluate(
        protocol: EvaluationProtocol,
        *,
        training_dataset: ValidationDataset,
        holdout_dataset: ValidationDataset,
        estimator_factory: EstimatorFactory,
        model_spec: ModelSpec,
        metrics: tuple[MetricSpec, ...],
        transformer_factories: tuple[TransformerFactory, ...],
        transformer_specs: tuple[TransformerSpec, ...],
    ) -> EvaluationResult:
        train_features = np.array(training_dataset.features, copy=True)
        holdout_features = np.array(holdout_dataset.features, copy=True)
        train_targets = np.array(training_dataset.targets, copy=True)
        for factory in transformer_factories:
            transformer = factory()
            if not callable(getattr(transformer, "fit", None)) or not callable(
                getattr(transformer, "transform", None)
            ):
                raise HoldoutEvaluationError(
                    "transformer factory must create an unfitted fit/transform object"
                )
            transformer.fit(train_features, train_targets)
            train_features = np.asarray(transformer.transform(train_features))
            holdout_features = np.asarray(transformer.transform(holdout_features))
            _finite_matrix(train_features, len(training_dataset.sample_ids), "training")
            _finite_matrix(holdout_features, len(holdout_dataset.sample_ids), "holdout")

        estimator = estimator_factory()
        if not callable(getattr(estimator, "fit", None)) or not callable(
            getattr(estimator, "predict", None)
        ):
            raise HoldoutEvaluationError(
                "estimator factory must create an unfitted fit/predict object"
            )
        estimator.fit(train_features, train_targets)
        predictions = np.asarray(estimator.predict(holdout_features))
        if (
            predictions.ndim != 1
            or len(predictions) != len(holdout_dataset.sample_ids)
            or not np.issubdtype(predictions.dtype, np.number)
            or not bool(np.isfinite(predictions).all())
        ):
            raise HoldoutEvaluationError(
                "holdout predictions must be finite and align with holdout rows"
            )
        run_id = canonical_digest(
            {
                "kind": "holdout-evaluation-run",
                "protocol_digest": protocol.digest,
                "model_digest": model_spec.digest,
                "transformer_spec_digests": [spec.digest for spec in transformer_specs],
            }
        )
        snapshot = holdout_dataset.pit_snapshot
        assert snapshot is not None
        ledger = OOSLedger(
            tuple(
                OOSObservation(
                    run_id=run_id,
                    sample_id=sample_id,
                    session=holdout_dataset.sessions[index],
                    asset_id=(
                        None
                        if holdout_dataset.asset_ids is None
                        else holdout_dataset.asset_ids[index]
                    ),
                    fold_index=0,
                    split_id=protocol.digest,
                    target=float(holdout_dataset.targets[index]),
                    prediction=float(predictions[index]),
                    dataset_digest=holdout_dataset.digest,
                    split_spec_digest=protocol.split_spec_digest,
                    model_digest=model_spec.digest,
                    pit_snapshot_digest=snapshot.provenance_digest,
                    feature_manifest_digest=holdout_dataset.feature_manifest_digest,
                    transformer_spec_digests=tuple(
                        spec.digest for spec in transformer_specs
                    ),
                    evidence_channel=EvidenceChannel.HOLDOUT_CONFIRMATION,
                )
                for index, sample_id in enumerate(holdout_dataset.sample_ids)
            )
        )
        derived: list[DerivedMetric] = []
        for metric in metrics:
            value = float(metric.function(ledger.targets, ledger.predictions))
            if not np.isfinite(value):
                raise HoldoutEvaluationError(
                    f"holdout metric {metric.name!r} must return a finite value"
                )
            derived.append(
                DerivedMetric(
                    name=metric.name,
                    version=metric.version,
                    overall=value,
                    per_fold=(value,),
                    observation_count=len(ledger.observations),
                    observation_coverage=1.0,
                    fold_count=1,
                    fold_coverage=1.0,
                    ledger_digest=ledger.digest,
                    metric_digest=metric.digest,
                )
            )
        return EvaluationResult(
            run_id=run_id,
            plan_digest=protocol.split_spec_digest,
            ledger=ledger,
            metrics=tuple(derived),
            evidence_channel=EvidenceChannel.HOLDOUT_CONFIRMATION,
        )

    def _persist_receipt(self, receipt: HoldoutReceipt) -> None:
        target = self.root / "receipts" / f"{receipt.digest}.json"
        try:
            with target.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    receipt.canonical(), handle, ensure_ascii=False, sort_keys=True
                )
                handle.write("\n")
        except OSError as exc:
            raise HoldoutEvaluationError(
                "holdout receipt could not be persisted"
            ) from exc


def _finite_matrix(features: np.ndarray, rows: int, name: str) -> None:
    if (
        features.ndim != 2
        or features.shape[0] != rows
        or not np.issubdtype(features.dtype, np.number)
        or not bool(np.isfinite(features).all())
    ):
        raise HoldoutEvaluationError(
            f"{name} transformed features must be a finite two-dimensional matrix"
        )
