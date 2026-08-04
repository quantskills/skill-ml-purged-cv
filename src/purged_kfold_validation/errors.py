"""Typed public failures for leakage-safe validation."""

from __future__ import annotations


class ValidationError(ValueError):
    """Base class for rejected validation evidence."""


class DatasetValidationError(ValidationError):
    """The canonical dataset contract is invalid."""


class TemporalValidationError(DatasetValidationError):
    """Temporal evidence is missing, ambiguous, or inconsistent."""


class PointInTimeValidationError(DatasetValidationError):
    """Declared feature evidence is insufficient for formal historical scoring."""


class AdapterValidationError(DatasetValidationError):
    """An explicit ecosystem mapping cannot construct the canonical dataset."""


class UploadLimitError(ValidationError):
    """A bounded local upload exceeds an explicitly configured resource limit."""


class HoldoutProtocolError(ValidationError):
    """A frozen holdout protocol or its supplied components do not match."""


class ReusedHoldoutError(HoldoutProtocolError):
    """A holdout identity has already been consumed by an evaluation attempt."""


class HoldoutEvaluationError(ValidationError):
    """A one-time holdout evaluation failed after consuming the holdout identity."""


class ForwardEvidenceError(ValidationError):
    """A temporal forward-evidence protocol or ledger operation is invalid."""


class DuplicateForwardEvidenceError(ForwardEvidenceError):
    """An append-only forward prediction or settlement identity already exists."""


class RankingStabilityError(ValidationError):
    """Cross-regime model-ranking evidence is incomplete or incomparable."""


class StrategyBenchmarkError(ValidationError):
    """Strategy-return or selection-overfitting evidence is invalid."""


class SplitPlanError(ValidationError):
    """A split configuration or formal split plan is invalid."""


class InvalidFoldError(SplitPlanError):
    """At least one requested fold cannot be evaluated."""


class EvaluationError(ValidationError):
    """Fold-local evaluation failed without producing partial evidence."""


class FactoryLifecycleError(EvaluationError):
    """A fold factory violated the fresh-object lifecycle contract."""


class PredictionShapeError(EvaluationError):
    """A fold prediction does not align with its test observations."""


class MetricEvaluationError(EvaluationError):
    """A derived metric could not be evaluated."""
