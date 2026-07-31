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
