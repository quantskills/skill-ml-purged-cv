"""Leakage-safe financial time-series model-selection evidence."""

from .domain import (
    DerivedMetric,
    EvaluationResult,
    ExclusionRecord,
    ExclusionSummary,
    FoldAssignment,
    InformationInterval,
    InvalidFold,
    MetricSpec,
    ModelSpec,
    OOSLedger,
    OOSObservation,
    PITSnapshot,
    SplitPlan,
    TestBlock,
    ValidationDataset,
)
from .evaluation import LeakageSafeEvaluator
from .errors import (
    AdapterValidationError,
    DatasetValidationError,
    EvaluationError,
    FactoryLifecycleError,
    InvalidFoldError,
    MetricEvaluationError,
    PredictionShapeError,
    PointInTimeValidationError,
    SplitPlanError,
    TemporalValidationError,
    ValidationError,
)
from .splitters import PurgedKFold

__version__ = "0.1.0"

__all__ = [
    "AdapterValidationError",
    "DatasetValidationError",
    "DerivedMetric",
    "EvaluationError",
    "EvaluationResult",
    "ExclusionRecord",
    "ExclusionSummary",
    "FactoryLifecycleError",
    "FoldAssignment",
    "InformationInterval",
    "InvalidFold",
    "InvalidFoldError",
    "LeakageSafeEvaluator",
    "MetricSpec",
    "MetricEvaluationError",
    "ModelSpec",
    "OOSLedger",
    "OOSObservation",
    "PITSnapshot",
    "PointInTimeValidationError",
    "PredictionShapeError",
    "PurgedKFold",
    "SplitPlan",
    "SplitPlanError",
    "TemporalValidationError",
    "TestBlock",
    "ValidationDataset",
    "ValidationError",
]
