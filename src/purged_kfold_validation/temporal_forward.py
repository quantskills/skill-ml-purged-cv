"""Append-only prediction-before-label evidence for frozen temporal models."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, cast

import numpy as np

from .domain import canonical_digest
from .errors import DuplicateForwardEvidenceError, ForwardEvidenceError


Clock = Callable[[], datetime]


class ForwardEvidenceStatus(str, Enum):
    """Governed maturity state of a temporal forward-evidence ledger."""

    WAITING_FOR_FUTURE_DATA = "WAITING_FOR_FUTURE_DATA"
    COLLECTING = "COLLECTING"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    FAIL = "FAIL"


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ForwardEvidenceError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ForwardEvidenceError(f"{name} must be a non-empty string")
    return value.strip()


def _session(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    try:
        session = np.datetime64(text, "D")
    except ValueError as exc:
        raise ForwardEvidenceError(f"{name} must be an ISO calendar date") from exc
    if np.isnat(session) or str(session) != text:
        raise ForwardEvidenceError(f"{name} must be an ISO calendar date")
    return text


def _instant(value: object, *, name: str) -> datetime:
    text = _text(value, name=name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ForwardEvidenceError(f"{name} must be an ISO-8601 instant") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ForwardEvidenceError(f"{name} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _instant_text(value: object, *, name: str) -> str:
    return _instant(value, name=name).isoformat()


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ForwardEvidenceError(f"{name} must be a positive integer")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForwardEvidenceError(f"{name} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ForwardEvidenceError(f"{name} must be a finite number")
    return number


def _canonical_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    try:
        normalized = json.loads(
            json.dumps(dict(value), ensure_ascii=False, allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise ForwardEvidenceError(
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


@dataclass(frozen=True, slots=True)
class TemporalForwardProtocol:
    """Pre-registered identity and acceptance boundary for future forecasts."""

    protocol_id: str
    development_report_digest: str
    development_dataset_digest: str
    model_spec_digest: str
    temporal_dataset_spec_digest: str
    development_label_end_session: str
    forward_start_session: str
    label_horizon_sessions: int
    minimum_matured_sessions: int
    minimum_matured_observations: int
    minimum_assets: int
    minimum_mean_session_spearman_ic: float
    require_model_mse_not_worse_than_zero: bool
    selection_policy: Mapping[str, Any]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "protocol_id", _text(self.protocol_id, name="protocol_id")
        )
        for name in (
            "development_report_digest",
            "development_dataset_digest",
            "model_spec_digest",
            "temporal_dataset_spec_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        development_end = _session(
            self.development_label_end_session,
            name="development_label_end_session",
        )
        forward_start = _session(
            self.forward_start_session, name="forward_start_session"
        )
        if np.datetime64(forward_start) <= np.datetime64(development_end):
            raise ForwardEvidenceError(
                "forward_start_session must be strictly after development_label_end_session"
            )
        object.__setattr__(self, "development_label_end_session", development_end)
        object.__setattr__(self, "forward_start_session", forward_start)
        for name in (
            "label_horizon_sessions",
            "minimum_matured_sessions",
            "minimum_matured_observations",
            "minimum_assets",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name=name)
            )
        ic = _finite_number(
            self.minimum_mean_session_spearman_ic,
            name="minimum_mean_session_spearman_ic",
        )
        if not -1.0 <= ic <= 1.0:
            raise ForwardEvidenceError(
                "minimum_mean_session_spearman_ic must be between -1 and 1"
            )
        if not isinstance(self.require_model_mse_not_worse_than_zero, bool):
            raise ForwardEvidenceError(
                "require_model_mse_not_worse_than_zero must be boolean"
            )
        policy = _canonical_mapping(self.selection_policy, name="selection_policy")
        object.__setattr__(self, "minimum_mean_session_spearman_ic", ic)
        object.__setattr__(self, "selection_policy", policy)
        object.__setattr__(self, "digest", canonical_digest(self._identity()))

    def _identity(self) -> dict[str, object]:
        return {
            "kind": "temporal-forward-protocol",
            "schema_version": "1",
            "protocol_id": self.protocol_id,
            "development_report_digest": self.development_report_digest,
            "development_dataset_digest": self.development_dataset_digest,
            "model_spec_digest": self.model_spec_digest,
            "temporal_dataset_spec_digest": self.temporal_dataset_spec_digest,
            "development_label_end_session": self.development_label_end_session,
            "forward_start_session": self.forward_start_session,
            "label_horizon_sessions": self.label_horizon_sessions,
            "minimum_matured_sessions": self.minimum_matured_sessions,
            "minimum_matured_observations": self.minimum_matured_observations,
            "minimum_assets": self.minimum_assets,
            "minimum_mean_session_spearman_ic": self.minimum_mean_session_spearman_ic,
            "require_model_mse_not_worse_than_zero": (
                self.require_model_mse_not_worse_than_zero
            ),
            "selection_policy": _plain_json(self.selection_policy),
        }

    def canonical(self) -> dict[str, object]:
        return {**self._identity(), "digest": self.digest}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TemporalForwardProtocol:
        if value.get("kind") not in {None, "temporal-forward-protocol"}:
            raise ForwardEvidenceError("protocol kind is invalid")
        if value.get("schema_version") not in {None, "1"}:
            raise ForwardEvidenceError("protocol schema_version must equal '1'")
        allowed = set(cls.__dataclass_fields__) - {"digest"}
        unknown = set(value) - allowed - {"kind", "schema_version", "digest"}
        missing = allowed - set(value)
        if unknown or missing:
            raise ForwardEvidenceError(
                f"protocol fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        policy = value["selection_policy"]
        if not isinstance(policy, Mapping):
            raise ForwardEvidenceError("selection_policy must be a JSON object")
        protocol = cls(
            protocol_id=value["protocol_id"],  # type: ignore[arg-type]
            development_report_digest=value["development_report_digest"],  # type: ignore[arg-type]
            development_dataset_digest=value["development_dataset_digest"],  # type: ignore[arg-type]
            model_spec_digest=value["model_spec_digest"],  # type: ignore[arg-type]
            temporal_dataset_spec_digest=value["temporal_dataset_spec_digest"],  # type: ignore[arg-type]
            development_label_end_session=value["development_label_end_session"],  # type: ignore[arg-type]
            forward_start_session=value["forward_start_session"],  # type: ignore[arg-type]
            label_horizon_sessions=value["label_horizon_sessions"],  # type: ignore[arg-type]
            minimum_matured_sessions=value["minimum_matured_sessions"],  # type: ignore[arg-type]
            minimum_matured_observations=value["minimum_matured_observations"],  # type: ignore[arg-type]
            minimum_assets=value["minimum_assets"],  # type: ignore[arg-type]
            minimum_mean_session_spearman_ic=value["minimum_mean_session_spearman_ic"],  # type: ignore[arg-type]
            require_model_mse_not_worse_than_zero=value[
                "require_model_mse_not_worse_than_zero"
            ],  # type: ignore[arg-type]
            selection_policy=cast(Mapping[str, Any], policy),
        )
        supplied_digest = value.get("digest")
        if supplied_digest is not None and supplied_digest != protocol.digest:
            raise ForwardEvidenceError("protocol digest does not match its contents")
        return protocol


@dataclass(frozen=True, slots=True)
class ForwardPredictionReceipt:
    """Forecast durably recorded before its label becomes available."""

    protocol_digest: str
    sample_id: str
    asset_id: str
    decision_session: str
    label_end_session: str
    label_available_at: str
    prediction: float
    feature_snapshot_digest: str
    recorded_at: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protocol_digest",
            _digest(self.protocol_digest, name="protocol_digest"),
        )
        object.__setattr__(self, "sample_id", _text(self.sample_id, name="sample_id"))
        object.__setattr__(self, "asset_id", _text(self.asset_id, name="asset_id"))
        decision = _session(self.decision_session, name="decision_session")
        label_end = _session(self.label_end_session, name="label_end_session")
        if np.datetime64(label_end) <= np.datetime64(decision):
            raise ForwardEvidenceError(
                "label_end_session must be after decision_session"
            )
        available = _instant_text(self.label_available_at, name="label_available_at")
        recorded = _instant_text(self.recorded_at, name="recorded_at")
        if _instant(recorded, name="recorded_at") >= _instant(
            available, name="label_available_at"
        ):
            raise ForwardEvidenceError(
                "prediction must be recorded before label_available_at"
            )
        object.__setattr__(self, "decision_session", decision)
        object.__setattr__(self, "label_end_session", label_end)
        object.__setattr__(self, "label_available_at", available)
        object.__setattr__(
            self, "prediction", _finite_number(self.prediction, name="prediction")
        )
        object.__setattr__(
            self,
            "feature_snapshot_digest",
            _digest(self.feature_snapshot_digest, name="feature_snapshot_digest"),
        )
        object.__setattr__(self, "recorded_at", recorded)
        object.__setattr__(self, "digest", canonical_digest(self._identity()))

    def _identity(self) -> dict[str, object]:
        return {
            "kind": "forward-prediction-receipt",
            "schema_version": "1",
            "protocol_digest": self.protocol_digest,
            "sample_id": self.sample_id,
            "asset_id": self.asset_id,
            "decision_session": self.decision_session,
            "label_end_session": self.label_end_session,
            "label_available_at": self.label_available_at,
            "prediction": self.prediction,
            "feature_snapshot_digest": self.feature_snapshot_digest,
            "recorded_at": self.recorded_at,
        }

    def canonical(self) -> dict[str, object]:
        return {**self._identity(), "digest": self.digest}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ForwardPredictionReceipt:
        expected = {
            "kind",
            "schema_version",
            "protocol_digest",
            "sample_id",
            "asset_id",
            "decision_session",
            "label_end_session",
            "label_available_at",
            "prediction",
            "feature_snapshot_digest",
            "recorded_at",
            "digest",
        }
        if set(value) != expected:
            raise ForwardEvidenceError("prediction receipt fields do not match schema")
        if (
            value.get("kind") != "forward-prediction-receipt"
            or value.get("schema_version") != "1"
        ):
            raise ForwardEvidenceError("prediction receipt identity is invalid")
        receipt = cls(
            protocol_digest=value.get("protocol_digest"),  # type: ignore[arg-type]
            sample_id=value.get("sample_id"),  # type: ignore[arg-type]
            asset_id=value.get("asset_id"),  # type: ignore[arg-type]
            decision_session=value.get("decision_session"),  # type: ignore[arg-type]
            label_end_session=value.get("label_end_session"),  # type: ignore[arg-type]
            label_available_at=value.get("label_available_at"),  # type: ignore[arg-type]
            prediction=value.get("prediction"),  # type: ignore[arg-type]
            feature_snapshot_digest=value.get("feature_snapshot_digest"),  # type: ignore[arg-type]
            recorded_at=value.get("recorded_at"),  # type: ignore[arg-type]
        )
        if value.get("digest") != receipt.digest:
            raise ForwardEvidenceError("prediction digest does not match its contents")
        return receipt


@dataclass(frozen=True, slots=True)
class ForwardLabelSettlement:
    """Mature target appended after its matching prediction receipt."""

    protocol_digest: str
    prediction_digest: str
    target: float
    target_source_digest: str
    settled_at: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protocol_digest",
            _digest(self.protocol_digest, name="protocol_digest"),
        )
        object.__setattr__(
            self,
            "prediction_digest",
            _digest(self.prediction_digest, name="prediction_digest"),
        )
        object.__setattr__(self, "target", _finite_number(self.target, name="target"))
        object.__setattr__(
            self,
            "target_source_digest",
            _digest(self.target_source_digest, name="target_source_digest"),
        )
        object.__setattr__(
            self, "settled_at", _instant_text(self.settled_at, name="settled_at")
        )
        object.__setattr__(self, "digest", canonical_digest(self._identity()))

    def _identity(self) -> dict[str, object]:
        return {
            "kind": "forward-label-settlement",
            "schema_version": "1",
            "protocol_digest": self.protocol_digest,
            "prediction_digest": self.prediction_digest,
            "target": self.target,
            "target_source_digest": self.target_source_digest,
            "settled_at": self.settled_at,
        }

    def canonical(self) -> dict[str, object]:
        return {**self._identity(), "digest": self.digest}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ForwardLabelSettlement:
        expected = {
            "kind",
            "schema_version",
            "protocol_digest",
            "prediction_digest",
            "target",
            "target_source_digest",
            "settled_at",
            "digest",
        }
        if set(value) != expected:
            raise ForwardEvidenceError("settlement fields do not match schema")
        if (
            value.get("kind") != "forward-label-settlement"
            or value.get("schema_version") != "1"
        ):
            raise ForwardEvidenceError("settlement identity is invalid")
        settlement = cls(
            protocol_digest=value.get("protocol_digest"),  # type: ignore[arg-type]
            prediction_digest=value.get("prediction_digest"),  # type: ignore[arg-type]
            target=value.get("target"),  # type: ignore[arg-type]
            target_source_digest=value.get("target_source_digest"),  # type: ignore[arg-type]
            settled_at=value.get("settled_at"),  # type: ignore[arg-type]
        )
        if value.get("digest") != settlement.digest:
            raise ForwardEvidenceError("settlement digest does not match its contents")
        return settlement


@dataclass(frozen=True, slots=True)
class ForwardEvidenceReport:
    """Redacted deterministic projection of one forward-evidence ledger."""

    protocol_digest: str
    status: ForwardEvidenceStatus
    prediction_count: int
    settled_observation_count: int
    pending_prediction_count: int
    matured_session_count: int
    matured_asset_count: int
    model_mse: float | None
    zero_baseline_mse: float | None
    mean_session_spearman_ic: float | None
    contributing_ic_session_count: int
    checks: Mapping[str, bool]
    ledger_digest: str
    production_authorization: str = "NOT_AUTHORIZED"
    attestation_scope: str = "LOCAL_APPEND_ONLY_NOT_EXTERNALLY_NOTARIZED"
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        frozen = MappingProxyType(dict(sorted(self.checks.items())))
        object.__setattr__(self, "checks", frozen)
        object.__setattr__(self, "digest", canonical_digest(self._identity()))

    def _identity(self) -> dict[str, object]:
        return {
            "kind": "forward-evidence-report",
            "schema_version": "1",
            "protocol_digest": self.protocol_digest,
            "status": self.status.value,
            "prediction_count": self.prediction_count,
            "settled_observation_count": self.settled_observation_count,
            "pending_prediction_count": self.pending_prediction_count,
            "matured_session_count": self.matured_session_count,
            "matured_asset_count": self.matured_asset_count,
            "model_mse": self.model_mse,
            "zero_baseline_mse": self.zero_baseline_mse,
            "mean_session_spearman_ic": self.mean_session_spearman_ic,
            "contributing_ic_session_count": self.contributing_ic_session_count,
            "checks": dict(self.checks),
            "ledger_digest": self.ledger_digest,
            "production_authorization": self.production_authorization,
            "attestation_scope": self.attestation_scope,
        }

    def canonical(self) -> dict[str, object]:
        return {**self._identity(), "digest": self.digest}


@dataclass(frozen=True, slots=True)
class LocalTemporalForwardStore:
    """Atomic local store enforcing prediction-before-label ordering."""

    root: Path
    clock: Clock = field(
        default=lambda: datetime.now(timezone.utc), compare=False, repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    def register(self, protocol: TemporalForwardProtocol) -> Path:
        target = self.root / "protocols" / f"{protocol.digest}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = protocol.canonical()
        if target.exists():
            if self._read_mapping(target) != payload:
                raise ForwardEvidenceError(
                    "persisted protocol conflicts with supplied protocol"
                )
            return target
        self._exclusive_json(target, payload, "protocol")
        return target

    def record_prediction(
        self,
        protocol: TemporalForwardProtocol,
        *,
        sample_id: str,
        asset_id: str,
        decision_session: str,
        label_end_session: str,
        label_available_at: str,
        prediction: float,
        feature_snapshot_digest: str,
    ) -> ForwardPredictionReceipt:
        self._require_registered(protocol)
        decision = _session(decision_session, name="decision_session")
        if np.datetime64(decision) < np.datetime64(protocol.forward_start_session):
            raise ForwardEvidenceError(
                "decision_session precedes the frozen forward_start_session"
            )
        receipt = ForwardPredictionReceipt(
            protocol_digest=protocol.digest,
            sample_id=sample_id,
            asset_id=asset_id,
            decision_session=decision,
            label_end_session=label_end_session,
            label_available_at=label_available_at,
            prediction=prediction,
            feature_snapshot_digest=feature_snapshot_digest,
            recorded_at=self._now().isoformat(),
        )
        claim_digest = canonical_digest(
            {
                "kind": "forward-sample-claim",
                "protocol_digest": protocol.digest,
                "sample_id": receipt.sample_id,
            }
        )
        claim = self.root / "sample-claims" / f"{claim_digest}.json"
        self._exclusive_json(
            claim,
            {
                "schema_version": "1",
                "protocol_digest": protocol.digest,
                "sample_id": receipt.sample_id,
                "prediction_digest": receipt.digest,
            },
            "sample prediction",
        )
        prediction_path = self.root / "predictions" / f"{receipt.digest}.json"
        try:
            self._exclusive_json(prediction_path, receipt.canonical(), "prediction")
        except Exception as exc:
            raise ForwardEvidenceError(
                "prediction claim was consumed but its receipt could not be persisted"
            ) from exc
        return receipt

    def settle(
        self,
        protocol: TemporalForwardProtocol,
        *,
        prediction_digest: str,
        target: float,
        target_source_digest: str,
    ) -> ForwardLabelSettlement:
        self._require_registered(protocol)
        prediction_path = (
            self.root
            / "predictions"
            / f"{_digest(prediction_digest, name='prediction_digest')}.json"
        )
        if not prediction_path.is_file():
            raise ForwardEvidenceError("prediction receipt does not exist")
        prediction = ForwardPredictionReceipt.from_mapping(
            self._read_mapping(prediction_path)
        )
        if prediction.protocol_digest != protocol.digest:
            raise ForwardEvidenceError("prediction does not belong to this protocol")
        now = self._now()
        if now < _instant(prediction.label_available_at, name="label_available_at"):
            raise ForwardEvidenceError("label is not yet available for settlement")
        if now <= _instant(prediction.recorded_at, name="recorded_at"):
            raise ForwardEvidenceError(
                "settlement must occur after prediction recording"
            )
        settlement = ForwardLabelSettlement(
            protocol_digest=protocol.digest,
            prediction_digest=prediction.digest,
            target=target,
            target_source_digest=target_source_digest,
            settled_at=now.isoformat(),
        )
        target_path = self.root / "settlements" / f"{prediction.digest}.json"
        self._exclusive_json(
            target_path, settlement.canonical(), "prediction settlement"
        )
        return settlement

    def report(self, protocol: TemporalForwardProtocol) -> ForwardEvidenceReport:
        self._require_registered(protocol)
        predictions = self._load_predictions(protocol)
        settlements = self._load_settlements(protocol, predictions)
        settled_pairs = [
            (prediction, settlements[prediction.digest])
            for prediction in predictions
            if prediction.digest in settlements
        ]
        settled_count = len(settled_pairs)
        sessions = {prediction.decision_session for prediction, _ in settled_pairs}
        assets = {prediction.asset_id for prediction, _ in settled_pairs}
        model_mse: float | None = None
        zero_mse: float | None = None
        mean_ic: float | None = None
        ic_sessions = 0
        if settled_pairs:
            predicted = np.asarray([item.prediction for item, _ in settled_pairs])
            targets = np.asarray([item.target for _, item in settled_pairs])
            model_mse = float(np.mean((targets - predicted) ** 2))
            zero_mse = float(np.mean(targets**2))
            mean_ic, ic_sessions = _mean_session_spearman(settled_pairs)
        checks = {
            "minimum_matured_sessions": len(sessions)
            >= protocol.minimum_matured_sessions,
            "minimum_matured_observations": settled_count
            >= protocol.minimum_matured_observations,
            "minimum_assets": len(assets) >= protocol.minimum_assets,
            "model_mse_not_worse_than_zero": (
                model_mse is not None and zero_mse is not None and model_mse <= zero_mse
            ),
            "minimum_mean_session_spearman_ic": (
                mean_ic is not None
                and mean_ic >= protocol.minimum_mean_session_spearman_ic
            ),
        }
        sufficient = all(
            checks[name]
            for name in (
                "minimum_matured_sessions",
                "minimum_matured_observations",
                "minimum_assets",
            )
        )
        if not predictions:
            status = ForwardEvidenceStatus.WAITING_FOR_FUTURE_DATA
        elif not sufficient:
            status = ForwardEvidenceStatus.COLLECTING
        else:
            metric_pass = checks["minimum_mean_session_spearman_ic"]
            if protocol.require_model_mse_not_worse_than_zero:
                metric_pass = metric_pass and checks["model_mse_not_worse_than_zero"]
            status = (
                ForwardEvidenceStatus.READY_FOR_REVIEW
                if metric_pass
                else ForwardEvidenceStatus.FAIL
            )
        ledger_digest = canonical_digest(
            {
                "kind": "forward-evidence-ledger",
                "protocol_digest": protocol.digest,
                "prediction_digests": sorted(item.digest for item in predictions),
                "settlement_digests": sorted(
                    item.digest for item in settlements.values()
                ),
            }
        )
        return ForwardEvidenceReport(
            protocol_digest=protocol.digest,
            status=status,
            prediction_count=len(predictions),
            settled_observation_count=settled_count,
            pending_prediction_count=len(predictions) - settled_count,
            matured_session_count=len(sessions),
            matured_asset_count=len(assets),
            model_mse=model_mse,
            zero_baseline_mse=zero_mse,
            mean_session_spearman_ic=mean_ic,
            contributing_ic_session_count=ic_sessions,
            checks=checks,
            ledger_digest=ledger_digest,
        )

    def persist_report(self, report: ForwardEvidenceReport) -> Path:
        target = self.root / "reports" / f"{report.digest}.json"
        if not target.exists():
            self._exclusive_json(target, report.canonical(), "report")
        return target

    def _now(self) -> datetime:
        current = self.clock()
        if current.tzinfo is None or current.utcoffset() is None:
            raise ForwardEvidenceError(
                "store clock must return a timezone-aware instant"
            )
        return current.astimezone(timezone.utc)

    def _require_registered(self, protocol: TemporalForwardProtocol) -> None:
        target = self.root / "protocols" / f"{protocol.digest}.json"
        if not target.is_file():
            raise ForwardEvidenceError("protocol is not registered in this store")
        if self._read_mapping(target) != protocol.canonical():
            raise ForwardEvidenceError("registered protocol contents do not match")

    def _load_predictions(
        self, protocol: TemporalForwardProtocol
    ) -> tuple[ForwardPredictionReceipt, ...]:
        directory = self.root / "predictions"
        if not directory.exists():
            return ()
        result = tuple(
            ForwardPredictionReceipt.from_mapping(self._read_mapping(path))
            for path in sorted(directory.glob("*.json"))
        )
        if any(item.protocol_digest != protocol.digest for item in result):
            raise ForwardEvidenceError(
                "store contains a prediction from another protocol"
            )
        if len({item.sample_id for item in result}) != len(result):
            raise ForwardEvidenceError(
                "store contains duplicate prediction sample identities"
            )
        for item in result:
            prediction_path = directory / f"{item.digest}.json"
            if not prediction_path.is_file():
                raise ForwardEvidenceError(
                    "prediction filename does not match its digest"
                )
            claim_digest = canonical_digest(
                {
                    "kind": "forward-sample-claim",
                    "protocol_digest": protocol.digest,
                    "sample_id": item.sample_id,
                }
            )
            claim_path = self.root / "sample-claims" / f"{claim_digest}.json"
            expected_claim = {
                "schema_version": "1",
                "protocol_digest": protocol.digest,
                "sample_id": item.sample_id,
                "prediction_digest": item.digest,
            }
            if (
                not claim_path.is_file()
                or self._read_mapping(claim_path) != expected_claim
            ):
                raise ForwardEvidenceError(
                    "prediction sample claim is missing or inconsistent"
                )
        return result

    def _load_settlements(
        self,
        protocol: TemporalForwardProtocol,
        predictions: tuple[ForwardPredictionReceipt, ...],
    ) -> dict[str, ForwardLabelSettlement]:
        directory = self.root / "settlements"
        if not directory.exists():
            return {}
        result: dict[str, ForwardLabelSettlement] = {}
        prediction_digests = {item.digest for item in predictions}
        for path in sorted(directory.glob("*.json")):
            item = ForwardLabelSettlement.from_mapping(self._read_mapping(path))
            if item.protocol_digest != protocol.digest:
                raise ForwardEvidenceError(
                    "store contains a settlement from another protocol"
                )
            if item.prediction_digest not in prediction_digests:
                raise ForwardEvidenceError("store contains an orphan settlement")
            if path.stem != item.prediction_digest:
                raise ForwardEvidenceError(
                    "settlement filename does not match prediction"
                )
            if item.prediction_digest in result:
                raise ForwardEvidenceError("store contains duplicate settlements")
            result[item.prediction_digest] = item
        return result

    @staticmethod
    def _exclusive_json(path: Path, payload: Mapping[str, object], kind: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
        except FileExistsError as exc:
            raise DuplicateForwardEvidenceError(
                f"{kind} already exists in the append-only ledger"
            ) from exc
        except OSError as exc:
            raise ForwardEvidenceError(
                f"{kind} could not be persisted atomically"
            ) from exc

    @staticmethod
    def _read_mapping(path: Path) -> dict[str, object]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ForwardEvidenceError(
                f"ledger JSON cannot be read: {path.name}"
            ) from exc
        if not isinstance(value, dict):
            raise ForwardEvidenceError(f"ledger JSON must be an object: {path.name}")
        return cast(dict[str, object], value)


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + end - 1) / 2.0
        position = end
    return ranks


def _mean_session_spearman(
    pairs: list[tuple[ForwardPredictionReceipt, ForwardLabelSettlement]],
) -> tuple[float | None, int]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for prediction, settlement in pairs:
        grouped[prediction.decision_session].append(
            (prediction.prediction, settlement.target)
        )
    correlations: list[float] = []
    for rows in grouped.values():
        if len(rows) < 2:
            continue
        predicted_rank = _rank_average(np.asarray([item[0] for item in rows]))
        target_rank = _rank_average(np.asarray([item[1] for item in rows]))
        if np.std(predicted_rank) == 0.0 or np.std(target_rank) == 0.0:
            continue
        correlations.append(float(np.corrcoef(predicted_rank, target_rank)[0, 1]))
    if not correlations:
        return None, 0
    return float(np.mean(correlations)), len(correlations)
