"""Explicit offline PandaAI daily-frame adapter for validation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from ..domain import InformationInterval, PITSnapshot, ValidationDataset
from ..errors import AdapterValidationError


@dataclass(frozen=True, slots=True)
class PandaAIDailyMapping:
    """Explicit PandaAI-style daily field mapping."""

    session: str
    asset: str
    close: str
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        fields = (self.session, self.asset, self.close, *self.features)
        if any(not isinstance(value, str) or not value for value in fields):
            raise AdapterValidationError("PandaAI mappings must be non-empty strings")
        if not self.features:
            raise AdapterValidationError("at least one feature column is required")
        if len(set(self.features)) != len(self.features):
            raise AdapterValidationError("feature mappings must be unique")


@dataclass(frozen=True, slots=True)
class PandaAIDailyConfig:
    """Declared temporal and source evidence for one daily benchmark dataset."""

    label_horizon_sessions: int
    feature_lookback_sessions: int
    decision_time_offset_minutes: int
    snapshot_id: str
    source_digest: str

    def __post_init__(self) -> None:
        for name, value in (
            ("label_horizon_sessions", self.label_horizon_sessions),
            ("feature_lookback_sessions", self.feature_lookback_sessions),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AdapterValidationError(f"{name} must be an integer of at least 1")
        if (
            isinstance(self.decision_time_offset_minutes, bool)
            or not isinstance(self.decision_time_offset_minutes, int)
            or self.decision_time_offset_minutes < 0
        ):
            raise AdapterValidationError(
                "decision_time_offset_minutes must be a non-negative integer"
            )
        if not self.snapshot_id or not self.source_digest:
            raise AdapterValidationError(
                "snapshot_id and source_digest must be supplied explicitly"
            )


@dataclass(frozen=True, slots=True)
class PandaAIContinuousMapping:
    """Explicit identity fields for a mixed continuous/concrete futures frame."""

    series_symbol: str
    asset: str
    active_contract: str
    declared_underlying: str | None = None

    def __post_init__(self) -> None:
        required = (self.series_symbol, self.asset, self.active_contract)
        if any(not isinstance(value, str) or not value for value in required):
            raise AdapterValidationError(
                "continuous-contract mappings must be non-empty strings"
            )
        if self.declared_underlying is not None and (
            not isinstance(self.declared_underlying, str)
            or not self.declared_underlying
        ):
            raise AdapterValidationError(
                "declared_underlying must be a non-empty string when supplied"
            )


@dataclass(frozen=True, slots=True)
class PandaAIContinuousPolicy:
    """Explicit vendor-declared continuous-series selection policy."""

    dominant_token: str = "_DOMINANT."
    adjustment_status: str = "vendor-declared-unverified"

    def __post_init__(self) -> None:
        if not isinstance(self.dominant_token, str) or not self.dominant_token:
            raise AdapterValidationError("dominant_token must be a non-empty string")
        if self.adjustment_status != "vendor-declared-unverified":
            raise AdapterValidationError(
                "adjustment_status must remain vendor-declared-unverified"
            )

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "dominant_token": self.dominant_token,
                "adjustment_status": self.adjustment_status,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PandaAIContinuousGovernanceReceipt:
    """Redacted deterministic evidence for continuous-contract normalization."""

    input_rows: int
    declared_continuous_rows: int
    selected_continuous_rows: int
    discarded_concrete_rows: int
    discarded_insufficient_rows: int
    selected_assets: tuple[str, ...]
    excluded_assets: tuple[str, ...]
    active_contract_transitions: int
    eligible_rows: int
    label_roll_crossing_rows: int
    information_interval_roll_crossing_rows: int
    source_digest: str
    policy_digest: str
    digest: str = ""

    def __post_init__(self) -> None:
        payload = self.canonical(include_digest=False)
        object.__setattr__(
            self,
            "digest",
            sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
        )

    def canonical(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "input_rows": self.input_rows,
            "declared_continuous_rows": self.declared_continuous_rows,
            "selected_continuous_rows": self.selected_continuous_rows,
            "discarded_concrete_rows": self.discarded_concrete_rows,
            "discarded_insufficient_rows": self.discarded_insufficient_rows,
            "selected_assets": list(self.selected_assets),
            "excluded_assets": list(self.excluded_assets),
            "active_contract_transitions": self.active_contract_transitions,
            "eligible_rows": self.eligible_rows,
            "label_roll_crossing_rows": self.label_roll_crossing_rows,
            "information_interval_roll_crossing_rows": (
                self.information_interval_roll_crossing_rows
            ),
            "source_digest": self.source_digest,
            "policy_digest": self.policy_digest,
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True, eq=False)
class GovernedPandaAIDataset:
    """Canonical Validation Dataset plus aligned roll-sensitivity evidence."""

    dataset: ValidationDataset
    receipt: PandaAIContinuousGovernanceReceipt
    label_roll_clean: np.ndarray

    def __post_init__(self) -> None:
        mask = np.array(self.label_roll_clean, dtype=bool, copy=True)
        if mask.ndim != 1 or len(mask) != len(self.dataset.sample_ids):
            raise AdapterValidationError(
                "label_roll_clean must align with the Validation Dataset"
            )
        mask.setflags(write=False)
        object.__setattr__(self, "label_roll_clean", mask)


def governed_validation_dataset_from_pandaai_continuous_daily(
    frame: pd.DataFrame,
    *,
    mapping: PandaAIDailyMapping,
    config: PandaAIDailyConfig,
    continuous_mapping: PandaAIContinuousMapping,
    policy: PandaAIContinuousPolicy,
) -> GovernedPandaAIDataset:
    """Select and receipt vendor-declared continuous rows before normalization."""

    if not isinstance(frame, pd.DataFrame):
        raise AdapterValidationError("frame must be a pandas DataFrame")
    if not isinstance(continuous_mapping, PandaAIContinuousMapping):
        raise AdapterValidationError(
            "continuous_mapping must be PandaAIContinuousMapping"
        )
    if not isinstance(policy, PandaAIContinuousPolicy):
        raise AdapterValidationError("policy must be PandaAIContinuousPolicy")
    required = {
        continuous_mapping.series_symbol,
        continuous_mapping.asset,
        continuous_mapping.active_contract,
    }
    if continuous_mapping.declared_underlying is not None:
        required.add(continuous_mapping.declared_underlying)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AdapterValidationError(
            f"continuous-contract frame is missing columns: {missing}"
        )

    identity = frame.loc[:, list(required)].copy()
    series = _nonempty_identity_series(
        identity[continuous_mapping.series_symbol], field_name="series symbol"
    )
    assets = _nonempty_identity_series(
        identity[continuous_mapping.asset], field_name="continuous asset"
    )
    dominant_mask = series.str.contains(policy.dominant_token, regex=False)
    declared = frame.loc[dominant_mask].copy()
    if declared.empty:
        raise AdapterValidationError(
            "frame contains no rows matching the declared dominant token"
        )
    declared[continuous_mapping.series_symbol] = series.loc[dominant_mask]
    declared[continuous_mapping.asset] = assets.loc[dominant_mask]
    declared[continuous_mapping.active_contract] = _nonempty_identity_series(
        declared[continuous_mapping.active_contract], field_name="active contract"
    )
    if continuous_mapping.declared_underlying is not None:
        underlying = _nonempty_identity_series(
            declared[continuous_mapping.declared_underlying],
            field_name="declared underlying",
        )
        if not bool(
            underlying.eq(declared[continuous_mapping.asset].astype(str)).all()
        ):
            raise AdapterValidationError(
                "declared underlying does not match the authoritative asset"
            )

    try:
        declared["__governance_session"] = pd.to_datetime(
            declared[mapping.session], errors="raise"
        ).dt.normalize()
    except (TypeError, ValueError) as exc:
        raise AdapterValidationError("session column contains invalid times") from exc
    if declared.duplicated([continuous_mapping.asset, "__governance_session"]).any():
        raise AdapterValidationError("continuous rows contain duplicate asset sessions")
    series_per_asset = declared.groupby(continuous_mapping.asset)[
        continuous_mapping.series_symbol
    ].nunique()
    if bool(series_per_asset.gt(1).any()):
        raise AdapterValidationError(
            "each asset must map to exactly one continuous series identity"
        )

    required_source_sessions = (
        config.feature_lookback_sessions + config.label_horizon_sessions
    )
    source_counts = declared.groupby(continuous_mapping.asset).size()
    selected_assets = tuple(
        sorted(
            str(asset)
            for asset, count in source_counts.items()
            if int(count) >= required_source_sessions
        )
    )
    all_assets = {str(value) for value in assets}
    excluded_assets = tuple(sorted(all_assets.difference(selected_assets)))
    selected = declared.loc[
        declared[continuous_mapping.asset].isin(selected_assets)
    ].copy()
    if selected.empty:
        raise AdapterValidationError(
            "no continuous asset has enough history for the declared horizons"
        )
    selected = selected.sort_values(
        [continuous_mapping.asset, "__governance_session"], kind="stable"
    )

    roll_clean_by_sample: dict[tuple[str, str], bool] = {}
    active_contract_transitions = 0
    label_roll_crossing_rows = 0
    information_interval_roll_crossing_rows = 0
    lookback = config.feature_lookback_sessions
    horizon = config.label_horizon_sessions
    for asset, group in selected.groupby(continuous_mapping.asset, sort=False):
        active = group[continuous_mapping.active_contract].astype(str).to_numpy()
        session_values = group["__governance_session"].tolist()
        transitions = active[1:] != active[:-1]
        active_contract_transitions += int(transitions.sum())
        for position in range(lookback - 1, len(group) - horizon):
            label_crosses = bool(
                np.any(
                    active[position + 1 : position + horizon + 1] != active[position]
                )
            )
            interval_start = position - lookback + 1
            interval_end = position + horizon
            interval_crosses = bool(
                np.any(
                    active[interval_start + 1 : interval_end + 1]
                    != active[interval_start:interval_end]
                )
            )
            label_roll_crossing_rows += int(label_crosses)
            information_interval_roll_crossing_rows += int(interval_crosses)
            roll_clean_by_sample[
                (str(asset), session_values[position].strftime("%Y-%m-%d"))
            ] = not label_crosses

    dataset = validation_dataset_from_pandaai_daily(
        selected.drop(columns="__governance_session"),
        mapping=mapping,
        config=config,
    )
    try:
        clean_values: list[bool] = []
        for sample_id in dataset.sample_ids:
            if not (
                isinstance(sample_id, tuple)
                and len(sample_id) == 2
                and all(isinstance(value, str) for value in sample_id)
            ):
                raise AdapterValidationError(
                    "normalized sample identities must be string pairs"
                )
            clean_values.append(roll_clean_by_sample[(sample_id[0], sample_id[1])])
        label_roll_clean = np.array(clean_values, dtype=bool)
    except KeyError as exc:
        raise AdapterValidationError(
            "governance evidence does not align with normalized sample identities"
        ) from exc
    receipt = PandaAIContinuousGovernanceReceipt(
        input_rows=len(frame),
        declared_continuous_rows=len(declared),
        selected_continuous_rows=len(selected),
        discarded_concrete_rows=len(frame) - len(declared),
        discarded_insufficient_rows=len(declared) - len(selected),
        selected_assets=selected_assets,
        excluded_assets=excluded_assets,
        active_contract_transitions=active_contract_transitions,
        eligible_rows=len(dataset.sample_ids),
        label_roll_crossing_rows=label_roll_crossing_rows,
        information_interval_roll_crossing_rows=(
            information_interval_roll_crossing_rows
        ),
        source_digest=config.source_digest,
        policy_digest=policy.digest,
    )
    return GovernedPandaAIDataset(
        dataset=dataset,
        receipt=receipt,
        label_roll_clean=label_roll_clean,
    )


def _nonempty_identity_series(series: pd.Series, *, field_name: str) -> pd.Series:
    if series.isna().any():
        raise AdapterValidationError(f"{field_name} contains missing identities")
    normalized = series.astype(str).str.strip()
    if normalized.eq("").any():
        raise AdapterValidationError(f"{field_name} contains empty identities")
    return normalized


def validation_dataset_from_pandaai_daily(
    frame: pd.DataFrame,
    *,
    mapping: PandaAIDailyMapping,
    config: PandaAIDailyConfig,
) -> ValidationDataset:
    """Build a canonical PIT-declared dataset from loaded PandaAI daily rows."""

    if not isinstance(frame, pd.DataFrame):
        raise AdapterValidationError("frame must be a pandas DataFrame")
    if not isinstance(mapping, PandaAIDailyMapping):
        raise AdapterValidationError("mapping must be PandaAIDailyMapping")
    if not isinstance(config, PandaAIDailyConfig):
        raise AdapterValidationError("config must be PandaAIDailyConfig")
    required = {mapping.session, mapping.asset, mapping.close, *mapping.features}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AdapterValidationError(f"PandaAI frame is missing columns: {missing}")

    working = frame.loc[:, list(required)].copy()
    try:
        working["__session"] = pd.to_datetime(
            working[mapping.session], errors="raise"
        ).dt.normalize()
    except (TypeError, ValueError) as exc:
        raise AdapterValidationError("session column contains invalid times") from exc
    if working[mapping.asset].isna().any():
        raise AdapterValidationError("asset column contains missing identities")
    working["__asset"] = working[mapping.asset].astype(str).str.strip()
    if working["__asset"].eq("").any():
        raise AdapterValidationError("asset column contains empty identities")
    if working.duplicated(["__asset", "__session"]).any():
        raise AdapterValidationError("PandaAI frame contains duplicate asset sessions")

    numeric_columns = tuple(dict.fromkeys((mapping.close, *mapping.features)))
    for column in numeric_columns:
        try:
            working[column] = pd.to_numeric(working[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise AdapterValidationError(
                f"PandaAI numeric column {column!r} contains invalid values"
            ) from exc
    working = working.sort_values(["__asset", "__session"], kind="stable")
    grouped = working.groupby("__asset", sort=False, group_keys=False)
    working["__feature_start"] = grouped["__session"].shift(
        config.feature_lookback_sessions - 1
    )
    working["__label_end"] = grouped["__session"].shift(-config.label_horizon_sessions)
    future_close = grouped[mapping.close].shift(-config.label_horizon_sessions)
    working["__target"] = future_close / working[mapping.close] - 1.0

    eligible = working["__feature_start"].notna() & working["__label_end"].notna()
    working = working.loc[eligible].copy()
    if working.empty:
        raise AdapterValidationError(
            "PandaAI frame has insufficient history for the declared horizons"
        )
    finite_columns = [*numeric_columns, "__target"]
    values = working.loc[:, finite_columns].to_numpy(dtype=float)
    if not bool(np.isfinite(values).all()):
        raise AdapterValidationError(
            "eligible PandaAI rows must contain finite numeric values"
        )

    working = working.sort_values(["__session", "__asset"], kind="stable")
    sessions = working["__session"].to_numpy(dtype="datetime64[ns]")
    axis = np.sort(frame_session_axis(frame[mapping.session]))
    assets = tuple(working["__asset"].tolist())
    sample_ids = tuple(
        (asset, session.strftime("%Y-%m-%d"))
        for asset, session in zip(assets, working["__session"])
    )
    offset = np.timedelta64(config.decision_time_offset_minutes, "m")
    decision_times = sessions + offset
    return ValidationDataset(
        sample_ids=sample_ids,
        asset_ids=assets,
        session_axis=axis,
        sessions=sessions,
        information_intervals=tuple(
            InformationInterval(start, end)
            for start, end in zip(working["__feature_start"], working["__label_end"])
        ),
        decision_times=decision_times,
        feature_availability=decision_times,
        pit_snapshot=PITSnapshot(
            snapshot_id=config.snapshot_id,
            source_digest=config.source_digest,
        ),
        features=working.loc[:, list(mapping.features)].to_numpy(dtype=float),
        targets=working["__target"].to_numpy(dtype=float),
    )


def frame_session_axis(values: pd.Series) -> np.ndarray:
    """Normalize the source's observed daily sessions into an ordered axis."""

    try:
        sessions = pd.to_datetime(values, errors="raise").dt.normalize()
    except (TypeError, ValueError) as exc:
        raise AdapterValidationError("session column contains invalid times") from exc
    if sessions.isna().any():
        raise AdapterValidationError("session column contains missing times")
    return sessions.drop_duplicates().sort_values().to_numpy(dtype="datetime64[ns]")
