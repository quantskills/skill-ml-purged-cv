"""Time-series strategy return construction and selection-overfitting evidence."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from itertools import combinations, product
from math import comb, exp, log, sqrt
from statistics import NormalDist

import numpy as np

from .domain import TestBlock, canonical_digest
from .errors import StrategyBenchmarkError
from .paths import build_cpcv_path_decomposition


def _sessions(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) < 3:
        raise StrategyBenchmarkError(
            "sessions must be one-dimensional with at least 3 values"
        )
    if not np.issubdtype(array.dtype, np.datetime64):
        raise StrategyBenchmarkError("sessions must use a numpy datetime64 dtype")
    normalized = np.asarray(array, dtype="datetime64[ns]").copy()
    if np.any(np.isnat(normalized)):
        raise StrategyBenchmarkError("sessions cannot contain NaT")
    if np.any(normalized[1:] <= normalized[:-1]):
        raise StrategyBenchmarkError("sessions must be strictly increasing and unique")
    normalized.setflags(write=False)
    return normalized


def _identifiers(
    value: tuple[str, ...], *, name: str, minimum: int = 1
) -> tuple[str, ...]:
    identifiers = tuple(value)
    if len(identifiers) < minimum:
        raise StrategyBenchmarkError(f"{name} must contain at least {minimum} values")
    if any(not isinstance(item, str) or not item.strip() for item in identifiers):
        raise StrategyBenchmarkError(f"{name} must contain non-empty strings")
    if len(set(identifiers)) != len(identifiers):
        raise StrategyBenchmarkError(f"{name} must be unique")
    return identifiers


def _matrix(
    value: np.ndarray,
    *,
    name: str,
    shape: tuple[int, int],
    positive: bool = False,
    non_negative: bool = False,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise StrategyBenchmarkError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise StrategyBenchmarkError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0.0):
        raise StrategyBenchmarkError(f"{name} must contain only positive values")
    if non_negative and np.any(array < 0.0):
        raise StrategyBenchmarkError(f"{name} must contain only non-negative values")
    frozen = array.copy()
    frozen.setflags(write=False)
    return frozen


def _positive_integer(value: int, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StrategyBenchmarkError(f"{name} must be an integer of at least {minimum}")
    return value


def _finite_number(value: float, *, name: str, minimum: float | None = None) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise StrategyBenchmarkError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise StrategyBenchmarkError(f"{name} must be at least {minimum}")
    return number


def _text(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyBenchmarkError(f"{name} must be a non-empty string")
    return value.strip()


def _array_digest(value: np.ndarray) -> str:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.datetime64):
        payload = np.ascontiguousarray(array.astype("datetime64[ns]").astype("<i8"))
    else:
        payload = np.ascontiguousarray(array.astype("<f8"))
    return sha256(payload.tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True)
class DensePricePanel:
    """A governed dense panel used only by the built-in TSMOM adapter."""

    sessions: np.ndarray
    asset_ids: tuple[str, ...]
    signal_prices: np.ndarray
    tradable_returns: np.ndarray
    source_digest: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        sessions = _sessions(self.sessions)
        asset_ids = _identifiers(self.asset_ids, name="asset_ids")
        shape = (len(sessions), len(asset_ids))
        prices = _matrix(
            self.signal_prices, name="signal_prices", shape=shape, positive=True
        )
        returns = _matrix(self.tradable_returns, name="tradable_returns", shape=shape)
        if np.any(returns <= -1.0):
            raise StrategyBenchmarkError("tradable_returns must be greater than -1")
        source_digest = _text(self.source_digest, name="source_digest")
        object.__setattr__(self, "sessions", sessions)
        object.__setattr__(self, "asset_ids", asset_ids)
        object.__setattr__(self, "signal_prices", prices)
        object.__setattr__(self, "tradable_returns", returns)
        object.__setattr__(self, "source_digest", source_digest)
        object.__setattr__(
            self,
            "digest",
            canonical_digest(
                {
                    "sessions_digest": _array_digest(sessions),
                    "asset_ids": list(asset_ids),
                    "signal_prices_digest": _array_digest(prices),
                    "tradable_returns_digest": _array_digest(returns),
                    "shape": list(shape),
                    "source_digest": source_digest,
                }
            ),
        )

    def canonical(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "source_digest": self.source_digest,
            "session_count": len(self.sessions),
            "asset_count": len(self.asset_ids),
            "first_session": str(self.sessions[0]),
            "last_session": str(self.sessions[-1]),
        }


@dataclass(frozen=True, slots=True)
class TSMOMCandidate:
    """One pre-registered member of the per-asset TSMOM reference family."""

    candidate_id: str
    lookback_sessions: int
    volatility_window: int
    rebalance_sessions: int
    leverage_cap: float
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        candidate_id = _text(self.candidate_id, name="candidate_id")
        lookback = _positive_integer(
            self.lookback_sessions, name="lookback_sessions", minimum=2
        )
        volatility = _positive_integer(
            self.volatility_window, name="volatility_window", minimum=2
        )
        rebalance = _positive_integer(
            self.rebalance_sessions, name="rebalance_sessions"
        )
        leverage = _finite_number(
            self.leverage_cap, name="leverage_cap", minimum=0.000001
        )
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "lookback_sessions", lookback)
        object.__setattr__(self, "volatility_window", volatility)
        object.__setattr__(self, "rebalance_sessions", rebalance)
        object.__setattr__(self, "leverage_cap", leverage)
        object.__setattr__(
            self,
            "digest",
            canonical_digest(
                {
                    "candidate_id": candidate_id,
                    "lookback_sessions": lookback,
                    "volatility_window": volatility,
                    "rebalance_sessions": rebalance,
                    "leverage_cap": leverage,
                }
            ),
        )

    def canonical(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "lookback_sessions": self.lookback_sessions,
            "volatility_window": self.volatility_window,
            "rebalance_sessions": self.rebalance_sessions,
            "leverage_cap": self.leverage_cap,
            "digest": self.digest,
        }


def default_tsmom_candidates() -> tuple[TSMOMCandidate, ...]:
    """Return the immutable 32-member registered TSMOM v1 grid."""

    return tuple(
        TSMOMCandidate(
            candidate_id=(
                f"tsmom-l{lookback:03d}-v{volatility:03d}-r{rebalance:02d}-c{cap:g}"
            ),
            lookback_sessions=lookback,
            volatility_window=volatility,
            rebalance_sessions=rebalance,
            leverage_cap=float(cap),
        )
        for lookback, volatility, rebalance, cap in product(
            (21, 63, 126, 252), (20, 60), (5, 21), (2, 4)
        )
    )


@dataclass(frozen=True, slots=True)
class TSMOMExecutionSpec:
    """Strictly causal offline return and simplified turnover-cost policy."""

    annualization_sessions: int = 252
    target_annual_volatility: float = 0.10
    cost_bps: float = 0.0

    def __post_init__(self) -> None:
        _positive_integer(
            self.annualization_sessions, name="annualization_sessions", minimum=2
        )
        _finite_number(
            self.target_annual_volatility,
            name="target_annual_volatility",
            minimum=0.000001,
        )
        _finite_number(self.cost_bps, name="cost_bps", minimum=0.0)

    @property
    def digest(self) -> str:
        return canonical_digest(self.canonical())

    def canonical(self) -> dict[str, float | int]:
        return {
            "annualization_sessions": self.annualization_sessions,
            "target_annual_volatility": self.target_annual_volatility,
            "cost_bps": self.cost_bps,
        }


@dataclass(frozen=True, slots=True)
class StrategyReturnMatrix:
    """The generic seam between strategy generation and overfitting analysis."""

    sessions: np.ndarray
    candidate_ids: tuple[str, ...]
    gross_returns: np.ndarray
    net_returns: np.ndarray
    turnover: np.ndarray
    source_digest: str
    cost_bps: float
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        sessions = _sessions(self.sessions)
        candidate_ids = _identifiers(
            self.candidate_ids, name="candidate_ids", minimum=2
        )
        shape = (len(sessions), len(candidate_ids))
        gross = _matrix(self.gross_returns, name="gross_returns", shape=shape)
        net = _matrix(self.net_returns, name="net_returns", shape=shape)
        turnover = _matrix(
            self.turnover, name="turnover", shape=shape, non_negative=True
        )
        if np.any(gross <= -1.0) or np.any(net <= -1.0):
            raise StrategyBenchmarkError("strategy returns must be greater than -1")
        source_digest = _text(self.source_digest, name="source_digest")
        cost_bps = _finite_number(self.cost_bps, name="cost_bps", minimum=0.0)
        object.__setattr__(self, "sessions", sessions)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "gross_returns", gross)
        object.__setattr__(self, "net_returns", net)
        object.__setattr__(self, "turnover", turnover)
        object.__setattr__(self, "source_digest", source_digest)
        object.__setattr__(self, "cost_bps", cost_bps)
        object.__setattr__(
            self,
            "digest",
            canonical_digest(
                {
                    "sessions_digest": _array_digest(sessions),
                    "candidate_ids": list(candidate_ids),
                    "gross_returns_digest": _array_digest(gross),
                    "net_returns_digest": _array_digest(net),
                    "turnover_digest": _array_digest(turnover),
                    "shape": list(shape),
                    "source_digest": source_digest,
                    "cost_bps": cost_bps,
                }
            ),
        )

    def canonical(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "source_digest": self.source_digest,
            "session_count": len(self.sessions),
            "candidate_ids": list(self.candidate_ids),
            "cost_bps": self.cost_bps,
        }


def build_tsmom_return_matrix(
    panel: DensePricePanel,
    candidates: tuple[TSMOMCandidate, ...],
    execution: TSMOMExecutionSpec,
) -> StrategyReturnMatrix:
    """Build strictly lagged TSMOM candidate net returns from a dense panel."""

    registered = tuple(candidates)
    _identifiers(
        tuple(candidate.candidate_id for candidate in registered),
        name="candidate_ids",
        minimum=2,
    )
    rows = len(panel.sessions)
    columns = len(registered)
    gross = np.zeros((rows, columns), dtype=np.float64)
    net = np.zeros((rows, columns), dtype=np.float64)
    turnover = np.zeros((rows, columns), dtype=np.float64)
    asset_count = len(panel.asset_ids)
    annualizer = sqrt(execution.annualization_sessions)
    cost_rate = execution.cost_bps / 10_000.0

    for column, candidate in enumerate(registered):
        warmup = max(candidate.lookback_sessions + 1, candidate.volatility_window + 1)
        if warmup >= rows:
            raise StrategyBenchmarkError(
                f"candidate {candidate.candidate_id!r} requires "
                f"{warmup + 1} sessions, got {rows}"
            )
        weights = np.zeros(asset_count, dtype=np.float64)
        for row in range(warmup, rows):
            if (row - warmup) % candidate.rebalance_sessions == 0:
                previous_price = panel.signal_prices[
                    row - 1 - candidate.lookback_sessions
                ]
                latest_price = panel.signal_prices[row - 1]
                lagged_return = latest_price / previous_price - 1.0
                history = panel.tradable_returns[
                    row - candidate.volatility_window : row
                ]
                volatility = np.std(history, axis=0, ddof=1)
                direction = np.sign(lagged_return)
                scaled = np.divide(
                    execution.target_annual_volatility,
                    volatility * annualizer,
                    out=np.zeros_like(volatility),
                    where=volatility > 1e-12,
                )
                next_weights = (
                    direction
                    * np.clip(scaled, 0.0, candidate.leverage_cap)
                    / float(asset_count)
                )
            else:
                next_weights = weights
            turnover[row, column] = float(np.sum(np.abs(next_weights - weights)))
            weights = next_weights
            gross[row, column] = float(np.dot(weights, panel.tradable_returns[row]))
            net[row, column] = gross[row, column] - turnover[row, column] * cost_rate

    return StrategyReturnMatrix(
        sessions=panel.sessions,
        candidate_ids=tuple(candidate.candidate_id for candidate in registered),
        gross_returns=gross,
        net_returns=net,
        turnover=turnover,
        source_digest=canonical_digest(
            {
                "kind": "tsmom-v1",
                "panel_digest": panel.digest,
                "candidate_digests": [item.digest for item in registered],
                "execution_digest": execution.digest,
            }
        ),
        cost_bps=execution.cost_bps,
    )


@dataclass(frozen=True, slots=True)
class StrategyBenchmarkConfig:
    """Frozen partition and metric policy for one candidate-return audit."""

    annualization_sessions: int = 252
    cscv_groups: int = 8
    cpcv_groups: int = 6
    cpcv_test_groups: int = 2
    cpcv_embargo_sessions: int = 0
    walk_forward_windows: int = 5
    minimum_train_sessions: int = 63

    def __post_init__(self) -> None:
        _positive_integer(
            self.annualization_sessions, name="annualization_sessions", minimum=2
        )
        _positive_integer(self.cscv_groups, name="cscv_groups", minimum=4)
        if self.cscv_groups % 2:
            raise StrategyBenchmarkError("cscv_groups must be even")
        _positive_integer(self.cpcv_groups, name="cpcv_groups", minimum=3)
        _positive_integer(self.cpcv_test_groups, name="cpcv_test_groups", minimum=2)
        if self.cpcv_test_groups >= self.cpcv_groups:
            raise StrategyBenchmarkError(
                "cpcv_test_groups must be less than cpcv_groups"
            )
        _positive_integer(self.walk_forward_windows, name="walk_forward_windows")
        _positive_integer(
            self.minimum_train_sessions, name="minimum_train_sessions", minimum=2
        )
        if (
            isinstance(self.cpcv_embargo_sessions, bool)
            or not isinstance(self.cpcv_embargo_sessions, int)
            or self.cpcv_embargo_sessions < 0
        ):
            raise StrategyBenchmarkError(
                "cpcv_embargo_sessions must be a non-negative integer"
            )

    @property
    def digest(self) -> str:
        return canonical_digest(self.canonical())

    def canonical(self) -> dict[str, int]:
        return {
            "annualization_sessions": self.annualization_sessions,
            "cscv_groups": self.cscv_groups,
            "cpcv_groups": self.cpcv_groups,
            "cpcv_test_groups": self.cpcv_test_groups,
            "cpcv_embargo_sessions": self.cpcv_embargo_sessions,
            "walk_forward_windows": self.walk_forward_windows,
            "minimum_train_sessions": self.minimum_train_sessions,
        }


@dataclass(frozen=True, slots=True)
class StrategyCandidateMetrics:
    candidate_id: str
    annualized_sharpe: float
    total_return: float
    maximum_drawdown: float
    mean_session_return: float
    session_volatility: float
    observations: int

    def canonical(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "annualized_sharpe": self.annualized_sharpe,
            "total_return": self.total_return,
            "maximum_drawdown": self.maximum_drawdown,
            "mean_session_return": self.mean_session_return,
            "session_volatility": self.session_volatility,
            "observations": self.observations,
        }


@dataclass(frozen=True, slots=True)
class PBOEvidence:
    probability: float
    trial_count: int
    split_count: int
    negative_logit_count: int
    median_logit: float
    logits: tuple[float, ...]

    def canonical(self) -> dict[str, object]:
        return {
            "probability": self.probability,
            "trial_count": self.trial_count,
            "split_count": self.split_count,
            "negative_logit_count": self.negative_logit_count,
            "median_logit": self.median_logit,
            "logits": list(self.logits),
        }


@dataclass(frozen=True, slots=True)
class DeflatedSharpeEvidence:
    selected_candidate_id: str
    selected_annualized_sharpe: float
    benchmark_annualized_sharpe: float
    probability: float
    trial_count: int
    observations: int
    annualization_sessions: int
    skewness: float
    kurtosis: float

    def canonical(self) -> dict[str, object]:
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "selected_annualized_sharpe": self.selected_annualized_sharpe,
            "benchmark_annualized_sharpe": self.benchmark_annualized_sharpe,
            "probability": self.probability,
            "trial_count": self.trial_count,
            "observations": self.observations,
            "annualization_sessions": self.annualization_sessions,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
        }


@dataclass(frozen=True, slots=True)
class SelectedPathMetrics:
    path_index: int
    observations: int
    annualized_sharpe: float
    total_return: float
    maximum_drawdown: float
    selected_candidate_ids: tuple[str, ...]

    def canonical(self) -> dict[str, object]:
        return {
            "path_index": self.path_index,
            "observations": self.observations,
            "annualized_sharpe": self.annualized_sharpe,
            "total_return": self.total_return,
            "maximum_drawdown": self.maximum_drawdown,
            "selected_candidate_ids": list(self.selected_candidate_ids),
        }


@dataclass(frozen=True, slots=True)
class PathSharpeDistribution:
    median: float
    p10: float
    worst: float
    interquartile_range: float

    def canonical(self) -> dict[str, float]:
        return {
            "median": self.median,
            "p10": self.p10,
            "worst": self.worst,
            "interquartile_range": self.interquartile_range,
        }


@dataclass(frozen=True, slots=True)
class CPCVSelectionEvidence:
    group_count: int
    test_group_count: int
    combination_count: int
    path_count: int
    embargo_sessions: int
    paths: tuple[SelectedPathMetrics, ...]
    sharpe_distribution: PathSharpeDistribution

    def canonical(self) -> dict[str, object]:
        return {
            "group_count": self.group_count,
            "test_group_count": self.test_group_count,
            "combination_count": self.combination_count,
            "path_count": self.path_count,
            "embargo_sessions": self.embargo_sessions,
            "paths": [path.canonical() for path in self.paths],
            "sharpe_distribution": self.sharpe_distribution.canonical(),
        }


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    window_index: int
    train_end_index: int
    test_start_index: int
    test_end_index: int
    selected_candidate_id: str
    test_annualized_sharpe: float
    hindsight_best_candidate_id: str
    hindsight_best_annualized_sharpe: float
    selection_regret: float

    def canonical(self) -> dict[str, object]:
        return {
            "window_index": self.window_index,
            "train_end_index": self.train_end_index,
            "test_start_index": self.test_start_index,
            "test_end_index": self.test_end_index,
            "selected_candidate_id": self.selected_candidate_id,
            "test_annualized_sharpe": self.test_annualized_sharpe,
            "hindsight_best_candidate_id": self.hindsight_best_candidate_id,
            "hindsight_best_annualized_sharpe": (self.hindsight_best_annualized_sharpe),
            "selection_regret": self.selection_regret,
        }


@dataclass(frozen=True, slots=True)
class WalkForwardSelectionEvidence:
    windows: tuple[WalkForwardWindow, ...]
    annualized_sharpe: float
    total_return: float
    maximum_drawdown: float
    mean_selection_regret: float

    def canonical(self) -> dict[str, object]:
        return {
            "windows": [window.canonical() for window in self.windows],
            "annualized_sharpe": self.annualized_sharpe,
            "total_return": self.total_return,
            "maximum_drawdown": self.maximum_drawdown,
            "mean_selection_regret": self.mean_selection_regret,
        }


@dataclass(frozen=True, slots=True)
class StrategyOverfittingReport:
    matrix_digest: str
    config_digest: str
    candidates: tuple[StrategyCandidateMetrics, ...]
    pbo: PBOEvidence
    deflated_sharpe: DeflatedSharpeEvidence
    cpcv: CPCVSelectionEvidence
    walk_forward: WalkForwardSelectionEvidence
    selection_optimism_gap: float
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "digest", canonical_digest(self.canonical(include_digest=False))
        )

    def canonical(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "1",
            "matrix_digest": self.matrix_digest,
            "config_digest": self.config_digest,
            "candidates": [candidate.canonical() for candidate in self.candidates],
            "pbo": self.pbo.canonical(),
            "deflated_sharpe": self.deflated_sharpe.canonical(),
            "cpcv": self.cpcv.canonical(),
            "walk_forward": self.walk_forward.canonical(),
            "selection_optimism_gap": self.selection_optimism_gap,
            "warnings": [
                "Selection-overfitting evidence is not a profitability or "
                "deployment claim."
            ],
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


def _sharpe(returns: np.ndarray, annualization: int) -> float:
    values = np.asarray(returns, dtype=np.float64)
    mean = float(np.mean(values))
    deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean / max(deviation, 1e-12) * sqrt(annualization)


def _total_return(returns: np.ndarray) -> float:
    return float(np.prod(1.0 + np.asarray(returns, dtype=np.float64)) - 1.0)


def _maximum_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + np.asarray(returns, dtype=np.float64))
    peak = np.maximum.accumulate(np.concatenate((np.asarray([1.0]), wealth)))[1:]
    return float(np.max(1.0 - wealth / peak))


def _candidate_metric(
    candidate_id: str, returns: np.ndarray, annualization: int
) -> StrategyCandidateMetrics:
    values = np.asarray(returns, dtype=np.float64)
    return StrategyCandidateMetrics(
        candidate_id=candidate_id,
        annualized_sharpe=_sharpe(values, annualization),
        total_return=_total_return(values),
        maximum_drawdown=_maximum_drawdown(values),
        mean_session_return=float(np.mean(values)),
        session_volatility=float(np.std(values, ddof=1)),
        observations=len(values),
    )


def _best_candidate(returns: np.ndarray, annualization: int) -> int:
    scores = np.asarray(
        [
            _sharpe(returns[:, column], annualization)
            for column in range(returns.shape[1])
        ]
    )
    return int(np.argmax(scores))


def _average_rank(values: np.ndarray, selected_index: int) -> float:
    selected = values[selected_index]
    less = float(np.count_nonzero(values < selected))
    equal = float(np.count_nonzero(values == selected))
    return less + (equal + 1.0) / 2.0


def _pbo(matrix: StrategyReturnMatrix, config: StrategyBenchmarkConfig) -> PBOEvidence:
    if config.cscv_groups > len(matrix.sessions):
        raise StrategyBenchmarkError("cscv_groups cannot exceed session count")
    groups = tuple(
        np.asarray(group, dtype=np.int64)
        for group in np.array_split(np.arange(len(matrix.sessions)), config.cscv_groups)
    )
    logits: list[float] = []
    half = config.cscv_groups // 2
    for in_sample_groups in combinations(range(config.cscv_groups), half):
        in_set = set(in_sample_groups)
        train = np.concatenate([groups[index] for index in in_sample_groups])
        test = np.concatenate(
            [
                groups[index]
                for index in range(config.cscv_groups)
                if index not in in_set
            ]
        )
        winner = _best_candidate(
            matrix.net_returns[train], config.annualization_sessions
        )
        oos_scores = np.asarray(
            [
                _sharpe(matrix.net_returns[test, column], config.annualization_sessions)
                for column in range(len(matrix.candidate_ids))
            ]
        )
        rank = _average_rank(oos_scores, winner)
        relative_rank = rank / float(len(matrix.candidate_ids) + 1)
        logits.append(log(relative_rank / (1.0 - relative_rank)))
    values = np.asarray(logits, dtype=np.float64)
    negative = int(np.count_nonzero(values <= 0.0))
    return PBOEvidence(
        probability=negative / len(logits),
        trial_count=len(matrix.candidate_ids),
        split_count=len(logits),
        negative_logit_count=negative,
        median_logit=float(np.median(values)),
        logits=tuple(float(value) for value in values),
    )


def _moments(returns: np.ndarray) -> tuple[float, float]:
    values = np.asarray(returns, dtype=np.float64)
    centered = values - np.mean(values)
    deviation = float(np.std(values, ddof=0))
    if deviation <= 1e-12:
        return 0.0, 3.0
    normalized = centered / deviation
    return float(np.mean(normalized**3)), float(np.mean(normalized**4))


def _deflated_sharpe(
    matrix: StrategyReturnMatrix, config: StrategyBenchmarkConfig
) -> DeflatedSharpeEvidence:
    periodic_sharpes = np.asarray(
        [
            float(np.mean(matrix.net_returns[:, column]))
            / max(float(np.std(matrix.net_returns[:, column], ddof=1)), 1e-12)
            for column in range(len(matrix.candidate_ids))
        ]
    )
    winner = int(np.argmax(periodic_sharpes))
    trial_count = len(periodic_sharpes)
    trial_deviation = float(np.std(periodic_sharpes, ddof=1))
    normal = NormalDist()
    euler_gamma = 0.5772156649015329
    expected_maximum = trial_deviation * (
        (1.0 - euler_gamma) * normal.inv_cdf(1.0 - 1.0 / trial_count)
        + euler_gamma * normal.inv_cdf(1.0 - 1.0 / (trial_count * exp(1.0)))
    )
    selected = periodic_sharpes[winner]
    skewness, kurtosis = _moments(matrix.net_returns[:, winner])
    denominator = sqrt(
        max(
            1.0 - skewness * selected + ((kurtosis - 1.0) / 4.0) * selected * selected,
            1e-12,
        )
    )
    statistic = (
        (selected - expected_maximum) * sqrt(len(matrix.sessions) - 1) / denominator
    )
    annualizer = sqrt(config.annualization_sessions)
    return DeflatedSharpeEvidence(
        selected_candidate_id=matrix.candidate_ids[winner],
        selected_annualized_sharpe=float(selected * annualizer),
        benchmark_annualized_sharpe=float(expected_maximum * annualizer),
        probability=float(normal.cdf(statistic)),
        trial_count=trial_count,
        observations=len(matrix.sessions),
        annualization_sessions=config.annualization_sessions,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def _cpcv(
    matrix: StrategyReturnMatrix, config: StrategyBenchmarkConfig
) -> CPCVSelectionEvidence:
    if config.cpcv_groups > len(matrix.sessions):
        raise StrategyBenchmarkError("cpcv_groups cannot exceed session count")
    groups = tuple(
        np.asarray(group, dtype=np.int64)
        for group in np.array_split(np.arange(len(matrix.sessions)), config.cpcv_groups)
    )
    group_combinations = tuple(
        combinations(range(config.cpcv_groups), config.cpcv_test_groups)
    )
    blocks = tuple(
        TestBlock(
            start_session=matrix.sessions[group[0]],
            end_session=matrix.sessions[group[-1]],
            session_count=len(group),
        )
        for group in groups
    )
    decomposition = build_cpcv_path_decomposition(
        groups=blocks, combinations=group_combinations
    )
    selected_by_combination: list[int] = []
    for test_groups in group_combinations:
        train_mask = np.ones(len(matrix.sessions), dtype=bool)
        for group_index in test_groups:
            train_mask[groups[group_index]] = False
            end = int(groups[group_index][-1])
            embargo_end = min(
                len(matrix.sessions), end + 1 + config.cpcv_embargo_sessions
            )
            train_mask[end + 1 : embargo_end] = False
        train = np.flatnonzero(train_mask)
        if len(train) < config.minimum_train_sessions:
            raise StrategyBenchmarkError(
                "CPCV training sessions below minimum after exclusions"
            )
        selected_by_combination.append(
            _best_candidate(matrix.net_returns[train], config.annualization_sessions)
        )

    paths: list[SelectedPathMetrics] = []
    for path in decomposition.paths:
        path_returns = np.full(len(matrix.sessions), np.nan, dtype=np.float64)
        selected_ids: list[str] = []
        for occurrence in path.occurrences:
            selected = selected_by_combination[occurrence.combination_index]
            selected_ids.append(matrix.candidate_ids[selected])
            positions = groups[occurrence.group_index]
            path_returns[positions] = matrix.net_returns[positions, selected]
        if np.any(np.isnan(path_returns)):
            raise StrategyBenchmarkError(
                "CPCV path did not cover every session exactly once"
            )
        paths.append(
            SelectedPathMetrics(
                path_index=path.path_index,
                observations=len(path_returns),
                annualized_sharpe=_sharpe(path_returns, config.annualization_sessions),
                total_return=_total_return(path_returns),
                maximum_drawdown=_maximum_drawdown(path_returns),
                selected_candidate_ids=tuple(selected_ids),
            )
        )
    sharpes = np.asarray([path.annualized_sharpe for path in paths])
    return CPCVSelectionEvidence(
        group_count=config.cpcv_groups,
        test_group_count=config.cpcv_test_groups,
        combination_count=comb(config.cpcv_groups, config.cpcv_test_groups),
        path_count=decomposition.path_count,
        embargo_sessions=config.cpcv_embargo_sessions,
        paths=tuple(paths),
        sharpe_distribution=PathSharpeDistribution(
            median=float(np.median(sharpes)),
            p10=float(np.quantile(sharpes, 0.10)),
            worst=float(np.min(sharpes)),
            interquartile_range=float(
                np.quantile(sharpes, 0.75) - np.quantile(sharpes, 0.25)
            ),
        ),
    )


def _walk_forward(
    matrix: StrategyReturnMatrix, config: StrategyBenchmarkConfig
) -> WalkForwardSelectionEvidence:
    remaining = len(matrix.sessions) - config.minimum_train_sessions
    if remaining < config.walk_forward_windows * 2:
        raise StrategyBenchmarkError(
            "insufficient sessions for requested walk-forward windows"
        )
    test_groups = tuple(
        np.asarray(group, dtype=np.int64)
        for group in np.array_split(
            np.arange(config.minimum_train_sessions, len(matrix.sessions)),
            config.walk_forward_windows,
        )
    )
    windows: list[WalkForwardWindow] = []
    selected_returns: list[np.ndarray] = []
    for window_index, test in enumerate(test_groups):
        test_start = int(test[0])
        train = np.arange(test_start, dtype=np.int64)
        selected = _best_candidate(
            matrix.net_returns[train], config.annualization_sessions
        )
        hindsight = _best_candidate(
            matrix.net_returns[test], config.annualization_sessions
        )
        selected_sharpe = _sharpe(
            matrix.net_returns[test, selected], config.annualization_sessions
        )
        hindsight_sharpe = _sharpe(
            matrix.net_returns[test, hindsight], config.annualization_sessions
        )
        selected_returns.append(matrix.net_returns[test, selected])
        windows.append(
            WalkForwardWindow(
                window_index=window_index,
                train_end_index=test_start - 1,
                test_start_index=test_start,
                test_end_index=int(test[-1]),
                selected_candidate_id=matrix.candidate_ids[selected],
                test_annualized_sharpe=selected_sharpe,
                hindsight_best_candidate_id=matrix.candidate_ids[hindsight],
                hindsight_best_annualized_sharpe=hindsight_sharpe,
                selection_regret=hindsight_sharpe - selected_sharpe,
            )
        )
    joined = np.concatenate(selected_returns)
    return WalkForwardSelectionEvidence(
        windows=tuple(windows),
        annualized_sharpe=_sharpe(joined, config.annualization_sessions),
        total_return=_total_return(joined),
        maximum_drawdown=_maximum_drawdown(joined),
        mean_selection_regret=float(
            np.mean([window.selection_regret for window in windows])
        ),
    )


def analyze_strategy_return_matrix(
    matrix: StrategyReturnMatrix,
    config: StrategyBenchmarkConfig = StrategyBenchmarkConfig(),
) -> StrategyOverfittingReport:
    """Return generic candidate-selection evidence from an immutable return matrix."""

    minimum = max(
        config.cscv_groups * 2,
        config.cpcv_groups * 2,
        config.minimum_train_sessions + config.walk_forward_windows * 2,
    )
    if len(matrix.sessions) < minimum:
        raise StrategyBenchmarkError(
            f"strategy return matrix requires at least {minimum} sessions"
        )
    candidate_metrics = tuple(
        _candidate_metric(
            candidate_id,
            matrix.net_returns[:, column],
            config.annualization_sessions,
        )
        for column, candidate_id in enumerate(matrix.candidate_ids)
    )
    pbo = _pbo(matrix, config)
    deflated = _deflated_sharpe(matrix, config)
    cpcv = _cpcv(matrix, config)
    walk_forward = _walk_forward(matrix, config)
    best_full_sample_sharpe = max(
        candidate.annualized_sharpe for candidate in candidate_metrics
    )
    return StrategyOverfittingReport(
        matrix_digest=matrix.digest,
        config_digest=config.digest,
        candidates=candidate_metrics,
        pbo=pbo,
        deflated_sharpe=deflated,
        cpcv=cpcv,
        walk_forward=walk_forward,
        selection_optimism_gap=(
            best_full_sample_sharpe - walk_forward.annualized_sharpe
        ),
    )


@dataclass(frozen=True, slots=True)
class TimeSeriesBenchmarkConfig:
    """Registered TSMOM family, execution policy, scenarios, and analysis policy."""

    candidates: tuple[TSMOMCandidate, ...] = field(
        default_factory=default_tsmom_candidates
    )
    execution: TSMOMExecutionSpec = field(default_factory=TSMOMExecutionSpec)
    cost_scenarios_bps: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0)
    analysis: StrategyBenchmarkConfig = field(default_factory=StrategyBenchmarkConfig)

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        _identifiers(
            tuple(candidate.candidate_id for candidate in candidates),
            name="candidate_ids",
            minimum=2,
        )
        costs = tuple(
            _finite_number(value, name="cost_scenarios_bps", minimum=0.0)
            for value in self.cost_scenarios_bps
        )
        if not costs or len(set(costs)) != len(costs):
            raise StrategyBenchmarkError(
                "cost_scenarios_bps must be non-empty and unique"
            )
        if tuple(sorted(costs)) != costs:
            raise StrategyBenchmarkError("cost_scenarios_bps must be increasing")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "cost_scenarios_bps", costs)

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "candidate_digests": [item.digest for item in self.candidates],
                "execution": self.execution.canonical(),
                "cost_scenarios_bps": list(self.cost_scenarios_bps),
                "analysis": self.analysis.canonical(),
            }
        )


@dataclass(frozen=True, slots=True)
class StrategyScenarioReport:
    cost_bps: float
    matrix_digest: str
    analysis: StrategyOverfittingReport

    def canonical(self) -> dict[str, object]:
        return {
            "cost_bps": self.cost_bps,
            "matrix_digest": self.matrix_digest,
            "analysis": self.analysis.canonical(),
        }


@dataclass(frozen=True, slots=True)
class TimeSeriesBenchmarkReport:
    panel: dict[str, object]
    config_digest: str
    scenarios: tuple[StrategyScenarioReport, ...]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "digest", canonical_digest(self.canonical(include_digest=False))
        )

    def canonical(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "1",
            "benchmark_kind": "tsmom-selection-overfitting-benchmark",
            "panel": dict(self.panel),
            "config_digest": self.config_digest,
            "scenarios": [scenario.canonical() for scenario in self.scenarios],
            "warnings": [
                "The built-in TSMOM family is a validation fixture, not investment advice.",
                "Offline turnover costs do not model fills, slippage, capacity, or execution risk.",
            ],
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


def run_time_series_strategy_benchmark(
    panel: DensePricePanel,
    config: TimeSeriesBenchmarkConfig = TimeSeriesBenchmarkConfig(),
) -> TimeSeriesBenchmarkReport:
    """Run the registered TSMOM family across independent cost scenarios."""

    scenarios: list[StrategyScenarioReport] = []
    for cost_bps in config.cost_scenarios_bps:
        execution = replace(config.execution, cost_bps=cost_bps)
        matrix = build_tsmom_return_matrix(panel, config.candidates, execution)
        scenarios.append(
            StrategyScenarioReport(
                cost_bps=cost_bps,
                matrix_digest=matrix.digest,
                analysis=analyze_strategy_return_matrix(matrix, config.analysis),
            )
        )
    return TimeSeriesBenchmarkReport(
        panel=panel.canonical(),
        config_digest=config.digest,
        scenarios=tuple(scenarios),
    )
