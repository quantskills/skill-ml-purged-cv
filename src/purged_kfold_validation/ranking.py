"""Fail-closed model-ranking stability evidence across declared regimes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Literal, Mapping

import numpy as np

from .domain import canonical_digest
from .errors import RankingStabilityError


@dataclass(frozen=True, slots=True)
class ModelRankSummary:
    """One model's observed rank distribution across regimes."""

    model: str
    median_rank: float
    worst_rank: float
    first_place_count: int

    def canonical(self) -> dict[str, str | float | int]:
        return {
            "model": self.model,
            "median_rank": self.median_rank,
            "worst_rank": self.worst_rank,
            "first_place_count": self.first_place_count,
        }


@dataclass(frozen=True, slots=True)
class RankingStabilityReport:
    """Comparable rank evidence without converting it into a performance claim."""

    objective: Literal["maximize", "minimize"]
    regime_names: tuple[str, ...]
    model_names: tuple[str, ...]
    models: tuple[ModelRankSummary, ...]
    pairwise_spearman: tuple[float, ...]
    minimum_pairwise_spearman: float
    threshold: float
    stable: bool
    digest: str

    def canonical(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "regime_names": list(self.regime_names),
            "model_names": list(self.model_names),
            "models": [item.canonical() for item in self.models],
            "pairwise_spearman": list(self.pairwise_spearman),
            "minimum_pairwise_spearman": self.minimum_pairwise_spearman,
            "threshold": self.threshold,
            "stable": self.stable,
            "digest": self.digest,
        }


def assess_model_ranking_stability(
    scores: Mapping[str, Mapping[str, float]],
    *,
    objective: Literal["maximize", "minimize"] = "maximize",
    min_pairwise_spearman: float = 0.5,
) -> RankingStabilityReport:
    """Rank the same frozen models in every regime and quantify reversals."""

    if objective not in {"maximize", "minimize"}:
        raise RankingStabilityError("objective must be 'maximize' or 'minimize'")
    if not isfinite(min_pairwise_spearman) or not -1.0 <= min_pairwise_spearman <= 1.0:
        raise RankingStabilityError("min_pairwise_spearman must be between -1 and 1")
    normalized = {str(regime): dict(values) for regime, values in scores.items()}
    if len(normalized) < 2:
        raise RankingStabilityError("at least two regimes are required")
    if any(not name for name in normalized):
        raise RankingStabilityError("regime names must not be empty")

    first_models = set(next(iter(normalized.values())))
    if len(first_models) < 2 or any(not name for name in first_models):
        raise RankingStabilityError("at least two named models are required")
    for values in normalized.values():
        if set(values) != first_models:
            raise RankingStabilityError(
                "every regime must contain the same frozen model set"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in values.values()
        ):
            raise RankingStabilityError(
                "all regime scores must be finite numeric values"
            )

    regime_names = tuple(sorted(normalized))
    model_names = tuple(sorted(first_models))
    rank_rows = tuple(
        _rank_row(
            tuple(float(normalized[regime][model]) for model in model_names),
            maximize=objective == "maximize",
        )
        for regime in regime_names
    )
    correlations: list[float] = []
    for left_index, left in enumerate(rank_rows):
        for right in rank_rows[left_index + 1 :]:
            if np.array_equal(left, right):
                correlation = 1.0
            elif float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
                correlation = 0.0
            else:
                correlation = float(np.corrcoef(left, right)[0, 1])
            correlations.append(round(correlation, 12))
    minimum = min(correlations)
    summaries = tuple(
        sorted(
            (
                ModelRankSummary(
                    model=model,
                    median_rank=float(median(row[index] for row in rank_rows)),
                    worst_rank=float(max(row[index] for row in rank_rows)),
                    first_place_count=int(sum(row[index] == 1.0 for row in rank_rows)),
                )
                for index, model in enumerate(model_names)
            ),
            key=lambda item: (item.median_rank, item.worst_rank, item.model),
        )
    )
    digest = canonical_digest(
        {
            "kind": "ranking-stability-report",
            "objective": objective,
            "regimes": [
                {
                    "name": regime,
                    "scores": {
                        model: float(normalized[regime][model]) for model in model_names
                    },
                }
                for regime in regime_names
            ],
            "pairwise_spearman": correlations,
            "threshold": min_pairwise_spearman,
        }
    )
    return RankingStabilityReport(
        objective=objective,
        regime_names=regime_names,
        model_names=model_names,
        models=summaries,
        pairwise_spearman=tuple(correlations),
        minimum_pairwise_spearman=minimum,
        threshold=float(min_pairwise_spearman),
        stable=minimum >= min_pairwise_spearman,
        digest=digest,
    )


def _rank_row(values: tuple[float, ...], *, maximize: bool) -> np.ndarray:
    ordered = sorted(
        range(len(values)),
        key=lambda index: (-values[index] if maximize else values[index], index),
    )
    ranks = np.empty(len(values), dtype=float)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in ordered[cursor:end]:
            ranks[position] = average_rank
        cursor = end
    return ranks
