from __future__ import annotations

import json

import pytest

from purged_kfold_validation import (
    RankingStabilityError,
    assess_model_ranking_stability,
)


def test_stable_model_ranking_is_summarized_across_regimes() -> None:
    report = assess_model_ranking_stability(
        {
            "bull": {"linear": 0.30, "tree": 0.20, "mean": 0.10},
            "bear": {"linear": 0.25, "tree": 0.15, "mean": 0.05},
            "sideways": {"linear": 0.20, "tree": 0.10, "mean": 0.00},
        },
        objective="maximize",
        min_pairwise_spearman=0.5,
    )

    assert report.stable is True
    assert report.minimum_pairwise_spearman == 1.0
    assert report.models[0].model == "linear"
    assert report.models[0].median_rank == 1.0
    assert report.models[0].worst_rank == 1
    assert report.models[0].first_place_count == 3
    payload = json.loads(json.dumps(report.canonical(), allow_nan=False))
    assert payload["models"][0]["first_place_count"] == 3
    assert type(payload["models"][0]["first_place_count"]) is int


def test_adversarial_rank_reversal_is_explicitly_unstable() -> None:
    report = assess_model_ranking_stability(
        {
            "regime-a": {"linear": 3.0, "tree": 2.0, "mean": 1.0},
            "regime-b": {"linear": 1.0, "tree": 2.0, "mean": 3.0},
        },
        objective="maximize",
        min_pairwise_spearman=0.0,
    )

    assert report.stable is False
    assert report.minimum_pairwise_spearman == -1.0
    assert report.pairwise_spearman == (-1.0,)


@pytest.mark.parametrize(
    "scores",
    (
        {"only": {"a": 1.0, "b": 2.0}},
        {"one": {"a": 1.0, "b": 2.0}, "two": {"a": 1.0, "c": 2.0}},
        {"one": {"a": 1.0, "b": float("nan")}, "two": {"a": 2.0, "b": 1.0}},
    ),
)
def test_ranking_stability_fails_closed_on_incomparable_evidence(
    scores: dict[str, dict[str, float]],
) -> None:
    with pytest.raises(RankingStabilityError):
        assess_model_ranking_stability(scores)
