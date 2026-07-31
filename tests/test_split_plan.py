from __future__ import annotations

import numpy as np
import pytest

from purged_kfold_validation import (
    DatasetValidationError,
    FoldAssignment,
    InformationInterval,
    PurgedKFold,
    TemporalValidationError,
    ValidationDataset,
)


def test_user_can_inspect_a_minimal_leakage_safe_split_plan() -> None:
    sessions = np.array(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
        dtype="datetime64[D]",
    )
    dataset = ValidationDataset(
        sample_ids=("s0", "s1", "s2", "s3"),
        session_axis=sessions,
        sessions=sessions,
        information_intervals=(
            InformationInterval(sessions[0], sessions[0]),
            InformationInterval(sessions[1], sessions[2]),
            InformationInterval(sessions[2], sessions[2]),
            InformationInterval(sessions[3], sessions[3]),
        ),
        features=np.arange(8, dtype=float).reshape(4, 2),
        targets=np.array([0.0, 1.0, 0.0, 1.0]),
    )

    plan = PurgedKFold(n_splits=2).plan(dataset)

    first = plan.folds[0]
    assert isinstance(first, FoldAssignment)
    assert first.test_sample_ids == ("s0", "s1")
    assert first.train_sample_ids == ("s3",)
    assert first.test_positions.tolist() == [0, 1]
    assert first.train_positions.tolist() == [3]
    assert first.exclusion_summary.purged == 1
    assert first.exclusion_summary.embargoed == 0
    assert first.exclusion_summary.retained == 1
    assert first.dataset_digest == dataset.digest

    repeated = PurgedKFold(n_splits=2).plan(dataset).folds[0]
    assert repeated == first


def test_dataset_digest_changes_when_relevant_input_changes() -> None:
    sessions = np.array(["2025-01-02", "2025-01-03"], dtype="datetime64[D]")

    def dataset_with(features: np.ndarray) -> ValidationDataset:
        return ValidationDataset(
            sample_ids=("s0", "s1"),
            session_axis=sessions,
            sessions=sessions,
            information_intervals=(
                InformationInterval(sessions[0], sessions[0]),
                InformationInterval(sessions[1], sessions[1]),
            ),
            targets=np.array([0.0, 1.0]),
            features=features,
        )

    original = dataset_with(np.array([[1.0], [2.0]]))
    changed = dataset_with(np.array([[1.0], [3.0]]))

    assert original.digest != changed.digest


def test_public_arrays_cannot_mutate_audited_evidence() -> None:
    sessions = np.array(["2025-01-02", "2025-01-03"], dtype="datetime64[D]")
    dataset = ValidationDataset(
        sample_ids=("s0", "s1"),
        session_axis=sessions,
        sessions=sessions,
        information_intervals=(
            InformationInterval(sessions[0], sessions[0]),
            InformationInterval(sessions[1], sessions[1]),
        ),
        features=np.array([[1.0], [2.0]]),
        targets=np.array([0.0, 1.0]),
    )
    fold = PurgedKFold(n_splits=2).plan(dataset).folds[0]
    assert isinstance(fold, FoldAssignment)

    with pytest.raises(ValueError):
        dataset.features[0, 0] = 99.0
    with pytest.raises(ValueError):
        fold.test_positions[0] = 99


def test_invalid_dataset_evidence_fails_with_typed_errors() -> None:
    sessions = np.array(["2025-01-02", "2025-01-03"], dtype="datetime64[D]")

    with pytest.raises(DatasetValidationError, match="duplicate identities"):
        ValidationDataset(
            sample_ids=("same", "same"),
            session_axis=sessions,
            sessions=sessions,
            information_intervals=(
                InformationInterval(sessions[0], sessions[0]),
                InformationInterval(sessions[1], sessions[1]),
            ),
            features=np.ones((2, 1)),
            targets=np.ones(2),
        )

    with pytest.raises(DatasetValidationError, match="inconsistent lengths"):
        ValidationDataset(
            sample_ids=("s0", "s1"),
            session_axis=sessions,
            sessions=sessions,
            information_intervals=(InformationInterval(sessions[0], sessions[0]),),
            features=np.ones((2, 1)),
            targets=np.ones(2),
        )

    with pytest.raises(TemporalValidationError, match="outside session_axis"):
        ValidationDataset(
            sample_ids=("s0", "s1"),
            session_axis=sessions,
            sessions=(sessions[0], np.datetime64("2025-01-06")),
            information_intervals=(
                InformationInterval(sessions[0], sessions[0]),
                InformationInterval(sessions[1], sessions[1]),
            ),
            features=np.ones((2, 1)),
            targets=np.ones(2),
        )

    with pytest.raises(TemporalValidationError, match="start must not be after end"):
        InformationInterval(sessions[1], sessions[0])
