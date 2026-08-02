from __future__ import annotations

import pandas as pd

from purged_kfold_validation.adapters.pandaai import (
    PandaAIContinuousMapping,
    PandaAIContinuousPolicy,
    PandaAIDailyConfig,
    PandaAIDailyMapping,
    governed_validation_dataset_from_pandaai_continuous_daily,
)


def test_user_gets_dominant_only_asset_identity_and_roll_receipt() -> None:
    sessions = pd.date_range("2025-01-02", periods=6, freq="D")
    frame = pd.DataFrame(
        {
            "date": [*sessions, sessions[4], sessions[5], sessions[0], sessions[1]],
            "series": ["A_DOMINANT.X"] * 6
            + ["A2502.X", "A2503.X", "B_DOMINANT.X", "B_DOMINANT.X"],
            "file_asset": ["A"] * 8 + ["B", "B"],
            "underlying": ["A"] * 8 + ["B", "B"],
            "active": ["A2501"] * 3
            + ["A2502"] * 3
            + ["A2502", "A2503", "B2501", "B2501"],
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                104.0,
                105.0,
                50.0,
                51.0,
            ],
            "volume": [10.0] * 10,
            "open_interest": [20.0] * 10,
        }
    )

    governed = governed_validation_dataset_from_pandaai_continuous_daily(
        frame,
        mapping=PandaAIDailyMapping(
            session="date",
            asset="file_asset",
            close="close",
            features=("close", "volume", "open_interest"),
        ),
        config=PandaAIDailyConfig(
            label_horizon_sessions=1,
            feature_lookback_sessions=2,
            decision_time_offset_minutes=15 * 60,
            snapshot_id="governance-fixture",
            source_digest="governance-source",
        ),
        continuous_mapping=PandaAIContinuousMapping(
            series_symbol="series",
            asset="file_asset",
            active_contract="active",
            declared_underlying="underlying",
        ),
        policy=PandaAIContinuousPolicy(),
    )

    assert governed.dataset.asset_ids == ("A", "A", "A", "A")
    assert governed.receipt.input_rows == 10
    assert governed.receipt.selected_continuous_rows == 6
    assert governed.receipt.discarded_concrete_rows == 2
    assert governed.receipt.selected_assets == ("A",)
    assert governed.receipt.excluded_assets == ("B",)
    assert governed.receipt.active_contract_transitions == 1
    assert governed.receipt.eligible_rows == 4
    assert governed.receipt.label_roll_crossing_rows == 1
    assert governed.receipt.information_interval_roll_crossing_rows == 2
    assert governed.label_roll_clean.tolist() == [True, False, True, True]
