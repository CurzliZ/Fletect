"""Unit tests for dataset schema, split validation, and temporal feature engineering."""
from datetime import datetime, timezone
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from model.dataset_loader import FEATURES, TARGET_COL, SPLIT_COL, load_and_validate_dataset
from backend.risk_engine import compute_temporal_features


def test_dataset_splits_and_leakage():
    csv_path = Path("flood_risk_dnn_synthetic_120k.csv")
    if not csv_path.exists():
        pytest.skip("Synthetic dataset CSV not present in workspace")

    splits = load_and_validate_dataset(csv_path)

    # 1. Non-empty partitions
    assert len(splits.y_train) == 83940
    assert len(splits.y_val) == 17940
    assert len(splits.y_test) == 17940

    # 2. Features match specification
    assert splits.x_train.shape[1] == len(FEATURES)
    assert splits.x_test.shape[1] == len(FEATURES)

    # 3. Scaler was fitted strictly on train data (mean of train close to 0)
    assert np.allclose(splits.x_train.mean(axis=0), 0.0, atol=1e-2)

    # 4. Target classes present
    assert set(np.unique(splits.y_train)) == {0, 1, 2}
    assert set(np.unique(splits.y_test)) == {0, 1, 2}


def test_compute_temporal_features_cold_start():
    feat = compute_temporal_features(
        device_id="TEST_COLD_DEVICE",
        current_water_cm=35.0,
        current_rain_mm_h=12.0,
        channel_capacity_cm=100.0,
    )

    assert feat["water_level_cm"] == 35.0
    assert feat["water_level_pct"] == 35.0
    assert feat["rainfall_mm_h"] == 12.0
    assert feat["rainfall_3h_mm"] == 12.0
    assert feat["rainfall_6h_mm"] == 12.0
    assert feat["rainfall_24h_mm"] == 12.0
    assert feat["water_level_change_cm_h"] == 0.0
    assert 0.0 <= feat["soil_saturation_pct"] <= 100.0
    assert 0 <= feat["hour_of_day"] <= 23
    assert 1 <= feat["month"] <= 12
