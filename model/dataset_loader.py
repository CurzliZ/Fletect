"""Dataset loader and validation module for Flood Sentry ML pipeline.

Ensures strict chronological splits, reproducible feature extraction,
data validation, and protection against temporal leakage.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

FEATURES: List[str] = [
    "rainfall_mm_h",
    "rainfall_3h_mm",
    "rainfall_6h_mm",
    "rainfall_24h_mm",
    "water_level_cm",
    "water_level_pct",
    "water_level_change_cm_h",
    "soil_saturation_pct",
    "channel_capacity_cm",
    "hour_of_day",
    "month",
]

TARGET_COL: str = "risk_label_encoded_3h_ahead"
HUMAN_TARGET_COL: str = "risk_label_3h_ahead"
SPLIT_COL: str = "dataset_split"

CLASS_NAMES: Dict[int, str] = {
    0: "Low",
    1: "Medium",
    2: "High",
}


@dataclass
class DatasetSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    class_weights: Dict[int, float]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def load_and_validate_dataset(csv_path: Path) -> DatasetSplits:
    """Loads, validates, and partitions the dataset strictly by dataset_split.
    
    Fits scaler only on training data to prevent data leakage.
    Excludes boundary rows where future forecast target is undefined.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    required_cols = set(FEATURES + [TARGET_COL, SPLIT_COL])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    # Exclude rows without a valid future target (boundary rows at end of site splits)
    clean_df = df[df[TARGET_COL].notna()].copy()
    clean_df[TARGET_COL] = clean_df[TARGET_COL].astype(int)

    train_df = clean_df[clean_df[SPLIT_COL] == "train"].copy()
    val_df = clean_df[clean_df[SPLIT_COL] == "validation"].copy()
    test_df = clean_df[clean_df[SPLIT_COL] == "test"].copy()

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError(
            f"Invalid split counts: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    # Feature extraction
    x_train_raw = train_df[FEATURES].to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_COL].to_numpy(dtype=np.int64)

    x_val_raw = val_df[FEATURES].to_numpy(dtype=np.float32)
    y_val = val_df[TARGET_COL].to_numpy(dtype=np.int64)

    x_test_raw = test_df[FEATURES].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.int64)

    # Fit scaler ONLY on train partition
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    x_test = scaler.transform(x_test_raw)

    # Compute balanced class weights for training
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    return DatasetSplits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        scaler=scaler,
        class_weights=class_weights,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )
