"""Comprehensive Model Training, Benchmarking, and Evaluation Module for Flood Sentry.

Trains and rigorously benchmarks four distinct paradigms on the exact same
chronological test partition:
  1. Hydrological Rule-Based Baseline
  2. Tabular ML Baseline (XGBoost Classifier)
  3. Deep Neural Network (DNN with Dropout & Regularization)
  4. Sequence-aware Deep Learning Model (LSTM)

Generates full scientific comparative metrics:
  - Confusion matrices (3x3: Low, Medium, High)
  - Per-class Precision, Recall, and F1-score
  - Macro-F1 and Overall Accuracy
  - Critical High-Risk False Negative analysis
  - Latency benchmarks per prediction
  - Model artifact serialization for production deployment
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow as tf
from tensorflow.keras import Sequential, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import xgboost as xgb

from model.dataset_loader import (
    CLASS_NAMES,
    FEATURES,
    HUMAN_TARGET_COL,
    SPLIT_COL,
    TARGET_COL,
    DatasetSplits,
    load_and_validate_dataset,
)

# Output directory for production artifacts
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Rule-Based Baseline Engine
# ---------------------------------------------------------------------------
def rule_based_predict_vector(x_unscaled: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """Evaluates hydrological heuristics on unscaled features.
    
    Returns array of integer risk classes (0=Low, 1=Medium, 2=High).
    """
    pct_idx = feature_names.index("water_level_pct")
    r6_idx = feature_names.index("rainfall_6h_mm")
    delta_idx = feature_names.index("water_level_change_cm_h")
    soil_idx = feature_names.index("soil_saturation_pct")

    predictions = np.zeros(len(x_unscaled), dtype=int)
    for i in range(len(x_unscaled)):
        level_pct = x_unscaled[i, pct_idx]
        r6 = x_unscaled[i, r6_idx]
        delta = x_unscaled[i, delta_idx]
        soil = x_unscaled[i, soil_idx]

        # Domain rules aligned with hydrological thresholds
        if level_pct >= 76 or (level_pct >= 64 and delta >= 4.0) or (level_pct >= 53 and r6 >= 70 and delta >= 1.5):
            predictions[i] = 2  # High
        elif level_pct >= 43 or (level_pct >= 34 and r6 >= 25) or (level_pct >= 33 and delta >= 2.5) or (level_pct >= 36 and soil >= 82):
            predictions[i] = 1  # Medium
        else:
            predictions[i] = 0  # Low
    return predictions


# ---------------------------------------------------------------------------
# 2. Sequence Formatter for LSTM
# ---------------------------------------------------------------------------
def create_sequences_by_site(
    df: pd.DataFrame,
    features: List[str],
    scaler: Any,
    window_size: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Builds sliding sequence windows per site without crossing site boundaries."""
    x_seqs = []
    y_seqs = []
    
    for site_id, site_group in df.groupby("site_id"):
        site_group = site_group.sort_values("timestamp")
        x_vals = scaler.transform(site_group[features].to_numpy(dtype=np.float32))
        y_vals = site_group[TARGET_COL].to_numpy(dtype=np.int64)

        for i in range(len(site_group) - window_size + 1):
            x_seqs.append(x_vals[i : i + window_size])
            y_seqs.append(y_vals[i + window_size - 1])

    return np.array(x_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.int64)


# ---------------------------------------------------------------------------
# Metric Calculation Helper
# ---------------------------------------------------------------------------
def compute_comprehensive_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, latency_ms: float
) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    precision_per_class = precision_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)

    # High-Risk False Negative Analysis (Actual High predicted as Low or Medium)
    # Row 2 in confusion matrix is Actual High
    total_high_actual = int(cm[2].sum())
    high_missed_as_low = int(cm[2, 0])
    high_missed_as_medium = int(cm[2, 1])
    high_detected_correctly = int(cm[2, 2])
    false_negative_rate_high = float((high_missed_as_low + high_missed_as_medium) / total_high_actual) if total_high_actual > 0 else 0.0

    report_str = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Low", "Medium", "High"],
        digits=4,
        zero_division=0,
    )

    return {
        "model_name": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": {
            "Low": {
                "precision": float(precision_per_class[0]),
                "recall": float(recall_per_class[0]),
                "f1": float(f1_per_class[0]),
                "support": int(cm[0].sum()),
            },
            "Medium": {
                "precision": float(precision_per_class[1]),
                "recall": float(recall_per_class[1]),
                "f1": float(f1_per_class[1]),
                "support": int(cm[1].sum()),
            },
            "High": {
                "precision": float(precision_per_class[2]),
                "recall": float(recall_per_class[2]),
                "f1": float(f1_per_class[2]),
                "support": total_high_actual,
            },
        },
        "high_risk_false_negative_analysis": {
            "total_high_risk_actual": total_high_actual,
            "correctly_identified_high": high_detected_correctly,
            "dangerously_missed_as_low": high_missed_as_low,
            "missed_as_medium": high_missed_as_medium,
            "high_risk_false_negative_rate": round(false_negative_rate_high, 4),
        },
        "inference_latency_ms_per_sample": round(latency_ms, 4),
        "classification_report_text": report_str,
    }


# ---------------------------------------------------------------------------
# Main Training & Benchmarking Routine
# ---------------------------------------------------------------------------
def run_model_training_pipeline(dataset_path: Path) -> Dict[str, Any]:
    print(f"[*] Loading and partitioning dataset from: {dataset_path}")
    splits = load_and_validate_dataset(dataset_path)
    print(f"[*] Train rows: {len(splits.y_train)}, Val rows: {len(splits.y_val)}, Test rows: {len(splits.y_test)}")
    print(f"[*] Computed class weights: {splits.class_weights}")

    # Save scaler artifact
    scaler_path = ARTIFACTS_DIR / "flood_dnn_scaler.joblib"
    joblib.dump(splits.scaler, scaler_path)
    print(f"[+] Saved fitted scaler to {scaler_path}")

    # Unscaled test features for rule-based baseline
    x_test_unscaled = splits.test_df[FEATURES].to_numpy(dtype=np.float32)

    results = {}

    # -----------------------------------------------------------------------
    # 1. Evaluate Rule-Based Baseline
    # -----------------------------------------------------------------------
    print("\n--- 1. Evaluating Rule-Based Baseline ---")
    t0 = time.perf_counter()
    y_pred_rule = rule_based_predict_vector(x_test_unscaled, FEATURES)
    rule_latency = ((time.perf_counter() - t0) / len(x_test_unscaled)) * 1000.0
    results["rule_baseline"] = compute_comprehensive_metrics(
        splits.y_test, y_pred_rule, "Hydrological Rule-Based Baseline", rule_latency
    )
    print(f"Rule Baseline Accuracy: {results['rule_baseline']['accuracy']:.4f}, Macro-F1: {results['rule_baseline']['macro_f1']:.4f}")

    # -----------------------------------------------------------------------
    # 2. Train and Evaluate XGBoost Tabular Baseline
    # -----------------------------------------------------------------------
    print("\n--- 2. Training XGBoost Classifier Baseline ---")
    sample_weights_train = np.array([splits.class_weights[y] for y in splits.y_train])

    xgb_model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    xgb_model.fit(
        splits.x_train,
        splits.y_train,
        sample_weight=sample_weights_train,
        eval_set=[(splits.x_val, splits.y_val)],
        verbose=False,
    )
    xgb_model_path = ARTIFACTS_DIR / "xgboost_flood_model.json"
    xgb_model.save_model(str(xgb_model_path))
    print(f"[+] Saved XGBoost model to {xgb_model_path}")

    t0 = time.perf_counter()
    y_pred_xgb = xgb_model.predict(splits.x_test)
    xgb_latency = ((time.perf_counter() - t0) / len(splits.x_test)) * 1000.0
    results["xgboost"] = compute_comprehensive_metrics(
        splits.y_test, y_pred_xgb, "XGBoost Classifier", xgb_latency
    )
    print(f"XGBoost Accuracy: {results['xgboost']['accuracy']:.4f}, Macro-F1: {results['xgboost']['macro_f1']:.4f}")

    # -----------------------------------------------------------------------
    # 3. Train and Evaluate Deep Neural Network (DNN)
    # -----------------------------------------------------------------------
    print("\n--- 3. Training Deep Neural Network (DNN) ---")
    tf.keras.utils.set_random_seed(42)

    dnn_model = Sequential([
        layers.Input(shape=(len(FEATURES),)),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.20),
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.15),
        layers.Dense(16, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])
    dnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    dnn_callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ModelCheckpoint(filepath=str(ARTIFACTS_DIR / "flood_risk_dnn.keras"), monitor="val_loss", save_best_only=True),
    ]

    dnn_history = dnn_model.fit(
        splits.x_train,
        splits.y_train,
        validation_data=(splits.x_val, splits.y_val),
        epochs=35,
        batch_size=512,
        class_weight=splits.class_weights,
        callbacks=dnn_callbacks,
        verbose=1,
    )

    t0 = time.perf_counter()
    dnn_probs = dnn_model.predict(splits.x_test, verbose=0)
    dnn_latency = ((time.perf_counter() - t0) / len(splits.x_test)) * 1000.0
    y_pred_dnn = dnn_probs.argmax(axis=1)

    results["dnn"] = compute_comprehensive_metrics(
        splits.y_test, y_pred_dnn, "Deep Neural Network (DNN)", dnn_latency
    )
    results["dnn"]["epochs_trained"] = len(dnn_history.history["loss"])
    print(f"DNN Accuracy: {results['dnn']['accuracy']:.4f}, Macro-F1: {results['dnn']['macro_f1']:.4f}")

    # -----------------------------------------------------------------------
    # 4. Train and Evaluate Sequential LSTM Model
    # -----------------------------------------------------------------------
    print("\n--- 4. Training Sequential LSTM Network ---")
    window_size = 6  # 6-hour sequence history
    print(f"[*] Constructing {window_size}-hour temporal sequence tensors per site...")
    x_train_seq, y_train_seq = create_sequences_by_site(splits.train_df, FEATURES, splits.scaler, window_size)
    x_val_seq, y_val_seq = create_sequences_by_site(splits.val_df, FEATURES, splits.scaler, window_size)
    x_test_seq, y_test_seq = create_sequences_by_site(splits.test_df, FEATURES, splits.scaler, window_size)

    lstm_model = Sequential([
        layers.Input(shape=(window_size, len(FEATURES))),
        layers.LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.20),
        layers.Dense(16, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])
    lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    lstm_checkpoint_path = ARTIFACTS_DIR / "flood_risk_lstm.keras"
    lstm_callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=str(lstm_checkpoint_path), monitor="val_loss", save_best_only=True),
    ]

    lstm_history = lstm_model.fit(
        x_train_seq,
        y_train_seq,
        validation_data=(x_val_seq, y_val_seq),
        epochs=25,
        batch_size=512,
        class_weight=splits.class_weights,
        callbacks=lstm_callbacks,
        verbose=1,
    )

    t0 = time.perf_counter()
    lstm_probs = lstm_model.predict(x_test_seq, verbose=0)
    lstm_latency = ((time.perf_counter() - t0) / len(x_test_seq)) * 1000.0
    y_pred_lstm = lstm_probs.argmax(axis=1)

    results["lstm"] = compute_comprehensive_metrics(
        y_test_seq, y_pred_lstm, "Sequential Recurrent Network (LSTM)", lstm_latency
    )
    results["lstm"]["epochs_trained"] = len(lstm_history.history["loss"])
    print(f"LSTM Accuracy: {results['lstm']['accuracy']:.4f}, Macro-F1: {results['lstm']['macro_f1']:.4f}")

    # -----------------------------------------------------------------------
    # Comprehensive Summary & Serialization
    # -----------------------------------------------------------------------
    comparison_summary = {
        "metadata": {
            "prediction_target": "Simulated flood risk 3 hours ahead (0=Low, 1=Medium, 2=High)",
            "dataset_origin": str(dataset_path),
            "disclaimer": "Trained on synthetic dataset calibrated to CWC rainfall empirical distribution. Research/decision-support only.",
            "test_sample_count": len(splits.y_test),
            "feature_set": FEATURES,
            "classes": CLASS_NAMES,
        },
        "model_comparison": {
            "Rule-Based Baseline": {
                "accuracy": results["rule_baseline"]["accuracy"],
                "macro_f1": results["rule_baseline"]["macro_f1"],
                "high_risk_recall": results["rule_baseline"]["per_class_metrics"]["High"]["recall"],
                "high_risk_fn_rate": results["rule_baseline"]["high_risk_false_negative_analysis"]["high_risk_false_negative_rate"],
                "latency_ms": results["rule_baseline"]["inference_latency_ms_per_sample"],
            },
            "XGBoost Classifier": {
                "accuracy": results["xgboost"]["accuracy"],
                "macro_f1": results["xgboost"]["macro_f1"],
                "high_risk_recall": results["xgboost"]["per_class_metrics"]["High"]["recall"],
                "high_risk_fn_rate": results["xgboost"]["high_risk_false_negative_analysis"]["high_risk_false_negative_rate"],
                "latency_ms": results["xgboost"]["inference_latency_ms_per_sample"],
            },
            "Deep Neural Network (DNN)": {
                "accuracy": results["dnn"]["accuracy"],
                "macro_f1": results["dnn"]["macro_f1"],
                "high_risk_recall": results["dnn"]["per_class_metrics"]["High"]["recall"],
                "high_risk_fn_rate": results["dnn"]["high_risk_false_negative_analysis"]["high_risk_false_negative_rate"],
                "latency_ms": results["dnn"]["inference_latency_ms_per_sample"],
            },
            "Sequential LSTM": {
                "accuracy": results["lstm"]["accuracy"],
                "macro_f1": results["lstm"]["macro_f1"],
                "high_risk_recall": results["lstm"]["per_class_metrics"]["High"]["recall"],
                "high_risk_fn_rate": results["lstm"]["high_risk_false_negative_analysis"]["high_risk_false_negative_rate"],
                "latency_ms": results["lstm"]["inference_latency_ms_per_sample"],
            },
        },
        "detailed_results": results,
    }

    report_path = ARTIFACTS_DIR / "model_comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"\n[+] Master model comparison report saved to: {report_path.resolve()}")

    # Also update evaluation_report.json for backward compatibility
    compat_report = {
        "warning": "Synthetic-only evaluation; not evidence of real-world flood prediction accuracy.",
        "prediction_target": "risk class 3 hours ahead",
        "dataset": str(dataset_path.name),
        "rows": {"train": len(splits.y_train), "validation": len(splits.y_val), "test": len(splits.y_test)},
        "features": FEATURES,
        "class_mapping": CLASS_NAMES,
        "class_weights": splits.class_weights,
        "test_accuracy": results["dnn"]["accuracy"],
        "confusion_matrix_rows_true_cols_predicted": results["dnn"]["confusion_matrix"],
        "classification_report": results["dnn"]["classification_report_text"],
        "epochs_ran": results["dnn"].get("epochs_trained", 20),
    }
    with open(ARTIFACTS_DIR / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(compat_report, f, indent=2)

    return comparison_summary


if __name__ == "__main__":
    csv_file = Path("flood_risk_dnn_synthetic_120k.csv")
    run_model_training_pipeline(csv_file)
