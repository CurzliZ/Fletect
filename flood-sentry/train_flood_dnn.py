"""Train and evaluate a DNN on the synthetic flood-risk dataset.

This reports performance against synthetic rule-generated labels only. It is not
real-world flood forecasting validation.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input

DATA_FILE = Path("flood_risk_dnn_synthetic_120k.csv")
OUT_DIR = Path("artifacts")
FEATURES = [
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
CLASS_NAMES = ["Low", "Medium", "High"]


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE}. Put this script in the same folder as the dataset."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_FILE)
    TARGET = "risk_label_encoded_3h_ahead"
    required = set(FEATURES + [TARGET, "dataset_split"])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Rows at each split boundary have no safe 3-hour-ahead target, so exclude them.
    data = data[data[TARGET].notna()].copy()
    train = data[data["dataset_split"] == "train"]
    val = data[data["dataset_split"] == "validation"]
    test = data[data["dataset_split"] == "test"]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Expected non-empty train, validation, and test splits.")

    x_train = train[FEATURES].astype("float32")
    y_train = train[TARGET].astype("int64").to_numpy()
    x_val = val[FEATURES].astype("float32")
    y_val = val[TARGET].astype("int64").to_numpy()
    x_test = test[FEATURES].astype("float32")
    y_test = test[TARGET].astype("int64").to_numpy()

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    joblib.dump(scaler, OUT_DIR / "flood_dnn_scaler.joblib")

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(label): float(weight) for label, weight in zip(classes, weights)}

    tf.keras.utils.set_random_seed(42)
    model = Sequential([
        Input(shape=(len(FEATURES),)),
        Dense(64, activation="relu"),
        Dropout(0.20),
        Dense(32, activation="relu"),
        Dropout(0.10),
        Dense(16, activation="relu"),
        Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    history = model.fit(
        x_train_scaled,
        y_train,
        validation_data=(x_val_scaled, y_val),
        epochs=40,
        batch_size=512,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    probabilities = model.predict(x_test_scaled, verbose=0)
    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    accuracy = accuracy_score(y_test, predictions)
    model.save(OUT_DIR / "flood_risk_dnn.keras")
    result = {
        "warning": "Synthetic-only evaluation; not evidence of real-world flood prediction accuracy.",
        "prediction_target": "risk class 3 hours ahead",
        "dataset": str(DATA_FILE),
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "features": FEATURES,
        "class_mapping": {"0": "Low", "1": "Medium", "2": "High"},
        "class_weights": class_weights,
        "test_accuracy": float(accuracy),
        "confusion_matrix_rows_true_cols_predicted": matrix.tolist(),
        "classification_report": report,
        "epochs_ran": len(history.history.get("loss", [])),
    }
    (OUT_DIR / "evaluation_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nTEST ACCURACY (synthetic labels only):", round(accuracy, 4))
    print("\nCLASSIFICATION REPORT\n", report)
    print("\nCONFUSION MATRIX (rows=true, columns=predicted; Low/Medium/High)\n", matrix)
    print(f"\nSaved model and evaluation files in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
