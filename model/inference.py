"""Production Inference Engine for Flood Sentry.

Provides a unified, resilient inference interface for the FastAPI backend,
supporting:
  - Deep Neural Network (TensorFlow/Keras) primary model
  - XGBoost tabular alternate
  - Automatic hydrological rule-based fallback if models or dependencies fail
  - Calibrated confidence scoring and explanatory risk rationale
"""
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("flood_sentry.inference")

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

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

CLASS_MAP = {0: "Low", 1: "Medium", 2: "High"}


class FloodRiskInferenceEngine:
    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self.scaler = None
        self.dnn_model = None
        self.xgb_model = None
        self.load_status = {
            "scaler_loaded": False,
            "dnn_loaded": False,
            "xgb_loaded": False,
            "error_detail": None,
        }
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Attempts to load trained scaler and model weights."""
        scaler_file = self.artifacts_dir / "flood_dnn_scaler.joblib"
        dnn_file = self.artifacts_dir / "flood_risk_dnn.keras"
        xgb_file = self.artifacts_dir / "xgboost_flood_model.json"

        # 1. Load Scaler
        if scaler_file.exists():
            try:
                self.scaler = joblib.load(scaler_file)
                self.load_status["scaler_loaded"] = True
                logger.info(f"Loaded scaler from {scaler_file}")
            except Exception as e:
                logger.warning(f"Failed to load scaler from {scaler_file}: {e}")
                self.load_status["error_detail"] = f"Scaler error: {e}"

        # 2. Load DNN
        if dnn_file.exists() and self.scaler is not None:
            try:
                import tensorflow as tf
                self.dnn_model = tf.keras.models.load_model(dnn_file)
                self.load_status["dnn_loaded"] = True
                logger.info(f"Loaded Keras DNN model from {dnn_file}")
            except Exception as e:
                logger.warning(f"Failed to load Keras DNN model from {dnn_file}: {e}")
                self.load_status["error_detail"] = f"DNN error: {e}"

        # 3. Load XGBoost if available
        if xgb_file.exists():
            try:
                import xgboost as xgb
                self.xgb_model = xgb.XGBClassifier()
                self.xgb_model.load_model(str(xgb_file))
                self.load_status["xgb_loaded"] = True
                logger.info(f"Loaded XGBoost model from {xgb_file}")
            except Exception as e:
                logger.warning(f"Failed to load XGBoost model from {xgb_file}: {e}")

    def rule_based_fallback(self, features_dict: Dict[str, float]) -> Tuple[int, str, Dict[str, float], str]:
        """Hydrological deterministic rule-based decision fallback."""
        level_pct = features_dict.get("water_level_pct", 0.0)
        r6 = features_dict.get("rainfall_6h_mm", 0.0)
        delta = features_dict.get("water_level_change_cm_h", 0.0)
        soil = features_dict.get("soil_saturation_pct", 0.0)

        if level_pct >= 76 or (level_pct >= 64 and delta >= 4.0) or (level_pct >= 53 and r6 >= 70 and delta >= 1.5):
            code = 2
            reason = "High risk: Water level near channel capacity or rapid rise under heavy rainfall accumulation."
            probs = {"Low": 0.02, "Medium": 0.08, "High": 0.90}
        elif level_pct >= 43 or (level_pct >= 34 and r6 >= 25) or (level_pct >= 33 and delta >= 2.5) or (level_pct >= 36 and soil >= 82):
            code = 1
            reason = "Medium risk (Watch): Elevated water level, high soil saturation, or sustained precipitation accumulation."
            probs = {"Low": 0.15, "Medium": 0.75, "High": 0.10}
        else:
            code = 0
            reason = "Low risk (Normal): Water level and rainfall accumulations are within safe retention boundaries."
            probs = {"Low": 0.88, "Medium": 0.10, "High": 0.02}

        return code, CLASS_MAP[code], probs, reason

    def predict(self, feature_input: Dict[str, Any], prefer_model: str = "dnn") -> Dict[str, Any]:
        """Runs inference with full error handling and automatic rule fallback.
        
        Args:
            feature_input: Dictionary containing all 11 required features.
            prefer_model: "dnn" or "xgboost".
            
        Returns:
            Dictionary with prediction results, probabilities, confidence,
            rationale, and engine metadata.
        """
        t0 = time.perf_counter()

        # Validate feature completeness
        missing = [f for f in FEATURES if f not in feature_input]
        if missing:
            raise ValueError(f"Missing required features for inference: {missing}")

        # Try ML inference
        if prefer_model == "dnn" and self.dnn_model is not None and self.scaler is not None:
            try:
                row_array = np.array([[float(feature_input[f]) for f in FEATURES]], dtype=np.float32)
                scaled_x = self.scaler.transform(row_array)
                raw_probs = self.dnn_model.predict(scaled_x, verbose=0)[0]
                code = int(np.argmax(raw_probs))
                risk_label = CLASS_MAP[code]
                probs = {CLASS_MAP[i]: round(float(raw_probs[i]), 4) for i in range(3)}
                confidence = float(raw_probs[code])

                # Explainable rationale
                reason = (
                    f"Predicted {risk_label} 3 hours ahead by Deep Neural Network "
                    f"(confidence: {confidence*100:.1f}%). Features: Water level {feature_input['water_level_pct']:.1f}%, "
                    f"6h Rain {feature_input['rainfall_6h_mm']:.1f}mm, Rate {feature_input['water_level_change_cm_h']:+.1f}cm/h."
                )

                latency_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "predicted_risk_3h_ahead": risk_label,
                    "risk_code": code,
                    "class_probabilities": probs,
                    "confidence_score": round(confidence, 4),
                    "model_used": "Deep Neural Network (DNN)",
                    "is_fallback": False,
                    "risk_rationale": reason,
                    "latency_ms": round(latency_ms, 2),
                }
            except Exception as e:
                logger.error(f"DNN inference failed, engaging fallback: {e}")

        # Try XGBoost if DNN not available or selected
        if (prefer_model == "xgboost" or self.dnn_model is None) and self.xgb_model is not None and self.scaler is not None:
            try:
                row_array = np.array([[float(feature_input[f]) for f in FEATURES]], dtype=np.float32)
                scaled_x = self.scaler.transform(row_array)
                raw_probs = self.xgb_model.predict_proba(scaled_x)[0]
                code = int(np.argmax(raw_probs))
                risk_label = CLASS_MAP[code]
                probs = {CLASS_MAP[i]: round(float(raw_probs[i]), 4) for i in range(3)}
                confidence = float(raw_probs[code])

                reason = (
                    f"Predicted {risk_label} 3 hours ahead by XGBoost Classifier "
                    f"(confidence: {confidence*100:.1f}%)."
                )

                latency_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "predicted_risk_3h_ahead": risk_label,
                    "risk_code": code,
                    "class_probabilities": probs,
                    "confidence_score": round(confidence, 4),
                    "model_used": "XGBoost Classifier",
                    "is_fallback": False,
                    "risk_rationale": reason,
                    "latency_ms": round(latency_ms, 2),
                }
            except Exception as e:
                logger.error(f"XGBoost inference failed, engaging fallback: {e}")

        # Fallback to rule engine
        code, risk_label, probs, reason = self.rule_based_fallback(feature_input)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "predicted_risk_3h_ahead": risk_label,
            "risk_code": code,
            "class_probabilities": probs,
            "confidence_score": round(probs[risk_label], 4),
            "model_used": "Hydrological Rule Engine (Fallback)",
            "is_fallback": True,
            "risk_rationale": f"[FALLBACK MODE] {reason}",
            "latency_ms": round(latency_ms, 2),
        }


# Singleton engine instance
_engine_instance: Optional[FloodRiskInferenceEngine] = None


def get_inference_engine() -> FloodRiskInferenceEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = FloodRiskInferenceEngine()
    return _engine_instance
