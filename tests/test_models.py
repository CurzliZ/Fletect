"""Unit tests for ML models, inference engine, and rule-based fallback."""
import pytest
from model.inference import FloodRiskInferenceEngine, FEATURES


@pytest.fixture
def inference_engine():
    return FloodRiskInferenceEngine()


def test_inference_engine_fallback_mechanics(inference_engine):
    # Test deterministic fallback
    features_low = {
        "rainfall_mm_h": 0.0,
        "rainfall_3h_mm": 0.0,
        "rainfall_6h_mm": 0.0,
        "rainfall_24h_mm": 0.0,
        "water_level_cm": 20.0,
        "water_level_pct": 20.0,
        "water_level_change_cm_h": 0.0,
        "soil_saturation_pct": 25.0,
        "channel_capacity_cm": 100.0,
        "hour_of_day": 12,
        "month": 6,
    }
    code, label, probs, reason = inference_engine.rule_based_fallback(features_low)
    assert code == 0
    assert label == "Low"
    assert abs(sum(probs.values()) - 1.0) < 1e-4

    features_high = {
        "rainfall_mm_h": 80.0,
        "rainfall_3h_mm": 120.0,
        "rainfall_6h_mm": 180.0,
        "rainfall_24h_mm": 210.0,
        "water_level_cm": 85.0,
        "water_level_pct": 85.0,
        "water_level_change_cm_h": 4.5,
        "soil_saturation_pct": 95.0,
        "channel_capacity_cm": 100.0,
        "hour_of_day": 12,
        "month": 11,
    }
    code_h, label_h, probs_h, reason_h = inference_engine.rule_based_fallback(features_high)
    assert code_h == 2
    assert label_h == "High"


def test_predict_primary_model(inference_engine):
    features_vector = {
        "rainfall_mm_h": 15.0,
        "rainfall_3h_mm": 30.0,
        "rainfall_6h_mm": 45.0,
        "rainfall_24h_mm": 60.0,
        "water_level_cm": 50.0,
        "water_level_pct": 50.0,
        "water_level_change_cm_h": 1.2,
        "soil_saturation_pct": 65.0,
        "channel_capacity_cm": 100.0,
        "hour_of_day": 14,
        "month": 10,
    }

    result = inference_engine.predict(features_vector, prefer_model="dnn")

    assert "predicted_risk_3h_ahead" in result
    assert result["predicted_risk_3h_ahead"] in ["Low", "Medium", "High"]
    assert result["risk_code"] in [0, 1, 2]
    assert 0.0 <= result["confidence_score"] <= 1.0
    assert "class_probabilities" in result
    assert len(result["class_probabilities"]) == 3
    assert "risk_rationale" in result
    assert result["latency_ms"] >= 0.0


def test_missing_feature_raises_error(inference_engine):
    incomplete_features = {"rainfall_mm_h": 10.0}
    with pytest.raises(ValueError, match="Missing required features"):
        inference_engine.predict(incomplete_features)
