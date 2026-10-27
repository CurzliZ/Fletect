"""Integration tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "prediction_horizon_hours" in data


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert data["database_connected"] is True
    assert "models" in data


def test_ingest_telemetry_valid():
    payload = {
        "device_id": "TEST_UNIT_ESP32",
        "distance_cm": 65.0,
        "rainfall_raw": 1024,
        "rainfall_pct": 25.0,
        "flow_pct": 50.0,
        "channel_capacity_cm": 100.0,
        "data_source": "test_injection",
    }
    response = client.post("/api/telemetry", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["device_id"] == "TEST_UNIT_ESP32"
    assert data["water_level_cm"] == 35.0
    assert data["water_level_pct"] == 35.0
    assert data["predicted_risk"] in ["Low", "Medium", "High"]
    assert "confidence_score" in data
    assert "alert_triggered" in data


def test_ingest_telemetry_invalid_bounds():
    # Negative channel depth is invalid
    payload = {
        "device_id": "TEST_INVALID",
        "channel_capacity_cm": -50.0,
    }
    response = client.post("/api/telemetry", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_get_latest_and_history():
    # Fetch latest
    res_latest = client.get("/api/latest?device_id=TEST_UNIT_ESP32")
    assert res_latest.status_code == 200
    data = res_latest.json()
    assert data["status"] == "ok"
    assert data["data"]["device_id"] == "TEST_UNIT_ESP32"

    # Fetch history
    res_hist = client.get("/api/history?limit=10")
    assert res_hist.status_code == 200
    records = res_hist.json()
    assert isinstance(records, list)
    assert len(records) >= 1


def test_predict_endpoint_backward_compat():
    features = {
        "rainfall_mm_h": 10.0,
        "rainfall_3h_mm": 20.0,
        "rainfall_6h_mm": 30.0,
        "rainfall_24h_mm": 40.0,
        "water_level_cm": 45.0,
        "water_level_pct": 45.0,
        "water_level_change_cm_h": 0.5,
        "soil_saturation_pct": 50.0,
        "channel_capacity_cm": 100.0,
        "hour_of_day": 10,
        "month": 8,
    }
    response = client.post("/predict", json=features)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_risk_3h_ahead"] in ["Low", "Medium", "High"]
    assert "class_probabilities" in data


def test_alerts_and_acknowledge():
    res_alerts = client.get("/api/alerts?limit=10")
    assert res_alerts.status_code == 200
    assert isinstance(res_alerts.json(), list)

    res_ack = client.post("/api/alerts/acknowledge")
    assert res_ack.status_code == 200
    assert "acknowledged_count" in res_ack.json()


def test_model_info_endpoint():
    response = client.get("/api/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_comparison" in data or "status" in data
