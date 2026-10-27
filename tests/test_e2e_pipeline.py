"""Deterministic End-to-End Pipeline Verification Test for Flood Sentry.

Validates the complete signal pathway:
  Raw Sensor Readings (Ultrasonic Distance & Rain ADC)
  -> Rolling Temporal Feature Aggregation
  -> Model Inference / Rule Fallback
  -> Alert Hysteresis Engine
  -> SQLite Historical Persistence
  -> Endpoint Availability
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_deterministic_low_risk_pathway():
    """Scenario 1: Dry conditions with low water retention -> Low Risk, No Alert."""
    payload = {
        "device_id": "E2E_DEV_LOW",
        "distance_cm": 85.0,  # 85 cm distance => 15 cm water (15% capacity)
        "rainfall_raw": 0,
        "rainfall_pct": 0.0,
        "flow_pct": 70.0,
        "channel_capacity_cm": 100.0,
        "data_source": "test_injection",
    }
    res = client.post("/api/telemetry", json=payload)
    assert res.status_code == 201
    data = res.json()

    assert data["water_level_cm"] == 15.0
    assert data["water_level_pct"] == 15.0
    assert data["predicted_risk"] == "Low"
    assert data["risk_code"] == 0
    assert data["alert_triggered"] is False


def test_deterministic_medium_risk_with_hysteresis():
    """Scenario 2: Rising water & moderate rain -> Medium Risk (Watch).
    Tests that hysteresis requires sustained confirmation (2 consecutive samples) for Medium alert.
    """
    dev_id = "E2E_DEV_MED"
    # Sample 1: first medium reading (water level 42%, moderate rain 10 mm/h)
    payload_1 = {
        "device_id": dev_id,
        "distance_cm": 58.0,  # 42% water level
        "rainfall_pct": 10.0,
        "rainfall_mm_h": 10.0,
        "flow_pct": 50.0,
        "soil_saturation_pct": 45.0,
        "channel_capacity_cm": 100.0,
        "data_source": "test_injection",
    }
    res1 = client.post("/api/telemetry", json=payload_1)
    assert res1.status_code == 201
    data1 = res1.json()
    assert data1["predicted_risk"] == "Medium"
    # First sample should NOT trigger alert yet (hysteresis debounce)
    assert data1["alert_triggered"] is False

    # Sample 2: sustained medium reading -> triggers confirmed Medium alert
    payload_2 = {
        "device_id": dev_id,
        "distance_cm": 58.0,  # 42% water level
        "rainfall_pct": 10.0,
        "rainfall_mm_h": 10.0,
        "flow_pct": 50.0,
        "soil_saturation_pct": 46.0,
        "channel_capacity_cm": 100.0,
        "data_source": "test_injection",
    }
    res2 = client.post("/api/telemetry", json=payload_2)
    assert res2.status_code == 201
    data2 = res2.json()

    assert data2["predicted_risk"] == "Medium"
    assert data2["alert_triggered"] is True
    assert data2["alert_severity"] == "Medium"


def test_deterministic_high_risk_immediate_escalation():
    """Scenario 3: Severe flash flood deluge -> High Risk (Danger).
    Escalates immediately to High severity alert on the very first occurrence.
    """
    payload_high = {
        "device_id": "E2E_DEV_HIGH",
        "distance_cm": 15.0,  # 85% channel water depth
        "rainfall_raw": 3600,  # ~88% rain
        "rainfall_pct": 88.0,
        "flow_pct": 15.0,
        "channel_capacity_cm": 100.0,
        "data_source": "test_injection",
    }
    res = client.post("/api/telemetry", json=payload_high)
    assert res.status_code == 201
    data = res.json()

    assert data["predicted_risk"] == "High"
    assert data["risk_code"] == 2
    assert data["alert_triggered"] is True
    assert data["alert_severity"] == "High"


def test_invalid_sensor_timeout_handling():
    """Scenario 4: Ultrasonic sensor disconnected / no echo (-1 distance)."""
    payload_fault = {
        "device_id": "E2E_DEV_FAULT",
        "distance_cm": -1.0,
        "rainfall_pct": 5.0,
        "channel_capacity_cm": 100.0,
        "data_source": "test_injection",
    }
    res = client.post("/api/telemetry", json=payload_fault)
    assert res.status_code == 201
    data = res.json()

    # System handles invalid echo safely without crashing
    assert data["status"] == "success"
    assert "predicted_risk" in data


def test_database_persistence_and_alert_retrieval():
    """Scenario 5: Verify records and alert log persisted in SQLite."""
    # Check history contains our E2E device records
    res_hist = client.get("/api/history?device_id=E2E_DEV_HIGH&limit=10")
    assert res_hist.status_code == 200
    records = res_hist.json()
    assert len(records) >= 1

    # Check alert table recorded the high risk event
    res_alerts = client.get("/api/alerts?limit=20")
    assert res_alerts.status_code == 200
    alerts = res_alerts.json()
    assert len(alerts) >= 1
    assert any(a["device_id"] == "E2E_DEV_HIGH" for a in alerts)
