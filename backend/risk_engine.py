"""Risk Engine for Flood Sentry.

Handles:
  1. Temporal rolling feature aggregation (3h, 6h, 24h rainfall, water rate of change)
  2. Soil saturation proxy updating
  3. Real-time inference invocation via FloodRiskInferenceEngine
  4. Alert state machine with hysteresis to prevent flapping
"""
from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional, Tuple

from backend.database import get_db_connection, save_alert
from model.inference import get_inference_engine

logger = logging.getLogger("flood_sentry.risk_engine")

# In-memory hysteresis state tracking per device
# Structure: { device_id: { "last_risk_code": int, "consecutive_readings": int, "last_alert_time": str } }
_device_states: Dict[str, Dict[str, Any]] = {}


def compute_temporal_features(
    device_id: str,
    current_water_cm: float,
    current_rain_mm_h: float,
    channel_capacity_cm: float = 100.0,
    provided_soil: Optional[float] = None,
    current_time: Optional[datetime] = None,
) -> Dict[str, float]:
    """Computes rolling window temporal features using recent SQLite telemetry."""
    now = current_time or datetime.now(timezone.utc)
    hour_of_day = now.hour
    month = now.month
    water_pct = min(100.0, max(0.0, (current_water_cm / channel_capacity_cm) * 100.0))

    # Fetch recent telemetry for this device from SQLite
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT timestamp, water_level_cm, rainfall_mm_h, soil_saturation_pct
            FROM telemetry
            WHERE device_id = ?
            ORDER BY id DESC
            LIMIT 24
            """,
            (device_id,),
        )
        recent_rows = cursor.fetchall()

    if not recent_rows:
        # Cold start: first reading from this device
        soil = provided_soil if provided_soil is not None else min(100.0, max(15.0, current_rain_mm_h * 2.0 + 25.0))
        return {
            "rainfall_mm_h": float(current_rain_mm_h),
            "rainfall_3h_mm": float(current_rain_mm_h),
            "rainfall_6h_mm": float(current_rain_mm_h),
            "rainfall_24h_mm": float(current_rain_mm_h),
            "water_level_cm": float(current_water_cm),
            "water_level_pct": round(water_pct, 2),
            "water_level_change_cm_h": 0.0,
            "soil_saturation_pct": round(soil, 2),
            "channel_capacity_cm": float(channel_capacity_cm),
            "hour_of_day": int(hour_of_day),
            "month": int(month),
        }

    # Extract historical rain values
    past_rains = [row["rainfall_mm_h"] for row in recent_rows]
    past_waters = [row["water_level_cm"] for row in recent_rows]
    last_soil = recent_rows[0]["soil_saturation_pct"]

    # Rolling rainfall accumulations
    # 3-hour sum: current + up to 2 past hourly readings
    r3 = current_rain_mm_h + sum(past_rains[:2])
    # 6-hour sum: current + up to 5 past hourly readings
    r6 = current_rain_mm_h + sum(past_rains[:5])
    # 24-hour sum: current + all available up to 23 past readings
    r24 = current_rain_mm_h + sum(past_rains[:23])

    # Water level change rate (current vs reading ~1 step ago)
    delta_water = current_water_cm - past_waters[0]

    # Soil saturation proxy update
    if provided_soil is not None:
        soil = provided_soil
    else:
        # Hydrological saturation retention proxy: decay prior soil and absorb fraction of current rain
        soil = 0.985 * last_soil + 0.62 * min(current_rain_mm_h, 50.0) - 0.48
        soil = min(100.0, max(0.0, soil))

    return {
        "rainfall_mm_h": round(float(current_rain_mm_h), 2),
        "rainfall_3h_mm": round(float(r3), 2),
        "rainfall_6h_mm": round(float(r6), 2),
        "rainfall_24h_mm": round(float(r24), 2),
        "water_level_cm": round(float(current_water_cm), 2),
        "water_level_pct": round(water_pct, 2),
        "water_level_change_cm_h": round(float(delta_water), 2),
        "soil_saturation_pct": round(float(soil), 2),
        "channel_capacity_cm": float(channel_capacity_cm),
        "hour_of_day": int(hour_of_day),
        "month": int(month),
    }


def evaluate_alert_state(
    device_id: str,
    timestamp: str,
    risk_code: int,
    risk_label: str,
    water_level_pct: float,
    rainfall_6h_mm: float,
    risk_rationale: str,
) -> Tuple[bool, Optional[str]]:
    """Applies hysteresis to suppress flapping and triggers alerts on meaningful state transitions.
    
    Hysteresis logic:
      - Transition to High risk triggers immediate alert.
      - Transition to Medium risk triggers alert if sustained for >= 2 consecutive samples.
      - Return from High/Medium to Low suppresses alerts and resets state.
    """
    global _device_states
    state = _device_states.get(
        device_id,
        {"last_risk_code": 0, "consecutive_readings": 0, "last_alert_time": None},
    )

    last_code = state["last_risk_code"]
    consecutive = state["consecutive_readings"]

    alert_triggered = False
    severity = None

    if risk_code == 2:  # High Risk / Flood Danger
        # Urgent: alert immediately on entry or escalation
        if last_code < 2 or consecutive % 10 == 0:  # Re-alert periodically if persistent danger
            alert_triggered = True
            severity = "High"
            save_alert({
                "device_id": device_id,
                "timestamp": timestamp,
                "severity": severity,
                "trigger_reason": f"DANGER: {risk_rationale}",
                "water_level_pct": water_level_pct,
                "rainfall_6h_mm": rainfall_6h_mm,
                "acknowledged": 0,
            })
        consecutive = consecutive + 1 if last_code == 2 else 1

    elif risk_code == 1:  # Medium Risk / Watch
        # Require confirmation (2 consecutive samples) before raising new watch alert from Low
        consecutive = consecutive + 1 if last_code == 1 else 1
        if consecutive == 2 or (consecutive > 2 and consecutive % 15 == 0):
            alert_triggered = True
            severity = "Medium"
            save_alert({
                "device_id": device_id,
                "timestamp": timestamp,
                "severity": severity,
                "trigger_reason": f"WATCH: {risk_rationale}",
                "water_level_pct": water_level_pct,
                "rainfall_6h_mm": rainfall_6h_mm,
                "acknowledged": 0,
            })
    else:  # Low Risk / Normal
        consecutive = 0

    # Update state
    _device_states[device_id] = {
        "last_risk_code": risk_code,
        "consecutive_readings": consecutive,
        "last_alert_time": timestamp if alert_triggered else state["last_alert_time"],
    }

    return alert_triggered, severity


def process_device_telemetry(payload_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Coordinates telemetry transformation, temporal aggregation, ML inference, and alert checks."""
    device_id = payload_dict.get("device_id", "ESP32_FLOOD_01")
    timestamp = payload_dict.get("timestamp") or datetime.now(timezone.utc).isoformat()
    channel_capacity = float(payload_dict.get("channel_capacity_cm", 100.0))

    # Resolve water level from distance if distance given
    distance_cm = payload_dict.get("distance_cm")
    if payload_dict.get("water_level_cm") is not None:
        water_level_cm = float(payload_dict["water_level_cm"])
    elif distance_cm is not None and distance_cm >= 0:
        # Water level = Channel depth - Ultrasonic distance
        water_level_cm = max(0.0, channel_capacity - distance_cm)
    elif payload_dict.get("water_level_pct") is not None:
        water_level_cm = (float(payload_dict["water_level_pct"]) / 100.0) * channel_capacity
    else:
        water_level_cm = 25.0  # Safe default if sensor silent

    # Resolve rainfall rate mm/h
    if payload_dict.get("rainfall_mm_h") is not None:
        rain_mm_h = float(payload_dict["rainfall_mm_h"])
    elif payload_dict.get("rainfall_pct") is not None:
        # Map 0-100% pot to 0-100 mm/h
        rain_mm_h = float(payload_dict["rainfall_pct"])
    elif payload_dict.get("rainfall_raw") is not None:
        # Map 12-bit ADC 0-4095 to 0-100 mm/h
        rain_mm_h = (float(payload_dict["rainfall_raw"]) / 4095.0) * 100.0
    else:
        rain_mm_h = 0.0

    soil_pct = payload_dict.get("soil_saturation_pct")

    # 1. Temporal Feature Aggregation
    features = compute_temporal_features(
        device_id=device_id,
        current_water_cm=water_level_cm,
        current_rain_mm_h=rain_mm_h,
        channel_capacity_cm=channel_capacity,
        provided_soil=soil_pct,
    )

    # 2. Model Inference
    engine = get_inference_engine()
    inference_result = engine.predict(features, prefer_model="dnn")

    # 3. Alert State Machine & Hysteresis
    alert_triggered, severity = evaluate_alert_state(
        device_id=device_id,
        timestamp=timestamp,
        risk_code=inference_result["risk_code"],
        risk_label=inference_result["predicted_risk_3h_ahead"],
        water_level_pct=features["water_level_pct"],
        rainfall_6h_mm=features["rainfall_6h_mm"],
        risk_rationale=inference_result["risk_rationale"],
    )

    return {
        "features": features,
        "inference": inference_result,
        "alert_triggered": alert_triggered,
        "alert_severity": severity,
        "timestamp": timestamp,
        "device_id": device_id,
        "distance_cm": distance_cm,
        "flow_pct": payload_dict.get("flow_pct", 50.0),
        "data_source": payload_dict.get("data_source", "simulation"),
    }
