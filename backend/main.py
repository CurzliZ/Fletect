"""FastAPI Backend Application for Flood Sentry.

Coordinates:
  - Telemetry ingestion from Wokwi ESP32 & simulation bridge
  - Rolling temporal feature calculation
  - ML inference (DNN / XGBoost / Rule Fallback)
  - Alert generation with hysteresis
  - SQLite historical persistence
  - Dashboard integration endpoints
"""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

from backend.database import (
    acknowledge_alerts,
    get_db_connection,
    get_latest_telemetry,
    get_recent_alerts,
    get_telemetry_history,
    save_telemetry,
)
from backend.risk_engine import process_device_telemetry
from backend.schemas import (
    AlertEvent,
    PredictResponse,
    RawTelemetryPayload,
    SensorFeatures,
    SystemHealthResponse,
    TelemetryIngestResponse,
)
from model.inference import CLASS_MAP, FEATURES, get_inference_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flood_sentry.api")

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

app = FastAPI(
    title="Flood Sentry Backend API",
    description=(
        "IoT Telemetry Ingestion and Deep Neural Network Flood-Risk Decision Support API. "
        "Predicts 3-hour-ahead flood risk from ultrasonic water level and precipitation sensors."
    ),
    version="2.0.0",
)

# Enable CORS for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "Flood Sentry IoT & DNN Decision-Support System",
        "status": "operational",
        "docs_url": "/docs",
        "prediction_horizon_hours": 3,
        "disclaimer": (
            "Experimental research prototype. Calibrated to CWC rainfall distribution. "
            "Decision support only; not an official flood warning service."
        ),
    }


@app.get("/health", response_model=SystemHealthResponse)
def health() -> SystemHealthResponse:
    engine = get_inference_engine()
    db_connected = False
    telemetry_count = 0
    alerts_count = 0

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM telemetry")
            telemetry_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM alerts")
            alerts_count = cursor.fetchone()[0]
            db_connected = True
    except Exception as e:
        logger.error(f"DB health check error: {e}")

    overall_status = "ok" if db_connected else "degraded"

    return SystemHealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        database_connected=db_connected,
        telemetry_records_count=telemetry_count,
        alerts_count=alerts_count,
        models=engine.load_status,
        details=engine.load_status.get("error_detail"),
    )


@app.post("/api/telemetry", response_model=TelemetryIngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_telemetry(payload: RawTelemetryPayload) -> TelemetryIngestResponse:
    """Ingests raw sensor readings, runs rolling feature aggregation & inference, and persists."""
    try:
        processed = process_device_telemetry(payload.model_dump())
        feat = processed["features"]
        inf = processed["inference"]

        # Persist to database
        record_id = save_telemetry({
            "device_id": processed["device_id"],
            "timestamp": processed["timestamp"],
            "distance_cm": processed["distance_cm"],
            "water_level_cm": feat["water_level_cm"],
            "water_level_pct": feat["water_level_pct"],
            "rainfall_mm_h": feat["rainfall_mm_h"],
            "rainfall_3h_mm": feat["rainfall_3h_mm"],
            "rainfall_6h_mm": feat["rainfall_6h_mm"],
            "rainfall_24h_mm": feat["rainfall_24h_mm"],
            "water_level_change_cm_h": feat["water_level_change_cm_h"],
            "soil_saturation_pct": feat["soil_saturation_pct"],
            "flow_pct": processed["flow_pct"],
            "predicted_risk": inf["predicted_risk_3h_ahead"],
            "risk_code": inf["risk_code"],
            "confidence_score": inf["confidence_score"],
            "model_used": inf["model_used"],
            "is_fallback": 1 if inf["is_fallback"] else 0,
            "data_source": processed["data_source"],
        })

        return TelemetryIngestResponse(
            id=record_id,
            device_id=processed["device_id"],
            timestamp=processed["timestamp"],
            water_level_cm=feat["water_level_cm"],
            water_level_pct=feat["water_level_pct"],
            rainfall_mm_h=feat["rainfall_mm_h"],
            rainfall_3h_mm=feat["rainfall_3h_mm"],
            rainfall_6h_mm=feat["rainfall_6h_mm"],
            rainfall_24h_mm=feat["rainfall_24h_mm"],
            water_level_change_cm_h=feat["water_level_change_cm_h"],
            soil_saturation_pct=feat["soil_saturation_pct"],
            predicted_risk=inf["predicted_risk_3h_ahead"],
            risk_code=inf["risk_code"],
            confidence_score=inf["confidence_score"],
            model_used=inf["model_used"],
            is_fallback=inf["is_fallback"],
            alert_triggered=processed["alert_triggered"],
            alert_severity=processed["alert_severity"],
            status="success",
        )
    except Exception as e:
        logger.exception("Error ingesting telemetry")
        raise HTTPException(status_code=500, detail=f"Telemetry ingestion failed: {e}")


@app.get("/api/latest")
def get_latest(device_id: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves the most recent telemetry sample and active risk status."""
    latest = get_latest_telemetry(device_id)
    if not latest:
        return {
            "status": "no_telemetry_recorded",
            "message": "No sensor readings have been received yet. Start the Wokwi simulation or bridge feeder.",
            "data": None,
        }
    return {
        "status": "ok",
        "data": latest,
    }


@app.get("/api/history")
def get_history(
    device_id: Optional[str] = None, limit: int = Query(default=100, ge=1, le=1000)
) -> List[Dict[str, Any]]:
    """Retrieves historical readings for time-series charts."""
    return get_telemetry_history(device_id, limit)


@app.get("/api/alerts")
def get_alerts(limit: int = Query(default=50, ge=1, le=200)) -> List[Dict[str, Any]]:
    """Retrieves chronological alert transition events."""
    return get_recent_alerts(limit)


@app.post("/api/alerts/acknowledge")
def acknowledge(device_id: Optional[str] = None) -> Dict[str, Any]:
    """Silences and acknowledges active alerts."""
    count = acknowledge_alerts(device_id)
    return {
        "status": "acknowledged",
        "acknowledged_count": count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(features: SensorFeatures) -> PredictResponse:
    """Direct manual inference endpoint (maintains full backward compatibility)."""
    engine = get_inference_engine()
    res = engine.predict(features.model_dump(), prefer_model="dnn")

    return PredictResponse(
        predicted_risk_3h_ahead=res["predicted_risk_3h_ahead"],
        risk_code=res["risk_code"],
        prediction_horizon_hours=3,
        class_probabilities=res["class_probabilities"],
        confidence_score=res["confidence_score"],
        model_used=res["model_used"],
        is_fallback=res["is_fallback"],
        risk_rationale=res["risk_rationale"],
        data_note="Simulated 3-hour risk forecast. Decision-support prototype.",
    )


@app.get("/api/model/info")
def get_model_info() -> Dict[str, Any]:
    """Returns training benchmark metrics and comparative results."""
    comparison_file = ARTIFACTS_DIR / "model_comparison_report.json"
    if comparison_file.exists():
        with open(comparison_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "comparison_report_not_found"}
