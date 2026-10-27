"""Pydantic request and response schemas for Flood Sentry FastAPI service."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RawTelemetryPayload(BaseModel):
    """Payload received from Wokwi ESP32 or simulation bridge."""
    device_id: str = Field(default="ESP32_FLOOD_01", description="Unique device identifier")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp or auto-generated UTC")
    distance_cm: Optional[float] = Field(default=None, ge=-1.0, le=400.0, description="Ultrasonic distance in cm (-1 indicates timeout/invalid)")
    water_level_cm: Optional[float] = Field(default=None, ge=0.0, description="Water level height in cm")
    water_level_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Water depth percentage of capacity")
    rainfall_raw: Optional[int] = Field(default=None, ge=0, le=4095, description="Raw ADC 12-bit potentiometer value")
    rainfall_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Potentiometer rainfall % (0-100)")
    rainfall_mm_h: Optional[float] = Field(default=None, ge=0.0, le=150.0, description="Converted rainfall rate in mm/h")
    flow_raw: Optional[int] = Field(default=None, ge=0, le=4095, description="Flow sensor raw ADC value")
    flow_pct: Optional[float] = Field(default=50.0, ge=0.0, le=100.0, description="Channel drainage / flow rate percentage")
    soil_saturation_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Simulated soil saturation")
    channel_capacity_cm: float = Field(default=100.0, gt=0.0, description="Total channel/drain depth in cm")
    data_source: str = Field(default="simulation", description="'simulation', 'test_injection', or 'historical'")


class SensorFeatures(BaseModel):
    """Direct 11-feature vector for manual inference (backward-compatible)."""
    rainfall_mm_h: float = Field(ge=0, description="Current rainfall intensity in mm/h")
    rainfall_3h_mm: float = Field(ge=0, description="3-hour accumulated rainfall")
    rainfall_6h_mm: float = Field(ge=0, description="6-hour accumulated rainfall")
    rainfall_24h_mm: float = Field(ge=0, description="24-hour accumulated rainfall")
    water_level_cm: float = Field(ge=0, description="Water height in cm")
    water_level_pct: float = Field(ge=0, le=100, description="Water level % of capacity")
    water_level_change_cm_h: float = Field(description="Rate of water level change in cm/h")
    soil_saturation_pct: float = Field(ge=0, le=100, description="Soil saturation percentage")
    channel_capacity_cm: float = Field(gt=0, description="Channel capacity in cm")
    hour_of_day: int = Field(ge=0, le=23)
    month: int = Field(ge=1, le=12)


class PredictResponse(BaseModel):
    predicted_risk_3h_ahead: str
    risk_code: int
    prediction_horizon_hours: int = 3
    class_probabilities: Dict[str, float]
    confidence_score: float
    model_used: str
    is_fallback: bool
    risk_rationale: str
    data_note: str


class TelemetryIngestResponse(BaseModel):
    id: int
    device_id: str
    timestamp: str
    water_level_cm: float
    water_level_pct: float
    rainfall_mm_h: float
    rainfall_3h_mm: float
    rainfall_6h_mm: float
    rainfall_24h_mm: float
    water_level_change_cm_h: float
    soil_saturation_pct: float
    predicted_risk: str
    risk_code: int
    confidence_score: float
    model_used: str
    is_fallback: bool
    alert_triggered: bool
    alert_severity: Optional[str] = None
    status: str = "success"


class AlertEvent(BaseModel):
    id: int
    device_id: str
    timestamp: str
    severity: str
    trigger_reason: str
    water_level_pct: float
    rainfall_6h_mm: float
    acknowledged: bool
    acknowledged_at: Optional[str] = None


class SystemHealthResponse(BaseModel):
    status: str
    timestamp: str
    database_connected: bool
    telemetry_records_count: int
    alerts_count: int
    models: Dict[str, Any]
    details: Optional[str] = None
