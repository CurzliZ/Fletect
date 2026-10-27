# REST API SPECIFICATION: FLOOD SENTRY

Base URL: `http://localhost:8000`  
Interactive Swagger UI: `http://localhost:8000/docs`  
Interactive ReDoc: `http://localhost:8000/redoc`  

---

## 1. System Endpoints

### `GET /`
Returns service identification, operational status, and disclaimers.

**Response (200 OK):**
```json
{
  "service": "Flood Sentry IoT & DNN Decision-Support System",
  "status": "operational",
  "docs_url": "/docs",
  "prediction_horizon_hours": 3,
  "disclaimer": "Experimental research prototype. Calibrated to CWC rainfall distribution. Decision support only; not an official flood warning service."
}
```

### `GET /health`
Returns system component readiness, database connectivity, record counts, and model loading diagnostics.

**Response (200 OK):**
```json
{
  "status": "ok",
  "timestamp": "2026-10-23T19:45:00.000Z",
  "database_connected": true,
  "telemetry_records_count": 142,
  "alerts_count": 4,
  "models": {
    "scaler_loaded": true,
    "dnn_loaded": true,
    "xgb_loaded": true,
    "error_detail": null
  },
  "details": null
}
```

---

## 2. Telemetry & Ingestion Endpoints

### `POST /api/telemetry`
Primary ingestion endpoint for edge nodes (ESP32 Wokwi simulation, physical device, or virtual bridge). Performs range validation, rolling temporal feature extraction, model inference, alert hysteresis evaluation, and SQLite persistence.

**Request Body:**
```json
{
  "device_id": "ESP32_FLOOD_01",
  "distance_cm": 35.0,
  "rainfall_pct": 45.0,
  "rainfall_mm_h": 45.0,
  "flow_pct": 50.0,
  "channel_capacity_cm": 100.0,
  "data_source": "simulation"
}
```

**Response (201 Created):**
```json
{
  "id": 143,
  "device_id": "ESP32_FLOOD_01",
  "timestamp": "2026-10-23T19:45:02.124Z",
  "water_level_cm": 65.0,
  "water_level_pct": 65.0,
  "rainfall_mm_h": 45.0,
  "rainfall_3h_mm": 72.0,
  "rainfall_6h_mm": 98.0,
  "rainfall_24h_mm": 110.0,
  "water_level_change_cm_h": 4.2,
  "soil_saturation_pct": 74.5,
  "predicted_risk": "High",
  "risk_code": 2,
  "confidence_score": 0.9421,
  "model_used": "Deep Neural Network (DNN)",
  "is_fallback": false,
  "alert_triggered": true,
  "alert_severity": "High",
  "status": "success"
}
```

### `GET /api/latest`
Retrieves the most recent telemetry sample and active risk status.

**Query Parameters:**
* `device_id` (optional): Filter by device ID.

**Response (200 OK):**
```json
{
  "status": "ok",
  "data": {
    "id": 143,
    "device_id": "ESP32_FLOOD_01",
    "timestamp": "2026-10-23T19:45:02.124Z",
    "water_level_pct": 65.0,
    "predicted_risk": "High",
    "confidence_score": 0.9421,
    "model_used": "Deep Neural Network (DNN)"
  }
}
```

### `GET /api/history`
Returns chronological telemetry readings for time-series charts.

**Query Parameters:**
* `limit` (default: 100, max: 1000): Number of records.
* `device_id` (optional): Filter by device.

---

## 3. Alert Management Endpoints

### `GET /api/alerts`
Returns recent alert state transitions.

**Query Parameters:**
* `limit` (default: 50, max: 200)

### `POST /api/alerts/acknowledge`
Marks pending alerts as acknowledged/silenced.

**Query Parameters:**
* `device_id` (optional): Filter to acknowledge specific device.

---

## 4. Model & Benchmark Endpoints

### `POST /predict`
Direct inference endpoint for raw 11-feature vector (maintains backward compatibility).

**Request Body:**
```json
{
  "rainfall_mm_h": 12.0,
  "rainfall_3h_mm": 25.0,
  "rainfall_6h_mm": 35.0,
  "rainfall_24h_mm": 50.0,
  "water_level_cm": 42.0,
  "water_level_pct": 42.0,
  "water_level_change_cm_h": 0.5,
  "soil_saturation_pct": 55.0,
  "channel_capacity_cm": 100.0,
  "hour_of_day": 14,
  "month": 10
}
```

**Response (200 OK):**
```json
{
  "predicted_risk_3h_ahead": "Medium",
  "risk_code": 1,
  "prediction_horizon_hours": 3,
  "class_probabilities": {
    "Low": 0.152,
    "Medium": 0.781,
    "High": 0.067
  },
  "confidence_score": 0.781,
  "model_used": "Deep Neural Network (DNN)",
  "is_fallback": false,
  "risk_rationale": "Predicted Medium 3 hours ahead by Deep Neural Network...",
  "data_note": "Simulated 3-hour risk forecast. Decision-support prototype."
}
```

### `GET /api/model/info`
Returns full benchmark comparative metrics across Rule Baseline, XGBoost, DNN, and LSTM.
