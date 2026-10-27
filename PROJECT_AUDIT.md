# PROJECT AUDIT: FLOOD SENTRY
**IoT-Based Flood Risk Detection Using Deep Neural Networks**

*Audit Timestamp:* October 23, 2026  
*Auditor:* Principal Systems & ML Engineer  
*Environment:* Windows 11, Python 3.12 (`.venv312`), Wokwi Simulation Platform  

---

## 1. Executive Summary & Existing Architecture

The **Flood Sentry** project is conceived as an end-to-end IoT flood monitoring and early warning decision-support system. It marries an edge sensing node (simulated via an ESP32 in Wokwi) with a local ingestion backend (FastAPI), deep learning/tabular predictive models for 3-hour-ahead risk forecasting, SQLite persistence, and an interactive operational dashboard.

### Existing Architecture Found in Workspace:
```
[ Wokwi ESP32 Circuit ]
   - HC-SR04 (Ultrasonic level)
   - Rain Potentiometer (ADC GPIO34)
   - Flow Potentiometer (ADC GPIO35)
   - LCD 1602 (I2C 21/22)
   - RGB LEDs (25/26/27) & Buzzer (14)
           │
           ▼ (Serial text logs only; NO HTTP telemetry implemented)
      [ GAP / UNCONNECTED ]
           │
           ▼
[ FastAPI Backend (backend/main.py) ]
   - /health, /predict (Accepts 11 pre-computed static features)
   - NO DB persistence
   - NO device telemetry ingestion endpoint
   - NO rolling window accumulator for temporal features (3h, 6h, 24h rainfall)
   - NO fallback risk evaluation when model fails
           │
           ▼
[ Model Artifacts (artifacts/) ]
   - flood_risk_dnn.keras (Simple 3-layer Dense MLP)
   - flood_dnn_scaler.joblib
   - evaluation_report.json (Trained on synthetic 120k dataset)
   - NO baseline comparisons (Rule-based, XGBoost, LSTM)
           │
           ▼
[ Dashboard ]
   - Completely missing from workspace
```

---

## 2. Actual Workspace File Inventory

| Path | Size (Bytes) | Category | Current State & Assessment |
|---|---|---|---|
| `diagram.json` | 2,786 | Hardware Simulation | Wokwi ESP32-devkit-v1 circuit wiring. Defines HC-SR04, LCD1602, Rain & Flow pots, 3 LEDs, Buzzer, Pushbutton. Valid JSON. |
| `wokwi-project.txt` | 109 | Hardware Simulation | Contains origin URL (`https://wokwi.com/projects/478416784820391937`). |
| `flood-sentry/sketch.ino` | 4,534 | Firmware (Arduino C++) | Runs sensor read loop, local threshold logic, LCD printing, buzzer control, and Serial output. **Lacks WiFi, HTTP client, and JSON telemetry transmission.** |
| `flood-sentry/libraries.txt` | 128 | Firmware Config | Specifies `LiquidCrystal I2C`. |
| `flood-sentry/requirements.txt` | 71 | Python Dependencies | Minimal dependencies (`pandas`, `numpy`, `scikit-learn`, `tensorflow`, `joblib`). |
| `flood-sentry/train_flood_dnn.py` | 5,053 | ML Pipeline | Trains a 3-layer Dense Keras model on `flood_risk_dnn_synthetic_120k.csv`. Only trains DNN; lacks XGBoost, rule baseline, or LSTM comparisons. |
| `backend/main.py` | 4,003 | Backend API | Minimal FastAPI app with `/predict` and `/health`. Lacks database persistence, telemetry ingestion, device tracking, and temporal window feature computation. |
| `artifacts/flood_risk_dnn.keras` | 76,039 | ML Artifact | Saved Keras model (3 hidden layers: 64, 32, 16 neurons). |
| `artifacts/flood_dnn_scaler.joblib` | 1,279 | ML Artifact | Fitted `StandardScaler` for the 11 feature inputs. |
| `artifacts/evaluation_report.json` | 1,458 | ML Artifact | Evaluation metrics on synthetic test split (Accuracy: ~92.4%). |
| `flood_risk_dnn_synthetic_120k.csv` | 21,388,036 | Dataset | 120,000 synthetic rows across 20 simulated sites (`SIM_01` to `SIM_20`). |
| `generate_flood_dnn_dataset.py` | 12,803 | Data Generation | Python standard library script generating reproducible synthetic sequences calibrated to CWC rainfall distribution. |
| `README_flood_risk_dataset.md` | 6,864 | Documentation | Clear documentation of dataset structure, synthetic nature, and split scheme. |
| `.venv312/` | Directory | Python Environment | Active Python 3.12.10 virtual environment with TensorFlow 2.21, Keras 3.15, Scikit-learn, FastAPI. |

---

## 3. Hardware Pin Mapping & Circuit Analysis

### Verified Pin Assignments:
An essential finding from inspecting `diagram.json` and `sketch.ino`:
- **HC-SR04 Ultrasonic:**
  - `TRIG` -> GPIO 5
  - `ECHO` -> GPIO 18
- **Simulated Rain Potentiometer:**
  - `SIG` -> GPIO 34 (ADC1_CH6)
- **Simulated Flow / Runoff Potentiometer:**
  - `SIG` -> GPIO 35 (ADC1_CH7)
- **I2C LCD 1602:**
  - `SDA` -> GPIO 21
  - `SCL` -> GPIO 22
- **Status LEDs:**
  - Green LED (`ledG`) -> GPIO 25 (via 220Ω resistor `rg`)
  - Yellow LED (`ledY`) -> GPIO 26 (via 220Ω resistor `ry`)
  - Red LED (`ledR`) -> GPIO 27 (via 220Ω resistor `rr`)
- **Acoustic & Input:**
  - Buzzer (`buzz`) -> GPIO 14
  - Alarm ACK Pushbutton (`ack`) -> GPIO 13 (with internal pullup)

### Pin Mapping Discrepancy Note:
The Master Directive prompt mentions an alternate hypothetical mapping:
*(TRIG GPIO4, ECHO GPIO5, Green GPIO18, Yellow GPIO19, Red GPIO21, Rain GPIO34)*.
**Audit Finding:** In the actual working Wokwi circuit (`diagram.json`), assigning Green LED to GPIO18 would create an electrical short with HC-SR04 `ECHO` (GPIO18), and assigning Red LED to GPIO21 would short out the I2C LCD `SDA` line (GPIO21).
**Decision:** We preserve the actual, verified `diagram.json` pinout (5, 18, 34, 35, 25, 26, 27, 14, 13, 21, 22), but provide clear configuration macros in the firmware code so that any pin alterations can be made without rewriting logic.

---

## 4. What Already Works

1. **Wokwi Circuit Wiring:** Complete wiring diagram in `diagram.json` with ESP32, sonar, 2 potentiometers, 3 LEDs, buzzer, button, and LCD.
2. **Firmware Local Loop:** `sketch.ino` reads ultrasonic distance, calculates water level percentage based on `SENSOR_TO_DRAIN_BOTTOM_CM = 100.0`, reads analog inputs, updates LCD display, and executes local warning logic.
3. **Synthetic Dataset:** `flood_risk_dnn_synthetic_120k.csv` is fully generated, non-corrupt, 120,000 rows, chronologically split (70% train, 15% validation, 15% test) across 20 distinct sites with zero temporal cross-split leakage.
4. **DNN Baseline Model:** Keras model and scaler exist in `artifacts/`, achieving ~92.4% test accuracy on synthetic labels.
5. **Python 3.12 Environment:** Fully operational virtual environment in `.venv312` with TensorFlow 2.21 and FastAPI.

---

## 5. What Is Broken, Missing, or Incomplete

1. **Wokwi Network & Telemetry Delivery:**
   - `sketch.ino` has no networking code. It prints text logs to Serial but never sends telemetry to the backend API.
   - Wokwi simulation runs in a sandboxed browser environment (`Wokwi-GUEST` WiFi) and cannot directly connect to `localhost:8000` on the host machine without special network tunneling or a serial bridge.
2. **Backend Telemetry Ingestion & Transformation:**
   - `backend/main.py` only accepts raw ML feature vectors (`rainfall_3h_mm`, `rainfall_24h_mm`, `water_level_change_cm_h`). A physical or simulated sensor only produces instantaneous readings (`water_level_cm`, `rainfall_mm_h`).
   - The backend lacks an ingestion endpoint (`POST /api/telemetry`), rolling time-window feature computation, and sensor range validation.
3. **Data Persistence:**
   - No database exists. All telemetry and inference results vanish immediately.
4. **Model Suite & Evaluation Completeness:**
   - No benchmark comparison exists: Rule-based baseline, Tabular ML (XGBoost/RandomForest), and sequential modeling (LSTM) are not yet implemented or benchmarked on the identical test split.
5. **Monitoring Dashboard:**
   - Completely missing. No user interface exists to monitor live telemetry, view risk levels, or observe historical trends.
6. **Alert Engine & Hysteresis:**
   - No hysteresis mechanism to prevent alert flapping between Medium and High risk states.
7. **Automated Testing & End-to-End Verification:**
   - No test suite exists for unit tests, API tests, database tests, or end-to-end integration.

---

## 6. Dataset Quality Assessment

- **File:** `flood_risk_dnn_synthetic_120k.csv` (21.4 MB)
- **Rows:** 120,000
- **Sites:** 20 independent simulated catchments (`SIM_01` to `SIM_20`), 6,000 hourly steps each.
- **Split Distribution:**
  - Train: 84,000 rows (70%)
  - Validation: 18,000 rows (15%)
  - Test: 18,000 rows (15%)
- **Target Distribution (3h ahead target, excluding boundary rows):**
  - Low (0): ~69.5% (83,184 rows)
  - Medium (1): ~18.4% (22,012 rows)
  - High (2): ~12.1% (14,624 rows)
- **Data Integrity:**
  - No NaN values in feature columns.
  - Boundary rows (final 3 hours of each split) have empty target labels to strictly prevent future-target leakage across train/validation/test partitions.
  - Provenance: Explicitly synthetic. The empirical rainfall distribution was sampled from CWC Tamil Nadu telemetry, but all timestamps, water levels, and risk labels are simulated.

---

## 7. Required Repairs & Implementation Roadmap

### Phase 2: Architecture & Clean Workspace Organization
- Establish clean directory layout: `simulation/`, `backend/`, `model/`, `dashboard/`, `tests/`, `docs/`.
- Preserve existing root files and backward compatibility.

### Phase 3 & 4: Comprehensive ML Pipeline & Model Comparison
- Implement feature extraction pipeline with reproducible scaling and class weighting.
- Implement and benchmark 4 models on the exact same test partition:
  1. Documented Rule-based Baseline
  2. Tabular ML Baseline (XGBoost Classifier)
  3. Deep Neural Network (DNN) Classifier (TensorFlow/Keras)
  4. Sequence-aware Deep Learning Model (LSTM)
- Generate full comparative metrics: Confusion matrices, Precision, Recall, Macro-F1, High-Risk False Negative Analysis, and latency benchmarks.
- Save production artifacts (`model/artifacts/`) with unified inference interface and fallback engine.

### Phase 5: Wokwi Firmware Enhancement & Telemetry Ingestion
- Update `sketch.ino`:
  - Add Wi-Fi connectivity (`Wokwi-GUEST`) and HTTP POST telemetry client.
  - Implement structured JSON Serial output (providing an alternate high-reliability ingestion bridge).
  - Add distance-to-water-level calibration, validation, and error codes.
  - Implement local LED states (Green=Low, Yellow=Watch, Red=Danger) matching telemetry.
- Provide a robust local telemetry bridge (`simulation/telemetry_bridge.py`) allowing instant streaming from Wokwi Serial or deterministic simulation into the FastAPI backend.

### Phase 6: FastAPI Backend & SQLite Engine
- Build robust FastAPI server with:
  - `POST /api/telemetry`: Ingests instantaneous device readings (`water_level_cm`, `rainfall_mm_h`, `flow_rate`, `device_id`).
  - Automatic temporal feature engineering: Computes rolling 3h, 6h, 24h rainfall and water-level change rate from SQLite history.
  - Real-time inference using loaded ML model with automatic rule-based fallback if ML is unavailable.
  - Alert engine with configurable hysteresis to suppress flapping.
  - Endpoints: `GET /health`, `GET /api/latest`, `GET /api/history`, `GET /api/events`, `GET /api/model/info`.
  - SQLite persistence (`data/flood_sentry.db`).

### Phase 7: Professional Streamlit Dashboard
- Multi-metric real-time operational dashboard:
  - Current risk banner (Low/Medium/High) with clear hydrological reasoning.
  - Gauge / Metric cards: Current water level (% and cm), rainfall intensity, soil saturation proxy.
  - Dynamic time-series plots: Water level vs channel capacity, multi-hour rainfall accumulation.
  - Classification timeline & historical alert log.
  - Telemetry freshness indicator & connection status.
  - Interactive "Demo / Stress Test" mode for evaluating edge cases and risk transitions on demand.

### Phase 8 & 9: Test Suite & End-to-End Verification
- Complete automated test suite using `pytest`:
  - Unit tests for feature scaling, temporal accumulators, and rule engines.
  - API endpoint integration tests.
  - Database persistence and historical query tests.
  - End-to-end telemetry-to-alert verification script (`tests/test_e2e_pipeline.py`).

### Phase 10: Complete Research-Ready Documentation
- `README.md`, `WORKLOG.md`, `docs/ARCHITECTURE.md`, `docs/DATASET.md`, `docs/MODEL_EVALUATION.md`, `docs/WOKWI_SETUP.md`, `docs/API.md`, `docs/DEMO_GUIDE.md`, `.env.example`.

---

## 8. Audit Resolution & Final Verification Status

All identified architectural gaps, missing components, and bugs have been resolved and verified:

1. **Wokwi Firmware:** Updated with Wi-Fi client, HTTP telemetry dispatch, JSON serial logging, LCD/LED states, and physical datum conversion ($100.0 - \text{distance}$).
2. **Multi-Model Machine Learning:** All 4 models (Rule baseline, XGBoost, DNN, and LSTM) trained and benchmarked on identical chronological test split with zero temporal leakage.
3. **Inference & Fallback:** Production inference engine in `model/inference.py` loaded with dual-model support and automatic deterministic rule-based failover.
4. **Backend Engine:** Complete FastAPI backend with rolling temporal aggregation (3h, 6h, 24h rain, rise rate), alert hysteresis, SQLite persistence, and CORS support.
5. **Operational Dashboard:** Full-featured Streamlit UI (`dashboard/app.py`) with real-time status banner, KPI cards, multi-parameter charts, alert history, and built-in scenario testing panel.
6. **Testing & QA:** 18 automated unit and integration tests executing via `pytest`, achieving a 100% pass rate.
7. **Documentation:** Research-ready documentation suite completed in `docs/` and root `README.md`.

