# 🌊 Flood Sentry: IoT-Based Flood Risk Detection Using Deep Neural Networks

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Tests-18%20Passed-brightgreen.svg)](https://pytest.org/)

**Flood Sentry** is an end-to-end edge-to-decision IoT early warning decision-support system. It integrates an ultrasonic and precipitation sensing node (simulated via an ESP32 in Wokwi) with a local ingestion API (FastAPI), dynamic rolling temporal feature engineering, multi-model predictive inference (Deep Neural Network, XGBoost, and sequential LSTM), persistent SQLite telemetry storage, an alert engine with anti-flapping hysteresis, and an operational monitoring dashboard (Streamlit).

---

## 🏛️ System Architecture

```
+──────────────────────────+     HTTP / Serial     +──────────────────────────+
|  Wokwi ESP32 Edge Node   | ────────────────────> |   FastAPI Local API      |
|  - HC-SR04 Ultrasonic    |     JSON Telemetry    |  - Temporal Aggregator   |
|  - Rain & Flow Pots      |                       |  - Model Inference       |
|  - 16x2 LCD + RGB LEDs   |                       |  - Alert Hysteresis      |
|  - Buzzer + ACK Button   |                       |  - SQLite Persistence    |
+──────────────────────────+                       +────────────┬─────────────+
                                                                │
                                                                ▼
                                                   +──────────────────────────+
                                                   |    Streamlit Dashboard   |
                                                   |  - Real-time Risk Banner |
                                                   |  - KPI Cards & Charts    |
                                                   |  - Alert Log & Silence   |
                                                   |  - Scenario Test Panel   |
                                                   +──────────────────────────+
```

---

## 🚀 Key Features

* **Calibrated Physical Sensing:** Converts ultrasonic echo distance to true water level depth based on an overhead mount datum ($100.0\text{ cm}$ channel capacity).
* **Dynamic Temporal Aggregation:** Ingests instantaneous raw sensor readings and dynamically computes rolling 3h, 6h, and 24h rainfall sums, hourly water-level rise rates ($\text{cm/h}$), and soil saturation proxies from SQLite historical data.
* **Multi-Paradigm ML Benchmarking:** Evaluates four distinct models on the identical chronological test split ($N=17,940$ samples) for 3-hour-ahead flood forecasting:
  * **Deep Neural Network (DNN):** 92.36% Accuracy, 0.895 Macro-F1, 92.71% High-Risk Recall.
  * **XGBoost Classifier:** 92.70% Accuracy, 0.902 Macro-F1, 93.72% High-Risk Recall.
  * **Sequential LSTM:** 92.67% Accuracy, 0.900 Macro-F1, 93.18% High-Risk Recall.
  * **Hydrological Rule Baseline:** 87.99% Accuracy, 0.817 Macro-F1, 83.12% High-Risk Recall.
* **Resilient Failover:** Automatic fallback to deterministic hydrological heuristics if model weights or dependencies are unavailable.
* **Alert Engine with Anti-Flapping Hysteresis:** Escalates to critical alarms immediately on High risk, but requires sustained confirmation for Medium watch conditions, preventing false flapping on boundary fluctuations.
* **Full Local Execution:** Runs 100% locally on Windows 11 without requiring paid cloud APIs or external accounts.

---

## 📁 Repository Structure

```
├── artifacts/                     # Serialized production models & evaluation reports
│   ├── flood_risk_dnn.keras       # Trained Keras DNN model
│   ├── xgboost_flood_model.json   # Trained XGBoost model
│   ├── flood_risk_lstm.keras      # Trained Sequential LSTM model
│   ├── flood_dnn_scaler.joblib    # Fitted StandardScaler
│   ├── model_comparison_report.json # Master benchmark metrics
│   └── evaluation_report.json    # Backward-compatible evaluation file
├── backend/                       # Production FastAPI application
│   ├── main.py                    # API entry point & routes
│   ├── database.py                # SQLite persistence layer (telemetry & alerts)
│   ├── risk_engine.py             # Rolling feature aggregator & hysteresis
│   └── schemas.py                 # Pydantic request/response schemas
├── dashboard/                     # Operational monitoring UI
│   └── app.py                     # Streamlit dashboard application
├── docs/                          # In-depth technical documentation
│   ├── ARCHITECTURE.md            # System architecture & equations
│   ├── DATASET.md                 # Dataset provenance & schema
│   ├── MODEL_EVALUATION.md        # Comprehensive benchmark report
│   ├── WOKWI_SETUP.md             # Wokwi wiring & networking guide
│   ├── API.md                     # REST API reference
│   └── DEMO_GUIDE.md              # Live demonstration walkthrough
├── flood-sentry/                  # Wokwi ESP32 edge firmware
│   ├── sketch.ino                 # Microcontroller code (WiFi, LCD, LEDs, Sonar)
│   ├── libraries.txt              # Wokwi library specifications
│   └── requirements.txt           # Minimal training dependencies
├── simulation/                    # Local simulator package
│   ├── sketch.ino                 # Firmware copy
│   ├── diagram.json               # Wokwi circuit schematic
│   └── telemetry_bridge.py        # Stream feeder & scenario injector
├── model/                         # ML training & inference pipeline
│   ├── dataset_loader.py          # Anti-leakage chronological partitioner
│   ├── train_and_evaluate.py      # Master benchmark training pipeline
│   └── inference.py               # Production inference engine with fallback
├── tests/                         # Automated test suite
│   ├── test_data_and_features.py  # Dataset & rolling feature unit tests
│   ├── test_models.py             # Inference engine & fallback unit tests
│   ├── test_api.py                # FastAPI endpoint integration tests
│   └── test_e2e_pipeline.py       # Deterministic end-to-end pipeline test
├── flood_risk_dnn_synthetic_120k.csv # 120k row calibrated dataset
├── requirements.txt               # Unified project dependencies
├── .env.example                   # Configuration template
├── PROJECT_AUDIT.md               # Complete audit report
└── WORKLOG.md                     # Engineering task and test log
```

---

## ⚡ Quick Start Instructions (Windows 11)

### 1. Activate Environment & Install Dependencies
```powershell
# Using the pre-configured Python 3.12 virtual environment:
& ".\.venv312\Scripts\python.exe" -m pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```powershell
& ".\.venv312\Scripts\python.exe" -m pytest tests/ -v
```
*Expected: 18 passed in ~12 seconds.*

### 3. Launch Backend API Server
```powershell
& ".\.venv312\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
* OpenAPI Swagger Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
* Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### 4. Launch Operational Monitoring Dashboard
In a second terminal:
```powershell
& ".\.venv312\Scripts\python.exe" -m streamlit run dashboard\app.py
```
* Dashboard URL: [http://localhost:8501](http://localhost:8501)

### 5. Stream Live Telemetry / Test Scenarios
In a third terminal:
```powershell
# Continuous realistic storm progression simulation:
& ".\.venv312\Scripts\python.exe" simulation\telemetry_bridge.py --mode stream --interval 2.0

# Or inject a specific deterministic condition:
& ".\.venv312\Scripts\python.exe" simulation\telemetry_bridge.py --mode scenario --scenario high
```

---

## 🔬 Scientific & Dataset Provenance Notice

* **Empirical Rainfall Calibration:** The rainfall magnitude distribution was sampled from empirical telemetry collected by the Central Water Commission (CWC), Tamil Nadu region (2021–2025).
* **Synthetic Sequence Generation:** Sensor sequences, timestamps, channel water depths, and risk labels were generated using `generate_flood_dnn_dataset.py`.
* **Decision-Support Scope:** This software is an engineering and research decision-support prototype. It does not replace official meteorological agency bulletins, certified hydrological engineering models, or local civil protection directives.
