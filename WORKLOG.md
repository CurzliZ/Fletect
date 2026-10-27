# WORKLOG: FLOOD SENTRY
**IoT-Based Flood Risk Detection Using Deep Neural Networks**

## 2026-10-23

### Session 1: System Audit & Discovery
- **Action:** Inspected full workspace directory tree, files, and metadata.
- **Action:** Audited existing Wokwi circuit (`diagram.json`) and firmware (`flood-sentry/sketch.ino`).
- **Discovered:** Verified pin configuration in `diagram.json`: HC-SR04 on TRIG=5, ECHO=18; Potentiometers on GPIO34 (Rain) and GPIO35 (Flow); LEDs on GPIO25 (Green), GPIO26 (Yellow), GPIO27 (Red); Buzzer on GPIO14; Button on GPIO13; LCD1602 on I2C SDA=21, SCL=22.
- **Discovered:** Reconciled prompt's hypothetical pinout (Green=18, Red=21) with actual circuit: in `diagram.json`, GPIO 18 is used by ECHO and GPIO 21 is used by LCD SDA. Preserved actual working circuit pins while adding clear macros.
- **Discovered:** Audited dataset `flood_risk_dnn_synthetic_120k.csv` (120,000 rows across 20 sites, chronological split 70/15/15, target `risk_label_encoded_3h_ahead`).
- **Discovered:** Audited existing backend `backend/main.py` and model artifacts in `artifacts/`. Found lack of DB persistence, missing telemetry ingestion endpoint, missing temporal rolling window feature calculation, missing comparative models, and missing dashboard.
- **Discovered:** Found Python 3.12 environment in `.venv312` with TensorFlow 2.21 pre-installed.
- **Action:** Initiated installation of missing packages (`pytest`, `httpx`, `matplotlib`, `xgboost`, `streamlit`) into `.venv312`.
- **Created:** `PROJECT_AUDIT.md` providing comprehensive system audit, pinout reconciliation, and execution plan.

---

### Session 2: Multi-Model Machine Learning Pipeline & Benchmarking
- **Created:** `model/dataset_loader.py` with strict chronological partitioning (70% train, 15% validation, 15% test) and zero-leakage `StandardScaler` fitted strictly on train data.
- **Created:** `model/train_and_evaluate.py` implementing 4 distinct modeling paradigms:
  1. Hydrological Rule-Based Baseline
  2. Tabular XGBoost Classifier Baseline
  3. Deep Neural Network (DNN) with BatchNormalization, Dropout, and L2 regularization
  4. Sequential LSTM Network with 6-hour sliding temporal tensors
- **Executed Command:**
  ```powershell
  & ".\.venv312\Scripts\python.exe" -m model.train_and_evaluate
  ```
- **Benchmark Outcomes ($N=17,940$ test samples):**
  - **Rule Baseline:** Accuracy 87.99%, Macro-F1 0.817, High-Risk Recall 83.12%, FN Rate 16.88%, Latency 0.0005 ms
  - **XGBoost:** Accuracy 92.70%, Macro-F1 0.902, High-Risk Recall 93.72%, FN Rate 6.28%, Latency 0.0009 ms
  - **DNN:** Accuracy 92.36%, Macro-F1 0.895, High-Risk Recall 92.71%, FN Rate 7.29%, Latency 0.0315 ms
  - **LSTM:** Accuracy 92.67%, Macro-F1 0.900, High-Risk Recall 93.18%, FN Rate 6.82%, Latency 0.0576 ms
- **Artifacts Saved:**
  - `artifacts/flood_risk_dnn.keras`
  - `artifacts/xgboost_flood_model.json`
  - `artifacts/flood_risk_lstm.keras`
  - `artifacts/flood_dnn_scaler.joblib`
  - `artifacts/model_comparison_report.json`
  - `artifacts/evaluation_report.json`
- **Created:** `model/inference.py` production inference engine with dual-model support, calibrated probabilities, explainable rationales, and automatic fallback to rule-based evaluation.

---

### Session 3: Backend API, Feature Accumulator & Alert Engine
- **Created:** `backend/database.py` SQLite persistence layer for `telemetry` and `alerts` tables with compound indexing on `(device_id, timestamp)`.
- **Created:** `backend/schemas.py` Pydantic models for raw telemetry ingestion, 11-feature manual inference, system health, and alert events.
- **Created:** `backend/risk_engine.py` providing:
  - Temporal feature aggregation from SQLite history: rolling 3h, 6h, 24h rainfall sums, hourly water level delta, and recursive soil saturation proxy calculation.
  - Alert state machine with anti-flapping hysteresis: immediate escalation for High danger, 2-sample debounce confirmation for Medium watch condition.
- **Updated:** `backend/main.py` complete production FastAPI application with endpoints:
  - `GET /`, `GET /health`, `POST /api/telemetry`, `GET /api/latest`, `GET /api/history`, `GET /api/alerts`, `POST /api/alerts/acknowledge`, `POST /predict`, `GET /api/model/info`.

---

### Session 4: Edge Firmware & Simulation Feeder
- **Updated:** `flood-sentry/sketch.ino` and `simulation/sketch.ino`:
  - Added Wi-Fi client (`Wokwi-GUEST`) and HTTPClient POST to `BACKEND_URL`.
  - Added structured JSON Serial logging (`TELEMETRY:{...}`).
  - Calibrated physical datum conversion: $\text{Level} = 100.0 - \text{Distance (cm)}$.
  - Added sensor bounds validation (2 to 100 cm) and timeout handling.
  - Implemented LED indicators (Green=Normal, Yellow=Watch, Red=Danger) and piezo buzzer with push-button silence/ACK.
- **Created:** `simulation/telemetry_bridge.py`:
  - Mode 1: Continuous realistic multi-phase storm stream generator.
  - Mode 2: Deterministic scenario injector (`low`, `medium`, `high`, `fault`, `flapping_test`).
  - Mode 3: Serial COM port bridge for live hardware forwarding.

---

### Session 5: Operational Monitoring Dashboard
- **Created:** `dashboard/app.py` Streamlit operational monitoring dashboard:
  - Dark modern UI aesthetic with custom CSS metric cards and risk badges.
  - Live KPI cards: Water Level (%, cm), Rainfall Rate, 6h Accumulation, Soil Saturation, Flow Rate.
  - Dynamic time-series charts (water level vs capacity, multi-window rainfall, risk code history).
  - Alert state transition log with Acknowledge/Silence action.
  - Telemetry freshness and device connectivity indicator.
  - Model suite benchmark cards.
  - Integrated sidebar Demo & Test Scenario control panel.

---

### Session 6: Automated Testing & Verification
- **Created Test Suite:**
  - `tests/test_data_and_features.py`: Dataset schema, split non-leakage, temporal feature engineering.
  - `tests/test_models.py`: Model loading, inference outputs, probability normalization, fallback heuristics.
  - `tests/test_api.py`: FastAPI endpoints (`/health`, `/api/telemetry`, `/predict`, `/api/latest`, `/api/history`, `/api/alerts`).
  - `tests/test_e2e_pipeline.py`: Deterministic end-to-end signal pathway verification for Low, Medium (hysteresis), High, and Sensor Fault scenarios.
- **Executed Command:**
  ```powershell
  & ".\.venv312\Scripts\python.exe" -m pytest tests/ -v
  ```
- **Test Results:** 18 passed in 12.33 seconds (100% pass rate).

---

### Session 7: Research-Grade Documentation & Packaging
- **Created:**
  - `requirements.txt`: Unified dependency specification.
  - `.env.example`: Configuration template.
  - `.gitignore`: Clean ignore rules for Python, SQLite, and IDE caches.
  - `docs/ARCHITECTURE.md`: Complete system architecture and hydrological equations.
  - `docs/DATASET.md`: Data provenance, empirical reference pool, and schema.
  - `docs/MODEL_EVALUATION.md`: Multi-model comparative benchmark report.
  - `docs/WOKWI_SETUP.md`: Wiring schematic, pin reconciliation, and networking guide.
  - `docs/API.md`: Full REST API reference with JSON payloads.
  - `docs/DEMO_GUIDE.md`: Step-by-step evaluator demonstration walkthrough.
  - `README.md`: Master project guide.
