# Flood Sentry
## IoT Flood-Risk Monitoring and Decision-Support Prototype

**Project report — October 2026**

**Prepared by:** ____________________  
**Institution / course:** ____________________  
**Supervisor:** ____________________

> **Scope and safety note:** Flood Sentry is an educational and research prototype. It uses simulated sensors and a synthetic-trained prediction model. Its outputs are not a certified flood forecast, public warning, or substitute for official hydrological and emergency-management guidance.

## 1. Executive summary

Flood Sentry is an end-to-end prototype for demonstrating how an embedded sensor node can measure water level and simulated rainfall/runoff, classify local conditions, transmit readings to a backend, run a machine-learning (ML) risk estimate, retain telemetry, and display the result on a dashboard.

The project combines an ESP32 circuit in Wokwi, ultrasonic distance sensing, two potentiometer-based sensor inputs, a 16×2 I²C LCD, status LEDs, a buzzer, an acknowledge button, firmware written in Arduino C++, a Python/FastAPI API, SQLite storage, TensorFlow/Keras and XGBoost model artifacts, and a Streamlit web dashboard.

The normal demonstration flow is:

1. The ESP32 reads a distance and two analog inputs.
2. Firmware converts distance into estimated channel water depth and applies local alert thresholds.
3. Sensor readings are emitted over serial and, when configured, sent by HTTP.
4. The backend validates a telemetry request, computes features, predicts a three-hour risk class, evaluates alert hysteresis, and stores the result.
5. The dashboard polls the backend and shows the latest risk, measurements, history, and alerts.

During the October 26, 2026 demonstration, the Wokwi circuit reported **34.3 cm distance**, **65.7% estimated water level**, **24% rain-pot input**, and **78% flow-pot input**. Its firmware displayed **WATCH**. A reading was also ingested by the local API as `wokwi_live`; the trained DNN returned **High** for the dashboard forecast. That difference is expected to be discussed carefully: the device’s immediate threshold-based status and the backend’s model forecast are separate outputs.

## 2. Problem statement and objectives

Flooding can develop as water levels rise in drains, channels, or small waterways, particularly during heavy rain and poor drainage. This prototype explores a low-cost monitoring pipeline that:

- measures the distance from an overhead sensor to a water surface;
- derives a water-level estimate relative to a configured channel depth;
- represents rainfall and flow inputs using adjustable potentiometers;
- gives local visual and audible feedback;
- transfers readings to a local software backend;
- classifies conditions as Low, Medium, or High using an ML model;
- records telemetry and alert events for dashboard display; and
- supports reproducible low-, medium-, high-, and sensor-fault test cases.

The project demonstrates system integration and software behavior. It does **not** establish the safety, reliability, or field accuracy required for real-world public warning.

## 3. System architecture

```text
Wokwi ESP32 circuit
  ├─ HC-SR04 ultrasonic distance sensor
  ├─ Rain-input potentiometer (simulated sensor)
  ├─ Flow-input potentiometer (simulated sensor)
  ├─ LCD, LEDs, buzzer, acknowledge button
  └─ Local threshold logic + serial / optional HTTP telemetry
              │ JSON over HTTP or a bridge
              ▼
FastAPI backend
  ├─ Request validation (Pydantic)
  ├─ Temporal feature aggregation (risk_engine.py)
  ├─ DNN inference; XGBoost / rule fallback paths
  ├─ Alert transition / hysteresis logic
  └─ SQLite persistence (telemetry and alerts)
              │ REST API
              ▼
Streamlit dashboard
  ├─ Current risk banner and explanation
  ├─ Water, rainfall, soil-proxy, and flow metrics
  ├─ History charts and alert list
  └─ Scenario injection and alert acknowledgement
```

### Main repository areas

| Area | Role |
|---|---|
| `flood-sentry/` | Wokwi-oriented ESP32 firmware and library list |
| `simulation/` | Local firmware/circuit copy and telemetry bridge/scenario generator |
| `backend/` | FastAPI routes, schemas, risk processing, and persistence |
| `model/` | Dataset loading, training/evaluation, and inference wrapper |
| `dashboard/` | Streamlit operations dashboard |
| `artifacts/` | Serialized scaler/models and evaluation reports |
| `tests/` | API, model, feature, and end-to-end tests |
| `data/` | Runtime SQLite telemetry database |

## 4. Hardware and circuit

### 4.1 Parts in the Wokwi diagram

| Component | Quantity | Purpose |
|---|---:|---|
| ESP32 DevKit V1 | 1 | Sensor acquisition, local rules, display, telemetry |
| HC-SR04 ultrasonic sensor | 1 | Measures distance to simulated water surface |
| 16×2 LCD with I²C interface | 1 | Local level/status display |
| Potentiometer | 2 | Simulated rainfall and flow analog inputs |
| Green, yellow, red LEDs | 3 | Local Normal, Watch, and Danger indicators |
| 220 Ω resistor | 3 | LED current limiting in the simulator |
| Buzzer | 1 | Audible alert |
| Pushbutton | 1 | Alarm acknowledge/silence input |

### 4.2 Pin assignment

| Function | ESP32 GPIO | Notes |
|---|---:|---|
| HC-SR04 TRIG | 5 | Firmware outputs a short trigger pulse |
| HC-SR04 ECHO | 18 | Echo pulse measured using `pulseIn` |
| Rain potentiometer signal | 34 | ADC input; rain percentage derived from ADC value |
| Flow potentiometer signal | 35 | ADC input; interpreted as flow percentage |
| Green LED | 25 | Normal / Low local status |
| Yellow LED | 26 | Watch / sensor-warning status |
| Red LED | 27 | Danger / High local status |
| Buzzer | 14 | Alert tones |
| Acknowledge button | 13 | Active-low input with pull-up |
| LCD SDA | 21 | I²C data |
| LCD SCL | 22 | I²C clock |

### 4.3 Measurement conversion

The firmware uses a configured 100 cm distance from the sensor face to the dry channel bed:

\[
\text{water depth (cm)} = 100 - \text{measured distance (cm)}
\]

\[
\text{water level (\%)} =
100 \times \frac{\text{water depth (cm)}}{\text{channel capacity (cm)}}
\]

For example, a measured distance of 34.3 cm yields an estimated depth of 65.7 cm, or 65.7% of the configured 100 cm capacity. Real installation requires a site-specific, measured datum and sensor calibration. A timeout or out-of-range echo should be treated as invalid rather than as a reliable water level.

The ADC inputs are scaled to a 0–100% range. In this circuit they are **potentiometers**, not actual rain-gauge or flow-meter instruments. The firmware maps the rain percentage to a nominal mm/h value for demonstration; this is not a calibrated physical rainfall measurement.

## 5. Embedded firmware operation

The Arduino-style ESP32 firmware:

1. configures GPIO and initializes the LCD over I²C;
2. performs an LED self-test;
3. attempts to join Wokwi’s `Wokwi-GUEST` Wi-Fi network;
4. samples the ultrasonic sensor and analog inputs on a periodic interval;
5. checks the distance is within the supported measurement range;
6. calculates water level and local Low/Watch/Danger states;
7. drives the LEDs, LCD, and buzzer;
8. allows the acknowledge button to silence an alarm until the risk state changes;
9. writes structured telemetry to Serial; and
10. attempts an HTTP POST if Wi-Fi is connected.

The local firmware uses preliminary threshold rules, not the DNN. In the local firmware copy, Danger is raised for water level at or above 75%, or a combination of level, high rain, and low flow; Watch is raised for level at or above 40%, or high rain combined with low flow. Invalid distance produces a yellow sensor-warning indication. These are prototype thresholds and must not be represented as official flood standards.

The firmware has a nominal three-second telemetry interval in the local copy. It includes a 30,000 μs ultrasonic echo timeout. The local firmware’s HTTP URL is configurable and is not by itself proof that Wokwi cloud networking can reach a host computer.

## 6. Telemetry and backend

### 6.1 Telemetry fields

The local firmware’s serial JSON contains a device identifier, sequence/uptime information, distance, computed water depth and percentage, raw and percentage rainfall/flow readings, channel capacity, sensor validity, and data-source metadata. The API schema accepts values including:

- `device_id`, optional `timestamp`;
- `distance_cm`, `water_level_cm`, `water_level_pct`;
- `rainfall_raw`, `rainfall_pct`, `rainfall_mm_h`;
- `flow_raw`, `flow_pct`;
- `soil_saturation_pct`, `channel_capacity_cm`; and
- `data_source`.

The backend validates request ranges with Pydantic. For example, distance accepts `-1` as a timeout marker and is bounded above; percentage features are constrained to 0–100; and capacity must be positive.

### 6.2 Feature processing

The risk engine builds an 11-value model input:

1. current rainfall rate;
2. 3-reading rainfall sum;
3. 6-reading rainfall sum;
4. 24-reading rainfall sum;
5. water level in cm;
6. water level percentage;
7. water-level change;
8. soil saturation or proxy;
9. channel capacity;
10. hour of day; and
11. month.

It obtains recent records for the same device from SQLite, sums the current rainfall with recent samples for the nominal rolling windows, computes the level difference to the prior sample, and either uses provided soil saturation or updates a simple proxy:

\[
S_t = \operatorname{clip}(0.985S_{t-1} + 0.62\min(R_t, 50) - 0.48,\;0,\;100)
\]

Here \(S_t\) is a percentage proxy and \(R_t\) is the current rainfall value. Since this project’s circuit has no soil sensor, any soil value in the live dashboard is supplied or estimated, not directly measured.

### 6.3 API routes

| Route | Purpose |
|---|---|
| `GET /` | Service status and forecast-horizon metadata |
| `GET /health` | Database and model-load status, record counts |
| `POST /api/telemetry` | Validate/process/infer/store one telemetry record |
| `GET /api/latest` | Retrieve latest stored telemetry |
| `GET /api/history` | Retrieve recent history, optionally by device |
| `GET /api/alerts` | Retrieve recent alert events |
| `POST /api/alerts/acknowledge` | Acknowledge stored alerts |
| `POST /predict` | Request a prediction directly from an 11-feature vector |
| `GET /api/model/info` | Return stored model comparison/evaluation report |

The API implementation is in `backend/main.py`; SQLite access and table setup are in `backend/database.py`.

### 6.4 Persistence and alert behavior

SQLite stores telemetry and computed prediction fields, plus alert events and acknowledgement state. Alert hysteresis in `backend/risk_engine.py` is intended to:

- trigger a High alert immediately on High-risk escalation;
- require two consecutive Medium readings before a Medium alert; and
- reset the consecutive-reading counter when risk returns to Low.

This is software alert logic, not a certified emergency-notification service. The prototype has no demonstrated SMS, public siren network, or municipal dispatch integration.

## 7. Prediction models and dataset

### 7.1 Risk classes and target

The models classify a three-hour-ahead risk label into:

| Code | Label | Intended demonstration meaning |
|---:|---|---|
| 0 | Low | Normal simulated conditions |
| 1 | Medium | Watch / elevated simulated conditions |
| 2 | High | Severe simulated conditions |

The primary backend inference path uses a trained DNN and a fitted `StandardScaler`. The DNN is a feed-forward network with 11 input features, dense layers of 64, 32, and 16 ReLU units, batch normalization, dropout, and a three-class softmax output. Training uses Adam, sparse categorical cross-entropy, class weights, early stopping, and model checkpointing.

Other evaluated options include:

- **XGBoost**, an alternate tabular classifier;
- **LSTM**, a sequence model using six consecutive hourly feature vectors; and
- **rule-based fallback**, used when model inference is unavailable.

The inference wrapper checks model availability and reports which path was used. Its output includes predicted class, class probabilities, confidence, model name, fallback flag, and rationale. The displayed “confidence” is the winning model probability; it should not be interpreted as a guarantee of correctness or as independently calibrated certainty.

### 7.2 Dataset provenance

The repository identifies `flood_risk_dnn_synthetic_120k.csv` as a 120,000-row **synthetic** dataset. The generator uses rainfall measurements as an empirical magnitude reference, but simulates timestamps, sites, water levels, soil saturation, rainfall sequences, and target labels. It intentionally reweights rainfall categories to create sufficient examples of different storm conditions.

The dataset generator describes 20 simulated sites and hourly sequences. The target is generated from simulated risk rules, not independently verified future flood observations. Therefore model evaluation measures agreement with this synthetic labeling setup. It is not evidence of prediction performance on real rivers, cities, sensor installations, or future flood events.

The dataset loader uses separate train, validation, and test splits and fits the scaler on training data only. The saved evaluation report lists 83,940 training rows, 17,940 validation rows, and 17,940 test rows after target filtering.

### 7.3 Recorded benchmark results

The project’s saved `artifacts/model_comparison_report.json` reports the following results on its synthetic test set of 17,940 samples:

| Model | Accuracy | Macro F1 | High-class recall | High-class false-negative rate |
|---|---:|---:|---:|---:|
| Rule-based baseline | 87.99% | 0.817 | 83.12% | 16.88% |
| XGBoost | 92.70% | 0.902 | 93.72% | 6.28% |
| DNN | 92.36% | 0.895 | 92.71% | 7.29% |
| LSTM | 92.67% | 0.900 | 93.18% | 6.82% |

The DNN is the deployed primary backend model even though XGBoost has slightly higher reported accuracy and macro-F1 in this saved comparison. These values are project-artifact results, not newly reproduced measurements for this report, and apply only to the synthetic dataset and labeling procedure.

## 8. Dashboard

The Streamlit dashboard at `dashboard/app.py` polls the configured backend and presents:

- backend health and stored-record count;
- current predicted risk and forecast horizon;
- confidence and model rationale;
- data source and telemetry freshness;
- current water level, rainfall, accumulated rainfall, soil proxy, and flow;
- time-series history and alert log;
- demo scenario injection controls; and
- an alert acknowledge action.

It is a local operational monitor for this prototype. By default it runs on port 8501 and contacts a local backend, typically on port 8000. A local URL is not a publicly hosted website.

## 9. Demonstration performed

### 9.1 Wokwi hardware simulation

The linked Wokwi project was opened and run in a browser. Its serial output was observed with these values:

| Observed value | Reading |
|---|---:|
| HC-SR04 distance | 34.3 cm |
| Derived channel level | 65.7% |
| Rain potentiometer | 24% |
| Flow potentiometer | 78% |
| Wokwi firmware status | WATCH |

The circuit also displays outputs using the LCD, indicator LEDs, buzzer, and acknowledge button.

### 9.2 Website/API ingestion

A browser-session relay forwarded an observed Wokwi serial reading to the local API using `data_source: "wokwi_live"`. The API returned HTTP 201 and persisted the record. The local dashboard then displayed the `wokwi_live` data source and the backend DNN risk result.

One verified example record contained:

- device ID `WOKWI_FLOOD_SENTRY_01`;
- 65.7% water level;
- 24 mm/h mapped rainfall input;
- 78% flow input;
- DNN forecast `High`; and
- `data_source` equal to `wokwi_live`.

In that example, the device itself showed **WATCH**, while the model forecast shown on the website was **High**. These represent different decision paths: local threshold logic versus a learned forecast based on backend features. The example was a successful software integration demonstration, not validation of a real flood event.

### 9.3 Important distinction between demo modes

The repository’s `simulation/telemetry_bridge.py --mode stream` generates a separate synthetic storm progression. It does not, by itself, read the Wokwi browser’s serial monitor. Actual Wokwi serial forwarding in the demonstration used a temporary browser-session relay; it is not a persistent, packaged integration service. Closing/reloading that Wokwi browser session ends that relay.

The project also contains local firmware copies and a public Wokwi project. They are not guaranteed to be byte-for-byte identical. For a repeatable report or deployment, synchronize the firmware, wiring diagram, and serial schema and clearly specify which version was run.

## 10. Known limitations and interpretation

These limitations are material and should be included in any presentation or report:

1. **Synthetic labels and data:** Model metrics do not demonstrate real-world flood accuracy.
2. **Surrogate sensors:** The rain and flow controls are potentiometers; they do not measure rainfall or discharge physically.
3. **Soil is not measured:** No soil-moisture sensor appears in the circuit. Soil saturation is an input/proxy.
4. **Time-window mismatch risk:** The backend rolling sums select a fixed number of recent POST records (up to 2, 5, and 23 previous records) rather than grouping or integrating them by elapsed time. The feature comments assume hourly samples. At the much faster cadence of a live simulation, those sums are not valid 3-, 6-, or 24-hour rainfall accumulations.
5. **Rate-of-change timing:** Water-level differences are calculated between adjacent readings, but the code does not normalize by the elapsed time between timestamps. The resulting field is labeled cm/h even when readings are seconds apart.
6. **Different edge and backend outputs:** Local firmware thresholds and DNN classification are separate. A disagreement is possible and must not be hidden.
7. **No operational alert delivery:** The alert table and dashboard are local. No verified external emergency notification or fail-safe redundancy is included.
8. **Connectivity is environment-dependent:** A Wokwi virtual ESP32 cannot automatically reach a host computer through `localhost`. HTTP forwarding needs an appropriate gateway/network configuration; Serial bridge or the temporary browser relay has different setup requirements.
9. **Calibration and physical installation:** Real sensors need suitable mounting, power, enclosure, electrical protection, range checks, site capacity calibration, and maintenance.
10. **Not safety certified:** The system has not been independently validated, field-tested, or certified as an official warning device.

The time-window behavior should be corrected before treating live model forecasts as meaningful: store and aggregate samples according to actual timestamps, convert rate readings into interval rainfall depth, normalize water-level change by elapsed time, and test irregular/missing sample intervals.

## 11. Testing and verification

The repository includes automated tests for:

- API root and health routes;
- telemetry validation, ingestion, latest/history endpoints;
- direct prediction and model-info routes;
- acknowledgement and alert retrieval;
- deterministic low-, medium-, high-, and fault-condition E2E scenarios;
- Medium-alert hysteresis; and
- SQLite persistence.

The live demonstration also verified:

- the backend health endpoint returned `ok`;
- the database and DNN/XGBoost artifacts loaded;
- `POST /api/telemetry` accepted a Wokwi-origin record and returned HTTP 201;
- `GET /api/latest` returned that persisted `wokwi_live` record; and
- the Streamlit dashboard displayed the Wokwi source and model result.

No claim that the full automated test suite passed is made here unless it is separately executed and logged for the final submission.

## 12. Reproduction instructions (Windows)

Run commands from the project root. The repository documents a preconfigured Python 3.12 environment.

**Terminal 1 — backend**

```powershell
& ".\.venv312\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — dashboard**

```powershell
& ".\.venv312\Scripts\python.exe" -m streamlit run dashboard\app.py
```

**Terminal 3 — synthetic stream**

```powershell
& ".\.venv312\Scripts\python.exe" simulation\telemetry_bridge.py --mode stream --interval 2.0
```

For deterministic backend-only cases:

```powershell
& ".\.venv312\Scripts\python.exe" simulation\telemetry_bridge.py --mode scenario --scenario low
& ".\.venv312\Scripts\python.exe" simulation\telemetry_bridge.py --mode scenario --scenario medium
& ".\.venv312\Scripts\python.exe" simulation\telemetry_bridge.py --mode scenario --scenario high
& ".\.venv312\Scripts\python.exe" simulation\telemetry_bridge.py --mode scenario --scenario fault
```

For automated tests:

```powershell
& ".\.venv312\Scripts\python.exe" -m pytest tests\ -v
```

Open:

- Wokwi project: https://wokwi.com/projects/478416784820391937
- Dashboard: http://localhost:8501
- API health: http://localhost:8000/health
- Interactive API docs: http://localhost:8000/docs

The synthetic telemetry bridge is suitable to demonstrate the dashboard and API pipeline. For a claim that values came directly from the ESP32 circuit, use a clearly identified Wokwi serial forwarding path and verify the stored `data_source`.

## 13. Future work

Priorities for improving the prototype are:

1. Correct temporal aggregation using actual sample timestamps and interval-based rainfall totals.
2. Add timestamp-normalized water-level rise rates and tests for irregular sampling.
3. Replace potentiometers with calibrated rainfall and flow sensors; add a real soil-moisture sensor only if it is needed and feasible.
4. Make the serial-to-API bridge a packaged, reconnecting service with explicit status, error handling, and duplicate suppression.
5. Add authenticated API access, restrictive CORS, secrets/configuration management, and deployment security.
6. Version the training dataset, synthetic generation seed, model artifacts, preprocessing, and evaluation procedure.
7. Evaluate on independent, real, representative hydrological observations before making performance claims.
8. Add sensor-fault, stale-data, network-loss, database-failure, and model-failure indicators to the operator UI.
9. Synchronize the public Wokwi sketch and local firmware, and maintain a single documented circuit/firmware source.
10. Conduct field, electrical, environmental, and safety testing before any real-world use.

## 14. Conclusion

Flood Sentry demonstrates an end-to-end IoT decision-support workflow: a simulated embedded node observes adjustable inputs, firmware produces a local status, an API processes telemetry and applies a trained classifier, SQLite records the outcome, and a dashboard displays it. The project is useful for illustrating sensor interfacing, telemetry design, backend APIs, model serving, alert logic, and monitoring UI development.

Its strongest substantiated result is that this complete demonstration path can operate in simulation and that Wokwi-origin readings were successfully displayed by the local website. The saved model benchmarks apply to a synthetic dataset, and the temporal aggregation mismatch makes high-frequency live model features unsuitable for real flood interpretation without correction. The prototype must therefore be presented as a demonstrator—not as a validated flood forecasting or public warning system.

## 15. Project reference files

- `README.md` — quick start and project overview.
- `simulation/diagram.json` — circuit parts and wiring.
- `simulation/sketch.ino` — local ESP32 firmware copy.
- `flood-sentry/sketch.ino` — Wokwi-oriented firmware copy.
- `simulation/telemetry_bridge.py` — synthetic telemetry stream and scenario injector.
- `backend/main.py` — REST endpoints.
- `backend/risk_engine.py` — feature processing and alert-state logic.
- `backend/database.py` — SQLite schema and data access.
- `backend/schemas.py` — API payload and response models.
- `model/inference.py` — inference selection and fallback.
- `model/dataset_loader.py` — feature set and split/scaler handling.
- `model/train_and_evaluate.py` — model training and metrics generation.
- `generate_flood_dnn_dataset.py` — synthetic data generation.
- `artifacts/model_comparison_report.json` — saved benchmark comparison.
- `artifacts/evaluation_report.json` — DNN evaluation summary.
- `docs/WOKWI_SETUP.md` — pinout and simulator networking notes.
- `docs/ARCHITECTURE.md` — documented architecture and algorithms.
- `docs/DEMO_GUIDE.md` — demonstration steps.
