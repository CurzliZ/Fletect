# SYSTEM ARCHITECTURE: FLOOD SENTRY
**IoT-Based Flood Risk Detection Using Deep Neural Networks**

## 1. High-Level System Overview

Flood Sentry is an integrated, end-to-end edge-to-decision IoT flood monitoring and early warning system. It bridges physical/simulated telemetry acquisition, temporal rolling feature aggregation, deep neural network risk forecasting, resilient SQLite persistence, and interactive operational monitoring.

```
+-------------------------------------------------------------------------+
|                        EDGE TIER (Wokwi / Hardware)                     |
|                                                                         |
|  [HC-SR04 Sonar]     [Rain Pot (ADC 34)]     [Flow Pot (ADC 35)]        |
|    GPIO 5 (TRIG)            │                       │                   |
|    GPIO 18 (ECHO)           │                       │                   |
|          │                  │                       │                   |
|          ▼                  ▼                       ▼                   |
|  +───────────────────────────────────────────────────────────────────+  |
|  |             ESP32 Microcontroller (FreeRTOS Firmware)             |  |
|  |  - Ultrasonic echo to water level conversion (100cm datum)        |  |
|  |  - Sensor bounds checking & timeout fault detection               |  |
|  |  - Dual local output: 16x2 I2C LCD (21/22) + RGB Status LEDs     |  |
|  |  - Buzzer alarm with Pushbutton Silence / ACK (GPIO 13)           |  |
|  |  - Wokwi-GUEST WiFi HTTP Client + Structured Serial JSON Stream   |  |
|  +──────────────────────────────────┬────────────────────────────────+  |
+─────────────────────────────────────┼───────────────────────────────────+
                                      │ HTTP POST / Serial Bridge
                                      ▼
+-------------------------------------------------------------------------+
|                        INGESTION & PERSISTENCE TIER                     |
|                                                                         |
|  FastAPI Server (backend/main.py)                                       |
|  ├── Ingestion Endpoint: POST /api/telemetry                            |
|  ├── Pydantic Input Validation & Range Bounds                           |
|  ├── Risk Engine (backend/risk_engine.py)                               |
|  │     ├── Temporal Rolling Accumulator (3h, 6h, 24h Rain)              |
|  │     ├── Hourly Water Level Delta (cm/h)                              |
|  │     └── Hydrological Soil Saturation Proxy Calculation               |
|  ├── Persistence (backend/database.py)                                  |
|  │     └── SQLite Database (data/flood_sentry.db)                       |
|  │           ├── telemetry table (Indexed by device_id & timestamp)     |
|  │           └── alerts table (Transition events & ACK status)          |
|  └── Alert State Machine with Hysteresis (Anti-Flapping Engine)         |
+─────────────────────────────────────┬───────────────────────────────────+
                                      │ Feature Vector (11 Dims)
                                      ▼
+-------------------------------------------------------------------------+
|                        ANALYTICS & INFERENCE TIER                       |
|                                                                         |
|  FloodRiskInferenceEngine (model/inference.py)                          |
|  ├── Scaler: StandardScaler fitted strictly on Train split              |
|  ├── Primary Model: Deep Neural Network (DNN, Keras / TensorFlow)       |
|  │     └── 3 Dense Layers + BatchNormalization + Dropout + L2           |
|  ├── Alternate Model: XGBoost Multi-Class Classifier                    |
|  ├── Research Benchmark: Sequential LSTM (6-hour sliding tensors)       |
|  └── Hydrological Rule-Based Fallback Engine (Zero-dependency failover) |
|                                                                         |
|  Forecast Horizon: +3 Hours Ahead                                       |
|  Classes: 0 = Low Risk (Normal), 1 = Medium (Watch), 2 = High (Danger)  |
+─────────────────────────────────────┬───────────────────────────────────+
                                      │ JSON Telemetry & Event Stream
                                      ▼
+-------------------------------------------------------------------------+
|                        OPERATIONAL DASHBOARD TIER                       |
|                                                                         |
|  Streamlit Application (dashboard/app.py)                               |
|  ├── Real-time Risk Status Banner & Hydrological Explainability         |
|  ├── Live KPI Metric Cards (Water Level %, Rain mm/h, Runoff Flow)      |
|  ├── Interactive Time-Series Graphs (Level vs Capacity, Accumulation)   |
|  ├── Alert History Table & Silence Action                               |
|  ├── Data Freshness Indicator & Device Health                           |
|  └── Integrated Demo & Stress Test Scenario Injector                    |
+-------------------------------------------------------------------------+
```

---

## 2. Sensor Interfacing & Physical Datum Calibration

### Ultrasonic Water Level Calculation
Ultrasonic sensors measure the time-of-flight from the sensor transducer to the nearest reflective surface. In an open channel or drainage canal, the sensor is mounted above the maximum water level looking downward.

$$\text{Water Level (cm)} = \text{Sensor Datum Height} - \text{Measured Distance (cm)}$$
$$\text{Water Level (\%)} = \left(\frac{\text{Water Level (cm)}}{\text{Sensor Datum Height}}\right) \times 100$$

* In the Flood Sentry reference installation, `SENSOR_TO_DRAIN_BOTTOM_CM = 100.0 cm`.
* A measured distance of 85.0 cm indicates a water depth of 15.0 cm (15% capacity, Normal).
* A measured distance of 18.0 cm indicates a water depth of 82.0 cm (82% capacity, High Risk).
* Timeouts (no echo within 30,000 microseconds) return `-1.0`, triggering sensor fault mode.

---

## 3. Dynamic Temporal Feature Engineering

Machine learning models forecasting hydrological risk 3 hours into the future require historical antecedent moisture and precipitation context. Instantaneous sensor readings are converted on ingestion via SQLite queries:

1. **Current Rainfall Rate ($R_t$):** Instantaneous rate in mm/hour.
2. **3-Hour Accumulated Rainfall ($R_{3h}$):** $\sum_{i=0}^{2} R_{t-i}$.
3. **6-Hour Accumulated Rainfall ($R_{6h}$):** $\sum_{i=0}^{5} R_{t-i}$.
4. **24-Hour Accumulated Rainfall ($R_{24h}$):** $\sum_{i=0}^{23} R_{t-i}$.
5. **Rate of Water Level Change ($\Delta W$):** $W_t - W_{t-1}$ (cm/hour).
6. **Soil Saturation Proxy ($S_t$):** Updated recursively:
   $$S_t = \text{clip}(0.985 \cdot S_{t-1} + 0.62 \cdot \min(R_t, 50.0) - 0.48, 0.0, 100.0)$$

---

## 4. Alert Hysteresis Engine (Anti-Flapping)

To prevent alerts from repeatedly toggling on boundary noise:
* **Escalation to High Risk:** Generates an immediate critical alert.
* **Escalation to Medium Risk:** Requires 2 consecutive readings confirming the watch condition before triggering an alert.
* **De-escalation:** Transition back to Normal automatically resets debounce counters and suppresses further alarm generation.
