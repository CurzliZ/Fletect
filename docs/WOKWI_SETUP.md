# WOKWI SIMULATOR CONFIGURATION & NETWORKING GUIDE: FLOOD SENTRY

## 1. Verified Hardware Pin Mapping

The physical wiring diagram in `diagram.json` and the firmware in `flood-sentry/sketch.ino` (and `simulation/sketch.ino`) use the following verified pin configuration:

| Component | Component Pin | ESP32 GPIO | Description / Role |
|---|---|---|---|
| **HC-SR04 Sonar** | `TRIG` | **GPIO 5** | Ultrasonic trigger pulse (10 µs) |
| **HC-SR04 Sonar** | `ECHO` | **GPIO 18** | Echo pulse width measurement |
| **Rain Potentiometer** | `SIG` | **GPIO 34** | Simulated rainfall intensity (ADC1_CH6) |
| **Flow Potentiometer** | `SIG` | **GPIO 35** | Channel drainage / runoff flow rate (ADC1_CH7) |
| **Green LED (`ledG`)** | Anode (via 220Ω) | **GPIO 25** | Low Risk / Normal status indicator |
| **Yellow LED (`ledY`)** | Anode (via 220Ω) | **GPIO 26** | Medium Risk / Watch status indicator |
| **Red LED (`ledR`)** | Anode (via 220Ω) | **GPIO 27** | High Risk / Danger alert indicator |
| **Buzzer (`buzz`)** | Terminal 1 | **GPIO 14** | Acoustic alert (1800 Hz continuous on danger) |
| **Pushbutton (`ack`)** | Pin 1.l | **GPIO 13** | Silence / Alarm Acknowledge (Active LOW) |
| **LCD 1602 (I2C)** | `SDA` | **GPIO 21** | I2C Data line |
| **LCD 1602 (I2C)** | `SCL` | **GPIO 22** | I2C Clock line |

> [!NOTE]
> **Pin Conflict Notice:** In the original master prompt, an alternate hypothetical pinout was listed (Green=18, Red=21). In the actual working Wokwi circuit (`diagram.json`), GPIO 18 is dedicated to the ultrasonic sensor `ECHO` line, and GPIO 21 is required for I2C LCD `SDA`. The pinout above is preserved to prevent electrical shorts while maintaining 100% circuit compatibility.

---

## 2. Water Level Datum Calibration

Ultrasonic sensors measure distance from the sensor downwards:
* Overhead mount height: `SENSOR_TO_DRAIN_BOTTOM_CM = 100.0 cm`.
* Water Level Height = $100.0 - \text{Measured Distance (cm)}$.
* To simulate high water in Wokwi: click the HC-SR04 sonar module and adjust the **Distance** slider down to ~15 cm (water depth 85 cm, 85% capacity).

---

## 3. Wokwi Networking & Telemetry Bridge

### How Wokwi Networking Works
* Wokwi simulates an open Wi-Fi network named **`Wokwi-GUEST`**.
* In Wokwi's cloud simulation sandbox, `localhost` (or `127.0.0.1`) refers to the browser/container environment, **not your host laptop**.

### Integration Options:

#### Option A: High-Reliability Local Bridge (Recommended)
The firmware continuously prints structured JSON telemetry to the Serial Monitor:
```text
TELEMETRY:{"device_id":"ESP32_FLOOD_01","seq":12,"distance_cm":34.0,"water_level_cm":66.0,"water_level_pct":66.0,"rainfall_mm_h":25.0,...}
```
Run the provided bridge script to stream or inject telemetry directly into the local FastAPI backend:
```powershell
.\.venv312\Scripts\python.exe simulation\telemetry_bridge.py --mode stream --interval 2.0
```

#### Option B: Wokwi IoT Gateway (Live Cloud-to-Host HTTP)
If you install the [Wokwi IoT Gateway](https://docs.wokwi.com/guides/esp32-wifi#the-private-gateway), the simulated ESP32 can send HTTP requests directly to your computer's LAN IP address.
1. Run the Wokwi IoT Gateway on your Windows machine.
2. Update `BACKEND_URL` in `sketch.ino` to:
   ```c
   const char* BACKEND_URL = "http://<YOUR_LOCAL_LAN_IP>:8000/api/telemetry";
   ```

---

## 4. How to Launch and Test in Wokwi

1. Open your project on [Wokwi.com](https://wokwi.com) or use VS Code Wokwi Extension with `diagram.json`.
2. Copy the contents of `flood-sentry/sketch.ino` into Wokwi's code tab.
3. Click the **Play (Start Simulation)** button.
4. **Normal State Demonstration:**
   - Sonar distance = 80 cm (Water Level = 20%).
   - Rain potentiometer = 0%.
   - **Result:** Green LED illuminates. LCD displays `STATUS: NORMAL`.
5. **Watch State Demonstration:**
   - Sonar distance = 58 cm (Water Level = 42%).
   - Rain potentiometer = 30%.
   - **Result:** Yellow LED illuminates. LCD displays `STATUS: WATCH`. Intermittent buzzer beep.
6. **High Flood Risk Demonstration:**
   - Sonar distance = 15 cm (Water Level = 85%).
   - Rain potentiometer = 80%.
   - **Result:** Red LED illuminates. LCD displays `STATUS: DANGER!`. Continuous 1800 Hz alarm tone.
7. **Alarm Acknowledge:** Press the blue `ACK` pushbutton (GPIO 13) to silence the buzzer.
