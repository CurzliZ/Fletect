# END-TO-END DEMONSTRATION GUIDE: FLOOD SENTRY

Follow these steps to demonstrate the complete Flood Sentry system live on Windows 11.

---

## 1. Prerequisites & Environment Check

Open PowerShell in the project directory (`c:\Users\Siva Ranjan\Desktop\Fletect`). Verify the virtual environment:

```powershell
& ".\.venv312\Scripts\python.exe" --version
```
*Expected:* `Python 3.12.10`

---

## 2. Terminal 1: Launch FastAPI Backend Server

Start the local API on port 8000:

```powershell
& ".\.venv312\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

* Verify health in your browser: open [http://localhost:8000/health](http://localhost:8000/health).
* Interactive Swagger API docs: [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 3. Terminal 2: Launch Streamlit Operational Dashboard

Open a second terminal window and run:

```powershell
& ".\.venv312\Scripts\python.exe" -m streamlit run dashboard\app.py
```

* The dashboard will open automatically in your browser at [http://localhost:8501](http://localhost:8501).
* You will see the dark-themed operational monitoring console.

---

## 4. Terminal 3: Live Telemetry Simulation Options

You can feed telemetry into the system using either the **interactive simulation bridge** or the **Wokwi virtual circuit**:

### Method A: Automated Dynamic Storm Simulation (Recommended for Presentations)
Run the continuous weather progression generator:

```powershell
& ".\.venv312\Scripts\python.exe" simulation\telemetry_bridge.py --mode stream --interval 2.0
```

Watch the Streamlit dashboard live as it cycles through:
1. **Phase 1: Fair Weather** — Low water level (~20%), no rain, Green Normal status.
2. **Phase 2: Inflow Building** — Water rising to ~45%, rain ~25 mm/h, Yellow Watch condition.
3. **Phase 3: Severe Deluge Peak** — Water surging to ~85%, rain ~80 mm/h, Red High-Risk Flood Alert!
4. **Phase 4: Drainage Recession** — Water drops back below warning thresholds.

---

### Method B: Deterministic Scenario Injections

Inject specific risk states on demand from PowerShell (or via the dashboard sidebar buttons):

#### 1. Low Risk (Normal Operation)
```powershell
& ".\.venv312\Scripts\python.exe" simulation\telemetry_bridge.py --mode scenario --scenario low
```
* Dashboard banner: `GREEN | STATUS: LOW`

#### 2. Medium Risk (Watch Condition)
```powershell
& ".\.venv312\Scripts\python.exe" simulation\telemetry_bridge.py --mode scenario --scenario medium
```
* Injects 2 consecutive samples to confirm transition past hysteresis.
* Dashboard banner: `YELLOW | STATUS: MEDIUM`
* Yellow alert logged in Alert History table.

#### 3. High Risk (Flash Flood Surge)
```powershell
& ".\.venv312\Scripts\python.exe" simulation\telemetry_bridge.py --mode scenario --scenario high
```
* Dashboard banner: `RED | STATUS: HIGH (94%+ DNN confidence)`
* Red alert logged in Alert History table.

#### 4. Sensor Fault (Disconnected Cable / No Echo)
```powershell
& ".\.venv312\Scripts\python.exe" simulation\telemetry_bridge.py --mode scenario --scenario fault
```
* Demonstrates system resilience: invalid sensor readings do not crash the backend.

---

## 5. Method C: Interactive Wokwi Circuit Simulation

1. Open [Wokwi.com](https://wokwi.com) or your VS Code Wokwi extension.
2. Open `diagram.json` and `flood-sentry/sketch.ino`.
3. Click **Start Simulation**.
4. Move the **Distance slider** on the HC-SR04 sensor down towards 15 cm.
5. Turn the **Rain Potentiometer** clockwise towards maximum.
6. Observe:
   - Green LED turns OFF; Red LED turns ON.
   - 16x2 LCD changes from `STATUS: NORMAL` to `STATUS: DANGER!`.
   - Piezo buzzer sounds continuous 1800 Hz alarm.
   - Click the blue **ACK pushbutton** (GPIO 13) to silence the alarm.
