#!/usr/bin/env python3
"""Telemetry Bridge and Virtual Sensor Feeder for Flood Sentry.

Connects the physical/simulated edge node with the FastAPI backend.
Supports three modes:
  1. stream: Continuous dynamic weather simulation (ideal for live demos)
  2. serial: Reads lines from serial COM port (or Wokwi serial pipe)
  3. scenario: Injects deterministic test cases (low, medium, high, fault, flapping)
"""
import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

def get_default_backend() -> str:
    for url in ("http://localhost:8001/api/telemetry", "http://localhost:8000/api/telemetry"):
        try:
            base = url.replace("/api/telemetry", "")
            if requests.get(f"{base}/health", timeout=0.3).status_code == 200:
                return url
        except Exception:
            pass
    return "http://localhost:8001/api/telemetry"

DEFAULT_BACKEND = get_default_backend()


def send_payload(backend_url: str, payload: Dict[str, Any]) -> bool:
    try:
        resp = requests.post(backend_url, json=payload, timeout=3.0)
        if resp.status_code in (200, 201):
            data = resp.json()
            risk = data.get("predicted_risk", "UNKNOWN")
            conf = data.get("confidence_score", 0.0)
            model = data.get("model_used", "N/A")
            alert = " [ALERT TRIGGERED]" if data.get("alert_triggered") else ""
            print(f"[OK] Ingested: Level={payload.get('water_level_pct', 0):.1f}% | Rain={payload.get('rainfall_mm_h', 0):.1f}mm/h -> 3h Risk: {risk} ({conf*100:.1f}% via {model}){alert}")
            return True
        else:
            print(f"[ERR] Ingestion HTTP {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"[ERR] Failed to reach backend at {backend_url}: {e}")
        return False


def run_continuous_stream(backend_url: str, interval: float = 2.0, device_id: str = "ESP32_FLOOD_01"):
    print(f"[*] Starting continuous realistic weather stream to {backend_url}")
    print("[*] Simulating storm progression cycle: Fair -> Rain Building -> Peak Flash Flood -> Recession -> Fair")
    print("[*] Press Ctrl+C to terminate.")

    channel_capacity = 100.0  # cm
    t = 0
    water_level = 22.0  # initial baseline cm
    soil = 25.0

    while True:
        # Progress through an undulating multi-phase storm wave
        t += 1
        cycle = (t % 60)  # 60-step full storm cycle

        if cycle < 15:
            # Phase 1: Dry / Fair weather
            rain = max(0.0, random.gauss(0.5, 0.5))
            target_water = 20.0 + random.uniform(-1.0, 1.0)
            phase_name = "Fair Weather"
        elif cycle < 30:
            # Phase 2: Moderate rain shower building
            rain = 15.0 + (cycle - 15) * 2.5 + random.uniform(-3, 3)
            target_water = 38.0 + (cycle - 15) * 1.5 + random.uniform(-1.5, 1.5)
            phase_name = "Rainfall Inflow"
        elif cycle < 42:
            # Phase 3: Severe Monsoon Deluge / River Surge
            rain = 65.0 + random.uniform(5, 25)
            target_water = 75.0 + (cycle - 30) * 1.8 + random.uniform(-1.0, 1.0)
            target_water = min(96.0, target_water)
            phase_name = "Severe Deluge / Flood Risk Peak"
        else:
            # Phase 4: Rain clears, runoff drains
            rain = max(0.0, 10.0 - (cycle - 42) * 0.8 + random.uniform(-1, 1))
            target_water = max(22.0, water_level - 3.5 + random.uniform(-0.5, 0.5))
            phase_name = "Recession / Drainage"

        # Smooth water level movement
        water_level += 0.4 * (target_water - water_level)
        water_level = round(max(5.0, min(98.0, water_level)), 1)
        rain = round(max(0.0, rain), 1)
        soil = round(min(100.0, max(10.0, 0.98 * soil + 0.5 * rain - 0.2)), 1)
        water_pct = round((water_level / channel_capacity) * 100.0, 1)
        distance = round(channel_capacity - water_level, 1)

        payload = {
            "device_id": device_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "distance_cm": distance,
            "water_level_cm": water_level,
            "water_level_pct": water_pct,
            "rainfall_mm_h": rain,
            "rainfall_pct": min(100.0, rain),
            "flow_pct": round(max(10.0, min(95.0, 100.0 - water_pct * 0.6)), 1),
            "soil_saturation_pct": soil,
            "channel_capacity_cm": channel_capacity,
            "data_source": "simulation",
        }

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Phase: {phase_name} | ", end="")
        send_payload(backend_url, payload)
        time.sleep(interval)


def run_scenario(backend_url: str, scenario: str, device_id: str = "ESP32_FLOOD_01"):
    print(f"[*] Executing deterministic scenario: {scenario.upper()}")
    now_iso = datetime.now(timezone.utc).isoformat()

    if scenario == "low":
        payload = {
            "device_id": device_id,
            "timestamp": now_iso,
            "distance_cm": 80.0,
            "water_level_cm": 20.0,
            "water_level_pct": 20.0,
            "rainfall_mm_h": 0.0,
            "rainfall_pct": 0.0,
            "flow_pct": 60.0,
            "soil_saturation_pct": 22.0,
            "channel_capacity_cm": 100.0,
            "data_source": "test_injection",
        }
        send_payload(backend_url, payload)

    elif scenario == "medium":
        # Inject two consecutive medium readings to satisfy hysteresis
        for step in (1, 2):
            payload = {
                "device_id": device_id,
                "timestamp": now_iso,
                "distance_cm": 52.0,
                "water_level_cm": 48.0,
                "water_level_pct": 48.0,
                "rainfall_mm_h": 32.0,
                "rainfall_pct": 32.0,
                "flow_pct": 45.0,
                "soil_saturation_pct": 68.0,
                "channel_capacity_cm": 100.0,
                "data_source": "test_injection",
            }
            send_payload(backend_url, payload)
            time.sleep(0.5)

    elif scenario == "high":
        payload = {
            "device_id": device_id,
            "timestamp": now_iso,
            "distance_cm": 18.0,
            "water_level_cm": 82.0,
            "water_level_pct": 82.0,
            "rainfall_mm_h": 85.0,
            "rainfall_pct": 85.0,
            "flow_pct": 20.0,
            "soil_saturation_pct": 92.0,
            "channel_capacity_cm": 100.0,
            "data_source": "test_injection",
        }
        send_payload(backend_url, payload)

    elif scenario == "fault":
        payload = {
            "device_id": device_id,
            "timestamp": now_iso,
            "distance_cm": -1.0,  # No echo timeout code
            "water_level_cm": 0.0,
            "water_level_pct": 0.0,
            "rainfall_mm_h": 10.0,
            "channel_capacity_cm": 100.0,
            "data_source": "test_injection",
        }
        send_payload(backend_url, payload)

    elif scenario == "flapping_test":
        print("[*] Testing alert hysteresis suppression over 6 alternating samples...")
        for i in range(6):
            is_medium = (i % 2 == 1)
            water = 48.0 if is_medium else 22.0
            rain = 35.0 if is_medium else 0.0
            payload = {
                "device_id": device_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "distance_cm": 100.0 - water,
                "water_level_cm": water,
                "water_level_pct": water,
                "rainfall_mm_h": rain,
                "soil_saturation_pct": 50.0,
                "channel_capacity_cm": 100.0,
                "data_source": "test_injection",
            }
            send_payload(backend_url, payload)
            time.sleep(0.4)
    else:
        print(f"[ERR] Unknown scenario: {scenario}. Choose from: low, medium, high, fault, flapping_test")


def main():
    parser = argparse.ArgumentParser(description="Flood Sentry Telemetry Feeder and Bridge")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, help="Backend telemetry ingestion URL")
    parser.add_argument("--mode", choices=["stream", "scenario", "serial"], default="stream")
    parser.add_argument("--scenario", choices=["low", "medium", "high", "fault", "flapping_test"], default="high")
    parser.add_argument("--interval", type=float, default=2.0, help="Stream interval seconds")
    parser.add_argument("--device", default="ESP32_FLOOD_01", help="Device ID")
    parser.add_argument("--port", default="COM3", help="Serial port for serial mode")
    args = parser.parse_args()

    if args.mode == "stream":
        run_continuous_stream(args.backend, args.interval, args.device)
    elif args.mode == "scenario":
        run_scenario(args.backend, args.scenario, args.device)
    elif args.mode == "serial":
        try:
            import serial
            print(f"[*] Opening serial port {args.port} at 115200 baud...")
            with serial.Serial(args.port, 115200, timeout=2.0) as ser:
                while True:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if line.startswith("TELEMETRY:"):
                        raw_json = line[len("TELEMETRY:"):]
                        try:
                            payload = json.loads(raw_json)
                            send_payload(args.backend, payload)
                        except json.JSONDecodeError:
                            print(f"[WARN] Malformed serial JSON: {raw_json}")
        except ImportError:
            print("[ERR] pyserial is required for --mode serial. Install with: pip install pyserial")
        except Exception as e:
            print(f"[ERR] Serial bridge error: {e}")


if __name__ == "__main__":
    main()
