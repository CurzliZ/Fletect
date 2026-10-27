"""Flood Sentry — Interactive Operational Flood-Risk Monitoring Dashboard.

Streamlit-based real-time dashboard displaying:
  - Live flood risk status & hydrological explainability
  - Real-time gauge metrics (water level, rainfall rate, soil saturation)
  - Interactive multi-parameter time-series charts (Altair / Streamlit)
  - Risk classification history & alert log
  - Telemetry freshness & connection status indicators
  - Model diagnostics & comparative benchmark cards
  - Built-in Demo & Testing scenario injector
"""
from datetime import datetime, timezone
import json
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

# Configure page layout and aesthetics
st.set_page_config(
    page_title="Flood Sentry | AI Decision Support",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark modern dashboard aesthetic)
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 4px 0;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
    }

    /* Risk Status Badges */
    .badge-low {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: #78350f;
        color: #fbbf24;
        border: 1px solid #d97706;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
    .badge-high {
        background-color: #7f1d1d;
        color: #f87171;
        border: 1px solid #dc2626;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
    .badge-unknown {
        background-color: #334155;
        color: #94a3b8;
        border: 1px solid #475569;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }

    /* Banner container */
    .status-banner {
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 20px;
        border-left: 6px solid;
    }
    .status-banner-low {
        background: rgba(6, 78, 59, 0.25);
        border-left-color: #10b981;
    }
    .status-banner-medium {
        background: rgba(120, 53, 15, 0.25);
        border-left-color: #f59e0b;
    }
    .status-banner-high {
        background: rgba(127, 29, 29, 0.35);
        border-left-color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/fluency/96/flood.png", width=64)
st.sidebar.title("Flood Sentry")
# Auto-detect active backend port (checks 8001 then 8000)
default_backend = "http://localhost:8001"
for test_url in ("http://localhost:8000", "http://localhost:8001"):
    try:
        if requests.get(f"{test_url}/health", timeout=0.4).status_code == 200:
            default_backend = test_url
            break
    except Exception:
        pass

backend_host = st.sidebar.text_input("Backend API Base URL", value=default_backend)
auto_refresh = st.sidebar.toggle("Auto-Refresh Telemetry (3s)", value=True)
history_limit = st.sidebar.slider("Historical Points to Plot", min_value=20, max_value=300, value=80, step=10)

st.sidebar.divider()
st.sidebar.subheader("🕹️ Demo & Testing Controls")
st.sidebar.caption("Inject deterministic readings without physical hardware:")

scenario_selection = st.sidebar.selectbox(
    "Test Scenario",
    options=["low", "medium", "high", "fault", "flapping_test"],
    format_func=lambda x: {
        "low": "🟢 Low Risk (Normal Retention)",
        "medium": "🟡 Medium Risk (Watch Condition)",
        "high": "🔴 High Risk (Severe Flood Surge)",
        "fault": "⚠️ Sensor Fault (No Echo / Timeout)",
        "flapping_test": "🔄 Alert Flapping Hysteresis Test",
    }[x],
)

if st.sidebar.button("Inject Selected Scenario", use_container_width=True):
    try:
        from simulation.telemetry_bridge import run_scenario
        run_scenario(f"{backend_host}/api/telemetry", scenario_selection)
        st.sidebar.success(f"Injected: {scenario_selection.upper()}")
        time.sleep(0.5)
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Injection failed: {e}")

if st.sidebar.button("Acknowledge / Silence Alerts", use_container_width=True):
    try:
        r = requests.post(f"{backend_host}/api/alerts/acknowledge", timeout=2.0)
        if r.status_code == 200:
            st.sidebar.success("Alerts acknowledged!")
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"Ack failed: {e}")

st.sidebar.divider()
st.sidebar.info(
    "⚠️ **Disclaimer:** Experimental research prototype. Uses synthetic-calibrated "
    "deep neural networks for hydrological decision-support. Not an official municipal warning system."
)


# ============================================================================
# API Client Helper Functions
# ============================================================================
def fetch_api_data(endpoint: str, default: Any = None) -> Any:
    try:
        res = requests.get(f"{backend_host}{endpoint}", timeout=2.5)
        if res.status_code == 200:
            return res.json()
        return default
    except Exception:
        return default


health_data = fetch_api_data("/health", {"status": "offline"})
latest_response = fetch_api_data("/api/latest", {"status": "error", "data": None})
history_records = fetch_api_data(f"/api/history?limit={history_limit}", [])
alerts_records = fetch_api_data("/api/alerts?limit=20", [])
model_info = fetch_api_data("/api/model/info", {})

is_backend_online = health_data.get("status") in ("ok", "degraded")
latest = latest_response.get("data") if latest_response else None

# ============================================================================
# Header & Connectivity Status
# ============================================================================
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("🌊 Flood Sentry: Operational Monitor")
    st.caption("Edge Telemetry Ingestion • Rolling Hydrological Features • Deep Neural Network Forecasts")

with col_head2:
    if is_backend_online:
        db_count = health_data.get("telemetry_records_count", 0)
        st.success(f"● Backend Connected ({db_count} records)")
    else:
        st.error("● Backend Offline (Check localhost:8000)")

if not is_backend_online:
    st.warning("⚠️ Cannot communicate with FastAPI backend at `" + backend_host + "`. Please ensure the backend is started.")
    st.stop()


# ============================================================================
# Live Status Banner & Key Indicators
# ============================================================================
if latest:
    risk_level = latest.get("predicted_risk", "Low")
    risk_code = latest.get("risk_code", 0)
    conf = latest.get("confidence_score", 0.0)
    model_name = latest.get("model_used", "DNN")
    is_fallback = bool(latest.get("is_fallback", 0))
    source = latest.get("data_source", "simulation")
    ts_str = latest.get("timestamp", "")

    # Calculate freshness
    try:
        reading_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        age_sec = (datetime.now(timezone.utc) - reading_dt).total_seconds()
        freshness_label = f"{int(age_sec)}s ago" if age_sec < 120 else f"{int(age_sec/60)}m ago"
    except Exception:
        freshness_label = ts_str

    banner_class = "status-banner-high" if risk_code == 2 else "status-banner-medium" if risk_code == 1 else "status-banner-low"
    badge_class = "badge-high" if risk_code == 2 else "badge-medium" if risk_code == 1 else "badge-low"

    st.markdown(f"""
    <div class="status-banner {banner_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span class="{badge_class}">RISK STATUS: {risk_level.upper()}</span>
                <span style="margin-left: 14px; font-weight: 600; color: #cbd5e1;">
                    Forecast Horizon: +3 Hours Ahead | Confidence: {conf*100:.1f}%
                </span>
            </div>
            <div style="text-align: right; color: #94a3b8; font-size: 0.85rem;">
                Data Source: <code>{source}</code> | Freshness: <strong>{freshness_label}</strong>
            </div>
        </div>
        <div style="margin-top: 10px; color: #e2e8f0; font-size: 0.95rem;">
            <strong>Decision Rationale:</strong> Modeled via <em>{model_name}</em>{" (Fallback Engine Active)" if is_fallback else ""}. 
            Water level at {latest.get('water_level_pct', 0):.1f}% capacity, 6h rainfall accumulation {latest.get('rainfall_6h_mm', 0):.1f}mm, 
            hourly rate of change {latest.get('water_level_change_cm_h', 0):+.1f}cm/h.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("ℹ️ No telemetry recorded yet. Click 'Inject Selected Scenario' in the sidebar or run the Wokwi simulation.")

# ============================================================================
# KPI Metrics Row
# ============================================================================
c1, c2, c3, c4, c5 = st.columns(5)

water_pct = latest.get("water_level_pct", 0.0) if latest else 0.0
water_cm = latest.get("water_level_cm", 0.0) if latest else 0.0
rain_rate = latest.get("rainfall_mm_h", 0.0) if latest else 0.0
r6_mm = latest.get("rainfall_6h_mm", 0.0) if latest else 0.0
soil_pct = latest.get("soil_saturation_pct", 0.0) if latest else 0.0
flow_pct = latest.get("flow_pct", 50.0) if latest else 50.0

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Water Level</div>
        <div class="metric-value">{water_pct:.1f}%</div>
        <div class="metric-sub">{water_cm:.1f} cm / 100 cm capacity</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Rainfall Rate</div>
        <div class="metric-value">{rain_rate:.1f} mm/h</div>
        <div class="metric-sub">Instantaneous intensity</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">6h Accumulated Rain</div>
        <div class="metric-value">{r6_mm:.1f} mm</div>
        <div class="metric-sub">Temporal rolling accumulation</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Soil Saturation</div>
        <div class="metric-value">{soil_pct:.1f}%</div>
        <div class="metric-sub">Estimated infiltration proxy</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Drainage Runoff Flow</div>
        <div class="metric-value">{flow_pct:.1f}%</div>
        <div class="metric-sub">Channel outflow capacity</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Dynamic Multi-Param Time Series Charts
# ============================================================================
st.subheader("📈 Real-Time Sensor & Hydrological Trends")

if history_records:
    df_hist = pd.DataFrame(history_records)
    df_hist["datetime"] = pd.to_datetime(df_hist["timestamp"])
    df_chart = df_hist.sort_values("datetime").reset_index(drop=True)

    tab_overview, tab_rainfall, tab_risk_timeline = st.tabs([
        "🌊 Water Level & Channel Thresholds",
        "🌧️ Precipitation Dynamics (1h, 3h, 6h, 24h)",
        "📊 Risk & Probability History",
    ])

    with tab_overview:
        col_g1, col_g2 = st.columns([3, 1])
        with col_g1:
            st.line_chart(
                df_chart.set_index("datetime")[["water_level_pct", "water_level_cm"]],
                height=280,
            )
        with col_g2:
            st.metric("Rate of Level Change", f"{latest.get('water_level_change_cm_h', 0):+.1f} cm/h" if latest else "0 cm/h")
            st.metric("Ultrasonic Distance", f"{latest.get('distance_cm', 0):.1f} cm" if latest else "N/A")
            st.caption("Channel baseline depth: 100.0 cm. Sensor mounted at overhead zenith.")

    with tab_rainfall:
        st.area_chart(
            df_chart.set_index("datetime")[["rainfall_mm_h", "rainfall_3h_mm", "rainfall_6h_mm"]],
            height=280,
        )

    with tab_risk_timeline:
        st.line_chart(
            df_chart.set_index("datetime")[["risk_code", "confidence_score"]],
            height=280,
        )
        st.caption("Risk Code: 0 = Low Risk (Normal), 1 = Medium Risk (Watch), 2 = High Risk (Danger).")
else:
    st.info("No historical readings available yet to render charts.")


# ============================================================================
# Alert Log & ML Benchmarks
# ============================================================================
col_bottom1, col_bottom2 = st.columns([3, 2])

with col_bottom1:
    st.subheader("🚨 Alert & State-Transition Log")
    if alerts_records:
        df_alerts = pd.DataFrame(alerts_records)
        df_display = df_alerts[["timestamp", "severity", "trigger_reason", "acknowledged"]].copy()
        df_display["acknowledged"] = df_display["acknowledged"].apply(lambda x: "✅ Acknowledged" if x else "🔔 ACTIVE")
        st.dataframe(df_display, use_container_width=True, height=220)
    else:
        st.caption("No alerts logged. System is operating within normal parameters.")

with col_bottom2:
    st.subheader("🧠 Model Suite & Benchmark Summary")
    if model_info and "model_comparison" in model_info:
        comp = model_info["model_comparison"]
        rows = []
        for name, metrics in comp.items():
            rows.append({
                "Model Architecture": name,
                "Accuracy": f"{metrics['accuracy']*100:.1f}%",
                "Macro-F1": f"{metrics['macro_f1']:.3f}",
                "High-Risk Recall": f"{metrics['high_risk_recall']*100:.1f}%",
                "FN Rate (Danger)": f"{metrics['high_risk_fn_rate']*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=220)
    else:
        st.caption("Comparative benchmark metrics available in artifacts/model_comparison_report.json.")

# Auto-refresh loop
if auto_refresh:
    time.sleep(3)
    st.rerun()
