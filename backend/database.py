"""SQLite Database Persistence Layer for Flood Sentry.

Stores high-frequency sensor telemetry, computed rolling features, model inference results,
and alert transition events with full transaction safety.
"""
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "flood_sentry.db"


def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with Row factory enabled."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """Creates tables and indexes if they do not exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Telemetry records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                distance_cm REAL,
                water_level_cm REAL NOT NULL,
                water_level_pct REAL NOT NULL,
                rainfall_mm_h REAL NOT NULL,
                rainfall_3h_mm REAL NOT NULL,
                rainfall_6h_mm REAL NOT NULL,
                rainfall_24h_mm REAL NOT NULL,
                water_level_change_cm_h REAL NOT NULL,
                soil_saturation_pct REAL NOT NULL,
                flow_pct REAL DEFAULT 50.0,
                predicted_risk TEXT NOT NULL,
                risk_code INTEGER NOT NULL,
                confidence_score REAL NOT NULL,
                model_used TEXT NOT NULL,
                is_fallback INTEGER NOT NULL,
                data_source TEXT DEFAULT 'simulation'
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_device_time ON telemetry (device_id, timestamp);")

        # Alert transition events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                severity TEXT NOT NULL,
                trigger_reason TEXT NOT NULL,
                water_level_pct REAL NOT NULL,
                rainfall_6h_mm REAL NOT NULL,
                acknowledged INTEGER DEFAULT 0,
                acknowledged_at TEXT
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_device_time ON alerts (device_id, timestamp);")
        conn.commit()


def save_telemetry(record: Dict[str, Any]) -> int:
    """Inserts a telemetry and inference record."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO telemetry (
                device_id, timestamp, distance_cm, water_level_cm, water_level_pct,
                rainfall_mm_h, rainfall_3h_mm, rainfall_6h_mm, rainfall_24h_mm,
                water_level_change_cm_h, soil_saturation_pct, flow_pct,
                predicted_risk, risk_code, confidence_score, model_used,
                is_fallback, data_source
            ) VALUES (
                :device_id, :timestamp, :distance_cm, :water_level_cm, :water_level_pct,
                :rainfall_mm_h, :rainfall_3h_mm, :rainfall_6h_mm, :rainfall_24h_mm,
                :water_level_change_cm_h, :soil_saturation_pct, :flow_pct,
                :predicted_risk, :risk_code, :confidence_score, :model_used,
                :is_fallback, :data_source
            );
        """, record)
        conn.commit()
        return cursor.lastrowid


def save_alert(alert: Dict[str, Any]) -> int:
    """Inserts an alert event."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alerts (
                device_id, timestamp, severity, trigger_reason,
                water_level_pct, rainfall_6h_mm, acknowledged
            ) VALUES (
                :device_id, :timestamp, :severity, :trigger_reason,
                :water_level_pct, :rainfall_6h_mm, :acknowledged
            );
        """, alert)
        conn.commit()
        return cursor.lastrowid


def acknowledge_alerts(device_id: Optional[str] = None) -> int:
    """Marks pending alerts as acknowledged."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if device_id:
            cursor.execute(
                "UPDATE alerts SET acknowledged = 1, acknowledged_at = ? WHERE device_id = ? AND acknowledged = 0",
                (now_iso, device_id),
            )
        else:
            cursor.execute(
                "UPDATE alerts SET acknowledged = 1, acknowledged_at = ? WHERE acknowledged = 0",
                (now_iso,),
            )
        conn.commit()
        return cursor.rowcount


def get_latest_telemetry(device_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves the most recent telemetry record."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if device_id:
            cursor.execute(
                "SELECT * FROM telemetry WHERE device_id = ? ORDER BY id DESC LIMIT 1",
                (device_id,),
            )
        else:
            cursor.execute("SELECT * FROM telemetry ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return dict(row) if row else None


def get_telemetry_history(
    device_id: Optional[str] = None, limit: int = 100
) -> List[Dict[str, Any]]:
    """Retrieves chronological telemetry history up to limit."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if device_id:
            cursor.execute(
                "SELECT * FROM telemetry WHERE device_id = ? ORDER BY id DESC LIMIT ?",
                (device_id, limit),
            )
        else:
            cursor.execute("SELECT * FROM telemetry ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        # Return in ascending order for smooth charting
        return [dict(r) for r in reversed(rows)]


def get_recent_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent alerts ordered latest first."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in cursor.fetchall()]


# Initialize DB on module import
init_database()
