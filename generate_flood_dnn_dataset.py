#!/usr/bin/env python3
"""Generate a reproducible synthetic flood-risk dataset calibrated to a CWC rainfall file.

The output is synthetic. The input rainfall records are used only as an empirical
rainfall-magnitude reference pool; timestamps, temporal rainfall sequences,
water levels, soil saturation, and target labels are simulated.

Usage:
  python generate_flood_dnn_dataset.py --input rainfall_tel_hr_cwc_tn_2021_2025.csv \
      --output flood_risk_dnn_synthetic_120k.csv --rows 120000

Dependency-free: Python standard library only.
"""
import argparse
import csv
import datetime as dt
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

RAIN_COL = "Telemetry Hourly Rainfall (mm)"
TIME_COL = "Data Acquisition Time"
STATION_COL = "Station"
TIME_FORMATS = ("%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")


def parse_time(value):
    for fmt in TIME_FORMATS:
        try:
            return dt.datetime.strptime(value.strip(), fmt)
        except (ValueError, AttributeError):
            pass
    return None


def clip(value, low, high):
    return max(low, min(high, value))


def load_rainfall_reference(path):
    """Return one median value per station/timestamp; exclude missing and >100 mm values."""
    grouped = defaultdict(list)
    audit = Counter()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or RAIN_COL not in reader.fieldnames:
            raise ValueError(f"Expected column {RAIN_COL!r} was not found in {path}")
        for row in reader:
            audit["input_rows"] += 1
            station = (row.get(STATION_COL) or "unknown").strip() or "unknown"
            timestamp = parse_time(row.get(TIME_COL, ""))
            raw = (row.get(RAIN_COL) or "").strip()
            try:
                value = float(raw)
            except (ValueError, TypeError):
                audit["missing_or_unparseable_rainfall"] += 1
                continue
            if not math.isfinite(value):
                audit["missing_or_unparseable_rainfall"] += 1
                continue
            audit["finite_rainfall_rows"] += 1
            if value < 0:
                audit["negative_rainfall_rows"] += 1
                continue
            if value > 100:
                audit["rainfall_over_100_mm_flagged"] += 1
                continue
            if timestamp is None:
                audit["bad_timestamp_rows"] += 1
                continue
            grouped[(station, timestamp)].append(value)
    pool = [statistics.median(vs) for vs in grouped.values() if vs]
    pool = [v for v in pool if 0 < v <= 100]
    if len(pool) < 100:
        raise ValueError("Too few usable positive rainfall reference values to build a dataset.")
    audit["unique_station_timestamp_groups"] = len(grouped)
    audit["rainfall_reference_pool_size"] = len(pool)
    return pool, audit


def season_parameters(month, site_factor):
    # Deliberately event-rich synthetic seasonality for ML training, not a forecast.
    if month in (10, 11, 12):
        start_p, persist_p = 0.090, 0.82
    elif month in (6, 7, 8, 9):
        start_p, persist_p = 0.058, 0.76
    elif month in (4, 5):
        start_p, persist_p = 0.028, 0.66
    else:
        start_p, persist_p = 0.016, 0.60
    return clip(start_p * site_factor, 0.004, 0.14), persist_p


def classify_risk(level_pct, six_hour_rain, delta_cm_h, soil_pct):
    # Simulated proxy policy for creating a demonstrator target; not an official flood standard.
    if level_pct >= 76 or (level_pct >= 64 and delta_cm_h >= 4.0) or (
        level_pct >= 53 and six_hour_rain >= 70 and delta_cm_h >= 1.5
    ):
        return 2, "High", "water level near simulated capacity or rapid rise under heavy rain"
    if level_pct >= 43 or (level_pct >= 34 and six_hour_rain >= 25) or (
        level_pct >= 33 and delta_cm_h >= 2.5
    ) or (level_pct >= 36 and soil_pct >= 82):
        return 1, "Medium", "elevated simulated water level, rainfall accumulation, or rising trend"
    return 0, "Low", "simulated level and recent rainfall below configured warning thresholds"


def generate(input_path, output_path, rows=120000, seed=20261021, site_count=20):
    if rows <= 0 or rows % site_count != 0:
        raise ValueError(f"--rows must be a positive multiple of {site_count} (the number of sites).")
    steps = rows // site_count
    rng = random.Random(seed)
    rain_pool, audit = load_rainfall_reference(input_path)
    # Empirical bins are deliberately reweighted during simulation so the generated data
    # includes enough moderate/severe storm examples to train all three classes.
    rainfall_bins = {
        "light": [v for v in rain_pool if 0 < v <= 1],
        "moderate": [v for v in rain_pool if 1 < v <= 10],
        "heavy": [v for v in rain_pool if 10 < v <= 50],
        "very_heavy": [v for v in rain_pool if 50 < v <= 100],
    }
    rainfall_bins = {k: values for k, values in rainfall_bins.items() if values}

    # Randomized but fixed start dates distribute sequences across multiple years/seasons.
    first_day = dt.datetime(2021, 1, 1, 0, 0)
    last_start_day = dt.datetime(2024, 1, 1, 0, 0)
    days_span = (last_start_day - first_day).days
    out_fields = [
        "record_id", "timestamp", "site_id", "hour_of_day", "month",
        "rainfall_mm_h", "rainfall_3h_mm", "rainfall_6h_mm", "rainfall_24h_mm",
        "water_level_cm", "water_level_pct", "water_level_change_cm_h",
        "soil_saturation_pct", "channel_capacity_cm", "risk_label", "risk_label_encoded",
        "risk_label_3h_ahead", "risk_label_encoded_3h_ahead", "prediction_horizon_hours",
        "risk_reason", "dataset_split", "is_synthetic"
    ]

    global_counts = Counter()
    split_counts = Counter()
    rain_values = []
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        record_id = 0
        for s in range(site_count):
            site_id = f"SIM_{s+1:02d}"
            start = first_day + dt.timedelta(days=rng.randint(0, days_span), hours=rng.randint(0, 23))
            capacity = rng.randint(140, 320)
            base_level = rng.uniform(0.16, 0.34)
            sensitivity = rng.uniform(0.0085, 0.0140)
            site_rain_factor = rng.uniform(0.72, 1.42)
            dry_drain = rng.uniform(0.87, 0.93)
            soil_pct = rng.uniform(15, 45)
            runoff_memory = rng.uniform(0, 4)
            water_pct_prev = clip(base_level + sensitivity * runoff_memory + 0.0007 * soil_pct, 0.08, 0.95)
            prev_level_cm = water_pct_prev * capacity
            was_wet = False
            storm_class = "light"
            event_multiplier = 1.0
            rainfall_history = []

            site_rows = []
            for t in range(steps):
                timestamp = start + dt.timedelta(hours=t)
                p_start, p_persist = season_parameters(timestamp.month, site_rain_factor)
                if was_wet:
                    was_wet = (rng.random() < p_persist)
                    if not was_wet:
                        event_multiplier = 1.0
                else:
                    was_wet = (rng.random() < p_start)
                    if was_wet:
                        # Storm strength persists for the event, producing coherent sequences.
                        storm_class = rng.choices(
                            ["light", "moderate", "heavy", "very_heavy"],
                            weights=[0.34, 0.40, 0.21, 0.05], k=1
                        )[0]
                        event_multiplier = rng.choices([0.8, 1.0, 1.25, 1.6], weights=[0.15, 0.55, 0.25, 0.05], k=1)[0]

                if was_wet:
                    bin_values = rainfall_bins.get(storm_class) or rain_pool
                    base_rain = rng.choice(bin_values)
                    # Synthetic augmentation; this is not a copied timestamped observation.
                    rain = round(clip(base_rain * event_multiplier, 0.1, 100.0), 2)
                else:
                    rain = 0.0
                rainfall_history.append(rain)
                if len(rainfall_history) > 24:
                    rainfall_history.pop(0)
                r3 = round(sum(rainfall_history[-3:]), 2)
                r6 = round(sum(rainfall_history[-6:]), 2)
                r24 = round(sum(rainfall_history), 2)

                soil_pct = clip(0.985 * soil_pct + 0.62 * min(rain, 50.0) - 0.48, 0.0, 100.0)
                runoff_memory = clip(dry_drain * runoff_memory + rain, 0.0, 320.0)
                # First estimate the level implied by recent runoff, then limit hourly movement
                # to avoid implausible instantaneous jumps in the simulated sensor.
                target_pct = clip(base_level + sensitivity * runoff_memory + 0.0010 * soil_pct, 0.05, 0.99)
                level_step = clip(0.55 * (target_pct - water_pct_prev) + rng.gauss(0.0, 0.0025), -0.035, 0.065)
                water_pct = clip(water_pct_prev + level_step, 0.05, 0.99)
                level_cm = round(water_pct * capacity, 2)
                delta = round(level_cm - prev_level_cm, 2)
                level_pct = round(water_pct * 100.0, 2)
                label_id, label, reason = classify_risk(level_pct, r6, delta, soil_pct)

                if t < int(steps * 0.70):
                    split = "train"
                elif t < int(steps * 0.85):
                    split = "validation"
                else:
                    split = "test"

                record_id += 1
                global_counts[label] += 1
                split_counts[split] += 1
                rain_values.append(rain)
                site_rows.append({
                    "record_id": record_id,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "site_id": site_id,
                    "hour_of_day": timestamp.hour,
                    "month": timestamp.month,
                    "rainfall_mm_h": rain,
                    "rainfall_3h_mm": r3,
                    "rainfall_6h_mm": r6,
                    "rainfall_24h_mm": r24,
                    "water_level_cm": level_cm,
                    "water_level_pct": level_pct,
                    "water_level_change_cm_h": delta,
                    "soil_saturation_pct": round(soil_pct, 2),
                    "channel_capacity_cm": capacity,
                    "risk_label": label,
                    "risk_label_encoded": label_id,
                    "risk_label_3h_ahead": "",
                    "risk_label_encoded_3h_ahead": "",
                    "prediction_horizon_hours": 3,
                    "risk_reason": reason,
                    "dataset_split": split,
                    "is_synthetic": 1,
                })
                prev_level_cm = level_cm
                water_pct_prev = water_pct

            # Predict risk three hours ahead. Blank the final horizon rows at every split
            # boundary to prevent labels from looking into the next split.
            for i, current in enumerate(site_rows):
                future_i = i + 3
                if future_i < len(site_rows) and site_rows[future_i]["dataset_split"] == current["dataset_split"]:
                    future = site_rows[future_i]
                    current["risk_label_3h_ahead"] = future["risk_label"]
                    current["risk_label_encoded_3h_ahead"] = future["risk_label_encoded"]
            writer.writerows(site_rows)

    summary = {
        "output_file": str(out_path),
        "rows": rows,
        "sites": site_count,
        "hours_per_site": steps,
        "seed": seed,
        "split_counts": dict(split_counts),
        "class_counts": dict(global_counts),
        "rainfall_summary_synthetic": {
            "min_mm_h": min(rain_values),
            "max_mm_h": max(rain_values),
            "wet_hour_pct": round(100.0 * sum(v > 0 for v in rain_values) / len(rain_values), 2),
        },
        "input_audit": dict(audit),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the supplied CWC rainfall CSV")
    parser.add_argument("--output", default="flood_risk_dnn_synthetic_120k.csv")
    parser.add_argument("--rows", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--sites", type=int, default=20)
    args = parser.parse_args()
    summary = generate(args.input, args.output, args.rows, args.seed, args.sites)
    print(summary)

if __name__ == "__main__":
    main()
