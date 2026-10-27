# IoT-Based Flood Risk Detection using DNN — Dataset Package

## Files

- `flood_risk_dnn_synthetic_120k.csv` — 120,000 rows for model development.
- `generate_flood_dnn_dataset.py` — reproducible, standard-library-only dataset generator.
- `train_flood_dnn.py` — optional TensorFlow/Keras training and evaluation starter.
- `requirements.txt` — Python libraries used by the starter training script.

## Important: what is real and what is synthetic?

The supplied file `rainfall_tel_hr_cwc_tn_2021_2025.csv` is used only as a **reference distribution for rainfall magnitudes**. Valid finite readings between 0 and 100 mm are used as the empirical pool. The generator creates new storm sequences and timestamps; it does not claim that generated rows are historical observations.

The following fields are synthetic:

- timestamps and site identities (`SIM_01` to `SIM_20`)
- rainfall timing, dry periods, event persistence, and storm augmentation
- water level, channel capacity, rate of water-level change, and soil saturation
- `risk_label`, which is created by an illustrative rule-based policy for the prototype
- `risk_label_3h_ahead` / `risk_label_encoded_3h_ahead`, created by shifting the simulated risk label three hours forward within each timeline

The generated risk labels are **not official flood labels**, and thresholds are not universal hydrological warning thresholds. DNN performance measured against this dataset is evidence only that the model can learn these simulated patterns. It is not proof of real-world flood-warning accuracy. For real-world validation, obtain matched water-level/time-series records and verified flood-event or warning labels from a reliable source.

## Source-file quality audit

The uploaded rainfall CSV contains 81,008 rows. Inspection found:

- 55,272 finite numeric rainfall entries.
- 25,736 missing/NaN/unparseable rainfall entries (31.77%).
- 2,728 finite rainfall values above 100 mm in the stated hourly-rainfall column; these were flagged and excluded from the magnitude reference pool pending source/unit verification. The maximum recorded value was 1,013 mm.
- 3,310 extra rows sharing a station/timestamp combination with another row.
- Location metadata concerns: some records have coordinates around 25–26 degrees north while the `State` field says Tamil Nadu and some `District` entries say Chennai. Validate the source metadata before using location or district as a real feature.

The synthetic generator deduplicates station/timestamp observations to a median value when building its empirical pool, excludes missing/non-finite/negative values and excludes values above 100 mm from the empirical pool. Values sampled from this pool are used to simulate new timestamps/sequences and may be multiplied/capped as synthetic augmentation.

## Size and split

- 120,000 rows total
- 20 independent simulated sites, 6,000 hourly time steps per site
- Train: 84,000 rows (first 70% of each site's timeline)
- Validation: 18,000 rows (next 15%)
- Test: 18,000 rows (final 15%)
- Current-state class counts: Low 88,568; Medium 19,406; High 12,026
- The three-hour-ahead target uses the same simulated classes shifted forward within each split; the training script excludes boundary rows without a valid future target.

The split is chronological within each simulated site. Keep `dataset_split` intact; do not randomly shuffle all rows and create another split, because temporal leakage can make evaluation overly optimistic.

## Columns and meanings

| Column | Meaning | Use in DNN? |
|---|---|---|
| `record_id` | Unique row ID | No |
| `timestamp` | Simulated timestamp | No; use for ordering/plots |
| `site_id` | Simulated sensor/channel identity | Not in the provided numeric baseline |
| `hour_of_day` | Simulated hour | Yes |
| `month` | Simulated month | Yes |
| `rainfall_mm_h` | Simulated current rainfall intensity (mm/hour) | Yes |
| `rainfall_3h_mm` | Current plus previous 2 hours of rainfall | Yes |
| `rainfall_6h_mm` | Current plus previous 5 hours of rainfall | Yes |
| `rainfall_24h_mm` | Current plus previous 23 hours of rainfall | Yes |
| `water_level_cm` | Simulated current water height | Yes |
| `water_level_pct` | Simulated water level as percent of channel capacity | Yes |
| `water_level_change_cm_h` | Difference in water height from prior hourly sample | Yes |
| `soil_saturation_pct` | Simulated soil saturation proxy | Yes |
| `channel_capacity_cm` | Simulated channel depth/capacity | Yes |
| `risk_label` | Current simulated `Low`, `Medium`, or `High` class | Current-state label / dashboard |
| `risk_label_encoded` | Current class encoded as 0 = Low, 1 = Medium, 2 = High | No for the 3-hour forecast model |
| `risk_label_3h_ahead` | Simulated risk class three hours later | Human-readable forecast target |
| `risk_label_encoded_3h_ahead` | Three-hour-ahead target: 0 = Low, 1 = Medium, 2 = High | **Training target** |
| `prediction_horizon_hours` | Forecast horizon; always 3 | Metadata |
| `risk_reason` | Explanation produced by the current synthetic labeling policy | No; not a model input |
| `dataset_split` | `train`, `validation`, or `test` | Split selection only |
| `is_synthetic` | Always 1, to mark the data as simulated | No |

## Recommended DNN input columns

```python
FEATURES = [
    "rainfall_mm_h",
    "rainfall_3h_mm",
    "rainfall_6h_mm",
    "rainfall_24h_mm",
    "water_level_cm",
    "water_level_pct",
    "water_level_change_cm_h",
    "soil_saturation_pct",
    "channel_capacity_cm",
    "hour_of_day",
    "month",
]
TARGET = "risk_label_encoded_3h_ahead"
```

The model predicts the synthetic risk category three hours in the future using only features available at the current time. Do not include `risk_reason`, either current label field, or either future label field in `FEATURES`; that would leak the target into the model input. The last 3 records of each temporal split have blank forecast targets, because those target labels would cross a split boundary or fall beyond the sequence; the training script removes them.

## Run the training starter

Put the CSV and `train_flood_dnn.py` in the same folder, install dependencies, and run:

```bash
python -m pip install -r requirements.txt
python train_flood_dnn.py
```

It will create `artifacts/` with the Keras model, fitted scaler, and evaluation report for the 3-hour-ahead target. Report accuracy, precision, recall, F1 score, and the confusion matrix; do not report synthetic test metrics as real flood-prediction performance.

## Reproduce the dataset

With the original source CSV in the same folder:

```bash
python generate_flood_dnn_dataset.py --input rainfall_tel_hr_cwc_tn_2021_2025.csv --output flood_risk_dnn_synthetic_120k.csv --rows 120000 --sites 20 --seed 20261021
```

The default generation is deterministic for the same input file and seed.
