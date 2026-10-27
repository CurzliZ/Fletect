# DATASET PROVENANCE & METHODOLOGY: FLOOD SENTRY

## 1. Overview and Provenance

* **Dataset Filename:** `flood_risk_dnn_synthetic_120k.csv`
* **File Size:** ~21.4 MB (120,000 records)
* **Sites:** 20 independent simulated drainage catchments / stations (`SIM_01` to `SIM_20`)
* **Temporal Resolution:** 1-hour uniform steps (6,000 consecutive hourly observations per site)
* **Prediction Horizon:** 3 hours ahead (`risk_label_encoded_3h_ahead`)

### Provenance Distinction: Empirical vs. Synthetic
To satisfy rigorous scientific standards and non-negotiable directive rules:
* **Empirical Foundation:** The rainfall magnitude distribution pool was sampled from telemetry records of the Central Water Commission (CWC), Tamil Nadu region (2021–2025). Finite observations between 0 and 100 mm were deduplicated by median station/timestamp to construct the empirical rainfall pool.
* **Synthetic Generation:** All timestamps, site identifiers, precipitation temporal sequences, river stage / channel water levels, rates of water level change, soil saturation proxies, and risk labels were generated algorithmically via `generate_flood_dnn_dataset.py`.
* **Important Disclaimer:** These risk labels are illustrative decision-support proxies, not certified government flood warning declarations. Performance metrics against this dataset validate that models learn complex hydrological relationships; they do not constitute proof of field reliability in an uncalibrated physical river basin.

---

## 2. Partitioning & Strict Anti-Leakage Protocol

The dataset is partitioned chronologically within each simulated site:

| Partition | Time Interval | Row Count | Percentage | Purpose |
|---|---|---|---|---|
| **Train** | Steps 0 to 4,199 | 83,940 | 70% | Model parameter optimization & Scaler fitting |
| **Validation** | Steps 4,200 to 5,099 | 17,940 | 15% | Hyperparameter tuning & Early Stopping |
| **Test** | Steps 5,100 to 5,999 | 17,940 | 15% | Final unbiased benchmark evaluation |

### Anti-Leakage Rules Enforced:
1. **Chronological Integrity:** No random shuffling across time steps.
2. **Boundary Neutralization:** The final 3 hours of each temporal partition have their future target (`risk_label_encoded_3h_ahead`) set to blank/NaN and are excluded from training and evaluation, preventing target values from peeking across partition boundaries.
3. **Scaler Fitting:** `StandardScaler` is fitted strictly on the `train` partition ($N=83,940$) and transforms `validation` and `test` data without refitting.

---

## 3. Data Schema and Feature Definitions

| Column Name | Type | Unit | Description | Role in ML |
|---|---|---|---|---|
| `rainfall_mm_h` | float | mm/h | Instantaneous rainfall intensity | Input Feature |
| `rainfall_3h_mm` | float | mm | Rolling sum of current and prior 2 hours of rainfall | Input Feature |
| `rainfall_6h_mm` | float | mm | Rolling sum of current and prior 5 hours of rainfall | Input Feature |
| `rainfall_24h_mm` | float | mm | Rolling sum of current and prior 23 hours of rainfall | Input Feature |
| `water_level_cm` | float | cm | Channel water level depth | Input Feature |
| `water_level_pct` | float | % | Water depth as percentage of channel capacity | Input Feature |
| `water_level_change_cm_h` | float | cm/h | Hourly rate of water level rise or fall | Input Feature |
| `soil_saturation_pct` | float | % | Estimated infiltration saturation proxy | Input Feature |
| `channel_capacity_cm` | float | cm | Physical depth capacity of the channel | Input Feature |
| `hour_of_day` | int | 0-23 | Diurnal temporal proxy | Input Feature |
| `month` | int | 1-12 | Seasonal monsoon cycle proxy | Input Feature |
| `risk_label_encoded_3h_ahead` | int | 0, 1, 2 | Target: 0 = Low, 1 = Medium (Watch), 2 = High (Danger) | **Prediction Target** |

---

## 4. Class Distribution & Imbalance

Across the 17,940 test samples:
* **Low Risk (0):** 12,467 samples (69.5%)
* **Medium Risk (1):** 3,293 samples (18.4%)
* **High Risk (2):** 2,180 samples (12.1%)

Class weights computed via scikit-learn (`compute_class_weight="balanced"`) are applied during DNN and XGBoost training to heavily penalize false negatives on dangerous high-risk flood events.
