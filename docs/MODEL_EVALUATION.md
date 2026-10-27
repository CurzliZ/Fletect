# MODEL BENCHMARKING & EVALUATION REPORT: FLOOD SENTRY

## 1. Experimental Methodology

All candidate models were trained and benchmarked on the identical chronological test partition ($N=17,940$ test records) using identical features ($D=11$).

### Forecasting Target
* **Target Variable:** `risk_label_encoded_3h_ahead`
* **Horizon:** 3 hours in the future
* **Classes:**
  - `0`: **Low Risk (Normal)** — Safe retention levels; no imminent flood threat.
  - `1`: **Medium Risk (Watch)** — Elevated water levels, heavy precipitation accumulation, or rapid rise rate.
  - `2`: **High Risk (Danger)** — Imminent channel overtopping, flash flood surge, or critical multi-hour deluge.

---

## 2. Comparative Benchmark Results

| Model Architecture | Test Accuracy | Macro-F1 | High-Risk Recall | High-Risk FN Rate | Latency (ms/sample) | Parameter Count |
|---|---|---|---|---|---|---|
| **Hydrological Rule Baseline** | 87.99% | 0.817 | 83.12% | 16.88% | 0.0005 ms | 0 (Static heuristics) |
| **XGBoost Classifier** | 92.70% | 0.902 | 93.72% | 6.28% | 0.0009 ms | 120 gradient boosted trees |
| **Deep Neural Network (DNN)** | 92.36% | 0.895 | 92.71% | 7.29% | 0.0315 ms | 4,147 parameters |
| **Sequential LSTM** | 92.67% | 0.900 | 93.18% | 6.82% | 0.0576 ms | 6,243 parameters |

---

## 3. Detailed Confusion Matrices & Per-Class Metrics

### A. Deep Neural Network (DNN) — Deployed Primary Model
```
Confusion Matrix (Rows = Actual, Columns = Predicted):
                Pred Low    Pred Medium    Pred High
Actual Low       11,953         514             0
Actual Medium       587       2,595           111
Actual High          16         143         2,021
```

* **Low Class:** Precision = 95.20%, Recall = 95.88%, F1 = 95.54% (Support = 12,467)
* **Medium Class:** Precision = 79.80%, Recall = 78.80%, F1 = 79.30% (Support = 3,293)
* **High Class:** Precision = 94.79%, Recall = 92.71%, F1 = 93.74% (Support = 2,180)

### B. XGBoost Classifier — Tabular Baseline
```
Confusion Matrix:
                Pred Low    Pred Medium    Pred High
Actual Low       11,964         503             0
Actual Medium       582       2,624            87
Actual High          15         122         2,043
```
* **High Class:** Precision = 95.92%, Recall = 93.72%, F1 = 94.80%

### C. Sequential Recurrent Network (LSTM)
* Evaluated over 6-hour sliding temporal tensors ($N_{test\_seq} = 17,840$).
* Accuracy = 92.67%, Macro-F1 = 0.900, High Risk Recall = 93.18%.

### D. Hydrological Rule-Based Baseline
```
Confusion Matrix:
                Pred Low    Pred Medium    Pred High
Actual Low       11,748         719             0
Actual Medium       706       2,226           361
Actual High          15         353         1,812
```
* Accuracy = 87.99%, High Risk Recall = 83.12%, High-Risk False Negative Rate = 16.88%.

---

## 4. High-Risk Safety & False-Negative Analysis

In early warning systems, a false negative on an imminent flood (predicting Low when Actual is High) is catastrophic:

* **Hydrological Rule Baseline:** Missed 368 high-risk events (16.88% failure rate).
* **Deep Neural Network (DNN):** Reduced missed high-risk events down to 159 (7.29% failure rate). Only 16 actual High cases were predicted as Low.
* **XGBoost Classifier:** Achieved the lowest high-risk false-negative rate at 6.28% (137 missed events).

---

## 5. Overfitting & Generalization Verification

* **Train vs. Validation Loss:** The DNN model trained with EarlyStopping (patience=6) converged at epoch 16 with a validation loss of 0.282 and training loss of 0.297, confirming that Dropout ($p=0.20$) and L2 regularization prevented overfitting on training catchments.
* **Inference Interface:** `model/inference.py` wraps model loading, feature scaling, and calibrated prediction, with deterministic failover to the rule engine if model files are unreadable.
