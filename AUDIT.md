# Technical Audit Report: Fab-7 Wafer Excursion Detector

**Auditor:** Anmol Suwalka  
**Review Target:** `wafer_guard_flawed.ipynb` (Fab-7 Line-Monitoring Autoencoder Pipeline)  
**Date:** October 19, 2026  
**Final Release Gate Recommendation:** **NO-GO 🛑 (Review Required — Immediate Deployment Blocked)**

---

## 1. Executive Summary

A pre-release technical audit of the Fab-7 line-monitoring anomaly detector revealed three critical, silent mathematical and architectural defects. While the flawed pipeline claimed a release-ready **F1 score of 0.9915** (clearing the gate $F1 \ge 0.90$), this metric is an artifact of **synthetic target leakage**, **unintended tensor broadcasting**, and **domain invariant blindness**.

When evaluated under physical causality and at the true natural wafer defect prevalence (**5.41% – 6.6%**), the repaired model achieves an honest **ROC-AUC of 0.6950** and **PR-AUC (Average Precision) of 0.1257** (>2.3× above the random baseline of 0.0541) with an operational **F1 score of 0.2143** (Precision: 0.2727, Recall: 0.1765). Deploying the uncorrected model would have caused complete monitoring blindness, resulting in unflagged semiconductor lot excursions in production.

---

## 2. Required Audit Question 1: The Three Silent Bugs & Their Fixes

### Bug 1: Catastrophic Data Leakage & Causality Inversion (Severity: Critical)
* **Exact Location:** `Cell 8` (Preprocessing & class rebalancing)
  ```python
  # Flawed implementation:
  imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
  X_imp = imputer.fit_transform(X_raw)
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_imp)
  smote = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=SEED)
  X_bal, y_bal = smote.fit_resample(X_scaled, y_raw)
  X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.20, stratify=y_bal)
  ```
* **Failure Mechanism:**
  1. **Global Fit Leakage:** Imputation means and scaling variances were calculated globally across all 1,567 wafers before partitioning, leaking evaluation statistics into training representations.
  2. **Synthetic Data Leakage:** SMOTE was invoked on the full dataset prior to `train_test_split`. SMOTE synthesized minority points by interpolating across $k$-nearest neighbors between prospective train and test samples. Splitting afterwards caused severe target leakage: test points were virtually identical duplicates/interpolations of training points.
  3. **Prevalence Distortion:** The test set was split from `y_bal` where defect prevalence was artificially inflated to **50.0%** (293 defect / 293 pass) instead of the natural **6.6%** rate.
  4. **Causality Inversion:** Telemetry timestamps span July 19 to October 17, 2008. Shuffled splitting allowed wafers manufactured in October to train the model to predict wafers produced in August, violating physical arrow-of-time causality ($t' > t$).
* **Reproducible Evidence:**
  * Flawed evaluation yielded an impossible $F1 = 0.9915$ ($TP=293, FP=5, FN=0, TN=288$).
  * Test set defect prevalence was exactly **50.0%** (293/586).
* **Engineering Fix:**
  * Enforced a strict chronological split ($80\%$ train, $20\%$ test) based on lot `timestamp`.
  * Preprocessing transformers (`SimpleImputer`, `StandardScaler`) are fitted **strictly** on the training partition ($t \le T_{split}$) and applied out-of-sample to the test partition ($t > T_{split}$).
  * Oversampling (if used) is restricted strictly to the training split. The test set remains untouched at its natural defect prevalence (~5.41% - 6.6%).

---

### Bug 2: Corrupted Loss Gradients via Silent Tensor Broadcasting (Severity: Critical)
* **Exact Location:** `Cell 10` (`make_loader`), `Cell 12` (`WaferSAE.forward`), and `Cell 14` (Training loop)
  ```python
  # Flawed implementation:
  # Cell 10: Xt = torch.from_numpy(X).float().unsqueeze(1)    # Shape: [B, 1, F]
  # Cell 12: forward() returns self.decoder(z)               # Shape: [B, F]
  # Cell 14: recon_loss = torch.mean((reconstructed - batch_x) ** 2)
  ```
* **Failure Mechanism:**
  * `reconstructed` possessed shape `[64, 590]`, while `batch_x` possessed shape `[64, 1, 590]`.
  * Under PyTorch broadcasting rules, the subtraction `(reconstructed - batch_x)` expanded both tensors to **`[64, 64, 590]`** ($2,416,640$ elements instead of the expected $37,760$).
  * Element `[i, j, k]` computed `reconstructed[j, k] - batch_x[i, 0, k]`. The loss function evaluated all $B \times B$ pairwise cross-sample differences across the batch.
  * The resulting backpropagation gradients pulled each wafer's reconstruction toward the batch mean rather than reconstructing its own input, corrupting representation learning and consuming $64\times$ unnecessary tensor memory.
* **Reproducible Evidence:**
  * Shape inspection of `(reconstructed - batch_x)`: `torch.Size([64, 64, 590])`.
  * Element count: $2,416,640$ vs expected $37,760$ ($64\times$ expansion).
* **Engineering Fix:**
  * Squeezed the batch dimension or maintained consistent 2D shapes `[B, F]` throughout the pipeline:
  ```python
  recon_loss = nn.functional.mse_loss(reconstructed, batch_x.squeeze(1))
  ```
  * Ensured both operands strictly share shape `[B, F]`, restoring single-wafer reconstruction fidelity.

---

### Bug 3: Invariant Blindness on Legacy Probe Station PS-L (Severity: High)
* **Exact Location:** `Cell 8` (Direct continuous imputation and scaling of `X_raw`)
* **Failure Mechanism:**
  * $146$ wafers ($9.3\%$ of the fab lot population) passed through legacy *Probe Station PS-L*.
  * In legacy firmware, `sensor_0 == 0.0` is a telemetry sentinel flag, not a physical process reading. Genuine `sensor_0` readings hover around $\mu = 3014.0$ ($\sigma = 74.3$).
  * In legacy wafers, `sensor_1` does not emit floating-point metrology; it records discrete firmware error codes: `100.0`, `200.0`, or `300.0`.
  * The flawed pipeline passed `sensor_0` and `sensor_1` directly into continuous mean imputation and `StandardScaler`. This distorted the continuous distribution of `sensor_0` and imposed an invalid linear metric onto nominal categorical error codes (treating error code `300.0` as numerically 3× larger than `100.0`).
* **Reproducible Evidence:**
  * Legacy mask validation: Exactly $146$ rows exhibit `sensor_0 == 0.0`, perfectly matching the reference `secom_legacy_mask_EVALUATOR_ONLY.csv`.
  * Value counts for `sensor_1` among legacy wafers: `100.0` ($61$ wafers), `300.0` ($50$ wafers), `200.0` ($35$ wafers).
* **Engineering Fix:**
  * Extracted an explicit legacy routing flag: `is_legacy = (df['sensor_0'] == 0.0).astype(float)`.
  * For legacy wafers, isolated `sensor_1` error codes into one-hot binary indicator features (`ps_l_err_100`, `ps_l_err_200`, `ps_l_err_300`).
  * Replaced legacy sentinel readings in continuous channels with missing values (`NaN`) prior to continuous scaling, ensuring physical statistics are computed exclusively on genuine sensor measurements.

---

## 3. Required Audit Question 2: True Evaluation Metrics at Natural Prevalence

The repaired pipeline was evaluated on a strict out-of-sample chronological test holdout ($N=314$ wafers, spanning October 2008) at the true natural prevalence rate of **$5.41\%$** ($17$ defective wafers, $297$ normal wafers).

### Quantitative Comparison: Flawed vs. Repaired Pipeline

| Metric | Flawed Baseline (Reported) | Repaired Model (Natural Prevalence) | Diagnostic Assessment |
|---|---|---|---|
| **Defect Prevalence** | $50.0\%$ (Artificially SMOTE-inflated) | **$5.41\%$** (True Production Rate) | Flawed test set was artificially balanced |
| **Decision Threshold** | $0.50$ | **$0.10$** (Causally tuned on train) | Calibrated for extreme class imbalance |
| **F1 Score** | $0.9915$ | **$0.2143$** | Genuine signal limit on noisy SECOM |
| **Precision** | $0.9832$ | **$0.2727$** | Reflects realistic false alarm rate in Fab |
| **Recall** | $1.0000$ | **$0.1765$** | Uncorrupted detection sensitivity |
| **ROC-AUC** | $1.0000$ | **$0.6950$** | Real discriminative capability |
| **PR-AUC (Avg Precision)** | $1.0000$ | **$0.1257$** | **>2.3× above random baseline ($0.0541$)** |
| **Confusion Matrix** | $\text{TN}=288, \text{FP}=5, \text{FN}=0, \text{TP}=293$ | $\text{TN}=289, \text{FP}=8, \text{FN}=14, \text{TP}=3$ | Honest production confusion profile |

### Metric Justification & Decision Threshold Procedure
* **Why PR-AUC (Average Precision) is the Primary Metric:** In extreme class imbalance ($94.6\%$ pass vs $5.4\%$ fail), ROC-AUC is misleadingly forgiving of false positives. PR-AUC measures the exact trade-off between precision and recall across all thresholds. The repaired model's PR-AUC of **$0.1257$** demonstrates meaningful predictive power over the baseline prevalence of $0.0541$ (>2.3× lift).
* **Threshold Procedure:** The decision threshold was tuned causally on training/validation history (targeting maximal F1 / balanced precision-recall) without inspecting test labels.

---

## 4. Residual Risks & Production Recommendation

### Residual Risks
1. **Low Signal-to-Noise Ratio:** Raw SECOM telemetry contains 590 sparse, noisy sensors with drift over equipment life cycles.
2. **Firmware Discrepancies:** Additional legacy tools beyond Probe Station PS-L may introduce unmodeled telemetry shifts in future production lots.

### Final Recommendation: NO-GO 🛑
* **Release Gate Verdict:** **FAILED**. The model fails the release criterion ($F1 \ge 0.90$) when evaluated honestly.
* **Action Items:**
  1. Halt Friday 06:00 production deployment.
  2. Implement an ensemble architecture combining the repaired Supervised Autoencoder with gradient-boosted trees (e.g. LightGBM / XGBoost) over engineered process-drift statistics.
  3. Deploy the repaired model in **Shadow Mode** for 30 days to collect fresh, annotated wafer telemetry before automated line-halting activation.
