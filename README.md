# ICU Mortality Threshold Optimization (SOFA & APACHE II)

This repository contains code to **optimize classification thresholds** for ICU 90-day mortality prediction models based on **SOFA** and **APACHE II** scores using the **Youden J statistic**. The script supports both **repeated 80/20 internal splits** and **classic bootstrapping**, running **10,000 iterations per model** and returning **median probability thresholds** (and related summary metrics).

> **Note on validation:** This project uses **internal validation only** (repeated splits and bootstrap). The “independent test” refers to an **internal hold-out split**, *not* a separate external dataset.

---

## Key Features

- **Youden J optimization** for probabilistic classifiers  
- **Two resampling schemes:** repeated 80/20 splits and classic bootstrap  
- **High-replicate stability:** 10,000 runs per model to estimate **median optimal thresholds**  
- Designed for **SOFA**, **APACHE II**, and optionally a **combined** model  
- Outputs median thresholds and summary statistics such as sensitivity and specificity  

---

## Methods (at a glance)

- **Youden J** = *sensitivity + specificity − 1*  
- For each resample:  
  1. Fit the model on the resampled/training data.  
  2. Generate predicted probabilities on the corresponding evaluation data.  
  3. Sweep candidate thresholds to maximize **Youden J**.  
  4. Record the optimal threshold (and optional metrics).  
- After **10,000 iterations**, report **median** (and IQR/95% CI if enabled) thresholds per model.

---

## Repository Contents

- `Thresholds_Estimate.py` — main script for threshold optimization via Youden J  

---

## Environment

- **Python 3.13.5**
- **Recommended packages:**
  - `pandas ≥ 2.2`
  - `numpy ≥ 1.26`
  - `scikit-learn ≥ 1.5`
  - `scipy ≥ 1.13`
  - `joblib ≥ 1.4`
  - *(optional)* `matplotlib` for visualization  

Install with:
```bash
pip install -r requirements.txt
