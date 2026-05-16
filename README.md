# 🌍 African Systemic Crisis Prediction

> An end-to-end machine learning pipeline for early warning of banking, currency, and sovereign debt crises across 13 African nations - one year in advance.

---

## Overview

Predicting economic crises is a core challenge in risk management, policy-making, and international finance. This project builds a supervised binary classifier that forecasts the onset of a **systemic crisis** one year ahead, replicating the analytical workflow of a risk analyst at a central bank or international organisation.

The dataset spans **13 African countries** from the late 19th century to 2014, providing a rich historical panel for time-series classification under severe class imbalance.

> **Why this matters:** Early Warning Systems (EWS) for financial crises are actively used by the IMF, World Bank, and central banks globally. This project demonstrates how historical macroeconomic data can be transformed into actionable risk signals — without any future data leakage.

---

## Dataset

**Source:** `african_crises.csv` - a curated panel dataset of African economic history.


| Attribute | Detail |
|-----------|--------|
| Countries | Algeria, Angola, Central African Republic, Ivory Coast, Egypt, Kenya, Mauritius, Morocco, Nigeria, South Africa, Tunisia, Zambia, Zimbabwe |
| Period | ~1860 – 2014 |
| Observations | 1 059 country-year rows |
| Target rate | ~7.7 % systemic crisis years (severe class imbalance) |

**Key raw columns:**

| Column | Description                                             |
|--------|---------------------------------------------------------|
| `country`, `year` | Panel identifiers                                       |
| `exch_usd` | Exchange rate (local currency per USD)                  |
| `inflation_annual_cpi` | Annual CPI inflation rate (%)                           |
| `domestic_debt_in_default` | Domestic debt default flag (0/1)                        |
| `sovereign_external_debt_default` | Sovereign external default flag (0/1)                   |
| `gdp_weighted_default` | GDP-weighted default indicator                          |
| `banking_crisis` | Banking crisis flag (`"crisis"` / `"no_crisis"`)        |
| `currency_crises` | Currency crisis flag (0/1)                              |
| `inflation_crises` | Inflation crisis flag (0/1)                             |
| `systemic_crisis` | **Target** - country in systemic crisis that year (0/1) |

---

## Problem Statement

> **Can we predict whether a country will enter a systemic crisis *next year*, using only data available today?**

This is a **binary classification** task with a one-year prediction horizon:

- **Positive class:** `systemic_crisis = 1` in year *t + 1*
- **Features:** lagged macroeconomic indicators from year *t* and *t − 1*
- **Challenge:** crises are rare events (< 10 % of observations), requiring careful handling of class imbalance and strict avoidance of look-ahead bias

---

## Project Structure

```
crisis-prediction-africa/
│
├── data/
│   ├── african_crises.csv          # Raw source data
│   ├── cleaned_crisis.csv          # After notebook 01
│   ├── train.csv                   # Years ≤ 1999
│   ├── validation.csv              # Years 2000 – 2009
│   └── test.csv                    # Years 2010 – 2013
│
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_feature_engineering_and_data_splitting.ipynb
│   ├── 03_model_training_evaluation.ipynb
│   └── 04_shap_interpretation.ipynb
│
├── outputs/
│   └── figures/                    # Saved plots (PR curves, SHAP, waterfalls …)
│
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Data Cleaning

*(Notebook 01 - `01_eda_and_cleaning.ipynb`)*

| Step | Detail                                                                                    |
|------|-------------------------------------------------------------------------------------------|
| Binary encode | `banking_crisis`: `"crisis"` -> 1, `"no_crisis"` -> 0                                     |
| Outlier capping | `inflation_annual_cpi` and `exch_usd` clipped at the 99th percentile per the full dataset |
| Zero-rate fix | Exchange rate = 0 treated as missing, forward-filled within each country's time series    |
| Output | `cleaned_crisis.csv` - 1 059 rows, no nulls in key columns                                |

EDA highlights:
- Zimbabwe dominates crisis observations (hyperinflation + currency collapse post-2000)
- Crisis frequency accelerates post-independence
- Inflation and exchange rate are heavily right-skewed -> capping essential

---

### 2. Feature Engineering

*(Notebook 02 - `02_feature_engineering_and_data_splitting.ipynb`)*

All features are constructed from **past data only**, ensuring zero look-ahead bias:

| Feature group | Features created                                                                                        |
|---------------|---------------------------------------------------------------------------------------------------------|
| **Lag-1** | 8 base columns shifted back 1 year within each country                                                  |
| **Lag-2** | Same 8 columns shifted back 2 years                                                                     |
| **FX momentum** | `exch_usd_pct_change` - period-over-period % change                                                     |
| **Inflation volatility** | `inflation_volatility` - 3-year rolling standard deviation of capped CPI                                |
| **Target** | `crisis_next_year` = `systemic_crisis` shifted forward by 1 year (predicting *t + 1* from *t* features) |

**Total: 19 engineered features.** Rows with NaN in any lag feature or the target are dropped (necessary because lags do not exist for a country's first two observations).

---

### 3. Train / Validation / Test Split

A **strict chronological split** prevents any future information leaking into training:

| Split | Period | Rows | Crisis rows | Crisis rate |
|-------|--------|------|-------------|-------------|
| **Train** | ≤ 1999 | 866 | 66 | 7.6 % |
| **Validation** | 2000 – 2009 | 130 | 11 | 8.5 % |
| **Test** | 2010 – 2013 | 50 | 4 | 8.0 % |

- No country-year row appears in more than one split
- Median imputation values are computed **on training data only** and applied to val/test to prevent leakage
- `TimeSeriesSplit(n_splits=5)` used inside `GridSearchCV` so inner folds also respect time order

---

### 4. Handling Class Imbalance

With only ~8 % positive observations, a naïve model predicts "no crisis" every time and achieves 92 % accuracy while catching zero actual crises. Three strategies are evaluated:

| Strategy | Mechanism                                                                                   |
|----------|---------------------------------------------------------------------------------------------|
| **Baseline** | No adjustment - confirms the failure mode                                                   |
| **Class weights** | `class_weight='balanced'` or `scale_pos_weight` penalises misclassifying the minority class |
| **SMOTE** | Synthetic minority oversampling applied **to training data only**, inside a pipeline        |

Class weights consistently outperformed SMOTE on this small panel dataset - a typical result when the dataset is genuinely small rather than merely unbalanced.

---

### 5. Models & Hyperparameter Tuning

Three classifiers are trained and compared using `GridSearchCV` with `TimeSeriesSplit`:

| Model | Imbalance strategy | Tuned hyperparameters |
|-------|-------------------|-----------------------|
| **Logistic Regression** | SMOTE + `class_weight='balanced'` | `C` (inverse regularisation) |
| **Random Forest** | `class_weight='balanced'` | `n_estimators`, `max_depth`, `min_samples_leaf` |
| **XGBoost** | `scale_pos_weight` | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` |

**Scoring criterion inside CV:** `average_precision` (PR-AUC) - appropriate for imbalanced data and threshold-independent.

**Threshold selection:** after fitting, the decision threshold is set to the value that **maximises F2-score** on the validation set (with a minimum of 0.05 to avoid the degenerate "flag everything" case).

---

### 6. Evaluation Metrics

Accuracy is misleading when < 10 % of observations are positive. All evaluation is done on:

| Metric | Why it matters here                                                            |
|--------|--------------------------------------------------------------------------------|
| **Recall (Sensitivity)** | Fraction of actual crises correctly flagged - the most critical metric for EWS |
| **Precision** | Fraction of alarms that are genuine - controls alert fatigue                   |
| **F2-Score** | Weighted harmonic mean with β = 2, emphasising recall over precision           |
| **PR-AUC** | Area under the Precision-Recall curve - threshold-independent summary          |
| **Confusion matrix** | Full breakdown of TP / FP / FN / TN                                            |

ROC-AUC is deliberately avoided - it is optimistic under class imbalance and does not reflect the practical trade-off between catching crises and generating false alarms.

---

### 7. SHAP Interpretability

*(Notebook 04 - `04_shap_interpretation.ipynb`)*

`shap.TreeExplainer` provides exact Shapley values for both tree models. Four visualisation types are produced:

| Plot | What it shows                                                                                                   |
|------|-----------------------------------------------------------------------------------------------------------------|
| **Bar chart** | Mean \|SHAP\| per feature - global importance ranking                                                           |
| **Beeswarm** | Direction (positive = pushes toward crisis) + feature-value colour                                              |
| **Dependence plots** | How each top-3 feature's value maps to its SHAP contribution, coloured by the most strongly interacting feature |
| **Waterfall plots** | Single-observation decomposition for a True Positive, a False Positive, and a False Negative                    |

---

## Results

All models are trained on data up to 1999, tuned on 2000–2009 (validation), and evaluated — **exactly once** — on 2010–2013 (test).

### Validation Set (2000–2009)

| Model                    | Threshold | Recall | Precision | F2 | PR-AUC |
|--------------------------|-----------|--------|-----------|----|--------|
| LR Baseline (no weights) | 0.500 | 0.000 | 0.000 | 0.000 | 0.052 |
| LR + SMOTE               | 0.231 | 0.909 | 0.125 | 0.435 | 0.199 |
| Random Forest            | 0.583 | 0.909 | 0.400 | **0.725** | 0.321 |
| ***XGBoost***            | 0.270 | 1.000 | 0.344 | 0.724 | **0.396** |

### Test Set (2010–2013) - Final Evaluation

| Model | Recall | Precision | F2 | PR-AUC |
|-------|--------|-----------|----|--------|
| Random Forest | **1.000** | **0.571** | **0.870** | 0.762 |
| XGBoost | 1.000 | 0.103 | 0.362 | 0.854 |

> **Selected model for interpretation: XGBoost** (highest PR-AUC on validation = 0.396).
> **Best overall performance: Random Forest** (F2 = 0.87 on test; XGBoost generates excessive false alarms with threshold = 0.270).

---

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/yevqlzw/African_Crisis_Prediction.git
cd African_Crisis_Prediction
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the notebooks in order**

```bash
jupyter notebook
```

| Notebook | Purpose |
|----------|---------|
| `01_eda_and_cleaning.ipynb` | Understand the raw data, handle missing values |
| `02_feature_engineering.ipynb` | Build lag features, rolling stats, target variable |
| `03_model_training_evaluation.ipynb` | Train, tune, and compare all models |
| `04_shap_interpretation.ipynb` | Explain predictions with SHAP |

The notebooks are self-contained and include inline comments and visualisations.

---

## Dependencies

```
pandas==2.2.2
numpy==1.26.4
matplotlib==3.9.0
seaborn==0.13.2
scikit-learn==1.5.0
imbalanced-learn==0.12.3
xgboost==2.0.3
shap==0.45.1
scipy==1.13.1
plotly==5.22.0
jupyter==1.0.0
ipykernel==6.29.4
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Key Skills Demonstrated

- Time-series feature engineering (lags, rolling windows, percentage changes)
- Strict chronological train/test splitting to prevent data leakage
- Advanced class imbalance handling (SMOTE, class weights, stratified CV)
- Model comparison and hyperparameter tuning with time-aware cross-validation
- Model interpretability with SHAP - translating ML output into business insights

---

## Future Work

- Incorporate external data: commodity prices, global interest rates, political instability indices
- Extend the prediction horizon to two and three years ahead
- Build a multi-label classifier predicting crisis *type* (currency vs. banking vs. sovereign debt)
- Deploy the best model as a REST API (FastAPI) with an interactive Streamlit dashboard for real-time warning simulation

---

## Disclaimer

This project is for **educational and portfolio purposes only**. Predictions are based on historical patterns and should not be used for actual financial decision-making without further validation and domain expertise.

---

## Author

Kravchenko Yevhenia / GitHub - yevqlzw / LinkedIn - www.linkedin.com/in/zhenia-kravchenko-29a357372

