# 📈 Quantitative Marketing Analytics: Bayesian MMM, CLV & Distributed ML

[![Language](https://img.shields.io/badge/Language-Python%203.11+-3776AB?style=flat&logo=python)](https://www.python.org/)
[![Bayesian MMM](https://img.shields.io/badge/MMM-Google%20Meridian%20%7C%20SHAP-orange)](#)
[![Probabilistic CLV](https://img.shields.io/badge/CLV-BG%2FNBD%20%7C%20Gamma--Gamma-blue)](#)
[![Big Data](https://img.shields.io/badge/Distributed-Apache%20Spark%20%7C%20PySpark-E25A1C?logo=apache-spark)](#)
[![Deliverables](https://img.shields.io/badge/Reports-Executive%20PDFs-green)](#)

> Modern commercial decision engines require rigorous econometric modeling: Bayesian media attribution, probabilistic customer lifetime value, and cost-sensitive churn classification.

---

## 📌 Executive Summary

Superficial marketing analytics relies on last-click attribution, heuristic customer segmentation, and uncalibrated churn alerts, generating massive capital misallocation.

This repository implements a four-tier quantitative marketing engineering framework bridging Bayesian econometrics, probabilistic customer modeling, and distributed machine learning.

Across a $10.1M media budget, the Bayesian optimization engine disproved naive ROAS metrics, unlocking +$419K in incremental revenue by reallocating spend from saturated channels to high-marginal-ROI drivers.

---

## 🏛️ Analytical Framework Architecture

```
                    QUANTITATIVE MARKETING SUITE
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Bayesian MMM &  │    │  Probabilistic   │    │  Cost-Sensitive  │
│   Optimization   │    │   CLV Modeling   │    │  Churn Modeling  │
│(Project_colombini│    │ (HW2_fcolombini4)│    │(HW1 & HW3 Spark) │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         ▼                       ▼                       ▼
Google Meridian Bayesian  BG/NBD + Gamma-Gamma   Scikit-Learn & PySpark
Adstock & Hill Curves     14-Table FMCG Schema   Asymmetric Cost Matrix
mROI Budget Reallocation Survival Probabilities  Distributed Assembly
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    EXECUTIVE DECISION REPORTS
         Vector Executive Presentations & Notebook Evidence
```

The codebase progresses from single-customer behavioral economics to enterprise-scale distributed data pipelines processing millions of interaction logs.

---

## 🔬 Core Quantitative Modules

### 1. Bayesian Marketing Mix Modeling & Budget Optimization ([`Project_fcolombini4`](Project_fcolombini4))
- **Google Meridian Engine ([`Project_fcolombini4/src/modeling.py`](Project_fcolombini4/src/modeling.py)):** Hierarchical Bayesian MMM incorporating Adstock decay, Hill saturation functions, and territory priors (UK and US).
- **Marginal ROI Reallocation ([`Project_fcolombini4/src/optimization.py`](Project_fcolombini4/src/optimization.py)):** Capital reallocation based on [`outputs/tables/optimization_results.csv`](Project_fcolombini4/outputs/tables/optimization_results.csv). Cuts saturated Google Display spend by -44.4% while expanding Meta Facebook (+9.4%) and Meta Other (+50.0%).
- **Model Explainability ([`Project_fcolombini4/src/shap_analysis.py`](Project_fcolombini4/src/shap_analysis.py)):** SHAP tree explanations deconstructing non-linear media channel interactions and seasonality across territories.
- **Deliverables:** Verified executive slide deck [`Project_fcolombini4/Project_Presentation.pdf`](Project_fcolombini4/Project_Presentation.pdf) and interactive analysis [`Project_fcolombini4/notebooks/mmm_homework.ipynb`](Project_fcolombini4/notebooks/mmm_homework.ipynb).

### 2. Customer Lifetime Value & BTYD Modeling ([`HW2_fcolombini4`](HW2_fcolombini4))
- **Probabilistic Spend Engines ([`HW2_fcolombini4/HW2.py`](HW2_fcolombini4/HW2.py)):** Combines Beta-Geometric / Negative Binomial Distribution (BG/NBD) transaction processes with Gamma-Gamma monetary value modeling.
- **Relational FMCG Database:** Integrates 14 distinct enterprise entities documenting transactions, discounts, price changes, and support tickets via [`HW2_fcolombini4/data/ER Diagram.md`](HW2_fcolombini4/data/ER Diagram.md).
- **Customer Segmentation:** Calculates survival probabilities $P(\text{Active})$, expected transaction frequencies, and margin concentration curves.
- **Deliverables:** Methodological whitepaper [`HW2_fcolombini4/CLV_BG_NBG.pdf`](HW2_fcolombini4/CLV_BG_NBG.pdf) and executive deck [`HW2_fcolombini4/HW2_Presentation.pdf`](HW2_fcolombini4/HW2_Presentation.pdf).

### 3. Cost-Sensitive Churn Prediction ([`HW1_fcolombini4`](HW1_fcolombini4))
- **Asymmetric Loss Functions ([`HW1_fcolombini4/HW1_starter_code.py`](HW1_fcolombini4/HW1_starter_code.py)):** Calibrates classification thresholds against true financial payoff matrices rather than arbitrary 0.5 cutoffs.
- **Retention Trade-offs:** Balances false-negative attrition damage against the commercial cost of unnecessary retention incentives.
- **Deliverable:** Technical evaluation report [`HW1_fcolombini4/HW1_ChurnPrediction.pdf`](HW1_fcolombini4/HW1_ChurnPrediction.pdf).

### 4. Distributed Big Data Churn Pipeline with Apache Spark ([`HW3_fcolombini4`](HW3_fcolombini4))
- **PySpark MLlib Architecture ([`HW3_fcolombini4/HW3_starter_code.py`](HW3_fcolombini4/HW3_starter_code.py)):** Distributed feature extraction, `VectorAssembler` pipelines, and distributed gradient boosting.
- **Strict Schema Enforcement:** Validates input records against formal schemas via [`HW3_fcolombini4/schemas/dataset1_churn_schema.json`](HW3_fcolombini4/schemas/dataset1_churn_schema.json).
- **Deliverable:** Distributed architecture deck [`HW3_fcolombini4/HW3_ChurnPredictionSpark.pdf`](HW3_fcolombini4/HW3_ChurnPredictionSpark.pdf).

---

## 📊 Empirical Budget Optimization Matrix

Budget optimization results extracted from Google Meridian Bayesian model fit:

| Channel | Historical Spend | Optimized Spend | Spend Delta | Historical Revenue | Optimized Revenue | mROI | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Google Display** | \$90,000 | \$50,000 | **-44.4%** | \$233,731 | \$184,774 | 1.65 | Saturated / Cut |
| **Meta Facebook** | \$7,460,000 | \$8,160,000 | **+9.4%** | \$24,745,710 | \$26,335,322 | 2.21 | Under-invested |
| **Meta Instagram** | \$2,530,000 | \$1,850,000 | **-26.9%** | \$5,275,156 | \$4,131,028 | 1.79 | Rebalanced |
| **Meta Other** | \$20,000 | \$30,000 | **+50.0%** | \$106,339 | \$129,128 | 2.08 | High Potential |

---

## 🛠️ Environment & Reproduction

Each module maintains isolated execution scripts and dependencies:
```powershell
# 1. Marketing Mix Modeling (Google Meridian)
cd Project_fcolombini4
pip install -r requirements.txt
python src/modeling.py

# 2. Customer Lifetime Value (BTYD / Lifetimes)
cd ../HW2_fcolombini4
python HW2.py

# 3. Apache Spark Churn Pipeline
cd ../HW3_fcolombini4
spark-submit HW3_starter_code.py
```

---

**Author:** Francesco Colombini  
[GitHub Profile](https://github.com/FRA-0023) · [LinkedIn](https://www.linkedin.com/in/francescocolombini/)
