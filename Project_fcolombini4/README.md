# Marketing Analytics Project Assignment

## Title
Marketing Mix Modeling with Meridian and SHAP Explainability

Please, refer to the following tutorials before getting started:

1. [Merdian ROI and mROI](https://github.com/google/meridian/blob/bef37839848977458b86b29990df900818722c30/demo/ROI_mROI_Response_Curves.ipynb).
2. [Meridian Optimization](https://developers.google.com/meridian/docs/user-guide/budget-optimization-scenarios)
3. [SHAP tutorial](https://shap-readthedocs-io.translate.goog/en/latest/?_x_tr_sl=en&_x_tr_tl=it&_x_tr_hl=it&_x_tr_pto=sc)

---

# 1. Business Context

You are working as a Marketing Data Scientist for a multi-channel eCommerce company. The company invests across several acquisition and engagement channels including:

- Google Paid Search
- Google Shopping
- Google PMAX
- Meta Facebook
- Meta Instagram
- TikTok
- Email
- Organic Search
- Referral
- Direct Traffic

The company wants to:

1. Quantify the contribution of each marketing channel to purchases.
2. Estimate channel-level ROI and marginal ROI (mROI).
3. Understand diminishing returns and saturation effects.
4. Build future investment scenarios and predict business impact.
5. Explain model predictions using SHAP.
6. Simulate advertising budget scenarios

You have been provided with:

- `conjura_mmm_data.csv`
- `conjura_mmm_data_dictionary.xlsx`

The modeling framework must be implemented using the Meridian MMM library.
The dataset is peer reviewed:
Anderson, Anthony (2024). Multi-Region Marketing Mix Modelling (MMM) Dataset for Several eCommerce Brands. figshare. Dataset. https://doi.org/10.6084/m9.figshare.25314841.v3

---

# Required Libraries

Students MUST use the following libraries based on Python version at least 3.10:

```python
pandas
numpy
matplotlib
scikit-learn
google-meridian
shap
seborn
decimal
datetime
```

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn google-meridian shap seaborn decimal datetime
```
---

# 2. Learning Objectives

By completing this homework, students should be able to:

- Prepare marketing mix modeling datasets.
- Engineer media and control variables.
- Train a Meridian MMM model.
- Interpret adstock and saturation effects.
- Compute ROI and marginal ROI.
- Perform budget allocation simulations.
- Use SHAP for explainability.
- Present business recommendations from MMM outputs.

---

# 3. Dataset Overview

The dataset contains daily observations across multiple organizations and territories.

### Target Variables Examples:

Potential target variables include:

- `FIRST_PURCHASES`
- `ALL_PURCHASES`
- `FIRST_PURCHASES_ORIGINAL_PRICE`
- `ALL_PURCHASES_ORIGINAL_PRICE`

### Media Spend Variables Examples:

Examples:

- `GOOGLE_PAID_SEARCH_SPEND`
- `GOOGLE_SHOPPING_SPEND`
- `GOOGLE_PMAX_SPEND`
- `META_FACEBOOK_SPEND`
- `META_INSTAGRAM_SPEND`
- `TIKTOK_SPEND`

### Media Exposure Variables Examples:

Examples:

- `GOOGLE_PAID_SEARCH_CLICKS`
- `META_FACEBOOK_IMPRESSIONS`
- `EMAIL_CLICKS`
- `ORGANIC_SEARCH_CLICKS`

### Control Variables Examples:

Examples:

- Territory
- Seasonality
- Day of week
- Trend
- Promotions / discounts

---

# 4. Technical Requirements

## Mandatory Libraries

Students must use:

- Meridian MMM library
- SHAP
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

# 5. Homework Tasks

## Task 1 — Exploratory Data Analysis (10 points)

Perform exploratory data analysis on the dataset.

### Requirements

1. Inspect missing values.
2. Identify media channels.
3. Plot:
   - Daily purchases
   - Media spend trends
   - Correlation heatmap
4. Describe:
   - Seasonality
   - Outliers
   - Spend concentration by channel

### Deliverables

- Workign Python implementation on pushed on your Github Repository
- Visualizations to be repoted in the final presentation
- Report on your analysis and budget predicting scenarios in the form of presentation
- Kindly Remeber to push your presentation in PDF format on Github

---

## Task 2 — Data Preparation (15 points)

Prepare the dataset for MMM modeling.

### Requirements

1. Aggregate data at daily level.
2. Select a target variable.
3. Create:
   - Media variables
   - Control variables
   - Time variables
4. Handle missing values.
5. Split:
   - Train set
   - Validation set

### Deliverables

- Clean modeling dataset
- Feature engineering explanation

---

## Task 3 — Meridian MMM Model Development (25 points)

Build a Marketing Mix Model using Meridian.

### Requirements

1. Configure:
   - Media channels
   - Controls
   - Target
2. Train the Meridian model.
3. Estimate:
   - Adstock effects
   - Hill effects
   - Saturation curves
   - Channel contribution
4. Evaluate model quality.

### Suggested Metrics, not exhaustive

- R²
- MAPE
- RMSE

### Example Workflow Example

```python
from meridian import Meridian

model = Meridian(
    target='ALL_PURCHASES',
    media_channels=media_channels,
    controls=control_columns,
    date_column='DATE_DAY'
)

model.fit(train_df)
```

### Deliverables

- Trained MMM model
- Performance metrics
- Contribution plots
- Saturation analysis

---

## Task 4 — ROI and Marginal ROI Analysis (20 points)

Compute ROI and marginal ROI for each media channel.


### Requirements

1. Estimate incremental contribution by channel.
2. Compute ROI.
3. Compute mROI.
4. Compare channels.
5. Identify:
   - Over-invested channels
   - Under-invested channels

### Expected Analysis

Students should discuss:

- Diminishing returns
- Saturation effects
- Efficient spend allocation

### Deliverables

- ROI table
- mROI table
- Interpretation of channel efficiency

---

## Task 5 — SHAP Explainability (15 points)

Use SHAP to explain model predictions.

### Requirements

1. Create SHAP values.
2. Generate:
   - Summary plot
   - Dependence plots
   - Feature importance ranking
3. Compare SHAP importance vs MMM contribution.
4. Interpret:
   - Positive drivers
   - Negative drivers
   - Nonlinear effects

### Example

```python
import shap

explainer = shap.Explainer(model.predict, X)
shap_values = explainer(X)

shap.summary_plot(shap_values, X)
```

### Deliverables

- SHAP visualizations
- Written interpretation

---

## Task 6 — Marketing Investment Scenario Simulation (15 points)

Build a business case for future marketing investment.

### Scenarios

The company has an additional marketing budget of:

```text
1. +20% total spend
2. +50% total on most performing channel 
3. -30% total spend on already saturated channels
4. +10% randomly on half of the channels
```

Students must simulate:

1. Equal allocation across channels.
2. Allocation based on highest mROI.
3. Reduced investment in saturated channels.

### Requirements

For each scenario:

- Predict purchases
- Predict revenue uplift
- Estimate incremental ROI
- Compare outcomes

### Expected Outputs per each scenarios

| Scenario | Incremental Revenue | Incremental Purchases | ROI |
|---|---|---|---|
| Equal allocation | | | |
| mROI optimized | | | |
| Saturation-aware | | | |

### Deliverables

- Scenario comparison table
- Recommendation for optimal budget allocation

---

# 6. Suggested End-to-End Workflow

```text
Data Loading
    ↓
EDA
    ↓
Feature Engineering
    ↓
Meridian MMM Training
    ↓
Contribution Analysis
    ↓
ROI / mROI Computation
    ↓
SHAP Explainability
    ↓
Budget Optimization Scenarios
    ↓
Business Recommendations
```

---

# 7. Suggested Python Project Structure

```text
project/
│
├── data/
│   ├── conjura_mmm_data.csv
│   └── conjura_mmm_data_dictionary.xlsx
│
├── notebooks/
│   └── mmm_homework.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── roi.py
│   ├── shap_analysis.py
│   └── scenarios.py
│
├── outputs/
│   ├── charts/
│   ├── tables/
│   └── reports/
│
└── requirements.txt
```

---

# 8. Recommended Modeling Notes

## Recommended Aggregation

Because the dataset contains multiple organizations and territories, students may:

- Focus on a single organization
- Aggregate at territory level
- Aggregate globally

Students shall justify their decision. This project is mainly related to understandign Marketing Anlytics concepts. Meridian and Shap do most of the work for you. A correct justifycation and intepretation of your results are at core of the grading process.

---

## Recommended Controls

Students are encouraged to include:

- Day-of-week seasonality
- Monthly seasonality
- Trend terms
- Discount variables
- Organic traffic controls

---

## Meridian-Specific Expectations

Students should demonstrate understanding of:

- Adstock transformation
- Saturation curves
- Lagged media effects
- Bayesian estimation
- Incrementality vs Attribution

---

# 9. Grading Rubric

| Component | Points |
|---|---|
| EDA | 10 |
| Data Preparation | 15 |
| Meridian MMM Modeling | 25 |
| ROI and mROI Analysis | 20 |
| SHAP Explainability | 15 |
| Scenarios Simulation | 15 |
| Total | 100 |
| Extra Credit | 15 |

---

# 10. Expected Business Insights

Students should ultimately answer:

1. Which marketing channels drive the highest incremental purchases?
2. Which channels have the strongest ROI?
3. Which channels are saturated?
4. What is the optimal allocation of additional budget?
5. How explainable and trustworthy are the model outputs?

---

# 11. Bonus Tasks (Optional)

## Bonus 1 — Bayesian Credible Intervals

Estimate uncertainty intervals around contribution estimates.

## Bonus 2 — Geo-Level MMM

Build territory-specific MMM models.

## Bonus 3 — Optimization

Use optimization techniques to maximize revenue under budget constraints.

---

# 12. Deliverables

Students should submit:

1. Presentation Slides in PDF
2. Source Code
3. Business recommendation on Advertising Investment Plan

---

# 13. Suggested Report Structure

## Executive Summary

- Key findings
- ROI and mROI insights
- Recommended spend allocation

## Methodology

- Dataset preparation
- Modeling approach
- Meridian configuration

## Results

- Channel contributions
- ROI and mROI
- SHAP explainability

## Scenario Planning

- Investment simulations
- Business recommendations

## Limitations

- Data quality
- Causal assumptions
- Model uncertainty 

---

# 15. Final Questions

1. Based on your Meridian MMM model, what is your recommended marketing investment strategy for the next quarter, and why?
2. Define the best strategy toward revenue maximization


