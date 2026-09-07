# Customer Lifetime Value (CLV) Analysis
## Marketing Analytics HW2

**Course:** Marketing Analytics 
**Level:** Graduate  
**Assignment Weight:** 10% of Final Grade 
**Dataset:** FMCG Customer Behavior and Marketing Analytics Dataset 
**Submission Format:** Individual Assignment - Github submission

---

# 1. Assignment Overview

Customer Lifetime Value (CLV) is one of the most important metrics used by organizations to evaluate customer profitability and guide marketing investments. In this assignment, you will analyze customer transaction data from a Fast-Moving Consumer Goods (FMCG) retailer and estimate customer value using both descriptive and predictive analytics techniques.

You will explore customer purchasing behavior, perform customer segmentation using RFM analysis, estimate Customer Lifetime Value (CLV), and provide evidence-based business recommendations.

The goal is to demonstrate your ability to transform raw customer data into actionable managerial insights.

---

# 2. Learning Outcomes

Upon successful completion of this assignment, students will be able to:

- Prepare and clean transactional customer datasets.
- Conduct exploratory data analysis (EDA).
- Calculate and interpret RFM metrics.
- Estimate Customer Lifetime Value (CLV).
- Segment customers based on profitability.
- Develop data-driven marketing recommendations.
- Communicate analytical findings in a professional business report.

---

# 3. Dataset

### FMCG Customer Behavior and Marketing Analytics Dataset

Kaggle Dataset:

https://www.kaggle.com/datasets/shuchismitamallick/fmcg-customer-behavior-and-marketing-analytics-data

Students must import the dataset in a PostgreSQL Database accordingly to the provided ER diagram.

Python code must connect to the Database and SQL based queries shall be embedded in the code.

SQL based data ingestion scripts are deliverables and part of the final submission

```python
# psycopg2 library allows to connect to PostGREs DB

import psycopg2

db_connection = psycopg2.connect(dbname='test',
                                 user=<username>,
                                 password=<password>,
                                 host='localhost',
                                 port=5432)

print("Successfully connected to the database.")
```
Kind reminder, PostgreSQL has been shipped with your lab envirnment as a docker container. However, if you prefer a local installation, please follow below references.

Plese, here is where you can download PostgreSQL [installer](https://www.postgresql.org/download/)

[On Mac Install](https://www.geeksforgeeks.org/postgresql/install-postgresql-on-mac/)
[On Windows Install](https://www.geeksforgeeks.org/postgresql/install-postgresql-on-windows/)

[GUI tools for Postgre SQL](https://www.geeksforgeeks.org/postgresql/gui-tools-for-postgresql/)

Please, check the following [SQL in python](https://www.geeksforgeeks.org/python/postgresql-python-querying-data/)

Please, check [PostgreSQL tutorial](https://www.geeksforgeeks.org/postgresql/postgresql-tutorial/)

---

# Required Libraries

Students MUST use the following libraries based on Python version at least 3.10:

```python
pandas
numpy
matplotlib
scikit-learn
lifetimes
seaborn
decimal
datetime
```

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn lifetimes seaborn decimal datetime
```
---
# 4. Business Scenario

You have recently joined the marketing analytics team of an FMCG retailer.

Senior management would like to answer the following questions:

1. Which customers generate the highest value?
2. Which customers are at risk of churn?
3. How should marketing resources be allocated across customer segments?
4. What customer retention strategies should be implemented?
5. Which customers should be prioritized for loyalty programs?

Your analysis should provide evidence-based recommendations to support managerial decision making.

---

# 5. Assignment Tasks

## Part A – Data Understanding and Preparation (5 Points)

### Requirements

1. Load the dataset.
2. Describe the available variables.
3. Identify missing values and data quality issues.
4. Perform any necessary data cleaning.
5. Provide summary statistics.

### Deliverables

- Data dictionary summary
- Missing value assessment
- Descriptive statistics table
- Discussion (200–300 words)

---

## Part B – Exploratory Data Analysis (10 Points)

### Requirements

Conduct exploratory analysis to understand customer behavior.

Include visualizations for:

- Customer spending distribution
- Purchase frequency distribution
- Revenue by product category
- Customer demographics
- Revenue concentration among customers
- Time-based purchasing trends

### Deliverables

- Minimum of five visualizations
- Interpretation of each visualization
- Summary of key findings

---

## Part C – RFM Analysis (25 Points)

### Step 1: Calculate RFM Metrics

#### Recency

Number of days since the most recent purchase.

#### Frequency

Number of purchases made by the customer.

#### Monetary Value

Total spending by the customer.

### Step 2: RFM Scoring

Assign scores from 1–5 using quantiles.

Example:

```text
RFM Score = R × 100 + F × 10 + M
```

### Deliverables

- Customer-level RFM table
- Distribution of RFM scores
- Top 10 customers by RFM score
- Interpretation of customer behavior patterns

---

## Part D – Customer Lifetime Value Analysis (25 Points)

### Objective

Estimate customer lifetime value using a simplified CLV framework.

### Average Purchase Value

```text
APV = Total Revenue / Number of Transactions
```

### Purchase Frequency

```text
PF = Number of Transactions / Number of Customers
```

### Churn Rate

Assume customers who have not purchased within the previous 90 days are considered churned.

```text
Churn Rate = Churned Customers / Total Customers
```

### Customer Lifetime Value

```text
CLV = (APV × PF) / Churn Rate
```

### Deliverables

- CLV methodology
- Customer-level CLV table
- CLV distribution visualization
- Top 20 customers by CLV

---

# Part E - Predict CLV (25 Points)

Develop a predictive CLV model:

- Define the prediction model justifying your choice on dataset structure and analysis

hint: please check out the [BG/NBD](https://arxiv.org/html/2501.04719v1) 

Evaluate performance using:

- RMSE
- MAE
- R²

Discuss:

- Model performance
- Key predictive variables
- Practical implications

---

## Part F – Customer Segmentation (5 Points)

Using CLV estimates, create the following customer segments.

| Segment | Definition |
|----------|------------|
| Platinum | Top 20% |
| Gold | Next 30% |
| Silver | Next 30% |
| Bronze | Bottom 20% |

### Analyze

- Average spending
- Purchase frequency
- Customer demographics
- Product preferences

### Deliverables

- Segment summary table
- Segment visualizations
- Segment profiles

---

## Part G – Managerial Recommendations (5 Points)

Prepare recommendations for senior management.

Address:

- Customer retention
- Loyalty programs
- Resource allocation
- Promotional strategies
- Customer acquisition priorities

### Deliverables

500–750 word executive recommendation section.

---

# Submission Requirements

## Required Deliverables

###  Python and SQL

Include:
- Scripts or list of commands for data ingestion and DB creation
- SQL DB DUMP
- Code
- Visualizations
- Explanations

### 2. Presentation Report (PDF)

Recommended length:

**4–6 pages (excluding appendix)**

### Report Structure

1. Executive Summary
2. Data Overview
3. Exploratory Analysis
4. RFM Analysis
5. CLTV Analysis
6. Customer Segmentation
7. CLTV prediction
8. Recommendations
9. Conclusion

---

# 6. Assessment Rubric

| Criterion | Marks |
|------------|-------|
| Data Preparation | 5 |
| Exploratory Analysis | 10 |
| RFM Analysis | 25 |
| CLV Analysis | 25 |
| Predictive Modeling | 25 |
| Customer Segmentation | 5 |
| Recommendations | 5 |
| **Total** | **100** |

---

# 7. Academic Integrity

Students must submit original work.

You may discuss concepts with classmates; however:

- Code must be your own.
- Visualizations must be your own.
- Written interpretations must be your own.
- Any external sources must be properly cited.

Cheating and Use of GenAI for coding, please refer to class syllabus.

---


