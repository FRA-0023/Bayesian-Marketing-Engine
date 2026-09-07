# HW 1 Marketing Analytics 2026

## Overview

Students shall build a complete **Customer Churn Prediction Pipeline** using Python and Scikit-learn.

The project focuses on:
- Data preprocessing
- Exploratory data analysis
- Feature engineering
- Machine learning pipelines
- Model evaluation
- Prediction generation

Students must complete the provided Python starter code by implementing all required functions.
This can be accomplished using the course evirnment or creating your local Python environment

---

# Learning Objectives

By completing this assignment, students should be able to:

- Load and inspect tabular datasets
- Verify different datasets and define accordingly which training models to use
- Define train and test dataset
- Handle missing values
- Identify categorical and numerical features
- Build preprocessing pipelines
- Train machine learning classification models
- Evaluate classification performance
- Interpret feature importance
- Save prediction outputs
- Create a 3 to 5 slides presentation explaining your approach to solution

---

# Files Provided

| File | Description |
|---|---|
| `dataset1_HW1.csv` | dataset 1 |
| `dataset2_HW1.csv` | dataset 2 |
| `HW1_starter_code.py` | Assignment starter code |
| `README.md` | Assignment instructions |

---

# Required Libraries

Students MUST use the following libraries based on Python version at least 3.10:

```python
pandas
numpy
matplotlib
scikit-learn
```

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

# Assignment Requirements

Students must complete all TODO sections in the skeleton code.

The final implementation should include:

## Step 1 — Load Datasets
Implement dataset loading using Pandas.

Expected tasks:
- Read CSV files
- Define train and test dataset

---

## Step 2 — Dataset Overview
Perform basic exploratory analysis.

Expected tasks:
- Display dataset shapes
- Identify missing values
- Display summary statistics

---

## Step 3 — Split Features and Target
Separate:
- Feature variables (`X`)
- Target variable (`y`)

---

## Step 4 — Identify Column Types
Detect:
- Numerical columns
- Categorical columns

---

## Step 5 — Build Preprocessing Pipeline
Create preprocessing pipelines for:
- Numerical features
- Categorical features

Expected preprocessing:
- Missing value handling
- Feature scaling
- One-hot encoding

---

## Step 6 — Build Model Pipeline
Create a full machine learning pipeline using:
- Preprocessor
- Test at least 3 different models for comparison
---

## Step 7 — Train Model
Train the machine learning model using:

```python
model.fit()
```

---

## Step 8 — Generate Predictions
Generate:
- Predicted labels
- Prediction probabilities

---

## Step 9 — Evaluate Model
Compute:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Step 10 — Confusion Matrix
Generate and display a confusion matrix.

---

## Step 11 — Feature Importance Analysis
Analyze model interpretability:
- Extract feature importance values
- Visualize top features

---

## Step 12 — Save Training, Test and Predictions
Save predictions into:

```bash
churn_predictions_dataset1_train.csv
churn_predictions_dataset2_train.csv
churn_predictions_dataset1_test.csv
churn_predictions_dataset2_test.csv
churn_predictions_dataset1.csv
churn_predictions_dataset2.csv

```

---

# Submission Requirements

Students must submit:

| File | Required |
|---|---|
| Completed Python script | Yes |
| Generated predictions CSV | Yes |
| Feature importance plot | Yes |
| Presentation of 3-5 slides | Yes |

This applies to both dataset provided.

---

# Expected Output

The program should:
- Train successfully
- Print evaluation metrics
- Display feature importance plots
- Save train, test and predictions to CSV

---

# Academic Integrity

Students must:
- Write their own implementation
- Properly cite external resources, this applies to code snippets too
- The solution must be completed individually

Use of AI tools and cheating policies:
- Please, refer to course syllabus


---

# Grading Rubric (110 Points Total)

| Category | Points |
|---|---:|
| Step 1 — Dataset Loading | 5 |
| Step 2 — Dataset Overview | 5 |
| Step 3 — Feature/Target Split | 5 |
| Step 4 — Column Type Identification | 5 |
| Step 5 — Preprocessing Pipeline | 15 |
| Step 6 — Model Pipeline Construction | 10 |
| Step 7 — Model Training | 5 |
| Step 8 — Predictions | 10 |
| Step 9 — Evaluation Metrics | 10 |
| Step 10 — Confusion Matrix | 5 |
| Step 11 — Feature Importance Analysis | 10 |
| Step 12 — Save Predictions | 5 |
| Code Quality & Documentation | 20 |




# Bonus Opportunities (+10 Extra Credit)

Possible bonus enhancements:
- Multi models comparison implemented in Python
- ROC curve visualization
- Precision-Recall curve


---

# Suggested Development Workflow

1. Complete one function at a time
2. Test incrementally
3. Print intermediate outputs
4. Validate shapes and dimensions
5. Verify preprocessing before training
6. Create final presentation with path to solution description

The last push on you repository before the deadline is the one to be graded.

---

# Example Execution

```bash
python HW1_starter_code.py
```

Example of expected output:

```text
Training Shape: (1000, 20)
Test Shape: (300, 20)

===== MODEL PERFORMANCE =====
Accuracy : 0.84
Precision: 0.79
Recall   : 0.81
F1 Score : 0.80
ROC-AUC  : 0.88
```

---

# Deliverables Checklist

The last push on you repository before the deadline is the one to be graded.

Before submission, verify:

- [X] All TODOs completed
- [X] Program runs without errors
- [X] Predictions file generated
- [X] Metrics displayed
- [X] Feature importance plotted
- [X] Code properly commented
- [ ] Push presentation as PDF file on the repository


