# Research Methods for Business Analytics — Final Project

**Course:** Research Methods for Business Analytics
**Institution:** Coller School of Management, Tel Aviv University
**Semester:** B, 2026

**Authors:**
- Tal Cordova, 203868187
- Saar Cohen, 213499312

---

## Project Description

A retailer wants to maximise profit from a targeted marketing campaign. We are given a dataset of 20,000 frequent-flyer program (FFP) customers and must decide **which customers to contact**.

The business payoff is asymmetric:
- Contacting a customer who buys → **+$78.4**
- Contacting a customer who does not buy → **−$15.2**

We build a machine-learning pipeline that predicts purchase probability and uses a profit-optimal decision threshold derived from the payoff ratio.

---

## Repository Structure

| Notebook | Description |
|----------|-------------|
| [Course_Final_Project_TC_SC.ipynb](Course_Final_Project_TC_SC.ipynb) | **EDA** — Exploratory data analysis of the FFP dataset. Examines distributions, class imbalance, correlations, and multicollinearity to inform feature engineering decisions. |
| [Final_Project_SC_TC_fit_notebook.ipynb](Final_Project_SC_TC_fit_notebook.ipynb) | **Model Training** — Feature engineering, custom profit metric definition, and baseline comparison of four classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost) via 5-fold stratified cross-validation. |
| [Course_Project_Text_Mining_TC_SC.ipynb](Course_Project_Text_Mining_TC_SC.ipynb) | **Text Mining** — Trains a sentiment model on product reviews (Bag-of-Words + chi² selection + Logistic Regression) and outputs positive-review probabilities as an additional feature for the main model. |

---

## Data

The project uses two datasets (not included in this repository — stored on Google Drive):

| File | Description |
|------|-------------|
| `ffp_train.csv` | 20,000 FFP customer records with 22 features and a binary `BUYER_FLAG` target |
| `text_training.csv` | ~2,000-column bag-of-words matrix of product reviews with a binary `rating` target |
| `reviews_training.csv` | Review BoW features for the training-set customers |
| `reviews_rollout.csv` | Review BoW features for the rollout-set customers (no labels) |

### FFP Features

| Feature | Description |
|---------|-------------|
| `CUSTOMER_GRADE` | Continuous customer score |
| `STATUS_PANTINUM / GOLD / SILVER` | Loyalty tier indicators (mutually exclusive) |
| `NUM_DEAL` | Number of deals taken |
| `LAST_DEAL` | Days since last deal |
| `ADVANCE_PURCHASE` | Days booked in advance |
| `FARE_L_Y1–Y5` | Total fares paid in each of the last 5 years |
| `POINTS_L_Y1–Y5` | Loyalty points earned in each of the last 5 years |
| `ATT_FLAG` | Engagement/attention indicator |
| `ANIMAL_FLAG` | Animal-related indicator (dropped — uninformative) |
| `CREDIT_FLAG` | Credit-related indicator |
| `RELATED_FLAG` | Related-product indicator |
| `BUYER_FLAG` | **Target** — 1 if customer purchased, 0 otherwise |

---

## Methodology

### 1. Exploratory Data Analysis
- No missing values or duplicate IDs
- **Class imbalance:** ~9.5:1 (90.5% non-buyers, 9.5% buyers)
- `FARE_L_Y*` and `POINTS_L_Y*` columns are highly collinear (r > 0.92) across years
- `ATT_FLAG` is the strongest binary predictor (~3.4× conversion lift)
- `CUSTOMER_GRADE` and `ANIMAL_FLAG` have negligible predictive power

### 2. Feature Engineering
- **Recency-weighted aggregation:** replace 10 yearly fare/points columns with 2 weighted sums (weights: 1.0, 0.8, 0.6, 0.4, 0.2 from most to least recent)
- **Drop:** `ANIMAL_FLAG`, `CUSTOMER_GRADE` (based on EDA findings)

### 3. Custom Profit Metric
The break-even classification threshold is derived from the cost ratio:

```
threshold = cost_FP / (cost_FP + revenue_TP) = 15.2 / (15.2 + 78.4) ≈ 0.162
```

The primary evaluation metric is **average profit per customer** at this threshold.

### 4. Text Mining Feature
Product reviews are converted to a bag-of-words matrix. A Logistic Regression (with chi² feature selection) predicts the probability of a positive review. This probability is merged into the FFP dataset as an additional feature.

### 5. Model Comparison
Four models are evaluated with 5-fold stratified cross-validation:
- Logistic Regression (with StandardScaler)
- Decision Tree
- Random Forest (100 estimators)
- XGBoost (100 estimators)

---

## How to Run

All notebooks are designed to run on **Google Colab**. Click the "Open in Colab" badge at the top of each notebook.

**Recommended run order:**
1. `Course_Project_Text_Mining_TC_SC.ipynb` — generates review probability features
2. `Course_Final_Project_TC_SC.ipynb` — EDA (exploratory, no outputs required downstream)
3. `Final_Project_SC_TC_fit_notebook.ipynb` — model training and evaluation

Data files must be available on Google Drive at the paths configured in each notebook.
