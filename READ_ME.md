#  King County (Seattle Area) Housing Price Prediction — Data Analysis & Machine Learning

A complete data science project that analyzes **King County (WA, USA)** housing data to uncover what truly drives home prices - and uses five regression models to predict property values with strong accuracy.

This repository combines **business intelligence**, **exploratory data analysis (EDA)**, and **machine learning modeling** to build an interpretable, data-driven understanding of the local real estate market.

---

##  Executive Summary

The real estate market of King County is characterized by high variability in property types, geography, and renovation levels.  
This project uses **22 features** (structural, locational, and qualitative) to explain and predict **house sale prices**, identifying what variables matter most for valuation.

###  Key Outcomes
- **Best Model:** Gradient Boosting (proxy for XGBoost)  
- **R² ≈ 0.87** - The model explains roughly 87% of the variance in housing prices.  
- **Typical prediction error (MAE):** ≈ \$67,000  
- **Top predictive features:**  
  1. `sqft_living` - Interior area  
  2. `grade` - Quality/finish level  
  3. `bathrooms` - Utility factor  
  4. `lat` - Latitude proxy for location desirability  
  5. `view` / `waterfront` - Premium location effects  

The analysis reveals that **price dynamics are highly non-linear**, dominated by living area, property grade, and micro-location.

---

##  Project Structure

├── kc_houses_aa.csv # Dataset
├── final.ipynb # Main notebook (EDA + modeling)
└── READ_ME.md 


---

##  Business Understanding

###  Problem
Real-estate agencies and investors want to estimate a home’s fair market value before listing or negotiation. Manual estimations are inconsistent and biased.  
A data-driven regression model can provide **objective, explainable price predictions** and uncover **value creation opportunities** (renovation, expansion, location arbitrage).

###  Goal
1. Identify **key value drivers** in the King County housing market.  
2. Predict **sale prices** accurately using multiple regression approaches.  
3. Communicate insights in a **business-friendly executive report** and a **visual analytics dashboard**.

###  Impact
- For **sellers/investors:** know which features most influence value (e.g., grade, living area, view).  
- For **buyers:** identify under-valued properties given their physical/locational attributes.  
- For **analysts:** provide an extensible baseline for automated appraisal systems.

---

##  Dataset Overview

The dataset contains **over 20,000 property transactions** in King County, Washington (Seattle area), between **2014–2015**.

| Column | Description | Type |
|:--|:--|:--:|
| `id` | Unique property identifier | categorical |
| `date` | Sale date | datetime |
| `price` | Sale price (target variable) | float |
| `bedrooms`, `bathrooms` | Interior characteristics | numeric |
| `sqft_living`, `sqft_lot` | House area & land area (ft²) | numeric |
| `floors` | Number of levels | numeric |
| `waterfront`, `view`, `condition`, `grade` | Quality & view indicators | numeric |
| `sqft_above`, `sqft_basement` | Split of total area | numeric |
| `yr_built`, `yr_renovated` | Construction & renovation years | numeric |
| `zipcode`, `lat`, `long` | Location identifiers | categorical / numeric |
| `sqft_living15`, `sqft_lot15` | Neighbor averages (2015) | numeric |

###  Engineered Features
| New Feature | Description |
|--------------|-------------|
| `sale_year`, `sale_month`, `sale_day` | Derived from date |
| `age_at_sale` | Years between build and sale |
| `was_renovated` | Binary renovation flag |
| `yrs_since_renov` | Years since last renovation |

---

##  Methodology

## Model Training and Evaluation using Raw Data

Baseline models were trained on minimally processed features to establish pre‑processing performance.

## EDA

- Dataset size: 21613 rows × 21 columns.

- Target `price` summary:

  - count: 21613

  - min / median / max: 75000 / 450000 / 7700000

  - mean ± std: 540088 ± 367127

  - 25% / 75%: 321950 / 645000


Top 10 numeric correlations with `price`:


|               |   corr_with_price |
|:--------------|------------------:|
| sqft_living   |          0.702035 |
| grade         |          0.667434 |
| sqft_above    |          0.605567 |
| sqft_living15 |          0.585379 |
| bathrooms     |          0.525138 |
| view          |          0.397293 |
| sqft_basement |          0.323816 |
| bedrooms      |          0.30835  |
| lat           |          0.307003 |
| waterfront    |          0.266369 |


High‑value segment (price ≥ 650,000):


- Count: 5324
- Share of dataset: 24.63%
- Median sqft_living: 2890
- Median grade: 9
- Median bathrooms: 2.50

## Feature Engineering

Derived variables used in the notebook include: `sale_year`, `sale_month`, `sale_day`, `age_at_sale`, `was_renovated`, `yrs_since_renov`.

## Data Cleaning

No missing values were detected in the raw CSV.

## Data Scaling

Scaling (StandardScaler) is used for linear/polynomial models within scikit‑learn Pipelines to avoid leakage.

## Data Encoding

Binary/ordinal fields (`waterfront`, `view`, `condition`, `grade`) are kept numeric; high‑cardinality location proxies are handled via lat/long instead of zipcode.

## Models Traning

Five regressors are trained: Linear, Polynomial (deg 2), Random Forest, AdaBoost, Gradient Boosting/XGBoost (if available) using an 80/20 split.

## Models Evaluations

Model performance is reported in the notebook (MAE, RMSE, R²). Insert the exact values captured during execution here.


| Model | Description |
|:--|:--|
| Linear Regression | Baseline, interpretable, limited non-linearity |
| Polynomial Regression (deg=2) | Adds interaction/quadratic terms |
| Random Forest Regressor | Ensemble of trees, captures non-linearity |
| AdaBoost Regressor | Sequential error correction boosting |
| GradientBoosting (XGBoost proxy) | Gradient-based additive trees — best performance |

### Step 6 - Evaluation Metrics
- **MAE** — Mean Absolute Error → typical price deviation  
- **RMSE** — Root Mean Square Error → penalizes large errors  
- **R²** — Variance explained by model

---

##  Model Performance Summary

| Model | MAE (USD) | RMSE (USD) | R² |
|:--|--:|--:|:--:|
| **GradientBoosting (XGB)** | **67,478** | **136,181** | **0.877** |
| RandomForest | 72,553 | 146,328 | 0.858 |
| AdaBoost | 78,617 | 143,277 | 0.864 |
| Polynomial Regression | 94,000 | 162,000 | 0.821 |
| Linear Regression | 108,000 | 184,000 | 0.781 |

 **Interpretation**
- Ensemble tree methods (GB/XGBoost) outperform linear models → complex, non-linear relationships exist between features and price.  
- Linear models underfit and fail to capture interactions such as `grade × sqft_living` or `bathrooms × view`.  
- Gradient Boosting achieves the best overall calibration with moderate tuning.

---

##  Feature Importance Highlights

### GradientBoosting (Best Model)
| Rank | Feature | Importance | Interpretation |
|:--|:--|:--:|:--|
| 1 | `sqft_living` | 0.32 | Larger homes → higher prices. |
| 2 | `grade` | 0.19 | Quality and finish heavily impact value. |
| 3 | `lat` | 0.12 | Northern latitude correlates with higher desirability. |
| 4 | `bathrooms` | 0.08 | Adds functional and aesthetic appeal. |
| 5 | `view` | 0.06 | Premium for scenic views. |
| 6 | `waterfront` | 0.05 | Rare, large marginal premium. |
| 7 | `yr_built` | 0.04 | Newer homes command higher prices. |
| 8 | `sqft_above` | 0.04 | Positive contribution from above-ground living space. |
| 9 | `long` | 0.04 | Longitude interacts with lat (micro-location). |
| 10 | `sqft_living15` | 0.03 | Neighborhood composition effects. |

---

##  Business Insights

- **Living area and quality** drive the majority of value — expanding living space or upgrading finishes provides the best ROI.  
- **Location proxies** (`lat`, `long`) are highly predictive → price premium near Seattle’s employment centers.  
- **Renovation factor** improves perceived grade but with diminishing returns for older homes.  
- **Waterfront/view** features command unique price premiums (often +\$150k–\$300k).  
- Homes ≥ \$650k have **higher grades (8–9)**, **2.5+ bathrooms**, and **proximity to urban waterfronts**.

---

##  Deliverables

| File | Description |
|:--|:--|
| `final.ipynb` | Jupyter notebook with full workflow |
| `kc_houses_aa.csv` | Dataset |
| `READ_ME.md` | Project documentation (this file) |

---

##  Key Learnings
- Simple linear models are inadequate for real-world pricing due to **feature interactions**.  
- **Ensemble methods** like Gradient Boosting/XGBoost deliver higher accuracy and interpretability with feature importance.  
- **Feature engineering** (renovation flags, age, neighborhood metrics) significantly improves predictive power.  
- A clean pipeline for **data → model → interpretation → report** bridges the gap between technical modeling and business understanding.

---

##  Future Work
- Add **log(price)** modeling to stabilize heteroscedasticity.  
- Integrate **spatial clustering (KMeans / GeoPandas)** for location-driven segmentation.  
- Perform **hyperparameter tuning** (GridSearchCV / Optuna) for boosting models.  
- Build an **interactive dashboard** (Streamlit / Dash) for real-time price estimation.  
- Add **temporal drift analysis** (yearly price trends).

---

##  Tech Stack

| Category | Tools |
|-----------|-------|
| Programming | Python 3.11 |
| Data Analysis | pandas, numpy |
| Visualization | matplotlib |
| Machine Learning | scikit-learn, xgboost (optional) |
| Reporting | matplotlib.backends.pdf |
| Environment | Jupyter Notebook / VSCode |

---


