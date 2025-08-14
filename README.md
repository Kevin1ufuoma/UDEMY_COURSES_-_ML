Udemy Courses — End-to-End ML Project (Portfolio)

## What I set out to do
I built a complete, reproducible machine learning workflow that predicts **course popularity** (classification) and **subscriber counts** (regression) using Udemy course metadata. My goal was to demonstrate rigorous **EDA**, careful **data cleaning**, meaningful **feature engineering**, and transparent **model evaluation** with cross-validation.

## Dataset & Target
- **Source:** Provided Udemy Courses CSV (columns may vary by release).
- **Targets:**
  - **Classification:** `popular` — 1 if `num_subscribers` is in the top 25% (>= 75th percentile), else 0.
  - **Regression:** raw `num_subscribers` to estimate expected audience size.

## EDA Highlights
- Course **subjects** and **levels** show distinct pricing and review patterns.
- **Price**, **number of reviews**, **number of lectures**, and **content duration** correlate with subscriber counts.
- I engineered time-based features (e.g., `course_age_days`, `pub_year`, `pub_month`, `pub_dow`) from the published timestamp.

## Data Cleaning & Feature Engineering
- Normalized column names, parsed `published_timestamp`, converted `price`/`content_duration` to numeric, and enforced `price=0` for free courses.
- Imputed missing numeric values with medians and categorical values with `"Unknown"`.
- One-hot encoded categorical features (`subject`, `level`) and standardized numeric features before modeling.

## Modeling Approach
**Classification (Popularity):**
- Models: Logistic Regression, Random Forest, Gradient Boosting.
- Evaluation: **Accuracy, Precision, Recall, F1, ROC-AUC** on a holdout test set; **5-fold Stratified CV** on the train split.
- I selected the best classifier by **F1** (balances precision & recall).

**Regression (Subscribers):**
- Models: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor.
- Evaluation: **MAE, RMSE, R²** on the holdout; **5-fold CV** on the train split.
- I selected the best regressor by **MAE** (business-friendly absolute error).

## Results
- **Classification:** See `results/classification_results.csv`. I also export a **confusion matrix** and **ROC curve** for the best model.
- **Regression:** See `results/regression_results.csv` plus a **Predicted vs Actual** plot for the top regressor.
- **Figures**: saved under `figs/` (EDA charts, heatmap, importances, ROC).

## How to Reproduce
1. Create a fresh environment and install `requirements.txt`.
2. Run `python scripts/udemy_ml_project.py` — this will re-generate results and figures.
3. (Optional) Load the saved models from `models/` to make predictions without retraining.

## Business Interpretation
- The classification model can flag **high-potential courses** early, informing **marketing spend**, **instructor partnerships**, and **platform promotion**.
- The regression model quantifies expected **subscriber lift** from levers like **price**, **content duration**, and **lectures**, guiding **pricing** and **content strategy**.

## Files
- `scripts/udemy_ml_project.py` — a fully commented script that implements the entire pipeline.
- `results/` — CSVs with metrics and evaluation plots.
- `figs/` — EDA visualizations and model diagnostics.
- `models/` — serialized best models and scaler.
