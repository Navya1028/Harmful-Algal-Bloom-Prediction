# HAB Risk Monitor

A Streamlit dashboard that predicts Harmful Algal Bloom (HAB) risk for US lakes,
using a CatBoost model trained on EPA National Lakes Assessment data, PRISM
climate records, NNI nutrient inputs, and LakeCat watershed characteristics.

**Live dashboard:** _add your Streamlit Cloud link here once deployed (see
"Deploying so others can use it" below)_

---

## What the dashboard does

Enter lake sample measurements (nutrients, climate, lake geometry, watershed
nutrient pressure) and get back:

- A predicted chlorophyll-a concentration (µg/L)
- A WHO-aligned risk category: Safe / Caution / Warning / Danger
- A recommended action for that risk level
- Supporting charts: a WHO threshold gauge, N:P nutrient balance, which inputs
  are driving the prediction, and seasonal bloom context

## Project structure

```
.
├── app.py                      # Streamlit dashboard (run this)
├── hab_predictor.py            # HABPredictor class — required to load hab_model.pkl
├── hab_model.pkl                # Trained CatBoost pipeline
├── train_model.py               # Training script (reference only — not run by the dashboard)
├── RF_Inputs_Normalized.csv     # Training dataset
├── requirements.txt
├── .gitignore
├── plots/                       # All EDA / model evaluation charts (also shown below)
└── README.md
```

## Dataset

`RF_Inputs_Normalized.csv` — EPA National Lakes Assessment survey data
(May–October bloom season), combined with:
- PRISM climate variables (temperature, precipitation, land surface temperature, NPP)
- NNI net anthropogenic nutrient inputs (agricultural, urban, livestock, human waste)
- LakeCat watershed characteristics (wetlands, soils, baseflow index)

Target variable: `logchl_A`, the log10-transformed chlorophyll-a concentration (µg/L).

## Model

CatBoost Regressor, trained on engineered features (N:P ratio, log transforms,
nutrient pressure totals, seasonal flags, watershed ratios, climate anomalies,
baseflow interactions, legacy phosphorus interaction). Compared against Random
Forest, XGBoost, LightGBM, and several linear/kernel baselines — CatBoost gave
the best held-out test R².

Risk categories (WHO-aligned chlorophyll-a thresholds):

| Level | Category | Chlorophyll-a | Action |
|---|---|---|---|
| 1 | Safe | < 10 µg/L | No action needed |
| 2 | Caution | 10–50 µg/L | Post signage, monitor weekly |
| 3 | Warning | 50–100 µg/L | Limit water contact |
| 4 | Danger | > 100 µg/L | Close water body |

`train_model.py` contains the full pipeline: data cleaning, feature
engineering, train/validation/test split, model training and comparison,
cross-validation, and the final model-saving step. It's included for
reference — the dashboard only needs `hab_model.pkl` and `hab_predictor.py`
to run; it does not re-run training.

## Exploratory data analysis & model evaluation

**Target distribution and seasonal pattern**

![EDA overview](plots/brehob_eda.png)

**Feature correlations**

![Correlation heatmap](plots/correlation_heatmap_fixed.png)

**Outlier check on key raw variables**

![Outlier detection](plots/outlier_detection.png)

**Model comparison — R² across Random Forest, XGBoost, CatBoost**

![R2 comparison](plots/r2_comparison.png)

**Model comparison — RMSE**

![RMSE comparison](plots/rmse_comparison.png)

**Predicted vs. actual chlorophyll-a (test set)**

![Predicted vs actual](plots/combined_predicted_vs_actual.png)

**Residual plots**

![Residuals](plots/combined_residual_plots.png)

**5-fold cross-validation summary**

![CV summary](plots/cv_summary_all_models.png)

**Risk classification confusion matrix (CatBoost)**

![Confusion matrix](plots/risk_classification_confusion_matrix.png)

**Error analysis by bloom severity (CatBoost vs. Random Forest)**

![Error analysis](plots/error_analysis_catboost_vs_rf.png)

**Feature importance (SHAP)**

![SHAP importance](plots/shap_importance.png)

**SHAP waterfall example**

![SHAP waterfall](plots/shap_waterfall_example.png)

> If any filename above doesn't match what's in your `plots/` folder, rename
> the file to match, or edit the path in this README — GitHub just needs the
> exact filename to display the image.

## Running locally

```bash
git clone https://github.com/<your-username>/hab-risk-monitor.git
cd hab-risk-monitor
pip install -r requirements.txt
streamlit run app.py
```

The app looks for `hab_model.pkl` in the same folder as `app.py`. If you keep
your model file somewhere else, set an environment variable instead of
editing the code:

```bash
# Windows (PowerShell)
$env:HAB_MODEL_PATH = "D:\algal4\hab_model.pkl"
streamlit run app.py

# macOS/Linux
export HAB_MODEL_PATH=/path/to/hab_model.pkl
streamlit run app.py
```

## Re-training / re-saving the model

`hab_predictor.py` defines the `HABPredictor` class used both at training
time and inside the dashboard. When you retrain (see `train_model.py`),
import the class instead of redefining it inline, so the saved pickle stays
loadable from any script:

```python
from hab_predictor import HABPredictor
# ... build your hab_predictor object as usual ...
joblib.dump(hab_predictor, "hab_model.pkl")
```

## Deploying so others can interact with it live

Pushing this to GitHub makes the code and charts visible, but a Streamlit app
needs a running Python process to respond to clicks — GitHub alone can't do
that. Use [Streamlit Community Cloud](https://streamlit.io/cloud) (free):

1. Go to share.streamlit.io and sign in with GitHub
2. Click **New app**, select this repo, branch `main`, main file `app.py`
3. Click **Deploy**

This gives you a public URL where anyone can open the dashboard and use the
sidebar, Predict button, and charts live. Paste that link at the top of this
README once you have it. Every `git push` afterward redeploys automatically.

## Disclaimer

Built for exploratory and decision-support use. Not a substitute for
laboratory chlorophyll-a measurement or regulatory water-quality testing.
