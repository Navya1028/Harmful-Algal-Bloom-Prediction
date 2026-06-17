# HAB Risk Monitor

A machine learning-powered dashboard that predicts **Harmful Algal Bloom (HAB) risk** in US lakes. Given a lake's nutrient levels, climate measurements, and watershed characteristics, the system estimates **chlorophyll-a concentration** and classifies the lake into **Safe, Caution, Warning, or Danger** risk categories.

The project combines **environmental science, feature engineering, CatBoost machine learning, SHAP explainability, and interactive visualization** to support bloom-risk assessment and freshwater ecosystem monitoring.

---

### Live App

🔗 [Launch Dashboard](https://harmful-algal-bloom-prediction-eqstynrfgp3qjlyhbb5yoj.streamlit.app/)

---

### Why this Project Matters

- **Harmful algal blooms (HABs)** are becoming increasingly common in lakes and reservoirs due to **nutrient pollution, rising temperatures, and changing watershed conditions**.
- Severe blooms can release toxins that threaten **drinking water supplies, aquatic ecosystems, recreational activities, local economies, and public health**.
- Monitoring blooms at scale is challenging because **direct chlorophyll-a measurement requires field sampling, laboratory analysis, and significant operational resources**.
- Conducting frequent measurements across the **100,000+ lakes in the United States** is often impractical, expensive, and time-consuming.
- Many environmental factors that influence bloom formation—such as **nutrient concentrations, climate conditions, and watershed characteristics**—are already collected through large-scale monitoring programs.
- This project investigates whether **machine learning can leverage these readily available variables to estimate chlorophyll-a concentration and HAB risk**.
- The model predicts chlorophyll-a levels and classifies lakes into **Safe, Caution, Warning, and Danger** risk categories based on bloom severity thresholds.
- Rather than replacing laboratory testing, the system is intended as an **early-warning and decision-support tool** that can help identify lakes requiring closer monitoring before severe blooms develop.
- By combining **environmental science, data analysis, and machine learning**, this project demonstrates how predictive models can support **proactive freshwater ecosystem management and environmental risk assessment**.

---

### Project Objectives

- Predict **chlorophyll-a concentration** using environmental, climatic, and watershed variables.
- Classify lakes into **Safe, Caution, Warning, and Danger** HAB risk categories based on bloom severity thresholds.
- Compare multiple machine learning approaches and identify the **best-performing model**.
- Improve predictive performance through **domain-informed feature engineering**.
- Provide an **interactive Streamlit dashboard** for real-time bloom risk assessment.
- Enable users to explore how changes in **nutrient loading, climate conditions, and watershed characteristics** influence HAB risk.
- Demonstrate how **machine learning can support environmental monitoring, freshwater management, and early-warning systems**.

---

### Dataset

[Estimates of lake nitrogen, phosphorus, and chlorophyll-a concentrations to characterize harmful algal bloom risk across the United States](https://catalog.data.gov/dataset/estimates-of-lake-nitrogen-phosphorus-and-chlorophyll-a-concentrations-to-characterize-har)

Published by the EPA (Brehob et al.), this dataset combines information from:

- EPA National Lakes Assessment (NLA)
- PRISM climate records
- National Nutrient Inventory (NNI)
- LakeCat watershed characteristics

`RF_Inputs_Normalized.csv` in this repository is the version of the dataset used for model development and evaluation.

---

### What's in this Repository

```text
.
├── app.py
├── hab_predictor.py
├── hab_model.pkl
├── train_model.py
├── RF_Inputs_Normalized.csv
├── requirements.txt
├── plots/
└── README.md
```

---

### How the Model Works

Raw lake measurements undergo extensive feature engineering before reaching the model. Features include:

- Nitrogen-to-Phosphorus (N:P) ratio
- Log transformations of highly skewed variables
- Total anthropogenic nutrient pressure
- Peak bloom season indicators (July–August)
- Watershed-to-lake area ratio
- Climate anomaly features
- Nutrient interaction terms
- Legacy phosphorus loading indicators

The model predicts **log-transformed chlorophyll-a concentration**, which is converted back to **µg/L** and categorized into bloom risk levels.

| Level | Category | Chlorophyll-a | Interpretation |
|---------|----------|----------|----------|
| 1 | Safe | < 10 µg/L | No action needed |
| 2 | Caution | 10–50 µg/L | Increased monitoring recommended |
| 3 | Warning | 50–100 µg/L | Limit recreational water contact |
| 4 | Danger | > 100 µg/L | High bloom risk; intervention required |

---

## Results & Visualizations

### Target Distribution and Seasonal Bloom Pattern
![EDA overview](plots/brehob_eda.png)

### Feature Correlations with Chlorophyll-a
![Correlation heatmap](plots/correlation_heatmap_fixed.png)

### Outlier Detection
![Outlier detection](plots/outlier_detection.png)

### R² Comparison Across Models
![R2 comparison](plots/r2_comparison.png)

### RMSE Comparison Across Models
![RMSE comparison](plots/rmse_comparison.png)

### Predicted vs Actual Chlorophyll-a
![Predicted vs actual](plots/combined_predicted_vs_actual.png)

### Residual Analysis
![Residuals](plots/combined_residual_plots.png)

### 5-Fold Cross Validation Performance
![CV summary](plots/cv_summary_all_models.png)

### Risk Classification Confusion Matrix
![Confusion matrix](plots/risk_classification_confusion_matrix.png)

### Error Analysis: CatBoost vs Random Forest
![Error analysis](plots/error_analysis_catboost_vs_rf.png)

### SHAP Feature Importance
![SHAP importance](plots/shap_importance.png)

### SHAP Waterfall Explanation
![SHAP waterfall](plots/shap_waterfall_example.png)

---

### Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

pip install -r requirements.txt
streamlit run app.py
```

---

### Tech Stack

- **Machine Learning:** CatBoost, Random Forest, XGBoost, LightGBM
- **Data Processing:** pandas, NumPy
- **Visualization:** Plotly, Matplotlib, SHAP
- **Dashboard:** Streamlit
- **Model Persistence:** joblib
- **Deployment:** Streamlit Community Cloud

---

## Disclaimer

This project is intended for **research, exploration, and decision-support purposes only**. Predictions should not be treated as a substitute for laboratory-measured chlorophyll-a concentrations, field sampling, or official water-quality assessments.
