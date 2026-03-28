# 🌾 Crop Yield Prediction & Driver Analysis using Machine Learning

> An end-to-end Machine Learning pipeline to **predict crop yield** and **identify the key factors** driving agricultural productivity — powered by XGBoost and explained with SHAP.

<br>

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-orange?style=flat-square)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green?style=flat-square)
![R2 Score](https://img.shields.io/badge/R²%20Score-91.28%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📌 Overview

Most ML projects stop at making predictions. This one goes further.

The **Crop Yield Driver Discovery System** not only predicts how much a crop will yield — it explains *why*, using a combination of 6 feature selection methods and SHAP-based interpretability. This makes it useful not just for data scientists, but for farmers, agronomists, and policymakers who need to understand what actually moves the needle on yield.

---

## 🎯 Objectives

- Predict crop yield (tons/hectare) with high accuracy
- Discover the most influential factors (rainfall, irrigation, fertilizer, etc.)
- Compare multiple ML models and select the best one
- Provide explainable, actionable insights using SHAP
- Build a reproducible, production-quality ML pipeline

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Source | [Kaggle — Agriculture Crop Yield Dataset](https://www.kaggle.com/datasets/samuelotiattakorah/agriculture-crop-yield) |
| Size | 1,000,000+ records |
| Target | `Yield_tons_per_hectare` |

**Features:**

| Feature | Description |
|---------|-------------|
| `Region` | Geographic region of farming |
| `Soil_Type` | Type of soil (Sandy, Loamy, etc.) |
| `Crop` | Crop variety (Rice, Wheat, Cotton, etc.) |
| `Rainfall_mm` | Annual rainfall in mm |
| `Temperature_Celsius` | Average temperature (°C) |
| `Fertilizer_Used` | Whether fertilizer was used (Yes/No) |
| `Irrigation_Used` | Whether irrigation was used (Yes/No) |
| `Weather_Condition` | General weather condition |
| `Days_to_Harvest` | Days from planting to harvest |

---

## 🧠 Tech Stack

| Category | Libraries |
|----------|-----------|
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `ydata-profiling` |
| Feature Selection | `scikit-learn` (Lasso, RFE, F-test, MI) |
| ML Models | `scikit-learn`, `xgboost` |
| Hyperparameter Tuning | `RandomizedSearchCV` |
| Explainability | `shap` |
| Model Persistence | `joblib` |

---

## ⚙️ Project Workflow

```
Raw Data
   │
   ├── 1. EDA ──────────────── Distribution plots, heatmaps, boxplots
   │
   ├── 2. Feature Engineering ─ Rainfall×Temp Index, Water Availability,
   │                            Crop Category, Binary encodings
   │
   ├── 3. Train-Test Split ──── 80/20 split (leakage-free target encoding)
   │
   ├── 4. Feature Selection ─── 6 methods with voting ensemble:
   │                            Pearson Correlation, ANOVA F-test,
   │                            Mutual Information, Lasso (L1),
   │                            RFE (Random Forest), XGBoost Importance
   │
   ├── 5. Model Training ──────  Linear Regression
   │                             Random Forest
   │                             Linear SVR
   │                             XGBoost ⭐ (tuned with RandomizedSearchCV)
   │
   ├── 6. Evaluation ─────────── R², RMSE, MAE, MAPE + 5-fold CV
   │
   ├── 7. Model Saving ────────── joblib → xgboost_crop_yield_model.pkl
   │
   └── 8. SHAP Analysis ───────── Global summary, dependence plots,
                                  error analysis by region & crop
```

---

## 📈 Results

### Model Comparison

| Model | R² Score | RMSE | MAPE | Notes |
|-------|----------|------|------|-------|
| Linear Regression | ~0.65 | — | — | Baseline |
| Random Forest | ~0.85 | — | — | Good, slower |
| Linear SVR | ~0.70 | — | — | Fast, linear |
| **XGBoost ⭐** | **0.9128** | — | **12.68%** | Best model |

> Exact RMSE and MAE values are printed when you run the notebook.

### Key Drivers Identified (via SHAP)

1. 💧 **Fertilizer Usage** — highest impact on yield
2. 🌊 **Irrigation** — significantly boosts yield
3. 💦 **Water Availability** — engineered feature; strong predictor
4. 🌧️ **Rainfall (mm)** — direct climate driver
5. 🌡️ **Rainfall × Temperature Index** — interaction effect

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/crop-yield-prediction.git
cd crop-yield-prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn ydata-profiling scikit-learn xgboost shap joblib
```

### 3. Add the dataset

Download `crop_yield.csv` from [Kaggle](https://www.kaggle.com/datasets/samuelotiattakorah/agriculture-crop-yield) and place it in the **root of the project folder** (same directory as the notebook).

```
crop-yield-prediction/
├── crop_yield_prediction.ipynb   ← notebook
├── crop_yield.csv                ← dataset (download from Kaggle)
├── xgboost_crop_yield_model.pkl  ← saved after first run
└── README.md
```

### 4. Run the notebook

```bash
jupyter notebook crop_yield_prediction.ipynb
```

Run all cells from top to bottom. The trained model will be automatically saved as `xgboost_crop_yield_model.pkl`.

### 5. Load the saved model (optional)

```python
import joblib
model = joblib.load("xgboost_crop_yield_model.pkl")
predictions = model.predict(X_new)
```

---

## 📁 Project Structure

```
crop-yield-prediction/
│
├── crop_yield_prediction.ipynb   # Main notebook (EDA → Training → SHAP)
├── crop_yield.csv                # Dataset (not included — download from Kaggle)
├── xgboost_crop_yield_model.pkl  # Saved model (generated after running)
└── README.md                     # Project documentation
```

---

## 🔍 Key Features

- ✅ **End-to-end ML pipeline** from raw CSV to saved model
- ✅ **6 feature selection methods** with voting ensemble
- ✅ **Hyperparameter tuning** using `RandomizedSearchCV`
- ✅ **5-fold cross-validation** to verify no overfitting
- ✅ **Explainable AI** — SHAP global summary + dependence plots
- ✅ **Rich EDA** — distributions, boxplots, correlation heatmap
- ✅ **Error analysis** broken down by crop type and region
- ✅ **Model persistence** — save and reload without retraining
- ✅ **Leakage-free** target encoding (computed on train set only)

---
## 👤 Author

**Rishi Raghuvanshi**

- 🌐 GitHub: [@rishiraghu11](https://github.com/rishiraghu11)
- 💼 LinkedIn: [Rishi Raghuvanshi](https://www.linkedin.com/in/rishi-raj-singh-raghuvanshi/)
- 📧 Email: raghuvanshi11rishi@gmail.com

---

## 📚 References

1. Ravi, R. et al. — *Crop Yield Prediction using XGBoost Algorithm.* ResearchGate, 2022.
2. Shahhosseini, M. et al. — *Forecasting corn yield with machine learning ensembles.* Computers and Electronics in Agriculture, 2022.
3. van Klompenburg, T. et al. — *Crop yield prediction using machine learning: A systematic literature review.* Computers and Electronics in Agriculture, 2020.
4. You, J. et al. — *Deep Gaussian Process for Crop Yield Prediction.* arXiv, 2017.
5. Crane-Droesch, A. — *Machine learning methods for crop yield prediction.* PLOS ONE, 2018.
6. Dataset: Samuel Oti Attakorah — *Agriculture Crop Yield Dataset.* Kaggle.

---

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and distribute with attribution.
