# ML-CW1-KCL

Machine Learning Coursework 1 — Diamond Outcome Prediction

## Overview

A regression pipeline to predict a continuous `outcome` variable from diamond attributes and engineered features. Final model achieves **R² = 0.4780** using a blended XGBoost/LightGBM approach.

## Repository Structure

```
ML-CW1-KCL/
├── data/
│   └── raw/                  # CW1_train.csv, CW1_test.csv
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_model_selection.ipynb  # Baseline model comparison (8 models)
│   ├── 03_model_training.ipynb   # XGBoost vs LightGBM tuning
│   └── 04_model_tuning.ipynb     # Fine-tuning and final submission
├── reports/
│   └── figures/              # Generated plots
├── scripts/
│   └── CW1_eval_script.py   # Provided evaluation script
├── requirements.txt
└── .gitignore
```

## Pipeline

1. **EDA** — Inspected distributions, correlations, outliers. Identified `depth` as strongest predictor (r = -0.41).
2. **Preprocessing** — Ordinal encoding of categoricals, median imputation of zero x/y/z values, feature engineering (polynomial depth terms, interaction terms, squared features). 30 → 50 features.
3. **Model Selection** — Compared 8 models via 5-fold CV. Tree-based models (R² ≈ 0.46) outperformed linear models (R² ≈ 0.29).
4. **Tuning** — Fine-tuned XGBoost and LightGBM. Best configs use shallow trees (depth=3), slow learning rates, and strong regularisation.
5. **Blending** — 75% XGBoost / 25% LightGBM blend. Final OOF R² = 0.4780.

## Requirements

```
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```

## Author

K23115695
