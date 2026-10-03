# Atrial Fibrillation Detection and Early Pre-AF Warning from ECG

Detecting atrial fibrillation (AF) and identifying the **pre-AF period (up to 5 minutes before onset)** from single ECG segments, using wavelet-based features, classical machine learning and deep learning.

## Overview

| | |
|---|---|
| **Task** | 3-class classification: Normal Sinus Rhythm (NSR), AF, Pre-AF |
| **Data** | PhysioNet MIT-BIH AF Database (AFDB), MIT-BIH Normal Sinus Rhythm Database (NSRDB), PAF Prediction Challenge Database (AFPDB) |
| **Signal processing** | 10 s windows with 5 s overlap, Discrete Wavelet Transform (db8, 4 levels), Wavelet Packet Decomposition, Continuous Wavelet Transform (scalograms for visual validation) |
| **Models** | Random Forest, XGBoost, LightGBM, 1D-CNN, CNN+LSTM (PyTorch) |
| **Tools** | Python, NumPy, SciPy, PyWavelets, scikit-learn, XGBoost, LightGBM, PyTorch, SHAP |

## Results (record-based train/test split)

| Model | Accuracy | Macro-F1 | AUC (OvR) |
|---|---|---|---|
| CNN+LSTM | 95.71% | 95.50% | 99.42% |
| 1D-CNN | 95.29% | 95.11% | 99.13% |
| XGBoost | 88.24% | 87.65% | 96.21% |
| LightGBM | 87.50% | 86.89% | 95.82% |
| Random Forest | 85.45% | 84.79% | 95.43% |

**Early detection:** 96.9% of pre-AF segments in the test set were correctly flagged (95% CI: 94.1–98.3%), on average 2.6 minutes and at the earliest 5 minutes before AF onset.

## Methodology highlights

- **Record-based splitting** (GroupShuffleSplit / GroupKFold) so that segments from the same patient never appear in both training and test sets, preventing data leakage.
- **Ablation studies** for wavelet type and decomposition level (db8, level 4 selected) and a fair DWT vs. WPD comparison.
- **Dataset bias check:** per-dataset test analysis to make sure the model learns rhythm characteristics, not database-specific artefacts.
- **Explainability and statistics:** SHAP feature importance, McNemar test between models, confusion-matrix error analysis with clinical interpretation (missing AF is treated as riskier than missing pre-AF).

## How to run

The notebook was developed in Google Colab. Download the three PhysioNet databases, place them in your Google Drive, and update the data paths in the first cells. A GPU runtime is recommended for the deep learning models.

```
pip install -r requirements.txt
```

## Notes

Results are from a single record-based split; 5-fold record-based cross-validation for the classical models gave 78–82% accuracy, showing that performance varies across patients. Further validation on external data is needed before any clinical use.
