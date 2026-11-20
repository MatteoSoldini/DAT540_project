# DAT540 Project - Explainable Machine Learning for Gallstone Disease

DAT540 group project at the University of Stavanger.

We predict gallstone disease from routine, non-imaging data (demographic, bioimpedance and laboratory measurements), and investigate which clinical characteristics contribute most to the predictions.

## Dataset

The [Gallstone dataset](https://archive.ics.uci.edu/dataset/1150/gallstone-1) from the UCI Machine Learning Repository (Esen et al., 2024): 319 patients and 38 features, collected at Ankara VM Medical Park Hospital between 2022 and 2023. The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Place the dataset file at `dataset/dataset-uci.xlsx`.

## Repository structure

| File | Content |
|---|---|
| `1_data_exploration.ipynb` | Data review, visualizations, outlier handling, skewed features, removal of highly correlated features |
| `2_model_selection.ipynb` | Train/test split, model pipelines, hyperparameter tuning with cross-validation |
| `3_valid_eval.ipynb` | Model comparison on cross-validation and test set, permutation feature importance, EBM explanations |

Each notebook builds on the previous one through `%run`, so `3_valid_eval.ipynb` runs the whole pipeline.

## Setup

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

## Usage

```bash
jupyter notebook
```

Open and run the notebooks in order (`1_` → `2_` → `3_`), or run only `3_valid_eval.ipynb` to reproduce all results. The hyperparameter search takes a few minutes.

## Method

1. **Cleaning:** values more than 10 standard deviations from the mean were checked clinically. Two implausible records were removed and one decimal error was corrected, leaving 317 patients.
2. **Feature selection:** of each pair of features with correlation above 0.9, one was removed by a greedy rule, leaving 33 features.
3. **Models:** logistic regression, logistic regression with log-transformed skewed features, an Explainable Boosting Machine (EBM) and XGBoost.
4. **Evaluation:** stratified 80/20 train/test split, hyperparameters tuned with 5-fold cross-validation on the training set, and ROC AUC as the main metric.
5. **Feature importance:** permutation importance computed across 25 cross-validation folds.

## Results

| Model | CV ROC AUC | Test ROC AUC |
|---|---|---|
| Logistic regression + log | 0.890 ± 0.045 | 0.873 |
| EBM | 0.887 ± 0.046 | 0.892 |
| Logistic regression | 0.864 ± 0.041 | 0.854 |
| XGBoost | 0.837 ± 0.048 | 0.877 |

The interpretable models perform as well as XGBoost. C-reactive protein (CRP) and vitamin D are the most consistently important features across models.

## Authors

- Matteo Soldini: data exploration; logistic regression with and without log transformation
- Syed Sakibur Rahman: validation and evaluation; Explainable Boosting Machine
- Irtaza Bin Ashar: data exploration; XGBoost

## Reference

Esen, I., Arslan, H., Aktürk Esen, S., Gülşen, M., Kültekin, N., & Özdemir, O. (2024). Early prediction of gallstone disease with a machine learning-based method from bioimpedance and laboratory data. *Medicine*, 103(8), e37258. https://doi.org/10.1097/MD.0000000000037258