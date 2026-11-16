# %% [markdown]
# # Explainable Boosting Machine (EBM) for Gallstone Prediction
#
# **Model:** `Explainable Boosting Classifier (EBM)` from Microsoft's [`interpret`](https://interpret.ml) library.
#
# An EBM is a *glass-box* model. It is a Generalized Additive Model (GAM) trained with boosting:
#
# $$\text{logit}\big(P(\text{gallstone})\big) = \beta_0 + \sum_i f_i(x_i) + \sum_{(i,j)} f_{ij}(x_i, x_j)$$
#
# - Each feature gets its own learned shape function $f_i$, which shows exactly how that feature changes the risk.
# - A few pairwise interactions $f_{ij}$ are added automatically when they help.
# - Every prediction is a simple sum of per-feature contributions, so it can be explained exactly. No approximation such as SHAP is needed.
#
# That makes EBM a natural fit for this project. It is often as accurate as XGBoost or Random Forest, but it is as interpretable as logistic regression.
#
# **Pipeline** (kept identical to `script.ipynb` so the results are directly comparable)
# 1. Load the data and apply the same cleaning (drop rows 40 and 318, fix row 239, drop the 5 highly correlated features)
# 2. Stratified 80/20 train/test split (`random_state=42`, the same test set as `script.ipynb`)
# 3. Baseline EBM with default settings
# 4. Hyperparameter tuning with randomized search and 5-fold CV (scored on ROC-AUC)
# 5. Repeated stratified CV (5 folds × 5 repeats) compared with the existing models
# 6. Test-set evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix and ROC curves
# 7. Explainability: global term importances, shape functions, interactions and local explanations of single patients
# 8. Permutation importance across CV folds (same method as `script.ipynb`)
# 9. Comparison table (also printed as LaTeX for the report)

# %%
# %pip install interpret
#

# %% [markdown]
# ## 1. Imports and configuration

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV,
                                     cross_validate)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay)

import interpret
from interpret import show
from IPython.display import display
from interpret.glassbox import ExplainableBoostingClassifier

print("interpret version:", interpret.__version__)

RANDOM_STATE = 42
DATA_PATH    = "dataset/dataset-uci.xlsx"
TARGET       = "Gallstone Status"
TEST_SIZE    = 0.20
N_ITER       = 20     # random-search iterations

# %% [markdown]
# ## 2. Load data

# %%
# The file is saved in the Strict Open XML format, so `calamine` is needed to read it
df = pd.read_excel(DATA_PATH, engine="calamine")
print("Shape:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicated rows:", int(df.duplicated().sum()))
df.head()

# %% [markdown]
# ## 3. Cleaning (same steps as `script.ipynb`)
#
# | Row | Problem | Action |
# |---|---|---|
# | 40 | Glucose = 575 in a patient **without** diabetes (implausible) | drop the row |
# | 239 | Obesity = 1954 %, a misplaced decimal (BMI 29.3 fits 19.54 %) | set to 19.54 |
# | 318 | HDL = 273 is inconsistent with TC and LDL (the Friedewald estimate gives about 137) | drop the row |
#
# Then drop every feature that has |r| > 0.9 with an earlier feature (5 features).

# %%
y = df[TARGET]
X = df.drop(columns=TARGET)

print("Row 40 :", X.loc[40, ["Glucose", "Diabetes Mellitus (DM)"]].to_dict())
print("Row 239:", X.loc[239, ["Body Mass Index (BMI)", "Obesity (%)"]].to_dict())
print("Row 318:", X.loc[318, ["Total Cholesterol (TC)", "Low Density Lipoprotein (LDL)",
                              "High Density Lipoprotein (HDL)"]].to_dict())

X = X.drop(index=[40, 318]); y = y.drop(index=[40, 318])
X.loc[239, "Obesity (%)"] = 19.54

# Remove highly correlated features (keep the first of each correlated pair)
corr = X.corr().abs()
keep = []
for c in X.columns:
    if not (corr.loc[c, keep] > 0.9).any():
        keep.append(c)
dropped = [c for c in X.columns if c not in keep]
X = X[keep]

print(f"\nDropped {len(dropped)} correlated features:", dropped)
print("Final shape:", X.shape)

# %% [markdown]
# ### Class balance

# %%
counts = y.value_counts().sort_index()
print(pd.DataFrame({"count": counts, "%": (counts / counts.sum() * 100).round(1)}))

# %% [markdown]
# ## 4. Train / test split (80 / 20, stratified)
#
# This uses the same seed and the same data as `script.ipynb`, so the **test patients are identical** and the results can be compared directly.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
print("Train:", X_train.shape, " Test:", X_test.shape)

# %% [markdown]
# ## 5. Feature types
#
# The EBM needs **no scaling and no log transform**. It bins each feature, so skewed lab values such as CRP and AST cause no trouble.
# The yes/no columns are declared as *nominal* (categorical), so the EBM learns one score per category instead of a curve.

# %%
binary_cols = [c for c in X.columns if X[c].nunique() == 2]
feature_types = ["nominal" if c in binary_cols else "continuous" for c in X.columns]
print("Nominal (binary) features:", binary_cols)

# %% [markdown]
# ## 6. Baseline EBM (default settings)
#
# This uses repeated stratified 5-fold CV on the **training set only**.

# %%
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
SCORING = ["roc_auc", "accuracy", "f1"]

def cv_summary(model, name):
    s = cross_validate(model, X_train, y_train, cv=rcv, scoring=SCORING)
    out = {"Model": name}
    for m in SCORING:
        out[f"CV {m}"] = s[f"test_{m}"].mean()
        out[f"CV {m} std"] = s[f"test_{m}"].std()
    return out

ebm_default = ExplainableBoostingClassifier(feature_types=feature_types,
                                            random_state=RANDOM_STATE)
base = cv_summary(ebm_default, "EBM (default)")
print(f"Default EBM  CV ROC-AUC: {base['CV roc_auc']:.3f} ± {base['CV roc_auc std']:.3f}")
print(f"             CV accuracy: {base['CV accuracy']:.3f} ± {base['CV accuracy std']:.3f}")

# %% [markdown]
# ## 7. Hyperparameter tuning (randomized search, 5-fold CV, ROC-AUC)
#
# | Hyperparameter | What it controls |
# |---|---|
# | `learning_rate` | How much each boosting round changes the shape functions (smaller is smoother and slower) |
# | `max_bins` | How finely each continuous feature is split into bins |
# | `interactions` | How many pairwise interaction terms $f_{ij}$ are added (0 gives a pure additive model) |
# | `min_samples_leaf` | Minimum number of patients per leaf. Higher values give smoother, less overfit curves |
# | `max_leaves` | Leaves per small tree inside each boosting step |
# | `outer_bags` | Number of bagged models averaged together (more is more stable and gives error bars) |
#
# This takes a few minutes, because each EBM already trains `outer_bags` models internally.

# %%
param_dist = {
    "learning_rate":    [0.005, 0.01, 0.02, 0.05],
    "max_bins":         [32, 64, 128, 256],
    "interactions":     [0, 3, 5, 10],
    "min_samples_leaf": [2, 4, 8, 16],
    "max_leaves":       [2, 3, 4],
    "outer_bags":       [8, 14],
}

search = RandomizedSearchCV(
    ExplainableBoostingClassifier(feature_types=feature_types, random_state=RANDOM_STATE),
    param_distributions=param_dist,
    n_iter=N_ITER,
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    random_state=RANDOM_STATE,
    n_jobs=1,          # EBM parallelises internally
    verbose=1,
)
search.fit(X_train, y_train)

print("Best CV ROC-AUC:", round(search.best_score_, 4))
print("Best params:", search.best_params_)

# Top 5 configurations
(pd.DataFrame(search.cv_results_)
   .sort_values("rank_test_score")
   [["params", "mean_test_score", "std_test_score"]]
   .head(5))

# %% [markdown]
# ## 8. Repeated CV comparison with the existing models
#
# The tuned EBM is scored with the **same 5×5 repeated CV** used in `script.ipynb`. The three models from `script.ipynb` are re-run here on the identical folds, so the comparison is fair.

# %%
skewed = [c for c in X_train.columns
          if X_train[c].nunique() > 5 and X_train[c].skew() > 1 and (X_train[c] >= 0).all()]

ebm_params = search.best_params_
models = {
    "EBM (tuned)": ExplainableBoostingClassifier(feature_types=feature_types,
                                                 random_state=RANDOM_STATE, **ebm_params),
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=5000)),
    "logreg + log": make_pipeline(
        ColumnTransformer([("log", FunctionTransformer(np.log1p), skewed)], remainder="passthrough"),
        StandardScaler(), LogisticRegression(C=1, max_iter=5000)),
    "random forest": RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE),
}

cv_rows = [base] + [cv_summary(m, name) for name, m in models.items()]
cv_table = pd.DataFrame(cv_rows).set_index("Model")
cv_table.round(3)

# %% [markdown]
# ## 9. Final fit and evaluation on the 20% test set

# %%
fitted, test_rows, probas = {}, [], {}
for name, m in models.items():
    m = clone(m).fit(X_train, y_train)
    fitted[name] = m
    proba = m.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    probas[name] = proba
    test_rows.append({"Model": name,
                      "Accuracy":  accuracy_score(y_test, pred),
                      "Precision": precision_score(y_test, pred),
                      "Recall":    recall_score(y_test, pred),
                      "F1":        f1_score(y_test, pred),
                      "ROC-AUC":   roc_auc_score(y_test, proba)})

test_table = pd.DataFrame(test_rows).set_index("Model").round(4)
ebm = fitted["EBM (tuned)"]
test_table

# %%
print(classification_report(y_test, ebm.predict(X_test),
                            target_names=["No gallstones (0)", "Gallstones (1)"], digits=4))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, ebm.predict(X_test)),
                       display_labels=["0", "1"]).plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title("Confusion matrix: EBM (tuned)")

for name, p in probas.items():
    RocCurveDisplay.from_predictions(y_test, p, name=name, ax=ax[1])
ax[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
ax[1].set_title("ROC curves (test set)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 10. Global explanation: which features matter?
#
# **Term importance** is the mean absolute contribution of each term (feature or interaction) to the log-odds, averaged over the training patients.

# %%
term_imp = pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values()

top = term_imp.tail(15)
colors = ["tab:orange" if " & " in t else "tab:blue" for t in top.index]
plt.figure(figsize=(8, 6))
plt.barh(top.index, top.values, color=colors)
plt.xlabel("Mean |contribution| to log-odds")
plt.title("EBM term importance (orange = interaction)")
plt.tight_layout(); plt.show()

term_imp.sort_values(ascending=False).round(4).to_frame("importance").head(15)

# %% [markdown]
# ### Interactive dashboard
#
# `show()` opens interpret's interactive explorer, where you can click through every feature's shape function. It works in Jupyter and VS Code.

# %%
show(ebm.explain_global(name="EBM (tuned)"))

# %% [markdown]
# ## 11. Shape functions: *how* each feature changes the risk
#
# Each plot is the learned $f_i(x_i)$. A value **above 0 raises** the log-odds of gallstones and a value **below 0 lowers** them. The shaded band is the uncertainty across the bagged models.
# This is the key explainability result. For example, it shows directly whether the risk rises steadily with CRP or jumps above a threshold.

# %%
glob = ebm.explain_global()
term_index = {t: i for i, t in enumerate(ebm.term_names_)}

def plot_shape(term, ax):
    d = glob.data(term_index[term])
    scores = np.asarray(d["scores"])
    lo, hi = d.get("lower_bounds"), d.get("upper_bounds")
    names = d["names"]
    if len(names) == len(scores) + 1:              # continuous: bin edges
        edges = np.asarray(names, dtype=float)
        ax.step(edges, np.r_[scores, scores[-1]], where="post", color="tab:blue")
        if lo is not None and hi is not None:
            ax.fill_between(edges, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]],
                            step="post", alpha=0.2, color="tab:blue")
    else:                                           # nominal: one bar per category
        ax.bar([str(n) for n in names], scores, color="tab:blue")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_title(term, fontsize=9)
    ax.set_ylabel("contribution to log-odds")

main_terms = [t for t in term_imp.sort_values(ascending=False).index if " & " not in t][:8]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for t, ax in zip(main_terms, axes.ravel()):
    plot_shape(t, ax)
plt.suptitle("Shape functions of the 8 most important features")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 12. Interaction terms
#
# These are the pairs of features whose *combined* effect is more than the sum of their separate effects (only if `interactions > 0`).

# %%
inter = term_imp[[t for t in term_imp.index if " & " in t]].sort_values(ascending=False)
if inter.empty:
    print("The tuned model uses no interaction terms (purely additive).")
else:
    display(inter.round(4).to_frame("importance"))

# %% [markdown]
# ## 13. Local explanation: why did the model predict this for *one* patient?
#
# For a single patient the prediction is exactly `intercept + sum of contributions`. Below are one correctly classified gallstone patient and one misclassified patient from the test set.

# %%
proba_ebm = ebm.predict_proba(X_test)[:, 1]
pred_ebm = (proba_ebm >= 0.5).astype(int)
y_arr = y_test.to_numpy()

tp = np.where((pred_ebm == 1) & (y_arr == 1))[0]
wrong = np.where(pred_ebm != y_arr)[0]
examples = {"Correct gallstone prediction": tp[np.argmax(proba_ebm[tp])] if len(tp) else None,
            "Misclassified patient": wrong[0] if len(wrong) else None}

local = ebm.explain_local(X_test, y_test)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (title, i) in zip(axes, examples.items()):
    if i is None:
        ax.axis("off"); continue
    d = local.data(int(i))
    contrib = pd.Series(d["scores"], index=d["names"])
    contrib = contrib.reindex(contrib.abs().sort_values().index).tail(10)
    ax.barh(contrib.index, contrib.values,
            color=["tab:red" if v > 0 else "tab:green" for v in contrib.values])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"{title}\n(test row {X_test.index[i]}, true={y_arr[i]}, "
                 f"P(gallstone)={proba_ebm[i]:.2f})", fontsize=10)
    ax.set_xlabel("contribution to log-odds (red = towards gallstones)")
plt.tight_layout(); plt.show()

# %%
# Interactive version: browse the explanations of the first 10 test patients
show(ebm.explain_local(X_test.iloc[:10], y_test.iloc[:10], name="EBM local"))

# %% [markdown]
# ## 14. Permutation importance across CV folds (same method as `script.ipynb`)
#
# This is a model-agnostic check. Shuffle one feature at a time and measure how much ROC-AUC drops. `top5_rate` is the fraction of the 25 folds in which a feature ranked in the top 5. This makes the result directly comparable with the logistic regression and random forest tables in the report.

# %%
fold_imp = []
for tr, va in rcv.split(X_train, y_train):
    m = clone(models["EBM (tuned)"]).fit(X_train.iloc[tr], y_train.iloc[tr])
    r = permutation_importance(m, X_train.iloc[va], y_train.iloc[va],
                               scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE)
    fold_imp.append(r.importances_mean)

imp = pd.DataFrame(fold_imp, columns=X_train.columns)
perm_summary = pd.DataFrame({
    "mean": imp.mean(),
    "std": imp.std(),
    "top5_rate": (imp.rank(axis=1, ascending=False) <= 5).mean(),
}).sort_values("mean", ascending=False)
perm_summary.head(10).round(3)

# %%
# Do EBM's own term importances agree with permutation importance?
compare = pd.DataFrame({
    "EBM term importance rank": term_imp.drop([t for t in term_imp.index if " & " in t])
                                        .rank(ascending=False).astype(int),
    "Permutation rank": perm_summary["mean"].rank(ascending=False).astype(int),
}).sort_values("EBM term importance rank")
compare.head(10)

# %% [markdown]
# ## 15. Summary tables (LaTeX for the report)

# %%
summary = pd.DataFrame({
    "CV ROC AUC": [f"${cv_table.loc[n, 'CV roc_auc']:.3f} \\pm {cv_table.loc[n, 'CV roc_auc std']:.3f}$"
                   for n in models],
    "Test ROC AUC": [f"{test_table.loc[n, 'ROC-AUC']:.3f}" for n in models],
    "Test accuracy": [f"{test_table.loc[n, 'Accuracy']:.3f}" for n in models],
}, index=list(models)).rename_axis("Model").reset_index()

print(summary.to_string(index=False))
print()
print(summary.to_latex(index=False, column_format="lccc"))

# %%
top_feats = perm_summary.head(10)[["mean", "std", "top5_rate"]].round(3)
print(top_feats.to_latex(float_format="%.3f"))

# %% [markdown]
# ## 16. Notes
#
# - **Fair comparison:** The data cleaning, test split and CV folds are the same as in `script.ipynb`, so the numbers can sit in the same table in the report.
# - **Small data:** With 317 patients the test set has only 64. One patient more or less classified correctly changes accuracy by about 1.6 percentage points. Use the **repeated-CV mean ± std** as the main result, and the test set as confirmation.
# - **Hyperparameters were tuned on the training set only.** The test set is used once, in section 9.
# - **What to look for in the report:** whether the EBM's top features (section 10) and shape functions (section 11) agree with the logistic regression findings (Vitamin D, CRP, TBFR, bone mass, ECF/TBW, gender), and whether the shapes are non-linear. A non-linear shape is something a plain logistic regression cannot show.
