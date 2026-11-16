# %% [markdown]
# # Import dependencies

# %%
import numpy as np
import pandas as pd

print(f"NumPy: {np.__version__}")
print(f"pandas: {pd.__version__}")

# %% [markdown]
# # Load dataset

# %%
# Since the dataset file is saved in the Strict Open XML Spreadsheet format, we
# have to use `calamine` to read it
df = pd.read_excel("dataset/dataset-uci.xlsx", engine="calamine")

# Shape of dataset; data types; feature overview; missing values; duplicates; etc.
rows = len(df)
print(f"Number of rows: {rows}")
#how many columns we have of data
columns = len(df.columns)
print(f"Number of columns: {columns}")
##Now the actual shape of the dataset + types
print(f"Dataset shape: {df.shape}")
print(f"Dataset datatypes:\n{df.dtypes}")

print(f"# null values for each feature:\n{df.isnull().sum()}")
print(f"# null values: {df.isnull().sum().sum()}")

print(f"# duplicated values: {df.duplicated().sum()}")

y = df["Gallstone Status"]
X = df.drop(columns="Gallstone Status")

# %% [markdown]
# We have data of around 319 ppl.

# %%
#How many ppl have gallstones and how many do not
gallstone_counts = y.value_counts()[1]
print(f"Number of people with gallstones: {gallstone_counts}")
print(f"Number of people without gallstones: {len(X) - gallstone_counts}")


# %% [markdown]
# So we have half half ppl with and without gallstones

# %%


# %%
# Outlier
v_max =  X.max()
v_mean = X.mean()
v_min =  X.min()
v_std =  X.std()

z = (X - v_mean) / v_std
outliers = z.abs() > 10

print(outliers.sum())               # number of outliers per column
result = pd.DataFrame({
    "value": X[outliers].stack().dropna(),
    "z": z[outliers].stack().dropna(),
})

feature = result.index.get_level_values(1)
result["mean"] = v_mean[feature].values
result["std"] = v_std[feature].values

result = result[["value", "mean", "std", "z"]].rename_axis(["Record", "Feature"])
print(result.round(2))

# %%
# This is very high but clinically possible in poorly controlled diabetes
X.loc[40, "Diabetes Mellitus (DM)"]
X = X.drop(40)
y = y.drop(40)

# %%
# This isn’t physiologically possible. It may be a misplaced decimal, such as 19.54 or 195.4. Obesity % is usually derived from BMI
X.loc[239, ["Height", "Weight", "Body Mass Index (BMI)", "Obesity (%)"]]
X.loc[239, "Obesity (%)"] = 19.54

# %%
X.loc[318, ["Total Cholesterol (TC)", "Low Density Lipoprotein (LDL)", "High Density Lipoprotein (HDL)"]]
X = X.drop(318)
y = y.drop(318)

# %%
# Highly correlated features
corr = X.corr().abs()
upper = corr.where(np.triu(
    np.ones(corr.shape, dtype=bool),
    k=1
))  # upper triangle only

pairs = upper.stack()
high = pairs[pairs > 0.9]
print(high.sort_values(key=np.abs, ascending=False))


# %%
keep = []
for c in X.columns:
    if not (corr.loc[c, keep] > 0.9).any():
        keep.append(c)

dropped = [c for c in X.columns if c not in keep]
print(f"Dropped {len(dropped)}:", dropped)

X = X[keep]

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# %%
# Continuous features only (binary and ordinal ones have ≤ 5 distinct values)
cont = [c for c in X_train.columns if X_train[c].nunique() > 5]

skew = X_train[cont].skew()
skewed = skew[skew > 1].index.tolist()                   # strongly right-skewed
skewed = [c for c in skewed if (X_train[c] >= 0).all()]  # log1p needs x > -1

skew_table = pd.DataFrame({
    "Skewness": skew[skewed],
    "After log": np.log1p(X_train[skewed]).skew(),
}).sort_values("Skewness", ascending=False).round(2)

print(skew_table)
print(skew_table.to_latex())

# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def make_model(C):
    #return make_pipeline(
    #    StandardScaler(),
    #    LogisticRegression(C=C, max_iter=5000),
    #)
    return make_pipeline(
        ColumnTransformer([("log", FunctionTransformer(np.log1p), skewed)],
                          remainder="passthrough"),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=5000),
    )


# 2. Choose C with cross-validation on the training set only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    scores = cross_validate(make_model(C), X_train, y_train, cv=cv,
                            scoring=["accuracy", "roc_auc", "f1"])
    results[C] = scores["test_roc_auc"].mean()

    print(f"C={C}:")
    for metric in ["accuracy", "roc_auc", "f1"]:
        s = scores[f"test_{metric}"]
        print(f"{metric:>8}: {s.mean():.3f} ± {s.std():.3f}")

best_C = max(results, key=results.get)
print(f"\nBest C (by CV ROC AUC): {best_C}")

# 3. Refit on the full training set and evaluate once on the test set
final_model = make_model(best_C)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1]

print("\nTest set performance:")
print(f"accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f" roc_auc: {roc_auc_score(y_test, y_prob):.3f}")
print(f"      f1: {f1_score(y_test, y_pred):.3f}")

# %%
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.preprocessing import StandardScaler

models = {
    "logreg": make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10, max_iter=5000),
    ),
    "logreg + log": make_pipeline(
        ColumnTransformer([("log", FunctionTransformer(np.log1p), skewed)],
                          remainder="passthrough"),
        StandardScaler(),
        LogisticRegression(C=1, max_iter=5000),
    ),
    "random forest": make_pipeline(
        RandomForestClassifier(n_estimators=500, random_state=42),
    ),
    # "gradient boosting": HistGradientBoostingClassifier(   # handles NaN natively
    #     max_depth=3, learning_rate=0.05, max_iter=200, random_state=42),
}

# %%
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

rows = []
for name, model in models.items():
    # Validation (training set only)
    s = cross_val_score(model, X_train, y_train, cv=rcv, scoring="roc_auc")

    # Test (fit once on the full training set, evaluate once)
    model.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    rows.append({
        "Model": name,
        "CV ROC AUC": f"${s.mean():.3f} \\pm {s.std():.3f}$",
        "Test ROC AUC": f"{test_auc:.3f}",
    })

results = pd.DataFrame(rows)
print(results)
print(results.to_latex(index=False, column_format="lcc"))

# %%
from sklearn.inspection import permutation_importance
from sklearn.base import clone

summaries = {}

for name, model in models.items():
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    fold_imp = []
    for tr, va in cv.split(X_train, y_train):
        m = clone(model).fit(X_train.iloc[tr], y_train.iloc[tr])
        r = permutation_importance(m, X_train.iloc[va], y_train.iloc[va],
                                   scoring="roc_auc", n_repeats=10, random_state=42)
        fold_imp.append(r.importances_mean)

    imp = pd.DataFrame(fold_imp, columns=X_train.columns)
    summaries[name] = pd.DataFrame({
        "mean": imp.mean(),
        "std": imp.std(),
        "top5_rate": (imp.rank(axis=1, ascending=False) <= 5).mean(),
    }).sort_values("mean", ascending=False)
    print(f"\n{name}")
    print(summaries[name].head(10).round(3))

features = ["C-Reactive Protein (CRP)", "Vitamin D", "Bone Mass (BM)",
            "Extracellular Fluid/Total Body Water (ECF/TBW)", "Gender",
            "Total Body Fat Ratio (TBFR) (%)"]

table = pd.DataFrame({name: s["top5_rate"] for name, s in summaries.items()})
table["Sum"] = table.sum(axis=1)
table = table.sort_values("Sum", ascending=False)

print(table.head(8).round(2))
print(table.head(8).round(2).to_latex())

# %%
from sklearn.feature_selection import f_classif
from scipy.stats import mannwhitneyu

F, p = f_classif(X, y)

mw_p = [mannwhitneyu(X.loc[y == 1, c],
                     X.loc[y == 0, c]).pvalue
        for c in X.columns]

uni = pd.DataFrame({"F": F, "p_anova": p, "p_mannwhitney": mw_p},
                   index=X.columns).sort_values("F", ascending=False)
print(uni.head(10).round(4).to_latex())


