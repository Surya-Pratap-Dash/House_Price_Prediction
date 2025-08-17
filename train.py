import argparse
import pandas as pd
import json
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

# ----------------- Argument Parser -----------------
parser = argparse.ArgumentParser(description="Train House Price Prediction Model")
parser.add_argument("--data", type=str, required=True, help="Path to CSV file")
parser.add_argument("--target", type=str, required=True, help="Target column name")
args = parser.parse_args()

# ----------------- Load Data -----------------
df = pd.read_csv(args.data)

target = args.target
if target not in df.columns:
    raise ValueError(f"❌ Target column '{target}' not found in dataset! Available: {df.columns.tolist()}")

X = df.drop(columns=[target])
y = df[target]

categorical = X.select_dtypes(include=["object"]).columns.tolist()
numeric = X.select_dtypes(exclude=["object"]).columns.tolist()

# ----------------- Preprocessing -----------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric),
        ("cat", categorical_transformer, categorical)
    ]
)

# ----------------- Train Models -----------------
models = {
    "Ridge": Ridge(),
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor()
}

best_model = None
best_score = -999

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor),
                           ("model", model)])
    try:
        score = cross_val_score(pipe, X, y, cv=5, scoring="r2").mean()
        print(f"{name} CV R2: {score:.3f}")
        if score > best_score:
            best_score = score
            best_model = pipe
    except Exception as e:
        print(f"⚠️ Skipping {name} due to error: {e}")

# ----------------- Safe Fallback -----------------
if best_model is None:
    print("⚠️ No model succeeded with cross-validation. Using Ridge as fallback.")
    best_model = Pipeline(steps=[("preprocessor", preprocessor),
                                 ("model", Ridge())])

# ----------------- Save Best Model -----------------
best_model.fit(X, y)
joblib.dump(best_model, "model.joblib")

metadata = {
    "target": target,
    "numeric_features": numeric,
    "categorical_features": categorical,
    "categorical_options": {col: df[col].dropna().unique().tolist() for col in categorical}
}
with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

template = X.head(1)
template.to_csv("template.csv", index=False)

print("✅ Training complete! Model, metadata, and template saved.")
