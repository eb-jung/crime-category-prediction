"""
train.py
--------
End-to-end training pipeline for SF crime category prediction.

Loads data, engineers features, trains five classifiers, evaluates on a
temporal hold-out split, saves figures, and prints a summary table.

Usage
-----
    python scripts/train.py                    # full run
    python scripts/train.py --sample 50000     # subsample training set
    python scripts/train.py --model xgboost    # single model
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import features as F
from src import evaluate as E
from src import visualize as V

DATA_PATH = os.path.join("data", "train.csv")
FIGURES_DIR = os.path.join("outputs", "figures")


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = F.clean_coordinates(df)
    df = F.extract_temporal_features(df)
    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Odd week numbers → train, even → validation (mirrors original notebook split)."""
    start = df["Dates"].min()
    df = df.copy()
    df["_week"] = ((df["Dates"] - start).dt.days // 7) + 1
    train = df[df["_week"] % 2 != 0].drop(columns="_week")
    val = df[df["_week"] % 2 == 0].drop(columns="_week")
    return train, val


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", "passthrough", F.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), F.CATEGORICAL_FEATURES),
    ])


def build_pipelines(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=300, solver="lbfgs", n_jobs=-1)),
        ]),
        "Random Forest": Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)),
        ]),
        "XGBoost": Pipeline([
            ("pre", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                objective="multi:softprob", eval_metric="mlogloss",
                n_jobs=-1, random_state=42,
            )),
        ]),
        "KNN": Pipeline([
            ("pre", preprocessor),
            ("clf", KNeighborsClassifier(n_neighbors=10, weights="distance", n_jobs=-1)),
        ]),
        "Naive Bayes": Pipeline([
            ("pre", preprocessor),
            ("clf", MultinomialNB(alpha=0.5)),
        ]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate SF crime category classifiers.")
    parser.add_argument("--sample", type=int, default=50_000,
                        help="Training sample size (default: 50000). Use 0 for full dataset.")
    parser.add_argument("--val-sample", type=int, default=20_000,
                        help="Validation sample size (default: 20000).")
    parser.add_argument("--model", choices=["logistic", "rf", "xgboost", "knn", "nb", "all"],
                        default="all", help="Which model(s) to train.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- Load ---
    print(f"Loading data from {DATA_PATH}...")
    df = load_and_prepare(DATA_PATH)
    print(f"  {len(df):,} records, {df['Category'].nunique()} categories")

    # --- Spatial features (fit on full data so clusters are stable) ---
    print("Engineering spatial features...")
    df, geo_kmeans = F.add_spatial_features(df)

    # --- Split ---
    train_df, val_df = temporal_split(df)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")

    # --- Label encoding ---
    le = LabelEncoder()
    le.fit(df["Category"])
    y_train_full = le.transform(train_df["Category"])
    y_val_full = le.transform(val_df["Category"])

    # --- Sampling ---
    rng = np.random.default_rng(42)
    n_train = len(train_df) if args.sample == 0 else min(args.sample, len(train_df))
    n_val = min(args.val_sample, len(val_df))

    train_idx = rng.choice(len(train_df), n_train, replace=False)
    val_idx = rng.choice(len(val_df), n_val, replace=False)

    X_train = F.build_feature_matrix(train_df).iloc[train_idx]
    y_train = y_train_full[train_idx]
    X_val = F.build_feature_matrix(val_df).iloc[val_idx]
    y_val = y_val_full[val_idx]

    # Ensure all 39 classes appear in training sample
    missing_classes = np.setdiff1d(np.arange(len(le.classes_)), np.unique(y_train))
    if len(missing_classes):
        extra_idx = [np.where(y_train_full == c)[0][0] for c in missing_classes]
        X_train = pd.concat([X_train, F.build_feature_matrix(train_df).iloc[extra_idx]])
        y_train = np.concatenate([y_train, missing_classes])
        print(f"  Added {len(missing_classes)} rare-class samples → training: {len(X_train):,}")

    print(f"  Training on {len(X_train):,} samples, validating on {len(X_val):,}")

    # --- EDA figures ---
    print("Saving EDA figures...")
    V.plot_crime_distribution(df, save_path=os.path.join(FIGURES_DIR, "crime_distribution.png"))
    V.plot_temporal_patterns(df, save_path=os.path.join(FIGURES_DIR, "temporal_patterns.png"))
    V.plot_district_heatmap(df, save_path=os.path.join(FIGURES_DIR, "district_heatmap.png"))

    # --- Train & evaluate ---
    model_map = {
        "logistic": "Logistic Regression",
        "rf": "Random Forest",
        "xgboost": "XGBoost",
        "knn": "KNN",
        "nb": "Naive Bayes",
    }
    preprocessor = build_preprocessor()
    all_pipelines = build_pipelines(preprocessor)
    selected = list(all_pipelines.keys()) if args.model == "all" else [model_map[args.model]]

    results = {}
    for name in selected:
        print(f"\nTraining {name}...")
        pipe = all_pipelines[name]

        if name == "XGBoost":
            # XGBoost needs labels remapped to 0..K-1 for classes present in sample
            xgb_le = LabelEncoder()
            xgb_le.fit(np.unique(y_train))
            pipe.fit(X_train, xgb_le.transform(y_train))
            y_pred_enc = pipe.predict(X_val)
            y_pred = xgb_le.inverse_transform(y_pred_enc)
            y_proba = pipe.predict_proba(X_val)
            present = xgb_le.classes_
            full_proba = np.full((len(y_val), len(le.classes_)), 1e-9)
            full_proba[:, present] = y_proba
            full_proba /= full_proba.sum(axis=1, keepdims=True)
            result = E.evaluate(y_val, y_pred, full_proba, len(le.classes_), list(le.classes_))
        else:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            y_proba = pipe.predict_proba(X_val)
            result = E.evaluate(y_val, y_pred, y_proba, len(le.classes_), list(le.classes_))

        results[name] = result
        print(f"  Accuracy: {result['accuracy']:.1%} | Log Loss: {result['log_loss']:.4f}")

    # --- Summary table ---
    summary = E.summarise_results(results)
    print("\n=== Model Comparison (sorted by log loss) ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join("outputs", "model_comparison.csv"), index=False)

    # --- Result figures ---
    V.plot_model_comparison(summary, save_path=os.path.join(FIGURES_DIR, "model_comparison.png"))

    # Feature importance from best tree model
    best_tree = "XGBoost" if "XGBoost" in results else ("Random Forest" if "Random Forest" in results else None)
    if best_tree:
        tree_pipe = all_pipelines[best_tree]
        ohe = tree_pipe.named_steps["pre"].named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(F.CATEGORICAL_FEATURES))
        feat_names = F.NUMERIC_FEATURES + cat_names
        clf = tree_pipe.named_steps["clf"]
        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": clf.feature_importances_})
        imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)
        V.plot_feature_importance(imp_df, save_path=os.path.join(FIGURES_DIR, "feature_importance.png"))
        imp_df.to_csv(os.path.join("outputs", "feature_importance.csv"), index=False)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
