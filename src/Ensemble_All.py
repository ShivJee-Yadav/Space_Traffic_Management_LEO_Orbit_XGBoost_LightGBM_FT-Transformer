# Ensemble_All.py
# Combines 4 XGBoost + 4 LightGBM + optional FT-Transformer into one weighted ensemble.

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, recall_score, precision_score,
    f1_score, roc_auc_score
)
import optuna
from xgboost import XGBClassifier
import lightgbm as lgb



from src.LightGBM.Light_GBM import (
    LGB_Boost, LGB_Boost_NoLeak,
    LGB_Boost_Featured, LGB_Boost_NoLeak_Featured,
    CATEGORICAL_COLS
)

from src.XGBoost.Combined_XG_Boost import (
    XG_Boost, XG_Boost_NoLeak,
    XG_Boost_Featured, XG_Boost_NoLeak_Featured
)


# -----------------------------
# Feature Lists (same as training)
# -----------------------------
MODEL_LIST = [
    ("XG_Boost", XG_Boost, "xgb"),
    ("XG_Boost_NoLeak", XG_Boost_NoLeak, "xgb"),
    ("XG_Boost_Featured", XG_Boost_Featured, "xgb"),
    ("XG_Boost_NoLeak_Featured", XG_Boost_NoLeak_Featured, "xgb"),

    ("LGB_Boost", LGB_Boost, "lgb"),
    ("LGB_Boost_NoLeak", LGB_Boost_NoLeak, "lgb"),
    ("LGB_Boost_Featured", LGB_Boost_Featured, "lgb"),
    ("LGB_Boost_NoLeak_Featured", LGB_Boost_NoLeak_Featured, "lgb"),
]


# -----------------------------
# Helper: Load XGBoost model probs
# -----------------------------

def load_xgb_probs(model_name, feature_list, data_df):
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = model_name + ".json"
    model_path = os.path.join("models", model_file)
    
    
    st.write("Loading model from:", model_path)
    current_dir = os.getcwd()
    st.write("Current working directory:", current_dir)
    model_path2 = os.path.join(".." , "models", model_file)
    st.write(model_path2)
    model = XGBClassifier()
    model.load_model(model_path)
    
    X = data_df[feature_list]
    return model.predict_proba(X)[:, 1]
# -----------------------------
# Helper: Load LightGBM model probs
# -----------------------------
def load_lgb_probs(model_name, feature_list, data_df):
    model_file = model_name + "_LGB.txt"
    model_path = os.path.join("models", model_file )
    model = lgb.Booster(model_file=model_path)
    X = data_df[feature_list]
    return model.predict(X)




if __name__ == "__main__":

    # -----------------------------
    # Config
    # -----------------------------
    DATA_PATH = "data/Merged_Featured_DATA.xlsx"
    OUT_DIR = "outputs/Ensemble"
    os.makedirs(OUT_DIR, exist_ok=True)


    # -----------------------------
    # Load Data
    # -----------------------------
    df = pd.read_excel(DATA_PATH)
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype("category")

    # Train/Val/Test split (80/10/10)
    train_val_df, test_df = train_test_split(
        df, test_size=0.10, random_state=42, stratify=df["HighRisk"]
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.1111111, random_state=42, stratify=train_val_df["HighRisk"]
    )

    y_val = val_df["HighRisk"].values
    y_test = test_df["HighRisk"].values


    # -----------------------------
    # Collect probabilities for all models
    # -----------------------------

    val_probs = {}
    test_probs = {}

    for name, feats, mtype in MODEL_LIST:
        print(f"Loading predictions for {name} ...")
        if mtype == "xgb":
            val_probs[name] = load_xgb_probs(name, feats, val_df)
            test_probs[name] = load_xgb_probs(name, feats, test_df)
        else:
            val_probs[name] = load_lgb_probs(name, feats, val_df)
            test_probs[name] = load_lgb_probs(name, feats, test_df)

    # Convert to DataFrame
    val_probs_df = pd.DataFrame(val_probs)
    test_probs_df = pd.DataFrame(test_probs)

    # -----------------------------
    # Optuna: Tune ensemble weights
    # -----------------------------
    def objective(trial):
        w = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(len(val_probs_df.columns))])
        if w.sum() == 0:
            w = np.ones_like(w)
        w = w / w.sum()
        avg = val_probs_df.values.dot(w)
        return average_precision_score(y_val, avg)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=100)

    best_weights_raw = np.array([study.best_params[f"w{i}"] for i in range(len(val_probs_df.columns))])
    best_weights = best_weights_raw / best_weights_raw.sum()

    print("\nBest Weights:")
    for name, w in zip(val_probs_df.columns, best_weights):
        print(f"{name}: {w:.4f}")

    # -----------------------------
    # Threshold tuning (updated: precision >= 0.50)
    # -----------------------------
    best_thr = 0.5
    best_recall = 0.0
    min_precision = 0.50   # enforce meaningful precision

    val_avg = val_probs_df.values.dot(best_weights)

    for thr in np.linspace(0, 1, 1001):   # finer search
        preds = (val_avg >= thr).astype(int)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds)

        # skip thresholds that destroy precision
        if prec < min_precision:
            continue

        #  maximize recall under precision constraint
        if rec > best_recall:
            best_recall = rec
            best_thr = thr

    print(f"\nChosen threshold (precision >= {min_precision}): {best_thr:.4f}")

    # -----------------------------
    # Final Test Evaluation
    # -----------------------------
    test_avg = test_probs_df.values.dot(best_weights)
    test_pred = (test_avg >= best_thr).astype(int)

    metrics = {
        "auc_pr": float(average_precision_score(y_test, test_avg)),
        "auc_roc": float(roc_auc_score(y_test, test_avg)),
        "recall": float(recall_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0))
    }

    print("\nFinal Test Metrics:")
    print(json.dumps(metrics, indent=4))

    # -----------------------------
    # Save Outputs
    # -----------------------------
    out_df = test_df.copy().reset_index(drop=True)
    for name in test_probs_df.columns:
        out_df[f"{name}_prob"] = test_probs_df[name]

    out_df["ensemble_prob"] = test_avg
    out_df["ensemble_pred"] = test_pred

    excel_path = os.path.join(OUT_DIR, "Ensemble_All_Predictions.xlsx")
    json_path = os.path.join(OUT_DIR, "Ensemble_All_Summary.json")

    out_df.to_excel(excel_path, index=False)

    summary = {
        "model_names": list(test_probs_df.columns),
        "best_weights": best_weights.tolist(),
        "best_threshold": float(best_thr),
        "test_metrics": metrics,
        "n_test": len(test_df)
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nSaved ensemble predictions to {excel_path}")
    print(f"Saved summary to {json_path}")