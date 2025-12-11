# LightGBM_Train.py
# Contains four different LightGBM models trained on different feature sets:
# LGB_Boost -> Original dataset
# LGB_Boost_NoLeak -> Original minus cdmMissDistance and cdmPc (leakage removed)
# LGB_Boost_Featured -> Original + engineered features
# LGB_Boost_NoLeak_Featured -> Featured minus all Pc/missDistance dependencies

import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import lightgbm as lgb

# 1) Feature lists (mirroring your XGBoost lists)

LGB_Boost = [
    'cdmMissDistance', 'cdmPc',
    'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
    'rso1_objectType', 'rso2_objectType',
    'org1_displayName', 'org2_displayName',
    'condition_24H_tca_72H',
    'condition_Radial_100m',
    'condition_InTrack_500m', 'condition_CrossTrack_500m',
    'condition_sat2posUnc_1km', 'condition_sat2Obs_25',
    'hours_to_tca'      # derived but independent of target columns
]

LGB_Boost_NoLeak = [
    'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
    'rso1_objectType', 'rso2_objectType',
    'org1_displayName', 'org2_displayName',
    'condition_24H_tca_72H',
    'condition_Radial_100m',
    'condition_InTrack_500m', 'condition_CrossTrack_500m',
    'condition_sat2posUnc_1km', 'condition_sat2Obs_25',
    'hours_to_tca'
]

LGB_Boost_NoLeak_Featured = [
    # Original features
    'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
    'rso1_objectType', 'rso2_objectType',
    'org1_displayName', 'org2_displayName',
    'condition_24H_tca_72H',
    'condition_Radial_100m',
    'condition_InTrack_500m', 'condition_CrossTrack_500m',
    'condition_sat2posUnc_1km', 'condition_sat2Obs_25',
    'hours_to_tca',

    # Engineered features (no direct Pc/missDistance based ones)
    'tca_bin',
    'same_sat_type',
    'is_debris_pair',
    'close_all_axes',
    'risky_uncertainty',
    'distance_ratio',
    'object_type_match'
]

LGB_Boost_Featured = [
    # Original features
    'cdmMissDistance', 'cdmPc',
    'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
    'rso1_objectType', 'rso2_objectType',
    'org1_displayName', 'org2_displayName',
    'condition_24H_tca_72H',
    'condition_Radial_100m',
    'condition_InTrack_500m', 'condition_CrossTrack_500m',
    'condition_sat2posUnc_1km', 'condition_sat2Obs_25',
    'hours_to_tca',

    # Engineered features
    'log_cdmPc',
    'inv_miss_distance',
    'tca_bin',
    'same_sat_type',
    'is_debris_pair',
    'close_all_axes',
    'risky_uncertainty',
    'distance_ratio',
    'object_type_match'
]

CATEGORICAL_COLS = [
    'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
    'rso1_objectType', 'rso2_objectType',
    'org1_displayName', 'org2_displayName'
]

# ---------- CONFIG ----------
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from: {path}")
    df = pd.read_excel(path)
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype("category")
    return df


def compute_scale_pos_weight(y: pd.Series) -> float:
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0:
        return 1.0
    return neg / pos


def save_results(model_name: str, results_dict: dict) -> None:
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"{model_name}_LGB.json")
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"\nSaved LightGBM evaluation results to {out_path}")


def train_and_evaluate(df: pd.DataFrame, features: list, model_name: str) -> None:
    X = df[features].copy()
    y = df['HighRisk'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    spw = compute_scale_pos_weight(y_train)
    print(f"\n[{model_name}] scale_pos_weight:", spw)

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=[c for c in CATEGORICAL_COLS if c in features]
    )
    dtest = lgb.Dataset(
        X_test,
        label=y_test,
        reference=dtrain,
        categorical_feature=[c for c in CATEGORICAL_COLS if c in features]
    )
    # LightGBM parameters (fixed, simple, reasonable defaults)
    params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "scale_pos_weight": spw,
    "seed": RANDOM_STATE,
    "verbosity": -1,
    "early_stopping_round": 100   
}

    model = lgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    valid_sets=[dtest],
    valid_names=["valid"]
    )

    
    # Probabilities
    y_prob = model.predict(X_test, num_iteration=model.best_iteration)

    # 1) Evaluate at default threshold 0.5
    thr_default = 0.5
    y_pred_default = (y_prob >= thr_default).astype(int)

    print(f"\n================= {model_name} @ Threshold = 0.5 =================")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_default))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_default, digits=4))

    rec_default = recall_score(y_test, y_pred_default, zero_division=0)
    prec_default = precision_score(y_test, y_pred_default, zero_division=0)
    f1_default = f1_score(y_test, y_pred_default, zero_division=0)
    acc_default = (y_pred_default == y_test).mean()
    auc_pr = roc_auc_score(y_test, y_prob)
    auc_roc = roc_auc_score(y_test, y_prob)

    print(f"Recall: {rec_default:.4f}")
    print(f"Precision: {prec_default:.4f}")
    print(f"F1-score: {f1_default:.4f}")
    print(f"Accuracy: {acc_default:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    # 2) Scan thresholds for best recall/precision trade-off
    best_thr = 0.5
    best_score = 0.0

    for thr in np.arange(0.0, 1.01, 0.01):
        y_pred_thr = (y_prob >= thr).astype(int)
        rec = recall_score(y_test, y_pred_thr, zero_division=0)
        prec = precision_score(y_test, y_pred_thr, zero_division=0)
        score = rec * 0.7 + prec * 0.3
        if score > best_score:
            best_score = score
            best_thr = thr

    print(f"\nBest threshold for {model_name} = {best_thr:.2f}")

    # 3) Evaluate using best threshold
    y_pred_best = (y_prob >= best_thr).astype(int)

    print(f"\n================= {model_name} @ BEST Threshold =================")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_best, digits=4))

    rec_best = recall_score(y_test, y_pred_best, zero_division=0)
    prec_best = precision_score(y_test, y_pred_best, zero_division=0)
    f1_best = f1_score(y_test, y_pred_best, zero_division=0)
    acc_best = (y_pred_best == y_test).mean()

    print(f"Recall: {rec_best:.4f}")
    print(f"Precision: {prec_best:.4f}")
    print(f"F1-score: {f1_best:.4f}")
    print(f"Accuracy: {acc_best:.4f}")

    # Feature importance
    print("\nFeature importances:")
    importance = model.feature_importance(importance_type="gain")
    for feat, imp in sorted(zip(features, importance), key=lambda x: -x[1]):
        print(f"{feat}: {imp:.4f}")

    results = {
        "model_name": model_name,
        "default_threshold": float(thr_default),
        "best_threshold": float(best_thr),
        "confusion_matrix_default": confusion_matrix(y_test, y_pred_default).tolist(),
        "confusion_matrix_best": confusion_matrix(y_test, y_pred_best).tolist(),
        "metrics_default": {
            "recall": float(rec_default),
            "precision": float(prec_default),
            "f1": float(f1_default),
            "accuracy": float(acc_default),
            "auc_pr": float(auc_pr),
            "auc_roc": float(auc_roc)
        },
        "metrics_best_threshold": {
            "recall": float(rec_best),
            "precision": float(prec_best),
            "f1": float(f1_best),
            "accuracy": float(acc_best)
        }
    }

    save_results(model_name, results)

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", model_name + "_LGB.txt")
    model.save_model(model_path)
    print(f"\nLightGBM model saved to {model_path}")


DATA_PATH = os.path.join("data", "Merged_Featured_DATA.xlsx")


def main():
    df = load_data(DATA_PATH)
    print(df.info())

    train_and_evaluate(df, LGB_Boost, "LGB_Boost")
    train_and_evaluate(df, LGB_Boost_NoLeak, "LGB_Boost_NoLeak")
    train_and_evaluate(df, LGB_Boost_NoLeak_Featured, "LGB_Boost_NoLeak_Featured")
    train_and_evaluate(df, LGB_Boost_Featured, "LGB_Boost_Featured")


if __name__ == "__main__":
    main()