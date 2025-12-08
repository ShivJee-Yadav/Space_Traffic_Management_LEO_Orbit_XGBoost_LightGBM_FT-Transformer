# Contains Four Different XGBoost Models trained no Different Dataset .
# XG_Boost -> Trained on Original Dataset 
# XG_Boost_NoLeak -> Trained on Data After Removing CdmDistance and Pc  to avoid Data Leakage 
# XG_Boost_Featured -> Trained on Featured Data (10 new Columns added after Feature Engineering) 
# XG_Boost_NoLeak_Featured -> Trained on Data After Removing all dependencies on CdmDistance and Pc  to avoid Data Leakage completely


import os
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
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# 1) Feature Lists
XG_Boost = [
        'cdmMissDistance', 'cdmPc',
        'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
        'rso1_objectType', 'rso2_objectType',
        'org1_displayName', 'org2_displayName',
        'condition_24H_tca_72H',
        'condition_Radial_100m',
        'condition_InTrack_500m', 'condition_CrossTrack_500m',
        'condition_sat2posUnc_1km', 'condition_sat2Obs_25',
        'hours_to_tca'      # Derived But independent of Target Columns
    ]

XG_Boost_NoLeak = [
        'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
        'rso1_objectType', 'rso2_objectType',
        'org1_displayName', 'org2_displayName',
        'condition_24H_tca_72H', 
        'condition_Radial_100m',
        'condition_InTrack_500m', 'condition_CrossTrack_500m',
        'condition_sat2posUnc_1km', 'condition_sat2Obs_25',
        'hours_to_tca'
    ]

XG_Boost_NoLeak_Featured = [
    # Original features
    'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
    'rso1_objectType', 'rso2_objectType',
    'org1_displayName', 'org2_displayName',
    'condition_24H_tca_72H',
    'condition_Radial_100m',
    'condition_InTrack_500m', 'condition_CrossTrack_500m',
    'condition_sat2posUnc_1km', 'condition_sat2Obs_25',
    'hours_to_tca',

    # Engineered features
    'tca_bin',
    'same_sat_type',
    'is_debris_pair',
    'close_all_axes',
    'risky_uncertainty',
    'distance_ratio',
    'object_type_match'
    ]

XG_Boost_Featured = [
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


# ---------- CONFIG ----------

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from: {path}")
    df = pd.read_excel(path)
    return df


def preprocess(df: pd.DataFrame, features: list) -> tuple[pd.DataFrame, pd.Series]:
    
    categorical_cols = [
        'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
        'rso1_objectType', 'rso2_objectType',
        'org1_displayName', 'org2_displayName'
    ]
    df[categorical_cols] = df[categorical_cols].astype("category")

    X = df[features].copy()
    y = df['HighRisk'].copy()
    return X, y


def compute_scale_pos_weight(y: pd.Series) -> float:
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0:
        return 1.0
    return neg / pos

def load_params_from_txt(path):
    params = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=")
                key = key.strip()
                value = value.strip()

                # Convert numeric values automatically
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except:
                    pass  # keep as string if not numeric

                params[key] = value
    print(params)
    return params

def train_and_evaluate(X: pd.DataFrame, y: pd.Series, features: list , ModelName : str) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # preserves class ratio
    )

    spw = compute_scale_pos_weight(y_train)
    print("\nscale_pos_weight:", spw)

    # Load Tune HyperParameter from Text file
    HyperParameter_DIR = os.path.join("outputs", "Best_HyperParameter",ModelName) + ".txt" 
    best_params = load_params_from_txt(HyperParameter_DIR)

    model = XGBClassifier(
        **best_params,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        tree_method="hist",
        eval_metric='aucpr',    # BEST for rare events
        enable_categorical=True
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Probabilities and thresholding
    y_prob = model.predict_proba(X_test)[:, 1]

    # Try default threshold 0.5 first
    thr = 0.5
    y_pred = (y_prob >= thr).astype(int)

    
    best_thr = 0.5
    best_f1 = 0.0

    print("\nScanning thresholds from 0.00 to 1.00...")

    for thr in np.arange(0.0, 1.01, 0.01):
            y_pred_thr = (y_prob >= thr).astype(int)
            f1 = f1_score(y_test, y_pred_thr, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

    print(f"\nBest threshold based on F1 = {best_thr:.2f} (F1 = {best_f1:.4f})")

    # Feature importance
    print("\nFeature importances:")
    importance = model.feature_importances_
    for feat, imp in sorted(zip(features, importance), key=lambda x: -x[1]):
        print(f"{feat}: {imp:.4f}")
    
    ModelDIR = os.path.join("models",ModelName) + ".json"
    model.save_model(ModelDIR)
    print(f"\nXGBoost model saved to {ModelDIR}")

DATA_PATH = os.path.join("data", "Merged_Featured_DATA.xlsx")
def main():

    df = load_data(DATA_PATH)
    print(df.info())
    X, y= preprocess(df , XG_Boost)
    train_and_evaluate(X, y, XG_Boost , "XG_Boost")

    # X, y= preprocess(df , XG_Boost_NoLeak , )
    # train_and_evaluate(X, y, XG_Boost_NoLeak , "XG_Boost_NoLeak")
    
    # X, y= preprocess(df , XG_Boost_NoLeak_Featured)
    # train_and_evaluate(X, y, XG_Boost_NoLeak_Featured , "XG_Boost_NoLeak_Featured")
    
    # X, y= preprocess(df , XG_Boost_Featured)
    # train_and_evaluate(X, y, XG_Boost_Featured , "XG_Boost_Featured")


if __name__ == "__main__":
    main()