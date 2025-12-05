# src/train_xgb_small.py

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# ---------- CONFIG ----------

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from: {path}")
    df = pd.read_excel(path)
    print("Shape:", df.shape)
    print(df.info())
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    
    categorical_cols = [
        'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
        'rso1_objectType', 'rso2_objectType',
        'org1_displayName', 'org2_displayName'
    ]
    df[categorical_cols] = df[categorical_cols].astype("category")

    # 2) Feature list
    features = [
        'cdmMissDistance', 'cdmPc',
        'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
        'rso1_objectType', 'rso2_objectType',
        'org1_displayName', 'org2_displayName',
        'condition_24H_tca_72H',
        'condition_Radial_100m',
        'condition_InTrack_500m', 'condition_CrossTrack_500m',
        'condition_sat2posUnc_1km', 'condition_sat2Obs_25',
        'hours_to_tca'
    ]

    X = df[features].copy()
    y = df['HighRisk'].copy()

    print("\nHighRisk distribution:")
    print(y.value_counts())
    print("\nHighRisk distribution in Percentage %:")
    print((y.value_counts(normalize=True) * 100).round(3))

    return X, y, features


def compute_scale_pos_weight(y: pd.Series) -> float:
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0:
        return 1.0
    return neg / pos


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, features: list) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # preserves class ratio
    )

    spw = compute_scale_pos_weight(y_train)
    print("\nscale_pos_weight:", spw)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        eval_metric='aucpr',    # BEST for rare events
        scale_pos_weight=spw,
        tree_method="hist",
        enable_categorical=True
    )
    print(X_train , y_train)
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

    print("\n" + "#" * 50)
    print(f"Evaluation at threshold = {thr}")
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Optional: scan a few thresholds
    print("\nThreshold scan:")
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5]:
        y_pred_thr = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_test, y_pred_thr)
        print(f"\nthr={thr}")
        print("Confusion matrix:\n", cm)

    # Feature importance
    print("\nFeature importances:")
    importance = model.feature_importances_
    for feat, imp in sorted(zip(features, importance), key=lambda x: -x[1]):
        print(f"{feat}: {imp:.4f}")

DATA_PATH = os.path.join("data", "Merged_DATA.xlsx")
def main():

    df = load_data(DATA_PATH)
    X, y, features = preprocess(df)
    train_and_evaluate(X, y, features)


if __name__ == "__main__":
    main()