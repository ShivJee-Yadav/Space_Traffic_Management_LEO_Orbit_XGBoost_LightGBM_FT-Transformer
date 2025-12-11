# LightGBM_Predict.py
# Predicts using four LightGBM models and saves Excel + JSON outputs.

import os
import pandas as pd
import numpy as np
import lightgbm as lgb

# Feature lists (same as training)
LGB_Boost = [
    'cdmMissDistance', 'cdmPc',
    'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
    'rso1_objectType', 'rso2_objectType',
    'org1_displayName', 'org2_displayName',
    'condition_24H_tca_72H',
    'condition_Radial_100m',
    'condition_InTrack_500m', 'condition_CrossTrack_500m',
    'condition_sat2posUnc_1km', 'conditi on_sat2Obs_25',
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


def PredictModel(df, feature_list, model_name, best_thr=0.30):

    model_path = os.path.join("models", model_name + ".txt")
    model = lgb.Booster(model_file=model_path)

    X = df[feature_list]
    y_prob = model.predict(X)

    y_pred = (y_prob >= best_thr).astype(int)

    df[f"{model_name}_prob"] = y_prob
    df[f"{model_name}_pred"] = y_pred

    out_path = os.path.join("outputs", "Prediction", model_name + ".xlsx")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_excel(out_path, index=False)

    print(f"Saved predictions to {out_path}")

def main():
    DATA_DIR = "data/Merged_Featured_DATA.xlsx"
    df = pd.read_excel(DATA_DIR)
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype("category")

    PredictModel(df.copy(), LGB_Boost, "LGB_Boost")
    PredictModel(df.copy(), LGB_Boost_NoLeak, "LGB_Boost_NoLeak")
    PredictModel(df.copy(), LGB_Boost_Featured, "LGB_Boost_Featured")
    PredictModel(df.copy(), LGB_Boost_NoLeak_Featured, "LGB_Boost_NoLeak_Featured")

if __name__ == "__main__":
    main()