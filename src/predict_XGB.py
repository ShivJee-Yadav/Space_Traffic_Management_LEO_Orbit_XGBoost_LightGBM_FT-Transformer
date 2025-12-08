import os
import pandas as pd
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
        'hours_to_tca'
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

def PredictModel(df, feature_list, model_name):
    Model_DIR = os.path.join("models" ,model_name ) + ".json"
    # Load model
    model = XGBClassifier()
    model.load_model(Model_DIR)
    X_new = df[feature_list]
    
    # Predict probabilities
    y_prob = model.predict_proba(X_new)[:, 1]

    # Use your best threshold
    BEST_THR = 0.30   # example
    y_pred = (y_prob >= BEST_THR).astype(int)

    df["xgb_prob"] = y_prob
    df["xgb_pred"] = y_pred
    result_DIR = os.path.join("outputs","Prediction",model_name +".xlsx")
    df.to_excel(result_DIR, index=False)
    print(f"Saved predictions to {result_DIR}")



DATA_PATH = os.path.join('data' , 'Merged_Featured_DATA.xlsx')
df = pd.read_excel(DATA_PATH)
categorical_cols = [
        'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
        'rso1_objectType', 'rso2_objectType',
        'org1_displayName', 'org2_displayName'
    ]
df[categorical_cols] = df[categorical_cols].astype("category")

PredictModel(df, XG_Boost, "XG_Boost")
# PredictModel(df, XG_Boost_NoLeak, "XG_Boost_NoLeak")
# PredictModel(df, XG_Boost_Featured, "XG_Boost_Feature_data")
# PredictModel(df, XG_Boost_NoLeak_Featured, "XGBoost_NoLeak_Featured")
