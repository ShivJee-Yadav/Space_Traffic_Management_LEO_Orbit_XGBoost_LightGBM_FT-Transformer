# pip install optuna

import os
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from optuna.samplers import TPESampler # TPE = Tree‑structured Parzen Estimator
from sklearn.metrics import average_precision_score # AUC-PR
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from XG_Boost_Feature_data import compute_scale_pos_weight
# 6) Feature lists

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
# 6) Feature list
XG_Boost_NoLeak_Featured = [
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



def tune_model(df, feature_list, model_name):
    X = df[feature_list]
    y = df['HighRisk']
    ## change this to train, validation and test set After##
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    spw = compute_scale_pos_weight(y_train)

    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
        n_estimators = trial.suggest_int("n_estimators", 200, 800)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0)
        min_child_weight = trial.suggest_float("min_child_weight", 1, 10)
        gamma = trial.suggest_float("gamma", 0, 10)
        reg_lambda = trial.suggest_float("reg_lambda", 0.1, 10)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.5*spw, 2.0*spw)

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='aucpr',
            tree_method="hist",
            enable_categorical=True
        )

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        score = average_precision_score(y_test, y_prob)
        return score

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=20)

    # Save results
    result_path = f"{model_name}_best_params.txt"
    result_DIR = os.path.join('outputs','Best_HyperParameter',result_path)
    with open(result_DIR, "w") as f:
        f.write(f"Best AUC-PR: {study.best_value}\n")
        for k, v in study.best_params.items():
            f.write(f"{k}: {v}\n")

    print(f"\n Saved best params for {model_name} → {result_path}")




DATA_PATH = os.path.join('data' , 'Merged_Featured_DATA.xlsx')
df = pd.read_excel(DATA_PATH)
categorical_cols = [
        'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
        'rso1_objectType', 'rso2_objectType',
        'org1_displayName', 'org2_displayName'
    ]
df[categorical_cols] = df[categorical_cols].astype("category")
   
# tune_model(df, XG_Boost, "XGBoost_Original")
tune_model(df, XG_Boost_NoLeak, "XGBoost_NoLeak")
# tune_model(df, XG_Boost_Featured, "XGBoost_Featured")
# tune_model(df, XG_Boost_NoLeak_Featured, "XGBoost_NoLeak_Featured")
