import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# Load Dataset 
df = pd.read_excel("Train_2000.xlsx")
print(df.info())

df['cdmPc'] = df['cdmPc'].fillna(df['cdmPc'].median())
df['rso2_objectType'] = df['rso2_objectType'].fillna("DEBRIS")

df = df.rename(columns={
    'condition_24H<tca<72H': 'condition_24H_tca_72H',
    'condition_Radial<100m': 'condition_Radial_100m',
    'condition_InTrack<500m': 'condition_InTrack_500m',
    'condition_CrossTrack<500m': 'condition_CrossTrack_500m',
    'condition_sat2posUnc>1km': 'condition_sat2posUnc_1km',
    'condition_sat2Obs<25': 'condition_sat2Obs_25'
})

from sklearn.preprocessing import LabelEncoder # for ecnoding Categorical Features 

categorical_cols = ['SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
                    'rso1_objectType', 'rso2_objectType',
                    'org1_displayName', 'org2_displayName']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))


# Build Feature Matrix

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

# - Why: Conjunctions closer in time can be operationally more urgent.
df['creationTsOfCDM'] = pd.to_datetime(df['creationTsOfCDM'])
df['cdmTca'] = pd.to_datetime(df['cdmTca'])
df['hours_to_tca'] = (df['cdmTca'] - df['creationTsOfCDM']).dt.total_seconds() / 3600.0

df['HighRisk'] = ((df['cdmPc'] > 1e-6) & (df['cdmMissDistance'] < 2000)).astype(int)

X = df[features]
y = df['HighRisk']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

import xgboost 
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

model = XGBClassifier(
    n_estimators = 400,
    max_depth = 4,
    learning_rate = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    reg_lambda = 1.0,
    random_state = 42,
    eval_metric = 'logloss'
)

model.fit(
    X_train,y_train,
    eval_set = [(X_test,y_test)],
    verbose = False,
    
)
# print("Best iteration:", model.best_iteration)
print("#" * 50)
y_prob = model.predict_proba(X_test)[:, 1]   # probability of class 1 (HighRisk)
y_pred = (y_prob >= 0.3).astype(int)

print(classification_report(y_test, y_pred, digits=4))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
