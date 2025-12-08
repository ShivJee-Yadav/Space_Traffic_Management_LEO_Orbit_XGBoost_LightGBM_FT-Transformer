import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from FT_Transformer.model import FTTransformer

# ----------------------------------------------------
# 1. Load new data
# ----------------------------------------------------
df = pd.read_excel("data/Merged_DATA.xlsx")   # <-- your new CDM file

# ----------------------------------------------------
# 2. Rename columns (same as train.py)
# ----------------------------------------------------
df = df.rename(columns={
    "cdmMissDistance": "miss_distance",
    "cdmPc": "pc",
    "SAT1_CDM_TYPE": "sat1_type",
    "SAT2_CDM_TYPE": "sat2_type",
    "rso1_objectType": "obj1_type",
    "rso2_objectType": "obj2_type",
    "org1_displayName": "org1",
    "org2_displayName": "org2"
})

# ----------------------------------------------------
# 3. Convert condition columns into bool_0 ... bool_9
# ----------------------------------------------------
condition_cols = [
    "condition_cdmType=EPHEM:HAC",
    "condition_24H_tca_72H",
    "condition_Pc>1e-6",
    "condition_missDistance<2000m",
    "condition_Radial_100m",
    "condition_Radial<50m",
    "condition_InTrack_500m",
    "condition_CrossTrack_500m",
    "condition_sat2posUnc_1km",
    "condition_sat2Obs_25"
]

for col in condition_cols:
    df[col] = df[col].astype(int)

for i, col in enumerate(condition_cols):
    df[f"bool_{i}"] = df[col]

bool_cols = [f"bool_{i}" for i in range(10)]
df[bool_cols] = df[bool_cols].astype("int64")

# ----------------------------------------------------
# 4. Ensure numeric columns are float
# ----------------------------------------------------
df["miss_distance"] = df["miss_distance"].astype(float)
df["pc"] = df["pc"].astype(float)
df["hours_to_tca"] = df["hours_to_tca"].astype(float)

# ----------------------------------------------------
# 5. Label encode categorical columns
# ----------------------------------------------------
cat_cols = ["sat1_type", "sat2_type", "obj1_type", "obj2_type", "org1", "org2"]

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Compute cardinalities (must match training)
num_categories_sat1 = df["sat1_type"].nunique()
num_categories_sat2 = df["sat2_type"].nunique()
num_categories_obj1 = df["obj1_type"].nunique()
num_categories_obj2 = df["obj2_type"].nunique()
num_categories_org1 = df["org1"].nunique()
num_categories_org2 = df["org2"].nunique()

# ----------------------------------------------------
# 6. Load model
# ----------------------------------------------------
device = torch.device("cpu")

model = FTTransformer(
    num_categories_sat1=num_categories_sat1,
    num_categories_sat2=num_categories_sat2,
    num_categories_obj1=num_categories_obj1,
    num_categories_obj2=num_categories_obj2,
    num_categories_org1=num_categories_org1,
    num_categories_org2=num_categories_org2,
    num_boolean_features=10
).to(device)

model.load_state_dict(torch.load("ft_transformer.pth", map_location=device))
model.eval()

# ----------------------------------------------------
# 7. Run prediction row-by-row
# ----------------------------------------------------
results = []

for idx, row in df.iterrows():

    miss = torch.tensor([[row["miss_distance"]]], dtype=torch.float32)
    pc = torch.tensor([[row["pc"]]], dtype=torch.float32)
    hours_to_tca = torch.tensor([[row["hours_to_tca"]]], dtype=torch.float32)

    sat1 = torch.tensor([row["sat1_type"]], dtype=torch.long)
    sat2 = torch.tensor([row["sat2_type"]], dtype=torch.long)
    obj1 = torch.tensor([row["obj1_type"]], dtype=torch.long)
    obj2 = torch.tensor([row["obj2_type"]], dtype=torch.long)
    org1 = torch.tensor([row["org1"]], dtype=torch.long)
    org2 = torch.tensor([row["org2"]], dtype=torch.long)

    bools = torch.tensor([row[bool_cols].values], dtype=torch.float32)

    with torch.no_grad():
        pc_pred, class_pred = model(
            miss, pc, hours_to_tca,
            sat1, sat2, obj1, obj2, org1, org2,
            bools
        )

    highrisk_probability = float(class_pred.item())
    risk_label = "HighRisk" if highrisk_probability >= 0.5 else "LowRisk"

    results.append({
    "pc_pred": float(pc_pred.item()),
    "highrisk_prob": highrisk_probability,
    "risk_label": risk_label
    })

# ----------------------------------------------------
# 8. Save predictions
# ----------------------------------------------------
results_df = pd.DataFrame(results)
print(results_df)

results_df.to_excel("predictions.xlsx", index=False)
# print("Predictions saved to predictions.xlsx")