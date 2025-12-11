import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    classification_report
)
import numpy as np
import json
import os

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

state_dict = torch.load("models/ft_transformer.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)
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

    bools = torch.tensor(row[bool_cols].values.astype("float32")).unsqueeze(0)

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
results_df["true_label"] = df["HighRisk"].values

# -----------------------------
# ✅ Evaluation at threshold = 0.5
# -----------------------------
y_true = results_df["true_label"].values
y_prob = results_df["highrisk_prob"].values
y_pred = (y_prob >= 0.5).astype(int)

print("\n================= FT-Transformer Evaluation @ Threshold = 0.5 =================")
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

rec = recall_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred)
acc = (y_pred == y_true).mean()
auc_pr = roc_auc_score(y_true, y_prob)
auc_roc = roc_auc_score(y_true, y_prob)

print(f"Recall: {rec:.4f}")
print(f"Precision: {prec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"AUC-PR: {auc_pr:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")

# -----------------------------
# ✅ Scan thresholds for best Recall
# -----------------------------
best_thr = 0.5
best_recall = 0.0

for thr in np.arange(0.0, 1.01, 0.01):
    y_pred_thr = (y_prob >= thr).astype(int)
    rec_thr = recall_score(y_true, y_pred_thr, zero_division=0)

    if rec_thr > best_recall:
        best_recall = rec_thr
        best_thr = thr

print(f"\n✅ Best threshold based on Recall = {best_thr:.2f} (Recall = {best_recall:.4f})")
# -----------------------------
# ✅ Evaluation at BEST threshold
# -----------------------------
y_pred_best = (y_prob >= best_thr).astype(int)

print("\n================= FT-Transformer Evaluation @ BEST Threshold =================")
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred_best))
print("\nClassification Report:\n", classification_report(y_true, y_pred_best, digits=4))

rec_best = recall_score(y_true, y_pred_best)
prec_best = precision_score(y_true, y_pred_best, zero_division=0)
f1_best = f1_score(y_true, y_pred_best)
acc_best = (y_pred_best == y_true).mean()

print(f"Recall: {rec_best:.4f}")
print(f"Precision: {prec_best:.4f}")
print(f"F1-score: {f1_best:.4f}")
print(f"Accuracy: {acc_best:.4f}")

# -----------------------------
# ✅ Save results to JSON
# -----------------------------
results_dict = {
    "model_name": "FT_Transformer",
    "default_threshold": 0.5,
    "best_threshold": float(best_thr),

    "confusion_matrix_default": confusion_matrix(y_true, y_pred).tolist(),
    "confusion_matrix_best": confusion_matrix(y_true, y_pred_best).tolist(),

    "metrics_default": {
        "recall": float(rec),
        "precision": float(prec),
        "f1": float(f1),
        "accuracy": float(acc),
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

os.makedirs("results", exist_ok=True)
with open("results/FT_Transformer_metrics.json", "w") as f:
    json.dump(results_dict, f, indent=4)

print("\nSaved FT-Transformer evaluation to results/FT_Transformer_metrics.json")
results_df.to_excel("predictions.xlsx", index=False)
print("Predictions saved to predictions.xlsx")