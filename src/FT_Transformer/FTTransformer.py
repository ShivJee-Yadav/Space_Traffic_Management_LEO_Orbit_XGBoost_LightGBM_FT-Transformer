import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, average_precision_score
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.metrics import brier_score_loss, mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc


from dataset import SdcDataset
from model import FTTransformer

# ----------------------------------------------------
# 1. Load the cleaned dataset created by preprocess.py
# ----------------------------------------------------
df = pd.read_excel("data/Merged_DATA.xlsx")

# ----------------------------------------------------
# 2. Rename columns to match FT-Transformer expectations
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
print(df.info())
df["hours_to_tca"] = df["hours_to_tca"].astype(float)

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

# Step 1 — Convert booleans to integers
for col in condition_cols:
    df[col] = df[col].astype(int)

# Step 2 — Create bool_0 ... bool_9
for i, col in enumerate(condition_cols):
    df[f"bool_{i}"] = df[col]

# Step 3 — Force final dtype to int64
bool_cols = [f"bool_{i}" for i in range(10)]
df[bool_cols] = df[bool_cols].astype("int64")


# ----------------------------------------------------
# 4. Encode categorical columns
# ----------------------------------------------------
cat_cols = [
    "sat1_type",
    "sat2_type",
    "obj1_type",
    "obj2_type",
    "org1",
    "org2"
]

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

num_categories_sat1 = df["sat1_type"].nunique()  # 2
num_categories_sat2 = df["sat2_type"].nunique()  # 1
num_categories_obj1 = df["obj1_type"].nunique()  # 4
num_categories_obj2 = df["obj2_type"].nunique()  # 2
num_categories_org1 = df["org1"].nunique()       # 10
num_categories_org2 = df["org2"].nunique()       # 2input("DEBUGGER STop")
# ----------------------------------------------------
# 6. Shuffle and split into train/val
# ----------------------------------------------------
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(0.8 * len(df))
train_df = df[:split]
val_df = df[split:]

# ----------------------------------------------------
# 7. Dataset + DataLoader
# ----------------------------------------------------
train_dataset = SdcDataset(train_df)
val_dataset = SdcDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512)

# ----------------------------------------------------
# 8. Model, Loss, Optimizer
# ----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FTTransformer(
    num_categories_sat1=num_categories_sat1,
    num_categories_sat2=num_categories_sat2,
    num_categories_obj1=num_categories_obj1,
    num_categories_obj2=num_categories_obj2,
    num_categories_org1=num_categories_org1,
    num_categories_org2=num_categories_org2,
    num_boolean_features=10,
).to(device)

loss_pc_fn = nn.MSELoss()
loss_class_fn = nn.BCELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

epochs = 10

# ----------------------------------------------------
# 9. Training Loop
# ----------------------------------------------------
print("Training Started :")
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        miss, pc,hours_to_tca,  sat1, sat2, obj1, obj2, org1, org2, bools, pc_label, class_label = batch

        # Move to GPU if available
        miss = miss.to(device)
        pc = pc.to(device)
        hours_to_tca = hours_to_tca.to(device)
        sat1 = sat1.to(device)
        sat2 = sat2.to(device)
        obj1 = obj1.to(device)
        obj2 = obj2.to(device)
        org1 = org1.to(device)
        org2 = org2.to(device)
        bools = bools.to(device)
        pc_label = pc_label.to(device)
        class_label = class_label.to(device)

        # Forward pass
        pc_pred, class_pred = model(
            miss, pc, hours_to_tca, sat1, sat2, obj1, obj2, org1, org2, bools
        )

        # Loss
        loss_pc = loss_pc_fn(pc_pred, pc_label)
        loss_class = loss_class_fn(class_pred, class_label)
        loss = loss_pc + loss_class # combiner Loss BCE Loss + MSE loss

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss:.4f}")

    # -----------------------
    # Validation
    # -----------------------
    model.eval()
    val_losses = 0.0

    all_class_labels = []
    all_class_preds = []   # predicted probabilities
    all_pc_labels = []
    all_pc_preds = []

    with torch.no_grad():
        for batch in val_loader:
            # unpack batch (now includes hours_to_tca if you added it)
            miss, pc, hours_to_tca, sat1, sat2, obj1, obj2, org1, org2, bools, pc_label, class_label = batch

            miss = miss.to(device)
            pc = pc.to(device)
            hours_to_tca = hours_to_tca.to(device)
            sat1 = sat1.to(device)
            sat2 = sat2.to(device)
            obj1 = obj1.to(device)
            obj2 = obj2.to(device)
            org1 = org1.to(device)
            org2 = org2.to(device)
            bools = bools.to(device)
            pc_label = pc_label.to(device)
            class_label = class_label.to(device)

            pc_pred, class_pred = model(
                miss, pc, hours_to_tca, sat1, sat2, obj1, obj2, org1, org2, bools
            )

            loss_pc = loss_pc_fn(pc_pred, pc_label)
            loss_class = loss_class_fn(class_pred, class_label)
            loss = loss_pc + loss_class

            val_losses += loss.item()

            # store for metrics
            all_class_labels.append(class_label.cpu())
            all_class_preds.append(class_pred.cpu())
            all_pc_labels.append(pc_label.cpu())
            all_pc_preds.append(pc_pred.cpu())

    # concatenate all batches
    import torch as _torch  # avoid confusion with top-level torch
    all_class_labels = _torch.cat(all_class_labels).numpy().ravel()
    all_class_preds = _torch.cat(all_class_preds).numpy().ravel()
    all_pc_labels = _torch.cat(all_pc_labels).numpy().ravel()
    all_pc_preds = _torch.cat(all_pc_preds).numpy().ravel()

    # binarize predictions at 0.5 for recall
    class_preds_binary = (all_class_preds >= 0.5).astype(int)

    # compute metrics
    
    # --- Classification metrics (probabilities) ---
    auc_pr = average_precision_score(all_class_labels, all_class_preds)
    try:
        auc_roc = roc_auc_score(all_class_labels, all_class_preds)
    except ValueError:
        auc_roc = float("nan")

    # Threshold sweep to find best F1 and best recall subject to min precision
    best_f1 = 0.0
    best_f1_thr = 0.5
    best_recall_thr = 0.5
    best_recall = 0.0
    min_precision_for_recall = 0.2

    precisions, recalls, thrs = precision_recall_curve(all_class_labels, all_class_preds)
    # precision_recall_curve returns len(thrs)+1 for precisions/recalls; align by ignoring last precision/recall
    for i, thr in enumerate(np.append(thrs, 1.0)):
        preds_thr = (all_class_preds >= thr).astype(int)
        prec = precision_score(all_class_labels, preds_thr, zero_division=0)
        rec = recall_score(all_class_labels, preds_thr, zero_division=0)
        f1 = f1_score(all_class_labels, preds_thr, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_thr = thr
        if (prec >= min_precision_for_recall) and (rec > best_recall):
            best_recall = rec
            best_recall_thr = thr

    # Binarize at best F1 for reporting
    preds_best_f1 = (all_class_preds >= best_f1_thr).astype(int)

    # Confusion matrix and classification report
    cm_default = confusion_matrix(all_class_labels, class_preds_binary)
    cm_best = confusion_matrix(all_class_labels, preds_best_f1)
    report_default = classification_report(all_class_labels, class_preds_binary, digits=4)
    report_best = classification_report(all_class_labels, preds_best_f1, digits=4)

    # Brier score (calibration)
    brier = brier_score_loss(all_class_labels, all_class_preds)

    # Error counts
    tn, fp, fn, tp = cm_best.ravel()

    # --- Regression metrics for Pc ---
    mse = mean_squared_error(all_pc_labels, all_pc_preds)
    mae = mean_absolute_error(all_pc_labels, all_pc_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_pc_labels, all_pc_preds)
    
    val_loss_avg = val_losses / len(val_loader)
    recall = recall_score(all_class_labels, class_preds_binary)
    auc_pr = average_precision_score(all_class_labels, all_class_preds)

    # --- Print everything clearly ---
    print("\n===== FT-Transformer Validation Summary =====")
    print(f"Val Loss Avg: {val_loss_avg:.6f}")
    print(f"AUC-PR (avg precision): {auc_pr:.6f}")
    print(f"AUC-ROC: {auc_roc:.6f}")
    print(f"Brier score: {brier:.6f}")
    print(f"Best F1: {best_f1:.4f} at threshold {best_f1_thr:.3f}")
    print(f"Best recall (prec >= {min_precision_for_recall}): {best_recall:.4f} at threshold {best_recall_thr:.3f}")
    print("\n--- Default threshold (0.5) classification report ---")
    print(report_default)
    print("Confusion matrix (default 0.5):")
    print(cm_default)
    print("\n--- Best-F1 threshold classification report ---")
    print(report_best)
    print("Confusion matrix (best F1):")
    print(cm_best)
    print(f"\nTP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Precision (best F1): {precision_score(all_class_labels, preds_best_f1, zero_division=0):.4f}")
    print(f"Recall (best F1): {recall_score(all_class_labels, preds_best_f1):.4f}")
    print(f"F1 (best F1): {f1_score(all_class_labels, preds_best_f1):.4f}")

    print("\n--- Regression (Pc) metrics ---")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2: {r2:.6f}")

    # Show a few top failure examples for manual inspection (requires original val_df)
    try:
        val_df_reset = val_df.reset_index(drop=True)
        preds_series = pd.Series(all_class_preds, name="ft_prob")
        labels_series = pd.Series(all_class_labels, name="label")
        inspect_df = val_df_reset.copy()
        inspect_df["ft_prob"] = preds_series
        inspect_df["label"] = labels_series
        # top false positives (high prob but label=0)
        fp_examples = inspect_df[(inspect_df["label"]==0)].sort_values("ft_prob", ascending=False).head(5)
        # top false negatives (low prob but label=1)
        fn_examples = inspect_df[(inspect_df["label"]==1)].sort_values("ft_prob", ascending=True).head(5)
        print("\nTop 5 False Positives (high prob, label=0):")
        print(fp_examples[["ft_prob", "label"] + bool_cols + ["miss_distance", "pc", "hours_to_tca"]].to_string(index=False))
        print("\nTop 5 False Negatives (low prob, label=1):")
        print(fn_examples[["ft_prob", "label"] + bool_cols + ["miss_distance", "pc", "hours_to_tca"]].to_string(index=False))
    except Exception as e:
        print("Could not show failure examples (need val_df in scope). Error:", e)

    print("=============================================\n")
# ----------------------------------------------------
# 10. Save Model
# ----------------------------------------------------
torch.save(model.state_dict(), "ft_transformer.pth")
print("Model saved successfully!")