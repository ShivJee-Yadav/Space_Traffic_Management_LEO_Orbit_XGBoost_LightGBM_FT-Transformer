import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from FT_Transformer.dataset import SdcDataset
from FT_Transformer.model import FTTransformer

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

for i, col in enumerate(condition_cols):
    df[f"bool_{i}"] = df[col].astype(int)

# ----------------------------------------------------
# 4. HighRisk label (same as XGBoost)
# ----------------------------------------------------
df["HighRisk"] = (
    (df["pc"] > 1e-6) &
    (df["miss_distance"] < 2000)
).astype(int)

# ----------------------------------------------------
# 5. Shuffle and split into train/val
# ----------------------------------------------------
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(0.8 * len(df))
train_df = df[:split]
val_df = df[split:]

# ----------------------------------------------------
# 6. Dataset + DataLoader
# ----------------------------------------------------
train_dataset = SdcDataset(train_df)
val_dataset = SdcDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ----------------------------------------------------
# 7. Model, Loss, Optimizer
# ----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FTTransformer().to(device)

loss_pc_fn = nn.MSELoss()
loss_class_fn = nn.BCELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

epochs = 10

# ----------------------------------------------------
# 8. Training Loop
# ----------------------------------------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        miss, pc, sat1, sat2, obj1, obj2, org1, org2, bools, pc_label, class_label = batch

        # Move to GPU if available
        miss = miss.to(device)
        pc = pc.to(device)
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
            miss, pc, sat1, sat2, obj1, obj2, org1, org2, bools
        )

        # Loss
        loss_pc = loss_pc_fn(pc_pred, pc_label)
        loss_class = loss_class_fn(class_pred, class_label)
        loss = loss_pc + loss_class

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss:.4f}")

# ----------------------------------------------------
# 9. Save Model
# ----------------------------------------------------
torch.save(model.state_dict(), "ft_transformer.pth")
print("Model saved successfully!")