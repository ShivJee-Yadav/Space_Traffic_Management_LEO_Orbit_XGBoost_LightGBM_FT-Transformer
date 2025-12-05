import os
import pandas as pd
import numpy as np
# FIle size increased by 10 mb after Feature Engineering
# Paths
INPUT_PATH = os.path.join("data", "Merged_DATA.xlsx")
OUTPUT_PATH = os.path.join("data", "Merged_Featured_DATA.xlsx")

# Load cleaned data
df = pd.read_excel(INPUT_PATH)

# -----------------------------
# 1. Log transform of Pc
# -----------------------------
df['log_cdmPc'] = np.log1p(df['cdmPc'])   # log(1 + Pc)

# -----------------------------
# 2. Inverse miss distance
# -----------------------------
df['inv_miss_distance'] = 1 / (df['cdmMissDistance'] + 1)

# -----------------------------
# 3. TCA time bin (12-hour buckets)
# -----------------------------
df['tca_bin'] = (df['hours_to_tca'] // 12).astype(int)

# -----------------------------
# 4. Same satellite type
# -----------------------------
df['same_sat_type'] = (df['SAT1_CDM_TYPE'] == df['SAT2_CDM_TYPE']).astype(int)

# -----------------------------
# 5. Debris–debris pair
# -----------------------------
df['is_debris_pair'] = (
    (df['rso1_objectType'] == "DEBRIS") &
    (df['rso2_objectType'] == "DEBRIS")
).astype(int)

# -----------------------------
# 6. Close in all axes
# -----------------------------
df['close_all_axes'] = (
    df['condition_InTrack_500m'] &
    df['condition_CrossTrack_500m'] &
    df['condition_Radial_100m']
).astype(int)

# -----------------------------
# 7. Risky uncertainty
# -----------------------------
df['risky_uncertainty'] = (
    df['condition_sat2posUnc_1km'] &
    df['condition_InTrack_500m']
).astype(int)

# -----------------------------
# 8. Distance ratio
# -----------------------------
df['distance_ratio'] = df['cdmMissDistance'] / (df['hours_to_tca'] + 1)

# -----------------------------
# 9. Object type match
# -----------------------------
df['object_type_match'] = (
    df['rso1_objectType'] == df['rso2_objectType']
).astype(int)

# -----------------------------
# Save engineered dataset
# -----------------------------
print(df.info())
print("Writing File")
# df.to_excel(OUTPUT_PATH, index=False)
print(f"Feature engineering complete. Saved to: {OUTPUT_PATH}")


# ---

# ###  **1. `log_cdmPc`**  
# Makes tiny Pc values easier for the model to learn by applying a log transform.

# ###  **2. `inv_miss_distance`**  
# Gives higher importance to closer approaches by taking the inverse of miss distance.

# ###  **3. `tca_bin`**  
# Groups CDMs into 12‑hour time windows to capture time‑to‑TCA patterns.

# ###  **4. `same_sat_type`**  
# Checks whether both satellites belong to the same CDM type category.

# ###  **5. `is_debris_pair`**  
# Flags conjunctions where both objects are debris, which behave differently.

# ###  **6. `close_all_axes`**  
# Identifies cases where radial, in‑track, and cross‑track distances are all small.

# ###  **7. `risky_uncertainty`**  
# Marks events with high uncertainty combined with close in‑track distance.

# ###  **8. `distance_ratio`**  
# Normalizes miss distance by time‑to‑TCA to capture approach speed.

# ###  **9. `object_type_match`**  
# Checks whether both objects share the same physical type (payload, debris, etc.).
