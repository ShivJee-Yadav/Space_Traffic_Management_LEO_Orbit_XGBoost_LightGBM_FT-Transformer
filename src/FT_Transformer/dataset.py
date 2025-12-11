import torch
from torch.utils.data import Dataset

class SdcDataset(Dataset):
    def __init__(self, df):
        self.df = df

        self.num_cols = ["miss_distance", "pc","hours_to_tca"]
        self.cat_cols = ["sat1_type", "sat2_type", "obj1_type",
                         "obj2_type", "org1", "org2"]
        self.bool_cols = [f"bool_{i}" for i in range(10)]

        self.label_oc = "pc"          # FIXED
        self.label_class = "HighRisk"
 
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]       # FIXED: moved before debug prints

        # Debug (optional)
        # print("ROW BOOL TYPES:", row[self.bool_cols].apply(type).tolist())
        # print("ROW BOOL VALUES:", row[self.bool_cols].values)

        # Numeric features
        miss_distance = torch.tensor([row["miss_distance"]], dtype=torch.float32)
        pc = torch.tensor([row["pc"]], dtype=torch.float32)
        hours_to_tca = torch.tensor([row["hours_to_tca"]], dtype=torch.float32)
        # Categorical features
        sat1 = torch.tensor(row["sat1_type"], dtype=torch.long)
        sat2 = torch.tensor(row["sat2_type"], dtype=torch.long)
        obj1 = torch.tensor(row["obj1_type"], dtype=torch.long)
        obj2 = torch.tensor(row["obj2_type"], dtype=torch.long)
        org1 = torch.tensor(row["org1"], dtype=torch.long)
        org2 = torch.tensor(row["org2"], dtype=torch.long)

        # Boolean features
        bool_features = torch.tensor(row[self.bool_cols].astype(float).values, dtype=torch.float32)

        # Labels
        pc_label = torch.tensor([row["pc"]], dtype=torch.float32)        # FIXED
        class_label = torch.tensor([row["HighRisk"]], dtype=torch.float32)

        return (
            miss_distance,
            pc,
            hours_to_tca,
            sat1,
            sat2,
            obj1,
            obj2,
            org1,
            org2,
            bool_features,
            pc_label,
            class_label
        )