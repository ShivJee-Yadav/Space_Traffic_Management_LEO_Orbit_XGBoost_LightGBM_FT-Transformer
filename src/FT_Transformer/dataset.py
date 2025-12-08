import torch                # PyTorch only understands tensors
from torch.utils.data import Dataset

class SdcDataset(Dataset):
    def __init__(self , df):
        self.df = df
        self.num_cols = ["miss_distance" , "pc"]
        self.cat_cols = ["sat1_type" , "sat2_type" , "obj1_type",
                         "obj2_type" , "org1" , "org2"]
        self.bool_cols = [f"bool_{i}" for i in range(10)]

        self.label_oc = "Pc"
        self.label_class = "HighRisk"
    
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx): # heart of the Dataset.
    #         When the model asks for row 5:
    # - We extract row 5
    # - Convert each feature into a tensor
    # - Return everything as a tuple

        row = self.df.iloc[idx]

        miss_distance = torch.tensor([row["miss_distance"]], dtype = torch.float32)
        pc = torch.tensor([row["pc"]] , dtype = torch.float32)

        sat1 = torch.tensor(row["sat1_type"], dtype=torch.long)
        sat2 = torch.tensor(row["sat2_type"], dtype=torch.long)
        obj1 = torch.tensor(row["obj1_type"], dtype=torch.long)
        obj2 = torch.tensor(row["obj2_type"], dtype=torch.long)
        org1 = torch.tensor(row["org1"], dtype=torch.long)
        org2 = torch.tensor(row["org2"], dtype=torch.long)

        bool_features = torch.tensor(row[self.bool_cols].values, dtype = torch.float32)

        pc_label = torch([row["Pc"]] ,dtype = torch.flaot32)
        class_label = torch.tensor([row["HighRisk"]], dtype = torch.float32)

        return (
            miss_distance,
            pc,
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
    
