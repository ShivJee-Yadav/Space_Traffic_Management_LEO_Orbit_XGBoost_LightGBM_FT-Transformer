import torch
import torch.nn as nn

# print("CUDA available:", torch.cuda.is_available())
# print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

#  basic PyTorch module structure.
class FeatureTokenizer(nn.Module):
    """
    Converts raw tabular features into a sequence of token embeddings.
    Each feature becomes one token of size d_model.
    """

    def __init__(self , 
                 d_model = 64,
                 num_categories_sat1 =3,
                 num_categories_sat2 =3,
                 num_categories_obj1 =3,
                 num_categories_obj2 =3,
                 num_categories_org1 = 10,
                 num_categories_org2 = 10,
                 num_boolean_features = 10
    ):
        super().__init__()
        self.d_model = d_model
        self.num_embed_miss = nn.Linear(1,d_model)
        self.num_embed_pc = nn.Linear(1,d_model)
        self.cat_embed_sat1 = nn.Embedding(num_categories_sat1 , d_model)
        self.cat_embed_sat2 = nn.Embedding(num_categories_sat2 , d_model)
        self.cat_embed_obj1 = nn.Embedding(num_categories_obj1 , d_model)
        self.cat_embed_obj2 = nn.Embedding(num_categories_obj2 , d_model)
        self.cat_embed_org1 = nn.Embedding(num_categories_org1 , d_model)
        self.cat_embed_org2 = nn.Embedding(num_categories_org2 , d_model)
        self.bool_embed = nn.Linear(num_boolean_features, d_model)
        self.CLS = nn.Parameter(torch.randn(1,1,d_model))
    
    def forward(
            self ,
            miss_distance,
            pc,
            sat1_type,
            sat2_type,
            obj1_type,
            obj2_type,
            org1,
            org2,
            bool_features
    ):
        """
        Inputs:
            miss_distance: (B, 1)
            pc: (B, 1)
            sat1_type: (B,)
            sat2_type: (B,)
            obj1_type: (B,)
            obj2_type: (B,)
            org1: (B,)
            org2: (B,)
            bool_features: (B, num_boolean_features)
        """

        B= miss_distance.size(0)
        t_miss = self.num_embed_miss(miss_distance)
        t_pc = self.num_embed_pc(pc)
        t_sat1 = self.cat_embed_sat1(sat1_type)      # (B, d_model)
        t_sat2 = self.cat_embed_sat2(sat2_type)      # (B, d_model)
        t_obj1 = self.cat_embed_obj1(obj1_type)      # (B, d_model)
        t_obj2 = self.cat_embed_obj2(obj2_type)      # (B, d_model)
        t_org1 = self.cat_embed_org1(org1)           # (B, d_model)
        t_org2 = self.cat_embed_org2(org2)           # (B, d_model)
        t_bool = self.bool_embed(bool_features)      # (B, d_model)

        # stack all tokens 
        tokens = torch.stack([t_miss, t_pc , t_sat1 , t_sat2 , t_obj1 , t_obj2 , t_org1 , t_org2 , t_bool],
                           dim =1 )
        
        CLS = self.CLS.expand(B,1,self.d_model)
        tokens = torch.cat([CLS , tokens] , dim = 1)
        return tokens