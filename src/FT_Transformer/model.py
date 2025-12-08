import torch
import torch.nn as nn

from FT_Transformer.Tokenizer import FeatureTokenizer
from FT_Transformer.encoder import TransformerEncoder


class FTTransformer(nn.Module):
    def __init__(
        self,
        d_model=64,
        num_heads=4,
        ff_hidden=128,
        num_layers=4,
        num_categories_sat1=3,
        num_categories_sat2=3,
        num_categories_obj1=3,
        num_categories_obj2=3,
        num_categories_org1=10,
        num_categories_org2=10,
        num_boolean_features=10
    ):
        super().__init__()

        self.tokenizer = FeatureTokenizer(
            d_model=d_model,
            num_categories_sat1=num_categories_sat1,
            num_categories_sat2=num_categories_sat2,
            num_categories_obj1=num_categories_obj1,
            num_categories_obj2=num_categories_obj2,
            num_categories_org1=num_categories_org1,
            num_categories_org2=num_categories_org2,
            num_boolean_features=num_boolean_features
        )

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            ff_hidden=ff_hidden
        )

        self.pc_head = nn.Linear(d_model, 1)
        self.class_head = nn.Linear(d_model, 1)

    def forward(
        self,
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
        # 1. Tokenize features
        tokens = self.tokenizer(
            miss_distance,
            pc,
            sat1_type,
            sat2_type,
            obj1_type,
            obj2_type,
            org1,
            org2,
            bool_features
        )

        # 2. Pass through encoder stack
        encoded = self.encoder(tokens)

        # 3. Extract CLS token (first token)
        cls_token = encoded[:, 0, :]  # shape (B, d_model)

        # 4. Multi-task outputs
        pc_pred = self.pc_head(cls_token)  # regression
        class_pred = torch.sigmoid(self.class_head(cls_token))  # classification

        return pc_pred, class_pred