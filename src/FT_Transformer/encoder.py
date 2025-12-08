import torch
import torch.nn as nn

# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))


class TransformerEncoderLayer(nn.Module):
    def __init__(self , d_model = 64 , num_heads = 4, ff_hidden = 128):
        super().__init__()
        # LayerNorm for Normalization of Data
        self.norm1 = nn.LayerNorm(d_model)

        # MultiHead self - Attention
        self.attn = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = num_heads,
            batch_first = True
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Feed Forward Network

        self.ffn = nn.Sequential(
            nn.Linear(d_model , ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden , d_model)
        )
    def forward(self, x):

        # x : (B , T ,d_model ) , B =  Batch Size and T= Num Tokens
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm , x_norm , x_norm)

        x = x + attn_output # residual Connection


        # FeedForward Network Block
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output 

        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self , num_layers = 4 , d_model = 64, num_heads =4, ff_hidden = 128):
        super().__init__()

        # Create alist of Encoder Layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model = d_model ,
                num_heads= num_heads,
                ff_hidden = ff_hidden 
            )
            for _ in range(num_layers)
        ])
    def forward(self , x):
        # Pass Input through each encoder layer in sequence
        for layer in self.layers:
            x = layer(x)
        
        return x