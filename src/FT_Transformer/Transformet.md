# d_model is Most Important Hyperparameters
# d_model is 64 means each token have 64 dimensional vectors 
# A Transformer takes tokens (vectors), stacks them into a tensor, and processes them using attention.
# The size of each token vector is called d_model.
# Each token vector is a learned representation of a feature.
#This vector encodes:
#- how important the feature is
#- how it interacts with other features
#- patterns learned during training
# # CLS is a learnable vector (just like a parameter).
# Multi‑Head Self‑Attention
# This allows each feature token to “look at” other tokens.
# Attention learns feature interactions automatically.

# CDM Features (22 columns)
#         ↓
# Feature Tokenizer
#         ↓
# Tokens (22, d_model)
#         ↓
# Add CLS token → (23, d_model)
#         ↓
# Transformer Encoder (2–4 layers)
 [
    # A Transformer encoder is made of stacked layers.
    # Each layer contains:
    # - Multi‑head attention
    # - Feed‑forward network
    # - LayerNorm
    # - Residual connections
 ]
#         ↓
# CLS output (1, d_model)
#         ↓
# MLP Classifier
#         ↓
# Probability (0–1)



<!-- What Are Attention Heads
But instead of doing this once, the model does it multiple times in parallel — these are the attention heads. -->

<!-- **Final Architecture Summary (locked in)
-  d_model = 64
- num_heads = 4
- num_layers = 4
- Multi‑Task outputs:
- Pc regression
- HighRisk classification
This is a strong, balanced FT‑Transformer. -->



<!-- Encoder .py DAY 2 -->
<!-- A Transformer Encoder layer has four components:
- LayerNorm :LayerNorm normalizes each token vector so the model trains smoothly.
- Multi‑Head Self‑Attention
- Feed‑Forward Network (FFN)
- Residual Connections -->

# Input tokens (B, T, 64)
↓
# LayerNorm
↓
# MultiHeadAttention
↓
# Add residual
↓
# LayerNorm
# ↓
# FeedForward
# ↓
# Add residual
# ↓
# Output tokens (B, T, 64)



# Full FT atchitecture  multi‑task model.

<!-- Input features
      ↓
      FeatureTokenizer
      ↓
      Tokens (B, T, 64)
      ↓
      TransformerEncoder (4 layers)
      ↓
      CLS token extracted (B, 64)
      ↓
      Two heads:
         1. Regression head → predict Pc
         2. Classification head → predict HighRisk
      ↓
      Outputs:
    Pc_pred (B, 1)
    HighRisk_pred (B, 1)
 -->