import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    """
        Forward Args:
        queries (torch.Tensor): Query tensor of shape `(batch_size, seq_len, num_heads, embedding_dim)`.
        keys (torch.Tensor): Key tensor of shape `(batch_size, seq_len, num_heads, embedding_dim)`.
        values (torch.Tensor): Value tensor of shape `(batch_size, seq_len, num_heads, value_dim)`.
        sigma (torch.Tensor): Scaling factor for the Gaussian prior of shape `(batch_size, seq_len, num_heads)`.
        attn_mask (torch.Tensor or None): Attention mask of shape `(batch_size, seq_len, seq_len)`,
            applied if `mask_flag` is True.

    Returns:
        tuple:
            - `V` (torch.Tensor): The attended output of shape `(batch_size, seq_len, num_heads, value_dim)`.
            - `series` (torch.Tensor or None): Attention weights of shape `(batch_size, num_heads, seq_len, seq_len)`,
                returned if `output_attention` is True.
            - `prior` (torch.Tensor or None): The computed Gaussian prior of shape `(batch_size, num_heads, seq_len, seq_len)`,
                returned if `output_attention` is True.
            - `sigma` (torch.Tensor or None): The transformed sigma values used for the prior of shape `(batch_size, num_heads, seq_len)`,
                returned if `output_attention` is True.
    """

    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = (win_size - 3) // 2
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask, unmask_indices=None):
        B, L, H, E = queries.shape  # E: model dimension
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = (attn.shape[-1] - 3) // 2
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)[:, :, :, :]  # B H L L, with cls-token removed
        # ----- Prior with masked -----
        if unmask_indices is None:
            prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        else:
            prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1) \
                        [:, :, unmask_indices, :][:, :, :, unmask_indices].cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))
        # ----- ----- -----

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            series_ = series[:, :, 1:window_size+1, :][:, :, :, 1:window_size+1]
            series_ = torch.softmax(series_, dim=-1)
            prior = torch.softmax(prior, dim=-1)
            return V.contiguous(), series_, prior, sigma
        else:
            return V.contiguous(), None


class AnomalyAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, num_patches, dropout, d_keys=None, d_values=None, ):
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_values * n_heads)

        self.inner_attention = AnomalyAttention(num_patches + 3, False, attention_dropout=dropout,
                                                output_attention=True)
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.feedforward_layer = nn.Sequential(
            nn.Linear(d_values * n_heads, d_values * n_heads // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_values * n_heads // 2, d_model),
            nn.Dropout(dropout)
        )

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, unmask_indices=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        x = queries
        queries = self.norm(queries)
        keys = queries
        values = queries

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)
        sigma = sigma[:, 1:(L - 3)//2+1, :]
        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask,
            unmask_indices
        )
        out = out + queries
        out = out.view(B, L, -1)
        out = self.norm2(out)
        out = self.feedforward_layer(out)
        out = out.reshape(B, L, -1) + x
        return out, series, prior, sigma


if __name__ == '__main__':
    import torch
    import numpy as np
    import math

    # Define dummy input parameters
    B = 4  # Batch size
    L = 10  # Sequence length (window size)
    H = 8  # Number of attention heads
    E = 8  # Embedding dimension per head

    # Create dummy tensors
    queries = torch.randn(B, L, H * E).cuda()  # Queries [B, L, H, E]
    keys = torch.randn(B, L, H * E).cuda()  # Keys [B, L, H, E]
    values = torch.randn(B, L, H * E).cuda()  # Values [B, L, H, E]
    sigma = torch.randn(B, L, H).cuda()  # Sigma [B, L, H]
    attn_mask = None  # No attention mask

    # Initialize AnomalyAttention with dummy parameters
    win_size = L  # Window size should match sequence length
    anomaly_attention = AttentionLayer(AnomalyAttention(win_size, output_attention=True), 64, 8).cuda()

    # Forward pass through AnomalyAttention
    output, series, prior, sigma_out = anomaly_attention(queries, keys, values, attn_mask)

    # Print output shapes
    print("Output Shape:", output.shape)  # Expected: [B, L, H, E]
    print("Series Shape:", series.shape if series is not None else "None")  # Expected: [B, H, L, L]
    print("Prior Shape:", prior.shape if prior is not None else "None")  # Expected: [B, H, L, L]
    print("Sigma Shape:", sigma_out.shape if sigma_out is not None else "None")  # Expected: [B, H, L]
