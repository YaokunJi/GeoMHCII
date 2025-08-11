import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

class FusionLayer(nn.Module):
    """
    Fusion blocks defined in GeoMHCII.
    :param embed_dim (int): dimension of embedded features
    :param esm_embed (int): output embedding dimension of selected ESM model
    :param num_heads (int): number of heads for cross attention layer
    :param dropout (float): dropout rate for MLP
    """
    def __init__(
        self,
        emb_dim: int = 128,
        output_dim: int = 1280,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        # assert emb_dim % num_heads == 0
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.esm_proj = nn.Linear(output_dim, emb_dim)
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        esm_embedding: torch.Tensor, # (n, d_esm)
        h: torch.Tensor, # (n, d)
        batch: Union[Batch, ProteinBatch]
    ) -> torch.Tensor:
        """
        Compute the fused dense embeddings for batch data

        :param esm_embedding: computed ESM embedding tensor
        :param h: computed node-level embedding for features
        :param batch: data in batch

        :return: fused dense node-level embeddings
        """

        residual = h
        # dense tensor converted to batch for cross-attention
        h_dense, h_mask = to_dense_batch(h, batch.batch)
        h_esm = self.esm_proj(esm_embedding)
        esm_dense, _ = to_dense_batch(h_esm, batch.batch)
        B, pad_len, _ = h_dense.shape
        # calculate cross-attention scores
        q = self.q_proj(h_dense).view(B, pad_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(esm_dense).view(B, pad_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(esm_dense).view(B, pad_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_scores = attn_scores.masked_fill(~h_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = (attn_weights @ v).transpose(1, 2).reshape(B, pad_len, -1)
        output = self.out_proj(output)
        # convert back to dense
        output = output[h_mask]  # (B, pad_len, d) -> (n, d)
        return self.norm(output + residual)