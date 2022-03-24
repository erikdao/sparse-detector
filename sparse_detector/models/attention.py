"""
Sparse attention modules
"""
import math
from typing import Tuple, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, constant_

from entmax import sparsemax


def tvmax2d(x) -> None:
    pass


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None, dropout_p: float = 0.0,
    activation: str = "softmax"
) -> Tuple[Tensor, Tensor]:
    """
    Modified version of scaled dot-production attention, support sparse activation functions
    beside softmax
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    
    # This is the HEART of the first sparse experiment
    if activation == 'softmax':
        attn = F.softmax(attn, dim=-1)
    elif activation == 'sparsemax':
        attn = sparsemax(attn, dim=-1)
    elif activation == 'tva':  # Total variation 2D
        attn = tvmax2d(attn)
    else:
        raise ValueError(f"Unsupported activation function {activation}")

    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


class MultiheadAttention(nn.Module):
    """
    A simplified implementation of multihead attention, support sparse activation functions
    besides softmax
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, activation: str = 'softmax', device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.activation = activation

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))

        self.out_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        xavier_uniform_(self.out_proj.weight)
        constant_(self.out_proj.bias, 0.)
    

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        is_batched = query.dim() == 3

        # setup shape variables
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

        # compute in-projection
        q, k, v = F._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)

        # prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            key_padding_mask = key_padding_mask.to(torch.bool)

        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # update source sequence length after adjustments
        src_len = k.size(1)
    
        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask

        # adjust dropout probability
        dropout_p = self.dropout
        if not self.training:
            dropout_p = 0.0

        #
        # (deep breath) calculate attention and out projection
        #
        attn_output, attn_output_weights = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, self.activation)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
