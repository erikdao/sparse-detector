"""
Sparse attention modules
"""
import math
from typing import Any, Tuple, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, constant_

from entmax import sparsemax, entmax15, entmax_bisect

from sparse_detector.models.tvmax import TV2DFunction

VALID_ACTIVATION = ['softmax', 'sparsemax', 'tvmax', 'entmax15', 'entmax_alpha']


def tvmax2d(X: Tensor) -> None:
    """
    X: (B, Nt, Ns)
    """
    tvmax = TV2DFunction.apply
    # Hacky way, need to figure out how to make tvmax works on batch
    for i in range(X.size(0)):
        X[i] = tvmax(X[i])
    
    return X


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None, dropout_p: float = 0.0,
    activation: str = "softmax", entmax_alpha: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    Modified version of scaled dot-production attention, support sparse activation functions
    beside softmax
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)  # Normalisation
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    
    print(f"attn: {attn.shape}")

    # This is the HEART of the first sparse experiment
    if activation not in VALID_ACTIVATION:
        raise RuntimeError(f"Unsupported activation function {activation}")
    elif activation == 'softmax':
        attn = F.softmax(attn, dim=-1)
    elif activation == 'sparsemax':
        attn = sparsemax(attn, dim=-1)
    elif activation == 'entmax15':
        attn = entmax15(attn, dim=-1)
    elif activation == 'tvmax':  # Total variation 2D
        attn = tvmax2d(attn)
    elif activation == 'entmax_alpha':
        if entmax_alpha is None:
            raise ValueError(f"Activation {activation} requires a learnable alpha, i.e., a tensor that has gradients")
        attn = entmax_bisect(attn, entmax_alpha)
        

    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


class SparseMultiheadAttention(nn.Module):
    """
    A simplified implementation of multihead attention, support sparse activation functions
    besides softmax
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, activation: str = 'softmax', device: Any = None, dtype: Any = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.activation = activation

        if self.activation == 'entmax_alpha':
            # Initialise a learnable alpha if the activation funciton is alpha-entmax
            a = Parameter(torch.tensor(1.5, **factory_kwargs), requires_grad=True)
            self.entmax_alpha = Parameter(1 + torch.sigmoid(a), requires_grad=True)
        else:
            self.entmax_alpha = None

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
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        TODO: Implement the missing logics with attn_mask as in Pytorch's implementation.
        Technically, DETR forward pass doesn't use attn_mask (it's all zeros). However, it might have a role
        during the backward pass so it's better to implement the logic.
        """
        is_batched = query.dim() == 3

        # setup shape variables
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

        # compute in-projection
        q, k, v = F._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)

        # prepare attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                attn_mask = attn_mask.to(torch.bool)
            # ensure attention mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

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
        attn_output, attn_output_weights = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, self.activation, self.entmax_alpha)
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
