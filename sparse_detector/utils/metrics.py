"""
Useful metrics for measuring sparsity
"""
from typing import Any, Optional

import torch


def gini(w: torch.Tensor) -> torch.Tensor:
    r"""The Gini coeffeicent from the `"Improving Molecular Graph Neural
    Network Explainability with Orthonormalization and Induced Sparsity"
    <https://arxiv.org/abs/2105.04854>`_ paper

    Computes a regularization penalty for each row of a matrix according to:

    .. math::
        \mathcal{L}_\textrm{Gini}^i = \sum_j^n \sum_{j'}^n \frac{|w_{ij}
         - w_{ij'}|}{2 (n^2 - n)\bar{w_i}}

    And returns an average over all rows.

    Args:
        w (torch.Tensor): A two-dimensional tensor.

    Returns:
        The value of the Gini coefficient for this tensor :math:`\in [0, 1]`

    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/functional/gini.html
    """
    s = 0
    for row in w:
        t = row.repeat(row.size(0), 1)
        u = (t - t.T).abs().sum() / (2 * (row.size(-1)**2 - row.size(-1)) *
                                     row.abs().mean() + torch.finfo().eps)
        s += u
    s /= w.shape[0]
    return s


def gini_alternative(w: torch.Tensor) -> torch.Tensor:
    s = 0
    for row in w:
        y, _ = torch.sort(row)  # sort `row` in non-decreasing order
        n = y.size(0)
        indices = torch.tensor(torch.arange(n) + 1, device=y.device)  # So that indices is on the same device as y
        yp = torch.mul(indices, y)
        u = 1 - 2.0 * (n - (yp.sum() / (y.sum() + torch.finfo().eps))) / (n - 1)
        s += u
    
    s /= w.shape[0]
    return s


def gini_sorted(w: torch.Tensor) -> torch.Tensor:
    """
    Compute Gini score for rows in `w`.
    The rows in `w` are first sorted in ascending order
    Reference: https://arxiv.org/pdf/0811.4706.pdf
    """
    s = 0.0
    for row in w:
        y = torch.ravel(row)
        y, _ = torch.sort(y)
        N = y.size(0)
        norm1_y = y.sum()
        if norm1_y == 0.0:
            u = 0.0
        else:
            coeff = y.new_tensor([(N - (k+1) + 0.5) / N for k in range(N)])
            yp = torch.mul(coeff, y)
            u = 1 - 2 * (yp / norm1_y).sum()
        s += u
    
    s /= w.shape[0]
    return s


def gini_vectorized(w):
    """
    Compute the Gini score for matrix w.

    This function assumes that is row (of dim K) is a probability distribution, thus, their norms
    are not 0.0
    The last dimension of `w` is first sorted in ascending order
    Reference: https://arxiv.org/pdf/0811.4706.pdf

    Args:
        w: [nl, B, nh, Q, K] attention matrix, Q is the number of queries, K is the dim of attention map
            corresponding to each query
    
    Return:
        s: float - Gini score
    """
    assert w.dim() == 5
    nl, B, nh, Q, K = w.size()
    y, _ = torch.sort(w)  # [nl, B, nh, Q, K]
    norm1_y = torch.sum(y, dim=-1, keepdim=True) # [nl, B, nh, Q, 1]

    coeffs = y.new_tensor([(K - (k+1) + 0.5)/K for k in range(K)]).repeat(nl, B, nh, Q, 1) # [nl, B, nh, Q, K]
    yp = torch.mul(coeffs, y)

    key_scores = 1 - 2 * (yp / norm1_y).sum(dim=-1) # [nl, B, nh, Q,]
    gini = key_scores.view(key_scores.size(0), -1)
    return gini.mean(-1)


def zeros_ratio(w: torch.Tensor, threshold: Optional[float] = None) -> float:
    """
    Compute the zero ratio (i.e., # zero entries / # total entries) of a tensor

    Args:
        w: input tensor
        threshold: should be use for softmax tensors
    """
    s = 0.0
    for row in w:
        x = row
        if threshold is not None:
            x = torch.where(row > threshold, row, torch.tensor(0.0))
        
        r = (x == 0.0).type(torch.uint8).sum() / x.size()
        s += r
    s /= w.shape[0]
    return s


def zeros_ratio_vectorized(w: torch.Tensor, threshold: Optional[float] = None) -> float:
    """
    Compute the zero ratio (i.e., # zero entries / # total entries) of a tensor

    Args:
        w: [nl, B, nh, Q, K] - input tensor
        threshold: should be use for softmax tensors
    """
    K = w.shape[-1]

    if threshold is not None:
        x = torch.where(w > threshold, w, w.new_tensor(0.0))
    else:
        x = w
    
    s = (x == 0.0).type(torch.uint8).sum(dim=-1) / K
    s = s.view(s.size(0), -1)
    return s.mean(-1)


def paibb_vectorized(matcher: Any, outputs: Any, targets: Any):
    matching_indices = matcher(outputs, targets)
    # import pdb; pdb.set_trace()
