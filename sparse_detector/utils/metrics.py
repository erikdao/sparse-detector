"""
Useful metrics for measuring sparsity
"""
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


def gini_avg_mean(w: torch.Tensor) -> torch.Tensor:
    r"""The Gini coefficient computed using relative mean absolute difference
    https://en.wikipedia.org/wiki/Gini_coefficient
    """
    s = 0
    for row in w:
        t = row.repeat(row.size(0), 1)
        u = (t - t.T).abs().sum() / (2 * row.size(-1)**2 + torch.finfo().eps)
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
        u = 1 - 2.0 / (n - 1) * (n - yp.sum() / (y.sum() + torch.finfo().eps))
        s += u
    
    s /= w.shape[0]
    return s
