"""
Ad-hoc script to test the implementation of the function to compute gini scores
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore', 'UserWarning')

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)

import torch
import numpy as np

from sparse_detector.utils.metrics import gini, gini_avg_mean, gini_alternative

device = torch.device("cuda")


def test_higher_gini_for_sparser_tensor():
    """
    Expect: a sparse tensor will have higher gini score than a dense tensor
    """
    dense_tensor = torch.rand((128, 128), device=device)
    dense_gini = gini(dense_tensor)
    dense_gini_avg = gini_avg_mean(dense_tensor)
    dense_gini_alt = gini_alternative(dense_tensor)
    print(f"Dense gini: {dense_gini:.4f}\t Dense Gini avg: {dense_gini_avg:.4f}\t Dense Gini alternative: {dense_gini_alt:.4f}")

    for t in np.arange(0.1, 1.0, 0.1):
        sparse_tensor = dense_tensor.clone().detach()
        sparse_tensor = torch.where(sparse_tensor > t, sparse_tensor, torch.tensor(0.0, device=device))

        zero_count = (sparse_tensor == 0.0).type(torch.uint8).sum()

        sparse_gini = gini(sparse_tensor)
        sparse_gini_avg = gini_avg_mean(sparse_tensor)
        sparse_gini_alt = gini_alternative(sparse_tensor)
        print(f"Sparse gini (>{t:.2f}): {sparse_gini:.4f}\t Sparse Gini avg: {sparse_gini_avg:.4f}\t Sparse Gini alternative {sparse_gini_alt:.4f}\t Zeros count: {zero_count}")


if __name__ == "__main__":
    test_higher_gini_for_sparser_tensor()
