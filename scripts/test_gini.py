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
from sparse_detector.utils.metrics import gini

device = torch.device("cuda")


def test_higher_gini_for_sparser_tensor():
    """
    Expect: a sparse tensor will have higher gini score than a dense tensor
    """
    dense_tensor = torch.rand((10, 10), device=device)

    sparse_tensor = torch.tensor(dense_tensor)
    sparse_tensor = torch.where(sparse_tensor > 0.8, sparse_tensor, torch.tensor(0.0, device=device))

    dense_gini = gini(dense_tensor)
    sparse_gini = gini(sparse_tensor)
    print(f"Dense gini: {dense_gini}, sparse gini: {sparse_gini}")


if __name__ == "__main__":
    test_higher_gini_for_sparser_tensor()
