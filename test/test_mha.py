"""
Test suite for the custom MultiheadAttention
"""
import os
import sys

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)

import click
import torch

from sparse_detector.models.attention import MultiheadAttention, scaled_dot_product_attention

@click.command()
def main():
    torch.manual_seed(42)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    D = 2  # hidden dim
    seq_length = 3

    q = torch.randn((seq_length, D), device=device)
    k = torch.randn((seq_length, D), device=device)
    v = torch.randn((seq_length, D), device=device)

    attn, attn_weights = scaled_dot_product_attention(q, k, v, activation='softmax')
    print(attn)
    print(attn_weights)


if __name__ == "__main__":
    main()
