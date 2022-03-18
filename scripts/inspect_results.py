"""
Ad-hoc script to inspect the results
"""
import os
import sys
from pathlib import Path

import click
import torch

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)

def main():
    eval_dir = Path("./checkpoints/baseline_detr/eval")
    assert eval_dir.exists()

    paths = {"290": eval_dir / "290.pth", "latest": eval_dir / "latest.pth"}

    for name, fpath in paths.items():
        click.echo(fpath)
        data = torch.load(fpath, map_location="cpu")
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
