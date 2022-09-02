import os
import sys
package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)

from pathlib import Path
from typing import Any, Optional

import torch.utils.data as data_utils

from sparse_detector.datasets import get_coco_api_from_dataset
from sparse_detector.datasets.coco import CocoDetection, make_coco_transforms
from sparse_detector.utils import misc as misc_utils


def build_unrel(unrel_path):
    unrel_path = Path(unrel_path)
    img_folder = unrel_path / "images"
    ann_file = unrel_path / "instances_unrel.json"

    return CocoDetection(img_folder, ann_file, transforms=make_coco_transforms("val"))
    

def build_dataloaders(
    split: str,
    unrel_path: str,
    batch_size: int,
    distributed: bool,
    num_workers: Optional[int] = 24
) -> Any:
    if split not in ['train', 'val']:
        raise ValueError(f"Split {split} is not supported")

    dataset = build_unrel(unrel_path=unrel_path)

    if split == 'train':
        if distributed:
            sampler = data_utils.DistributedSampler(dataset, shuffle=True)
        else:
            sampler = data_utils.RandomSampler(dataset)
        
        batch_sampler = data_utils.BatchSampler(sampler, batch_size, drop_last=True)
        dataloader = data_utils.DataLoader(
            dataset, batch_sampler=batch_sampler,
            collate_fn=misc_utils.collate_fn, num_workers=num_workers
        )
        return dataloader, sampler
    elif split == 'val':
        if distributed:
            sampler = data_utils.DistributedSampler(dataset, shuffle=False)
        else:
            sampler = data_utils.SequentialSampler(dataset)

        dataloader = data_utils.DataLoader(
            dataset, batch_size, sampler=sampler, drop_last=False,
            collate_fn=misc_utils.collate_fn, num_workers=num_workers
        )

        base_ds = get_coco_api_from_dataset(dataset)
        return dataloader, base_ds
