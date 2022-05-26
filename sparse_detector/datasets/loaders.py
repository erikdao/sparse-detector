from typing import Any, Optional, Tuple

import torch.utils.data as data_utils

from sparse_detector.datasets import get_coco_api_from_dataset
from sparse_detector.datasets.coco import build as build_coco
from sparse_detector.utils import misc as misc_utils


def build_dataloaders(
    split: str,
    coco_path: str,
    batch_size: int,
    distributed: bool,
    num_workers: Optional[int] = 24
) -> Any:
    if split not in ['train', 'val']:
        raise ValueError(f"Split {split} is not supported")

    dataset = build_coco(image_set=split, coco_path=coco_path)

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
