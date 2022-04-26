from typing import Any, Optional, Tuple

import torch.utils.data as data_utils

from sparse_detector.datasets import build_dataset, get_coco_api_from_dataset
from sparse_detector.utils import misc as misc_utils


def build_dataloaders(
    split: str,
    coco_path: str,
    batch_size: int,
    distributed: bool,
    num_workers: Optional[int] = 12
) -> Any:
    if split not in ['train', 'val']:
        raise ValueError(f"Invalid data split {split}")

    dataset = build_dataset(image_set=split, coco_path=coco_path)

    if distributed:
        shuffle = True if split == 'train' else False
        sampler = data_utils.DistributedSampler(dataset, shuffle=shuffle)
    else:
        if split == 'train':
            sampler = data_utils.RandomSampler(dataset)
        else:
            sampler = data_utils.SequentialSampler(dataset)

    if split == 'train':
        batch_sampler = data_utils.BatchSampler(sampler, batch_size, drop_last=True)

        data_loader = data_utils.DataLoader(
            dataset, batch_sampler=batch_sampler,
            collate_fn=misc_utils.collate_fn, num_workers=num_workers
        )
        return data_loader, batch_sampler
    else:
        data_loader = data_utils.DataLoader(
            dataset, batch_size, sampler=sampler, drop_last=False,
            collate_fn=misc_utils.collate_fn, num_workers=num_workers
        )

        base_ds = get_coco_api_from_dataset(dataset)

        return data_loader, base_ds
