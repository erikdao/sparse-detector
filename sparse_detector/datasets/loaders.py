from typing import Any, Optional, Tuple

import torch.utils.data as data_utils

from sparse_detector.datasets import build_dataset, get_coco_api_from_dataset
from sparse_detector.util import misc as misc_utils


def build_dataloaders(
    dataset_file: str,
    coco_path: str,
    batch_size: int,
    distributed: bool,
    num_workers: Optional[int] = 12
) -> Tuple[Any, Any, Any, Any]:
    dataset_train = build_dataset(image_set='train', dataset_file=dataset_file, coco_path=coco_path)
    dataset_val = build_dataset(image_set='val', dataset_file=dataset_file, coco_path=coco_path)

    if distributed:
        sampler_train = data_utils.DistributedSampler(dataset_train)
        sampler_val = data_utils.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = data_utils.RandomSampler(dataset_train)
        sampler_val = data_utils.SequentialSampler(dataset_val)

    batch_sampler_train = data_utils.BatchSampler(sampler_train, batch_size, drop_last=True)

    data_loader_train = data_utils.DataLoader(
        dataset_train, batch_sampler=batch_sampler_train,
        collate_fn=misc_utils.collate_fn, num_workers=num_workers
    )
    data_loader_val = data_utils.DataLoader(
        dataset_val, batch_size, sampler=sampler_val, drop_last=False,
        collate_fn=misc_utils.collate_fn, num_workers=num_workers
    )

    base_ds = get_coco_api_from_dataset(dataset_val)

    return data_loader_train, data_loader_val, base_ds, sampler_train
