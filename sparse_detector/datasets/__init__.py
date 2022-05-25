import torch.utils.data
import torchvision

from sparse_detector.datasets.coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, coco_path):
    return build_coco(image_set, coco_path)
