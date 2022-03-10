# Work Logs

This file contains the log of the works that I have done in this project. I'm trying to log my works as details as possible to help my future self in tracing back to any problems, directions I've tried.

## March 12, 2022
- We continue to work on refactoring the `train_baseline` pipeline. Our major concerns at the moment include 1) how to manage the configurations of different experiments more efficiently; and 2) if there is any better alternative to the architecture of the codebase.
- The goal for today is two-fold:
  - Finish refactoring the code so we could start running the baseline experiment
  - Integrate W&B logging -- **This is a must**

## March 10, 2022
- Refactoring the DETR codebase to a more modular structure is pretty straightforward. All the parts related to segmentation have been removed.
- It seems that the refactoring works so far, i.e., results before and after refactoring are similar. Next, we're going to modularize the configurations.
- TODO: Figure out how to run the current code structure with `torchrun`

## March 9, 2022
- After consideration, we decided that we will not be porting the code to Pytorch Lightning. Instead, we will base on the original DETR repo and gradually add the features/functionalities we need to the repo.
- First off, we'll move it to a more modular structure, and clean up the segmentation parts as we're not doing segmentation in this project.

## March 5, 2022
- At this time of writing, the repository is hosted on Berzelius where the computational resources are abundant. However, we might need to soon move it to the RPL cluster in which only GPUs up to 12GB VRAM are available. Therefore, we might need to experiment with a smaller backbone (such as `resnet18` or `resnet34`) so that the whole DETR would fit into the GPUs.
- First off, we're exploring the [ResNet pretrained model](https://pytorch.org/hub/pytorch_vision_resnet/) provided by `torchvision`.
- We're still considering if we should move the DETR codebase to pytorch lightning.
   - Pros: remove many boilerplates, making debugging easier, handle distributed training on multiple GPUs
   - Cons: potential performance decrease, some special cases (even though we're not aware of any yet) might be difficult to implement in PL.
- Try to train DETR on COCO 2017 on 6 GPUs, hyperparameters are identical to the original DETR repo, except `batch_size = 8, num_workers = 4`. On average, an epoch takes `15` mins, which is twice faster than the original DETR trained on 8 V100s.
- **NOTE:** A potential can of worms is DETR's original implementation of transformer is mostly based on Pytorch's transformer, which is deeply nested with other pytorch's layers and functions. As a result, chaning self-attention mechanism by replacing `softmax` with other alternatives such as `sparsemax` wouldn't be straightforward.
- After checking the CNN-backbone part, it seems to be fairly straightforward.
- Next, we're looking into the COCO dataset loader.