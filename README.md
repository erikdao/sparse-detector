![Sparsity in DETR](./docs/img/detection-pipeline.png)
# Sparsity, Where and How? Incorporating Sparse Attentions into Transformer for Object Detection in Images

Master Thesis Project by Cuong Duc Dao -- Master's Programme in Machine Learning, KTH, Sweden.


## Work Logs

This section contains the log of the works that I have done in this project. I'm trying to log my works as details as possible to help my future self in tracing back to any problems, directions I've tried.

### March 15, 2022
- The baseline training was crashed after 17 epochs. The cause was that we switched the branch of the codebase leading to wrong data path. It has been resumed successfully.
- We ran the original DETR training for 25 epochs, and our **refactored** codebase for 25 epochs. Both for the Baseline experiment. We then plot the `class_error, loss_bbox_unscaled, mAP` for both cases. Our results are very similar to the original results. Therefore, we can be relieved that the refactoring did not change the expected behaviors of DETR.
![Our code](./docs/img/detr_our_25epochs.png)
Our implementation
![Original code](./docs/img/detr_original_25epochs.png)
DETR origianl implementation
- Some other things to do: include mAP plots on W&B, explore evaluation results, create visualizations.

### March 14, 2022
- It turned out that refactoring took more time than expected. The job now is to integrate W&B logging. The quick way to do so is to integrate logging right into the current `MetricLogger` class, which seems to be messy.
- Logging to W&B is working, but we need to handle the **global_step** problem. Currently, its value is not correct.
- After digging for about 3 hours, W&B logging is working as expected. There is still one problem, though. There are two many metrics logged, most of them are explaination of some major metrics. For more readability of visualizations and loggings, it is good to display on WandB metrics grouped as follow:
  - `train-main-metrics`: major metrics including `loss, loss_ce, loss_bbox, loss_giou, class_error, cardinality_error`. All should be scaled.
  - `train-metrics`: all variants of those in the main metrics section.
  - `train-extra-metrics`: metrics including `memory, iter_time, data_time`
  - `train-epoch-main-metrics`: major metrics as in `train-main-metrics` but for epoch
  - `val-epoch-metrics`: metrics for validations
- Keeping a global variable named `global_step` is much easier for experiment tracking

### March 12, 2022
- We continue to work on refactoring the `train_baseline` pipeline. Our major concerns at the moment include 1) how to manage the configurations of different experiments more efficiently; and 2) if there is any better alternative to the architecture of the codebase.
- The goal for today is two-fold:
  - Finish refactoring the code so we could start running the baseline experiment
  - Integrate W&B logging -- **This is a must**

### March 10, 2022
- Refactoring the DETR codebase to a more modular structure is pretty straightforward. All the parts related to segmentation have been removed.
- It seems that the refactoring works so far, i.e., results before and after refactoring are similar. Next, we're going to modularize the configurations.
- TODO: Figure out how to run the current code structure with `torchrun`

### March 9, 2022
- After consideration, we decided that we will not be porting the code to Pytorch Lightning. Instead, we will base on the original DETR repo and gradually add the features/functionalities we need to the repo.
- First off, we'll move it to a more modular structure, and clean up the segmentation parts as we're not doing segmentation in this project.

### March 5, 2022
- At this time of writing, the repository is hosted on Berzelius where the computational resources are abundant. However, we might need to soon move it to the RPL cluster in which only GPUs up to 12GB VRAM are available. Therefore, we might need to experiment with a smaller backbone (such as `resnet18` or `resnet34`) so that the whole DETR would fit into the GPUs.
- First off, we're exploring the [ResNet pretrained model](https://pytorch.org/hub/pytorch_vision_resnet/) provided by `torchvision`.
- We're still considering if we should move the DETR codebase to pytorch lightning.
   - Pros: remove many boilerplates, making debugging easier, handle distributed training on multiple GPUs
   - Cons: potential performance decrease, some special cases (even though we're not aware of any yet) might be difficult to implement in PL.
- Try to train DETR on COCO 2017 on 6 GPUs, hyperparameters are identical to the original DETR repo, except `batch_size = 8, num_workers = 4`. On average, an epoch takes `15` mins, which is twice faster than the original DETR trained on 8 V100s.
- **NOTE:** A potential can of worms is DETR's original implementation of transformer is mostly based on Pytorch's transformer, which is deeply nested with other pytorch's layers and functions. As a result, chaning self-attention mechanism by replacing `softmax` with other alternatives such as `sparsemax` wouldn't be straightforward.
- After checking the CNN-backbone part, it seems to be fairly straightforward.
- Next, we're looking into the COCO dataset loader.