![Sparsity in DETR](./docs/img/detection-pipeline.png)
# Sparsity, Where and How? Incorporating Sparse Attentions into Transformer for Object Detection in Images

Master Thesis Project by Cuong Duc Dao -- Master's Programme in Machine Learning, KTH, Sweden.


## Work Logs

This section contains the log of the works that I have done in this project. I'm trying to log my works as details as possible to help my future self in tracing back to any problems, directions I've tried.

**TODO**:
- Refactor the training and eval script to conform with the change in building model from config files

### April 27, 2022
- We're taking a look at the `entmax` with learnable alpha, and the `tvmax` implementation to see what should be done first, in terms of, what could get done easier.

### April 26, 2022
- We continue to work on the Gini score.
- However, as we create different files, scripts in our project, we feel the growing need to being able to load default model configs from files, and create a model instance from those configs. Thus, we spent some times on this technical debt first.
- Hooray! We've got some first Gini scores of the sparsemax model
```
|-------------|----------------|----------------|----------------|----------------|----------------|----------------|
| model       | layer 0        | layer 1        | layer 2        | layer 3        | layer 4        | layer 5        |
|-------------|----------------|----------------|----------------|----------------|----------------|----------------|
| sparsemax   |0.6796 - 0.2199 |0.5765 - 0.2219 |0.4841 - 0.1883 |0.5874 - 0.1888 |0.4196 - 0.1720 |0.6084 - 0.2047 |
|-------------|----------------|----------------|----------------|----------------|----------------|----------------|
| softmax     |0.7407 - 0.0575 |0.7267 - 0.0640 |0.7572 - 0.0626 |0.8083 - 0.0584 |0.7616 - 0.1087 |0.6702 - 0.0794 |
|-------------|----------------|----------------|----------------|----------------|----------------|----------------|
```

### April 24, 2022
- After too much pause, we've restarted the work on this thesis project.
- We'll try a new activation function `entmax15`, just to see if a *softer* level of sparsity would result in something different.
- Next,we'll compute the gini index for attention maps in each layers of the model. For a given input image, for each layer, the gini is average across all the queries. We then can report the gini scores for all layers of a model on the validation set.

### April 15, 2022
- Today, we've added more visualization of intermediate layers from the decoder.

### April 12, 2022
- Today, we're going to inspect some more samples with the `sparsemax` model to see if it always attends to the corner of the input images.Through two examples we've looked at, it seems that when there is only a prominent object in the image, `sparsemax` model doesn't really spread its attention to corners. However, there are attentions outside of the bounding boxes of the object.

### April 11, 2022
- We now add groundtruth boxes to the visualizations of attention maps and detection results.

### April 8, 2022
- After several days not working on the thesis, we're now restarting the work. First off, some changes need to be made to the figures.
- Then, we'll add more visualizations: drawing bounding boxes, showing groundtruth boxes on both the images and the attention maps.

### April 4, 2022
- Finally, after a couple of days ``resting'' in Vietnam, I've returned to my thesis work.
- The problem we're dealing with, at the moment, is the mismatch in tenson dimension between the attention matrix from transformer blocks vs the ones expected by `TVMax`.
- The original TVMax repo is not very well-documented and it's hard to even run the code. We'll need to dig deeper and figure out the proper dimension. Probably need to read the paper more carefully as well.

### March 30, 2022
- I'm at Arlanda waiting for my flight to Frankfurt, Singapore then Hanoi
- We're working on the implementation of TVMax. There is an [existing implementation](https://github.com/deep-spin/TVmax) that we'll try first.
- Seem that the default implementation of `TV2DFunction` cannot be directly plugged into our pipeline due to the mismatch of input shapes. Need to look into this problem.
- The default implementation of `TVMax` doesn't seem to operate on batched tensor. So a hack we're currently doing is to iteratively apply the `tvmax` function over all slices of the tensor. However, it makes the training hang after calling the function for 5, 6 times. The GPU consumption is just around `8GB/40GB`. We're investigating this problem.

### March 28, 2022
- The baseline training has stopped due to the excessive time on Berzelius. We'll restart it.
- The sparsemax version training follows the trends of the baseline, with lower performance. This is expected. We'll need to think about which kind of parameter tuning we can/should do. Also, we'll look more closely into the attention maps.

### March 26, 2022
- We're on a cruise trip to Helsinki 😍🛳.
- The run to verify if the replacement of `nn.MultiheadAttention` by the `SparseMultiheadAttention` works or not seems to confirm that it works. The losses and other metrics in both cases have close values and similar trend.
- Next, we're going to conduct the first experiment with `sparsemax` as the activation function in the multi-head cross attention. To this end, we'll use the same settings as the baseline.
 
### March 25, 2022
- We continue to test the custom `MultiheadAttention`.
- We've add a `decoder_mha` parameter to the Transformer class allowing to specify which type of attention layer the model is going to use. By default, it's the `nn.MultiheadAttention`. We've also completed implementing the `SparseMultiheadAttention` which supports sparse activation function `sparsemax`. The implementation is similar (and *simplified*) to Pytorch's implementation.
- A quick test with the `visualize_attentions` script, the forward pass of the whole model with the new custom MHA layer is successful.
- We'll run the `baseline_detr` training for 25 epochs to test if the new implementation (by default with softmax) is working properly.

### March 24, 2022
- After a discussion with our supervisor, we believe the best way to incorporate `sparsemax` into the current codebase is to create a custom MHA module that use sparse max. This custom MHA should then be used in the place of the current `self.multihead_attn` of the `TransformerDecoderLayer`.
- After we have had the first version of a custom MHA, the challenge is to unittest it to make sure that everything at least runs with properly-shaped inputs. However, we aren't not sure of the shape of queries, keys, values, etc. Thus, we need to do a surgery of the model during the forward pass.

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