# DETR Architecture and Training

## Backbone
### ResNet
- By default, DETR uses `resnet50` as its backbone. The model is loaded from torchvision hub, pretrained on ImageNet, the classification head is removed. The stride is replaced with dilated convolution for variant `DC5`. Another modification is that the normalization layer is changed to a `BatchNorm2d` where the batch statistics and the affine parameters are fixed.
- The backbone is jointly trained with the rest of DETR by default. However, there is an option to freeze the backbone's weights.
- Given an input `x (B, C, H, W)`, the backbone produces a fixed-size feature vector `feat (B, D, FH, FW)` where `D = 2048` for `resnet50` and `D = 512` for `resnet18, resnet34`; `FH = FW = H // 32 = W // 32` (TODO: figure out why this detail)

- Given an input tensor of shape `[1, 3, 800, 1182]`, the ResNet50 returns the following features at all intermediate layers:
  * `0`: `[1, 256, 200, 296]`
  * `1`: `[1, 512, 100, 148]`
  * `2`: `[1, 1024, 50, 74]`
  * `3`: `[1, 2048, 25, 37]`
- It is interesting to see that the total number of elements of those feature vectors doesn't change. However, the shape of each intermediate feature changes. This can be thought of the feature pyramid at different scales.


### Joiner
- Each of the above intermediate features are fed into the PositionalEmbedding layer to create the positional embedding
- Then the Joiner pack the backbone features, and positional embeddings together

## DETR
- It's **worth** to note that DETR only takes the last feature and positional encoding from the backbone and feeds them into the transformer:
  * `src`: `[1, 2048, 25, 37]`
  * `mask`: `[1, 25, 37]` which contains all zeros
  * `pos`: `[1, 256, 25, 37]`
