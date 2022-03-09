# DETR Architecture and Training

## Backbone
### ResNet
- By default, DETR uses `resnet50` as its backbone. The model is loaded from torchvision hub, pretrained on ImageNet, the classification head is removed. The stride is replaced with dilated convolution for variant `DC5`. Another modification is that the normalization layer is changed to a `BatchNorm2d` where the batch statistics and the affine parameters are fixed.
- The backbone is jointly trained with the rest of DETR by default. However, there is an option to freeze the backbone's weights.
- Given an input `x (B, C, H, W)`, the backbone produces a fixed-size feature vector `feat (B, D, FH, FW)` where `D = 2048` for `resnet50` and `D = 512` for `resnet18, resnet34`; `FH = FW = H // 32 = W // 32` (TODO: figure out why this detail)
