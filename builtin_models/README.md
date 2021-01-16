# builtin_models

Run `make generate` in the root `pytorch` (`..`) after updating model descriptions.

# model test
### Image Classification

| Name                        | Image                               | Label                  | Probability |
|:---------------------------:|:-----------------------------------:|:----------------------:|:-----------:|
| Caffe_ResNet_101            | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.913376    |
| DPN_68_v1.0                 | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.999947    |
| DPN_68_v2.0                 | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.999899    |
| DPN_92                      | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.999939    |
| DPN_98                      | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.999991    |
| DPN_107                     | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.999961    |
| DPN_131                     | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.999997    |
| Inception_ResNet_v2.0       | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.918950    |
| Inception_v3.0              | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.998987    |
| NasNet_A_Large              | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.909541    |
| NasNet_A_Mobile             | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.917232    |
| PNasNet_5_Large             | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.863879    |
| PolyNet                     | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.999995    |
| ResNext101_32x4D            | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.999645    |
| ResNext101_64x4D            | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.999981    |
| SE_ResNet_50                | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.572342    |
| SE_ResNet_101               | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.713485    |
| SE_ResNet_152               | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.793791    |
| SE_ResNext_50_32x4D         | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.998532    |
| SE_ResNext_101_32x4D        | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.889444    |
| SENet_154                   | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.904515    |
| TorchVision_AlexNet         | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.702554    |
| TorchVision_DenseNet_121    | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.999918    |
| TorchVision_DenseNet_161    | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 1.000000    |
| TorchVision_DenseNet_169    | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.996164    |
| TorchVision_DenseNet_201    | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.999987    |
| TorchVision_Resnet_18       | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.999424    |
| TorchVision_Resnet_34       | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.998857    |
| TorchVision_Resnet_50       | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.999784    |
| TorchVision_Resnet_101      | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.999767    |
| TorchVision_Resnet_152      | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.999978    |
| TorchVision_SqueezeNet_v1.0 | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.967674    |
| TorchVision_SqueezeNet_v1.1 | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.743228    |
| TorchVision_VGG_11          | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.999609    |
| TorchVision_VGG_11_BN       | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.982912    |
| TorchVision_VGG_13          | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.938878    |
| TorchVision_VGG_13_BN       | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.997749    |
| TorchVision_VGG_16          | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.994223    |
| TorchVision_VGG_16_BN       | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.999577    |
| TorchVision_VGG_19          | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.991907    |
| TorchVision_VGG_19_BN       | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.998358    |
| Xception                    | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 0.930981    |

### Image Object Detection

Note: Only recording first five detection, excluding background.
| Name                        | Image                                   | Label | Xmin  | Xmax  | Ymin  | Ymax  | Probability |
|:---------------------------:|:---------------------------------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|
| MobileNet_SSD_v1.0          | ../predictor/_fixtures/lane_control.jpg | car   | 0.574 | 0.990 | 0.634 | 1.004 | 0.998       |
|                             |                                         | car   | 0.567 | 0.801 | 0.001 | 0.344 | 0.998       |
|                             |                                         | car   | 0.575 | 0.990 | 0.629 | 0.998 | 0.997       |
|                             |                                         | car   | 0.584 | 0.801 | 0.006 | 0.342 | 0.996       |
|                             |                                         | car   | 0.569 | 0.806 | 0.013 | 0.339 | 0.994       |
| MobileNet_SSD_Lite_v2.0     | ../predictor/_fixtures/lane_control.jpg | car   | 0.581 | 0.992 | 0.612 | 0.992 | 0.999       |
|                             |                                         | car   | 0.585 | 0.800 | 0.020 | 0.331 | 0.999       |
|                             |                                         | car   | 0.584 | 0.806 | 0.011 | 0.337 | 0.998       |
|                             |                                         | car   | 0.578 | 0.992 | 0.608 | 0.996 | 0.993       |
|                             |                                         | car   | 0.583 | 0.807 | 0.007 | 0.335 | 0.987       |

### Image Enhancement

| Name                        | Image                               | (R, G, B) at (0, 0) (top-left corner) |
|:---------------------------:|:-----------------------------------:|:-------------------------------------:|
| SRGAN_v1.0                  | ../predictor/_fixtures/penguin.png  | (0xc2, 0xc2, 0xc6)                    |

### Image Semantic Segmentation

| Name                            | Image                                    | label at bottom-right corner |
|:-------------------------------:|:----------------------------------------:|:----------------------------:|
| TorchVision_DeepLabv3_Resnet101 | ../predictor/_fixtures/lane_control.jpg  | 7 (car)                      |
| TorchVision_Fcn_Resnet101       | ../predictor/_fixtures/lane_control.jpg  | 7 (car)                      |