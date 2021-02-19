###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
SimpleNet_v1 network with added residual layers for AI85.
Simplified version of the network proposed in [1].

[1] HasanPour, Seyyed Hossein, et al. "Lets keep it simple, using simple architectures to
    outperform deeper and more complex architectures." arXiv preprint arXiv:1608.06037 (2016).
"""
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

import ai8x

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        **kwargs,
    ) -> None:
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ai8x.FusedConv2dBNReLU(inplanes, planes, 3, stride=stride, padding=dilation, bias=True, **kwargs)
        self.conv2 = ai8x.FusedConv2dBNReLU(planes, planes, 3, stride=stride, padding=dilation, bias=True, **kwargs)
        self.downsample = downsample

        self.resid1 = ai8x.Add()

        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.resid1(out, identity)
        #out += identity
        #out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs,
    ) -> None:
        super(Bottleneck, self).__init__()
        #if norm_layer is None:
        #    norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ai8x.Conv2d(inplanes, width, 1, stride=stride, bias=True, batchnorm="NoAffine", **kwargs)
        #self.conv1 = conv1x1(inplanes, width)
        #self.bn1 = norm_layer(width)

        self.conv2 = ai8x.Conv2d(width, width, 3, stride=stride, padding=dilation, bias=True, batchnorm="NoAffine", **kwargs)
        #self.conv2 = conv3x3(width, width, stride, groups, dilation)
        #self.bn2 = norm_layer(width)

        self.conv3 = ai8x.FusedConv2dBNReLU(width, planes * self.expansion, 1, stride=stride, bias=True, **kwargs)
        #self.conv3 = conv1x1(width, planes * self.expansion)
        #self.bn3 = norm_layer(planes * self.expansion)
        #self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        #out = self.bn1(out)
        #out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        #out = self.relu(out)

        out = self.conv3(out)
        #out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        #out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_channels: int = 3,
        num_classes: int = 100,
        dimensions=(224,224),
        bias=False,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        **kwargs,
        #norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        #if norm_layer is None:
        #    norm_layer = nn.BatchNorm2d
        #self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = ai8x.FusedConv2dBNReLU(num_channels, self.inplanes, 3, stride=2, padding=2, bias=True, **kwargs)
        self.maxpool = ai8x.MaxPool2d(3, stride=2, padding=1, bias=False, **kwargs)
        #self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        #self.bn1 = norm_layer(self.inplanes)
        #self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool1 = ai8x.AvgPool2d(8, bias=False, **kwargs)
        self.avgpool2 = ai8x.AvgPool2d(7, bias=False, **kwargs)
        self.fc = ai8x.Linear(512 * block.expansion, num_classes, bias=False, **kwargs)

        #for m in self.modules():
        #    if isinstance(m, ai8x.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #    nn.init.constant_(m.weight, 1)
            #    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        #if zero_init_residual:
        #    for m in self.modules():
        #        if isinstance(m, Bottleneck):
        #            nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #        elif isinstance(m, BasicBlock):
        #            nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, **kwargs) -> nn.Sequential:
        #norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ai8x.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=True, batchnorm='NoAffine', **kwargs)
                #conv1x1(self.inplanes, planes * block.expansion, stride),
                #norm_layer(planes * block.expansion),

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool1(x)
        x = self.avgpool2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def ai85resnet18(pretrained=False, **kwargs):
    """
    Constructs a resnet18 model.
    """
    assert not pretrained
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


models = [
    {
        'name': 'ai85resnet18',
        'min_input': 1,
        'dim': 2,
    },
]
