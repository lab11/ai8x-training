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
import torch.nn as nn
from torch import Tensor

import ai8x

def _make_divisible(v: float, divisor: int, min_value = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        bias=False,
        **kwargs
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ai8x.FusedConv2dBNReLU(inp, hidden_dim, 1, bias=bias, **kwargs))
            #layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))

            # dw
            layers.append(ai8x.FusedConv2dBNReLU(hidden_dim, hidden_dim, 3, padding=1, bias=bias, **kwargs))
            #ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
        if stride == 2:
            layers.append(
                # pw-linear
                # instead of using stride in previous layer (unsupported), do a max pool before this layer
                ai8x.FusedMaxPoolConv2dBN(hidden_dim, oup, 1, pool_stride=stride, bias=bias),
            )
        else:
            layers.append(
                # pw-linear
                # instead of using stride in previous layer (unsupported), do a max pool before this layer
                ai8x.Conv2d(hidden_dim, oup, 1, pool_stride=stride, bias=bias, batchnorm="NoAffine"),
            )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class AI85MobileNetsV2(nn.Module):
    """
    Residual SimpleNet v1 Model
    """
    def __init__(
            self,
            num_classes=2,
            num_channels=3,
            dimensions=(224, 224),
            width_mult=.35,
            bias=False,
            inverted_residual_setting = None,
            round_nearest = 8,
            **kwargs
    ):
        super().__init__()

        block = InvertedResidual

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                    # t, c, n, s
                    [1, 16, 1, 1],
                    [6, 24, 2, 2],
                    [6, 32, 3, 2],
                    [6, 64, 4, 2],
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(min(last_channel * max(1.0, width_mult), round_nearest), 1024)
        features: List[nn.Module] = [ai8x.FusedConv2dBNReLU(num_channels, input_channel, 3, padding=1, bias=bias, **kwargs),
                                     ai8x.MaxPool2d(num_channels, 2, **kwargs)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, t, bias, **kwargs))
                input_channel = output_channel
        # building last several layers
        features.append(ai8x.FusedConv2dBNReLU(input_channel, self.last_channel, kernel_size=1, bias=bias, **kwargs))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            ai8x.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        #for m in self.modules():
        #    if isinstance(m, ai8x.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #        if m.bias is not None:
        #            nn.init.zeros_(m.bias)
        #    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #        nn.init.ones_(m.weight)
        #        nn.init.zeros_(m.bias)
        #    elif isinstance(m, ai8x.Linear):
        #        nn.init.normal_(m.weight, 0, 0.01)
        #        nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)

        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def ai85mobilenetsv2(pretrained=False, **kwargs):
    """
    Constructs a Residual SimpleNet v1 model.
    """
    assert not pretrained
    return AI85MobileNetsV2(**kwargs)


models = [
    {
        'name': 'ai85mobilenetsv2',
        'min_input': 1,
        'dim': 2,
    },
]
