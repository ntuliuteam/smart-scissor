from typing import Type, Any, Callable, Union, List, Optional,Dict

import torch
import torch.nn as nn

from configs.config import cfg
from networks.blocks import (
    activation,
    conv2d_1x1,
    conv2d_3x3,
    conv2d_cx,
    gap2d,
    gap2d_cx,
    init_weights,
    linear,
    linear_cx,
    norm2d,
    norm2d_cx,
    pool2d,
    pool2d_cx,
)

__all__ = [
    "AnyNet",
    "anynet_light",
]


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv2d_3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=cfg.MODEL.ACTIVATION_INPLACE)
        self.conv2 = conv2d_3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

        self.stride = stride
        self.inplanes = inplanes
        self.outplanes = planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def complexity(self, cx):
        cx = conv2d_cx(cx, self.inplanes, self.outplanes, 3, stride=self.stride)
        cx = norm2d_cx(cx, self.outplanes)
        cx = conv2d_cx(cx, self.outplanes, self.outplanes, 3)
        cx = norm2d_cx(cx, self.outplanes)

        if self.downsample is not None:
            cx['h'], cx['w'] = cx['h'] * self.stride, cx['w'] * self.stride
            cx = conv2d_cx(cx, self.inplanes, self.outplanes, 1, stride=self.stride)
            cx = norm2d_cx(cx, self.outplanes)

        return cx


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
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        err_message = "The planes must be divisible by groups. Got {} and {}, respectively.".format(planes, groups)
        assert planes % groups == 0, err_message

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv2d_1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv2d_3x3(planes, planes, stride, groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv2d_1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=cfg.MODEL.ACTIVATION_INPLACE)
        self.downsample = downsample

        self.stride = stride
        self.inplanes = inplanes
        self.width = planes
        self.stride = stride
        self.groups = groups
        self.outplanes = planes * self.expansion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def complexity(self, cx):
        cx = conv2d_cx(cx, self.inplanes, self.width, 1)
        cx = norm2d_cx(cx, self.width)
        cx = conv2d_cx(cx, self.width, self.width, 3, stride=self.stride, groups=self.groups)
        cx = norm2d_cx(cx, self.width)
        cx = conv2d_cx(cx, self.width, self.outplanes, 1)
        cx = norm2d_cx(cx, self.outplanes)

        if self.downsample is not None:
            cx['h'], cx['w'] = cx['h'] * self.stride, cx['w'] * self.stride
            cx = conv2d_cx(cx, self.inplanes, self.outplanes, 1, stride=self.stride)
            cx = norm2d_cx(cx, self.outplanes)

        return cx


class AnyNet(nn.Module):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 widths: List[int],
                 strides: List[int],
                 num_classes: int = 1000,
                 inplanes: int = 16,
                 groups: int = 1,
                 ) -> None:
        super(AnyNet, self).__init__()

        self.inplanes = inplanes
        self.groups = groups
        err_message = "Mismatch of the length of layers, widths, and strides. " \
                      "Got {}, {}, and {}. respectively.".format(len(layers), len(widths), len(strides))
        assert len(layers) == len(widths) and len(layers) == len(strides), err_message

        self.conv1 = conv2d_3x3(3, self.inplanes, stride=2)
        self.bn1 = norm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=cfg.MODEL.ACTIVATION_INPLACE)
        blocks = []
        for idx, (num_layers, width, stride) in enumerate(zip(layers, widths, strides)):
            blocks.append(self._make_layer(block, num_layers, width, stride=stride))
        self.blocks = nn.Sequential(*blocks)
        self.avgpool = gap2d(self.inplanes)
        self.fc = linear(self.inplanes, num_classes, bias=True)
        self.apply(init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_layers, width, stride=1):
        layers = []
        downsample = None

        outplanes = int(width * block.expansion)
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                conv2d_1x1(self.inplanes, outplanes, stride=stride),
                norm2d(outplanes),
            )
        layers.append(
            block(self.inplanes, width, stride, downsample, self.groups, norm2d)
        )
        self.inplanes = outplanes
        for _ in range(1, num_layers):
            layers.append(
                block(
                    self.inplanes,
                    width,
                    groups=self.groups,
                    norm_layer=norm2d,
                )
            )

        return nn.Sequential(*layers)

    def complexity(self, cx):
        inplanes = 0
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                cx = conv2d_cx(cx, m.in_channels, m.out_channels, m.kernel_size[0],
                               stride=m.stride[0], groups=m.groups, bias=m.bias)
                inplanes = m.out_channels
            elif isinstance(m, nn.BatchNorm2d):
                cx = norm2d_cx(cx, m.num_features)
            elif isinstance(m, nn.MaxPool2d):
                cx = pool2d_cx(cx, inplanes, m.kernel_size, stride=m.stride)
            elif isinstance(m, nn.Sequential):
                for n in m.children():
                    for k in n.children():
                        cx = k.complexity(cx)
                        inplanes = k.outplanes
            elif isinstance(m, nn.AdaptiveAvgPool2d):
                cx = gap2d_cx(cx, inplanes)
            elif isinstance(m, nn.Linear):
                cx = linear_cx(cx, m.in_features, m.out_features, bias=(m.bias is not None))

        return cx


def _anynet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    widths: List[int],
    strides: List[int],
    weights: str,
    **kwargs: Any,
) -> AnyNet:
    model = AnyNet(block, layers, widths, strides, **kwargs)
    if weights:
        ckpt = torch.load(weights)
        model.load_state_dict(ckpt['model_state'])

    return model


def anynet_light(**kwargs):
    """ Return the anynet_light model.

    :return: AnyNet
    """
    weights = cfg.MODEL.CROPNET_FEATURE_WEIGHTS
    kwargs["num_classes"] = cfg.MODEL.NUM_CLASSES
    return _anynet('anynet_light',
                   Bottleneck,
                   [2, 2, 2, 2],
                   [16, 32, 32, 64],
                   [2, 2, 2, 2],
                   weights,
                   **kwargs)


if __name__ == "__main__":
    cx = {"h": 224, "w": 224, "flops": 0, "params": 0, "acts": 0}
    model = anynet_light()
    print(model)
