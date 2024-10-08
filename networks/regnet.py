
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import nn, Tensor

from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation
from torchvision.models._utils import _make_divisible

import configs.config as config
from configs.config import cfg
from networks.blocks import (
    conv2d_cx,
    norm2d_cx,
    linear_cx,
    gap2d_cx
)


__all__ = [
    "RegNet",
    "regnet_x_400mf",
    "regnet_x_800mf",
    "regnet_x_1_6gf",
    "regnet_x_1_6gf_d09",
    "regnet_x_1_6gf_d08",
    "regnet_x_1_6gf_d07",
    "regnet_x_1_6gf_d06",
    "regnet_x_1_6gf_d05",
    "regnet_x_3_2gf",
]


model_urls = {
    "regnet_x_400mf": "https://download.pytorch.org/models/regnet_x_400mf-adf1edd5.pth",
    "regnet_x_800mf": "https://download.pytorch.org/models/regnet_x_800mf-ad17e45c.pth",
    "regnet_x_1_6gf": "https://download.pytorch.org/models/regnet_x_1_6gf-e3633e7f.pth",
    "regnet_x_3_2gf": "https://download.pytorch.org/models/regnet_x_3_2gf-f342aeae.pth",
}


class SimpleStemIN(ConvNormActivation):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__(
            width_in, width_out, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=activation_layer
        )

    def complexity(self, cx):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                cx = conv2d_cx(cx, m.in_channels, m.out_channels, m.kernel_size[0], stride=m.stride[0],
                               groups=m.groups, bias=m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                cx = norm2d_cx(cx, m.num_features)

        return cx


class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
    ) -> None:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        layers["a"] = ConvNormActivation(
            width_in, w_b, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=activation_layer
        )
        layers["b"] = ConvNormActivation(
            w_b, w_b, kernel_size=3, stride=stride, groups=g, norm_layer=norm_layer, activation_layer=activation_layer
        )

        if se_ratio:
            # The SE reduction ratio is defined with respect to the
            # beginning of the block
            width_se_out = int(round(se_ratio * width_in))
            layers["se"] = SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
                activation=activation_layer,
            )

        layers["c"] = ConvNormActivation(
            w_b, width_out, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=None
        )
        super().__init__(layers)

    def complexity(self, cx):
        for m in self.children():
            if isinstance(m, ConvNormActivation):
                for n in m.children():
                    if isinstance(n, nn.Conv2d):
                        cx = conv2d_cx(cx, n.in_channels, n.out_channels, n.kernel_size[0],
                                       stride=n.stride[0], groups=n.groups, bias=n.bias)
                    elif isinstance(n, nn.BatchNorm2d):
                        cx = norm2d_cx(cx, n.num_features)

        return cx


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()

        # Use skip connection with projection if shape changes
        self.proj = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = ConvNormActivation(
                width_in, width_out, kernel_size=1, stride=stride, norm_layer=norm_layer, activation_layer=None
            )
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )
        self.activation = activation_layer(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)

    def complexity(self, cx):
        for m in self.children():
            if isinstance(m, ConvNormActivation):
                for n in m.children():
                    if isinstance(n, nn.Conv2d):
                        cx = conv2d_cx(cx, n.in_channels, n.out_channels, n.kernel_size[0],
                                       stride=n.stride[0], groups=n.groups, bias=n.bias)
                    elif isinstance(n, nn.BatchNorm2d):
                        cx = norm2d_cx(cx, n.num_features)
                # Identity skip connection does not change the size of feature maps.
                cx["h"], cx["w"] = cx["h"] * 2, cx["w"] * 2
            elif isinstance(m, BottleneckTransform):
                cx = m.complexity(cx)

        return cx


class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        block_constructor: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float] = None,
        stage_index: int = 0,
    ) -> None:
        super().__init__()

        self.width_in = width_in
        self.width_out = width_out
        for i in range(depth):
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                norm_layer,
                activation_layer,
                group_width,
                bottleneck_multiplier,
                se_ratio,
            )

            self.add_module(f"block{stage_index}-{i}", block)

    def complexity(self, cx):
        for m in self.children():
            if isinstance(m, ResBottleneckBlock):
                cx = m.complexity(cx)

        return cx


class BlockParams:
    def __init__(
        self,
        depths: List[int],
        widths: List[int],
        group_widths: List[int],
        bottleneck_multipliers: List[float],
        strides: List[int],
        se_ratio: Optional[float] = None,
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(
        cls,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        width_mult: float = 1.0,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> "BlockParams":
        """
        Programatically compute all the per-block settings,
        given the RegNet parameters.

        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space

        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.

        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """

        QUANT = 8
        STRIDE = 2

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")
        # Compute the block widths. Each stage has one unique block width
        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), QUANT)) * QUANT).int().tolist()
        num_stages = len(set(block_widths))

        # Convert to per stage parameters
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [_make_divisible(w * width_mult, cfg.SCALING.ROUND) for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = torch.diff(torch.tensor([d for d, t in enumerate(splits) if t])).int().tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [_make_divisible(group_width * width_mult, 2)] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        print(stage_depths, stage_widths, group_widths, bottleneck_multipliers, strides, se_ratio)

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self):
        return zip(self.widths, self.strides, self.depths, self.group_widths, self.bottleneck_multipliers)

    @staticmethod
    def _adjust_widths_groups_compatibilty(
        stage_widths: List[int], bottleneck_ratios: List[float], group_widths: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Adjusts the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """
        # Compute all widths for the current settings
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        # Compute the adjusted widths so that stage and group widths fit
        ws_bot = [_make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


class RegNet(nn.Module):
    def __init__(
        self,
        block_params: BlockParams,
        num_classes: int = 1000,
        stem_width: int = 32,
        stem_type: Optional[Callable[..., nn.Module]] = None,
        block_type: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        # Ad hoc stem
        self.stem = stem_type(
            3,  # width_in
            stem_width,
            norm_layer,
            activation,
        )

        current_width = stem_width

        blocks = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):
            blocks.append(
                (
                    f"block{i+1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        block_type,
                        norm_layer,
                        activation,
                        group_width,
                        bottleneck_multiplier,
                        block_params.se_ratio,
                        stage_index=i + 1,
                    ),
                )
            )

            current_width = width_out

        self.trunk_output = nn.Sequential(OrderedDict(blocks))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=current_width, out_features=num_classes)

        # Performs ResNet-style weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.trunk_output(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x

    def complexity(self, cx):
        inplanes = 0
        for m in self.children():
            if isinstance(m, SimpleStemIN):
                cx = m.complexity(cx)
            elif isinstance(m, nn.Sequential):
                for n in m.children():
                    if isinstance(n, AnyStage):
                        cx = n.complexity(cx)
                        inplanes = n.width_out
            elif isinstance(m, nn.AdaptiveAvgPool2d):
                cx = gap2d_cx(cx, inplanes)
            elif isinstance(m, nn.Linear):
                cx = linear_cx(cx, m.in_features, m.out_features, bias=(m.bias is not None))

        return cx

def _regnet(arch: str, block_params: BlockParams, pretrained: bool, progress: bool, **kwargs: Any) -> RegNet:
    norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
    model = RegNet(block_params, norm_layer=norm_layer, **kwargs)
    if pretrained:
        if arch not in model_urls:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def regnet_x_400mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_400MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16, **kwargs)
    return _regnet("regnet_x_400mf", params, pretrained, progress, **kwargs)


def regnet_x_800mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_800MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=16, w_0=56, w_a=35.73, w_m=2.28, group_width=16, **kwargs)
    return _regnet("regnet_x_800mf", params, pretrained, progress, **kwargs)


def regnet_x_1_6gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_1.6GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24,
                                          width_mult=cfg.SCALING.WIDTH_MULT, **kwargs)
    kwargs["num_classes"] = cfg.MODEL.NUM_CLASSES
    return _regnet("regnet_x_1_6gf", params, pretrained, progress, **kwargs)


def regnet_x_1_6gf_d09(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_1.6GF with a depth scaling factor of 0.9 architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24,
                                          width_mult=cfg.SCALING.WIDTH_MULT, **kwargs)
    params.depths = [2, 4, 8, 2]
    kwargs["num_classes"] = cfg.MODEL.NUM_CLASSES
    return _regnet("regnet_x_1_6gf", params, pretrained, progress, **kwargs)


def regnet_x_1_6gf_d08(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_1.6GF with a depth scaling factor of 0.8 architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24,
                                          width_mult=cfg.SCALING.WIDTH_MULT, **kwargs)
    params.depths = [2, 3, 7, 2]
    kwargs["num_classes"] = cfg.MODEL.NUM_CLASSES
    return _regnet("regnet_x_1_6gf", params, pretrained, progress, **kwargs)


def regnet_x_1_6gf_d07(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_1.6GF with a depth scaling factor of 0.7 architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24,
                                          width_mult=cfg.SCALING.WIDTH_MULT, **kwargs)
    params.depths = [2, 3, 6, 2]
    kwargs["num_classes"] = cfg.MODEL.NUM_CLASSES
    return _regnet("regnet_x_1_6gf", params, pretrained, progress, **kwargs)


def regnet_x_1_6gf_d06(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_1.6GF with a depth scaling factor of 0.6 architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24,
                                          width_mult=cfg.SCALING.WIDTH_MULT, **kwargs)
    params.depths = [2, 2, 5, 2]
    kwargs["num_classes"] = cfg.MODEL.NUM_CLASSES
    return _regnet("regnet_x_1_6gf", params, pretrained, progress, **kwargs)


def regnet_x_1_6gf_d05(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_1.6GF with a depth scaling factor of 0.5 architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24,
                                          width_mult=cfg.SCALING.WIDTH_MULT, **kwargs)
    params.depths = [1, 2, 5, 1]
    kwargs["num_classes"] = cfg.MODEL.NUM_CLASSES
    return _regnet("regnet_x_1_6gf", params, pretrained, progress, **kwargs)


def regnet_x_3_2gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_3.2GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=25, w_0=88, w_a=26.31, w_m=2.25, group_width=48, **kwargs)
    return _regnet("regnet_x_3_2gf", params, pretrained, progress, **kwargs)

if __name__ == "__main__":
    cx = {"h": 224, "w": 224, "flops": 0, "params": 0, "acts": 0}
    model = regnet_x_400mf()
    cx = model.complexity(cx)
    print(cx)