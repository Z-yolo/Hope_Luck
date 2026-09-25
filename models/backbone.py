"""ResNet-18 feature encoders used by the audio/visual RCC model."""

from __future__ import annotations

import torch
from torch import nn


def conv3x3(
    input_channels: int,
    output_channels: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        input_channels,
        output_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(input_channels: int, output_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        input_channels, output_channels, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock supports only groups=1 and base_width=64.")
        if dilation > 1:
            raise NotImplementedError("BasicBlock does not support dilation > 1.")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            identity = self.downsample(inputs)
        return self.relu(output + identity)


class ResNet18Encoder(nn.Module):
    """Parameter-compatible feature-only ResNet-18 for audio or video frames."""

    def __init__(
        self,
        args,
        modality: str,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if modality not in {"audio", "visual"}:
            raise ValueError(f"Unsupported modality: {modality}")
        norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
        dilation_flags = (
            [False, False, False]
            if replace_stride_with_dilation is None
            else replace_stride_with_dilation
        )
        if len(dilation_flags) != 3:
            raise ValueError("replace_stride_with_dilation must contain three values.")

        self.modality = modality
        self.pool = "avgpool"
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        input_channels = 1 if modality == "audio" else 3
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2, dilate=dilation_flags[0])
        self.layer3 = self._make_layer(256, 2, stride=2, dilate=dilation_flags[1])
        self.layer4 = self._make_layer(512, 2, stride=2, dilate=dilation_flags[2])
        self.args = args

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(module.weight, mean=1, std=0.02)
                nn.init.constant_(module.bias, 0)
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def _make_layer(
        self, planes: int, blocks: int, stride: int = 1, dilate: bool = False
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride), norm_layer(planes)
            )
        layers = [
            BasicBlock(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.modality == "visual":
            batch, channels, frames, height, width = inputs.size()
            inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()
            inputs = inputs.view(batch * frames, channels, height, width)
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.maxpool(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        return self.layer4(output)


def resnet18_weight(modality: str, args, **_) -> ResNet18Encoder:
    """Keep the original factory name so checkpoint-facing model code stays stable."""
    return ResNet18Encoder(args=args, modality=modality)

