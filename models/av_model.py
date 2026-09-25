"""Shared two-stream model used for CREMA-D and AVSBench checkpoints."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import resnet18_weight


class ConcatMLPFusion(nn.Module):
    """Nonlinear fusion with masked audio-only and visual-only readouts."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, audio: torch.Tensor, visual: torch.Tensor):
        output = self.net(torch.cat([audio, visual], dim=1))
        audio_output = self.net(
            torch.cat([audio, torch.zeros_like(visual)], dim=1)
        )
        visual_output = self.net(
            torch.cat([torch.zeros_like(audio), visual], dim=1)
        )
        return audio_output, visual_output, output


class RCCGuardModel(nn.Module):
    """ResNet-18 audio/visual encoders and the released RCC fusion head.

    Attribute names and layer indices intentionally match the training model so
    historical DataParallel checkpoints can be loaded with ``strict=True``.
    """

    def __init__(self, args) -> None:
        super().__init__()
        if str(args.fusion_method) != "concat_mlp":
            raise ValueError("The released protocol supports only concat_mlp fusion.")
        num_classes = int(args.num_classes)
        if num_classes not in {6, 23}:
            raise ValueError(f"Unsupported class count: {num_classes}")
        self.args = args
        self.modality = "full"
        self.fusion_module = ConcatMLPFusion(
            input_dim=1024,
            hidden_dim=int(args.fusion_hidden_dim),
            output_dim=num_classes,
            dropout=float(args.fusion_dropout),
        )
        self.audio_net = resnet18_weight(modality="audio", args=args)
        self.visual_net = resnet18_weight(modality="visual", args=args)

    def encode(self, audio: torch.Tensor, visual: torch.Tensor):
        audio_feature = self.audio_net(audio)
        visual_feature = self.visual_net(visual)
        _, channels, height, width = visual_feature.size()
        batch_size = audio.size(0)
        visual_feature = visual_feature.view(
            batch_size, -1, channels, height, width
        ).permute(0, 2, 1, 3, 4)
        audio_feature = F.adaptive_avg_pool2d(audio_feature, 1).flatten(1)
        visual_feature = F.adaptive_avg_pool3d(visual_feature, 1).flatten(1)
        return audio_feature, visual_feature

    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        return_features: bool = False,
    ):
        audio_feature, visual_feature = self.encode(audio, visual)
        audio_output, visual_output, output = self.fusion_module(
            audio_feature, visual_feature
        )
        if return_features:
            features = {
                "feat_a": audio_feature,
                "feat_v": visual_feature,
                "feat_f": torch.cat([audio_feature, visual_feature], dim=1),
            }
            return output, audio_output, visual_output, features
        return output, audio_output, visual_output

