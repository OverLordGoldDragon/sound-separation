import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GlobalLayerNorm",
    "get_norm",
    "ScaleParam",
    "DepthwiseConv1d",
    "TDCNBlock",
    "TDCNpp",
]


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN) used in Conv-TasNet / TDCN++.

    Normalises over the *entire* time-frequency representation for each sample
    (all channels and time steps).
    Expected input shape: (B, C, T) – channels-first 1-D feature maps.
    """

    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)


class InstanceNorm1d(nn.InstanceNorm1d):
    """Alias to stay close to TF code nomenclature."""


class ScaleParam(nn.Module):
    """Learnable scalar multiplier alpha initialised to *scale*.

    If *scale* < 0 the module is a no-op (mimics TF implementation)."""

    def __init__(self, scale: float):
        super().__init__()
        self.enabled = scale >= 0
        if self.enabled:
            self.alpha = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            return x * self.alpha
        return x


def get_activation(name: str):
    name = name.lower()
    if name == "prelu":
        return nn.PReLU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01, inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "linear":
        return nn.Identity()
    raise ValueError(f"Unsupported activation {name}")


def get_norm(norm_type: str, channels: int):
    norm_type = norm_type.lower()
    if norm_type == "instance_norm":
        return nn.InstanceNorm1d(channels, affine=True)
    if norm_type == "layer_norm":
        return nn.LayerNorm(channels)
    if norm_type in {"global_layer_norm", "gln"}:
        return GlobalLayerNorm(channels)
    raise ValueError(f"Unsupported norm {norm_type}")


class DepthwiseConv1d(nn.Conv1d):
    """Depthwise separable 1-D convolution (channel-wise)."""

    def __init__(self, channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, padding: str = "same"):
        if padding == "same":
            # Compute padding such that output length equals input length when stride = 1.
            pad = (kernel_size - 1) // 2 * dilation
        else:
            pad = 0
        super().__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=channels,
            bias=False,
            padding=pad,
        )


class TDCNBlock(nn.Module):
    """One residual dilated convolutional block of TDCN++.

    Follows order: 1×1 conv → Norm+Act → depthwise-separable conv → Norm+Act →
    1×1 conv (+ optional scale) → residual add.
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck: int = 256,
        num_conv_channels: int = 512,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        separable: bool = True,
        norm_type: str = "instance_norm",
        middle_activation: str = "prelu",
        end_activation: str = "linear",
        scale: float = -1.0,
        resid: bool = True,
    ) -> None:
        super().__init__()
        self.resid = resid
        self.bottleneck = bottleneck

        self.dense1 = nn.Conv1d(bottleneck, num_conv_channels, kernel_size=1)
        self.norm1 = get_norm(norm_type, num_conv_channels)
        self.act1 = get_activation(middle_activation)

        if separable:
            self.depthwise = DepthwiseConv1d(num_conv_channels, kernel_size, stride=stride, dilation=dilation)
        else:
            pad = (kernel_size - 1) // 2 * dilation
            self.depthwise = nn.Conv1d(num_conv_channels, num_conv_channels, kernel_size, stride=stride, dilation=dilation, padding=pad, bias=False)

        self.norm2 = get_norm(norm_type, num_conv_channels)
        self.act2 = get_activation(middle_activation)

        self.dense2 = nn.Conv1d(num_conv_channels, bottleneck, kernel_size=1)
        self.scale = ScaleParam(scale)
        self.end_act = get_activation(end_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (B, bottleneck, T)"""
        residual = x
        out = self.dense1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.depthwise(out)
        out = self.norm2(out)
        out = self.act2(out)

        out = self.dense2(out)
        out = self.scale(out)
        out = self.end_act(out)

        if self.resid:
            out = residual + out
        return out


class TDCNpp(nn.Module):
    """Improved Time-Dilated Convolutional Network (TDCN ++)."""

    def __init__(
        self,
        bottleneck: int = 256,
        conv_channels: int = 512,
        kernel_size: int = 3,
        num_dilations: int = 8,
        num_repeats: int = 4,
        norm_type: str = "instance_norm",
        add_skip_residual_connections: bool = False,
        scale_type: str = "exponential",
        separable: bool = True,
    ) -> None:
        super().__init__()
        self.bottleneck = bottleneck

        # Initial 1×1 conv maps input features to bottleneck dim.
        self.initial = nn.Conv1d(in_channels, bottleneck, kernel_size=1, bias=True)

        # Build convolutional blocks.
        blocks: List[nn.Module] = []
        total_blocks = num_dilations * num_repeats
        dilations: List[int] = [2 ** i for i in range(num_dilations)] * num_repeats

        scale_fn = self._make_scale_fn(scale_type)

        for b in range(total_blocks):
            blocks.append(TDCNBlock(
                bottleneck=bottleneck,
                num_conv_channels=conv_channels,
                kernel_size=kernel_size,
                dilation=dilations[b],
                stride=1,
                separable=separable,
                norm_type=norm_type,
                middle_activation="prelu",
                end_activation="linear",
                scale=scale_fn(b),
                resid=True,
            ))
        self.blocks = nn.ModuleList(blocks)

        # Optionally create skip-residual dense layers. Simpler version: only from
        # first block of each repeat to first block of later repeats.
        self.skip_res_layers: List[nn.Conv1d] = []
        if add_skip_residual_connections:
            # Dense layer per repeat pair.
            for r in range(1, num_repeats):
                self.skip_res_layers.append(nn.Conv1d(bottleneck, bottleneck, kernel_size=1))
            self.skip_res_layers = nn.ModuleList(self.skip_res_layers)
        else:
            self.skip_res_layers = nn.ModuleList()

    # ---------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------
    @staticmethod
    def _make_scale_fn(scale_type: str):
        if scale_type == "exponential":
            return lambda b: 0.9 ** b
        if scale_type == "none":
            return lambda b: -1.0
        if scale_type == "linear":
            return lambda b: 1.0 - (b / 100)  # placeholder linear decay
        raise ValueError(f"Unsupported scale_type {scale_type}")

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor shape (B, F, T) where *F* is the feature dimension – e.g.
               STFT magnitudes or mixture embeddings.
        Returns:
            Tensor of shape (B, bottleneck, T).
        """
        out = self.initial(x)
        inputs_per_block: List[torch.Tensor] = []
        skip_idx = 0

        for idx, block in enumerate(self.blocks):
            # Skip-residuals (very simplified variant).
            if self.skip_res_layers and idx % len(self.blocks) == 0 and idx > 0:
                out = out + self.skip_res_layers[skip_idx](inputs_per_block[0])
                skip_idx += 1

            inputs_per_block.append(out)
            out = block(out)
        return out
