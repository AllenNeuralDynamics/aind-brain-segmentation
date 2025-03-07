"""
Neural network blocks
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

from .norm_layers import GRN, LayerNorm


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x, m=""):
        # Do your print / debug stuff here
        print(m, x.shape)
        return x


class ConvNeXtV2Block(nn.Module):
    """

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNextV2Block(torch.nn.Module):
    """
    3D Conv Next V2 block
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: Optional[int] = 1,
        padding: Optional[Union[int, str]] = "same",
        padding_mode: Optional[str] = "zeros",
        point_wise_scaling: Optional[int] = 4,
        device: Optional[str] = None,
        drop_path=0.0,
    ):
        """
        Initializes the convolutional block
        """
        super(ConvNextV2Block, self).__init__()
        self.conv_layer = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            device=device,
        )
        self.layer_norm = nn.LayerNorm(
            normalized_shape=out_channels, eps=1e-6, device=device
        )
        self.point_wise_conv1 = nn.Linear(
            in_features=out_channels,
            out_features=point_wise_scaling * out_channels,
            device=device,
        )
        self.activation_layer = nn.GELU()  # nn.ReLU()
        self.grn_norm = GRN(point_wise_scaling * out_channels)  # point_wise_scaling *
        self.point_wise_conv2 = nn.Linear(
            in_features=point_wise_scaling * out_channels,
            out_features=out_channels,
            device=device,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, input_feat: torch.Tensor):
        skip_con = input_feat
        input_feat = self.conv_layer(input_feat)

        # Changing channel axis to last
        # (N, C, D, H, W) -> (N, D, H, W, C)
        # input_feat = skip_con.permute(0, 2, 3, 4, 1)
        input_feat = input_feat.permute(0, 2, 3, 4, 1)

        input_feat = self.layer_norm(input_feat)
        input_feat = self.point_wise_conv1(input_feat)

        input_feat = self.activation_layer(input_feat)

        input_feat = self.grn_norm(input_feat)

        input_feat = self.point_wise_conv2(input_feat)

        # Back to channels in second pos
        # (N, D, H, W, C) -> (N, C, D, H, W)
        input_feat = input_feat.permute(0, 4, 1, 2, 3)
        # print("Check: ", skip_con.shape, input_feat.shape)

        if skip_con.shape[1] == input_feat.shape[1]:
            # print("Skip connection!")
            input_feat = skip_con + self.drop_path(input_feat)
        else:
            input_feat = self.drop_path(input_feat)

        return input_feat


class ConvNextV2(nn.Module):
    """

    Args:
        in_channels (int): Number of input image channels.
        depth (int): Number of stages.
        depths (tuple(int)): Number of blocks at each stage.
        dims (int): Feature dimension at each stage.
        stem_kernel_size (int): Kernel size of the stem conv. Default: 4
        stem_stride (int): Stride of the stem conv. Default: 4
    """

    def __init__(
        self,
        in_channels,
        depth,
        depths,
        dims,
        stem_kernel_size=4,
        stem_stride=4,
    ):
        super().__init__()

        self.depth = depth
        self.depths = depths
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                dims[0],
                kernel_size=stem_kernel_size,
                stride=stem_stride,
            ),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        for i in range(depth - 1):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        drop_path_rate = 0.0
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(depth):
            stage = nn.Sequential(
                *[
                    ConvNeXtV2Block(dim=dims[i], drop_path=dp_rates[cur + j])
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def get_stages(self):
        stages = [
            nn.Identity(),
            *[
                nn.Sequential(self.downsample_layers[i], self.stages[i])
                for i in range(self.depth)
            ],
        ]
        return stages

    def forward_features(self, x):
        outs = [x]
        for i in range(self.depth):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)

        return outs

    def forward(self, x):
        return self.forward_features(x)


class ConvolutionalBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding):
        conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
        )

        activation = nn.LeakyReLU()
        bn = nn.BatchNorm3d(out_channels)

        super(ConvolutionalBlock, self).__init__(conv, activation, bn)


class DecoderUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_rate, kernel_size=3, strides=1):
        super(DecoderUpsampleBlock, self).__init__()

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding="same",
        )

        self.batch_norm = nn.BatchNorm3d(out_channels, eps=norm_rate)
        self.activation = nn.LeakyReLU()
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = self.upsample(x)
        return x


class ConnectionComponents(nn.Module):
    def __init__(self, in_channels, out_channels, norm_rate, kernel_size=3, strides=1):
        super(ConnectionComponents, self).__init__()

        self.conv_1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        self.conv_2 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding="same",
        )

        self.activation_1 = nn.LeakyReLU()
        self.activation_2 = nn.LeakyReLU()

        self.bach_norm_1 = nn.BatchNorm3d(1, eps=norm_rate)
        self.bach_norm_2 = nn.BatchNorm3d(out_channels, eps=norm_rate)
        self.bach_norm_3 = nn.BatchNorm3d(out_channels, eps=norm_rate)

    def forward(self, x):
        shortcut = x
        path_1 = self.conv_1(shortcut)
        path_1 = self.bach_norm_1(path_1)

        # conv 3x3
        path_2 = self.conv_2(x)
        path_2 = self.bach_norm_2(path_2)
        path_2 = self.activation_2(path_2)

        # add layer
        out = path_1 + path_2
        out = self.activation_1(out)
        out = self.bach_norm_3(out)
        return out


class EncoderDecoderConnections(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_rate=1e-4):
        super(EncoderDecoderConnections, self).__init__()

        self.con_comp_1 = ConnectionComponents(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_rate=norm_rate,
            kernel_size=kernel_size,
        )

        self.con_comp_2 = ConnectionComponents(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_rate=norm_rate,
            kernel_size=kernel_size,
        )

        self.con_comp_3 = ConnectionComponents(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_rate=norm_rate,
            kernel_size=kernel_size,
        )

        self.con_comp_4 = ConnectionComponents(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_rate=norm_rate,
            kernel_size=kernel_size,
        )

    def forward(self, x):
        x = self.con_comp_1(x)
        x = self.con_comp_2(x)
        x = self.con_comp_3(x)
        x = self.con_comp_4(x)
        return x


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, strides=1):
        super(SegmentationHead, self).__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding="same",
        )

        # activation = nn.Softmax(dim=1)
        # self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        # print("Seg head: ", x.shape)
        # x = self.activation(x)
        return x
