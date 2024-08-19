import torch
from torch import nn
from typing import List, Sequence, Tuple, Union

from timm.models.vision_transformer import Block

from monai.utils import ensure_tuple_rep

from networks.layers import Layer, Relu, ResBlock
from networks.unetr_blocks import UnetOutBlock, UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock


class ViTDecoder(nn.Module):
    def __init__(self, dim, num_heads, depth, mlp_ratio, norm_layer=nn.LayerNorm):
        super(ViTDecoder, self).__init__()
        self.network = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.network:
            x = blk(x)
        x = self.norm(x)
        return x
        

class LinearDecoder(nn.Module):
    def __init__(self, in_size, dim, depth, layer_type:Layer=Relu):
        super(LinearDecoder, self).__init__()
        self.network = nn.ModuleList([layer_type(in_size, dim // 2, dropout=0.1)])
        self.network.extend([
            layer_type(dim // 2 ** i, dim // 2 ** (i + 1), dropout=0.1) for i in range(1, depth)])
        self.fc = nn.Linear(dim // 2 ** depth, 1, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.flatten(start_dim=1)
        for layer in self.network:
            x = layer(x)
        x = self.fc(x)
        return x


class UNETR_decoder(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        upsample_kernel_sizes: Union[list, Sequence[int]],
        feature_size: int = 16,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            spatial_dims: number of spatial dims.

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name="batch")

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name="batch", spatial_dims=2)

        """

        super().__init__()

        patch_size = (1, *patch_size)
        self.grid_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, patch_size))
        self.hidden_size = hidden_size
        self.slice_num = img_size[0]
        self.output_channel = out_channels * self.slice_num # times slice num
        self.upsample_kernel_sizes = upsample_kernel_sizes
        assert len(self.upsample_kernel_sizes) == 3, "Only support UNETR decoder depth equals 3"
            
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=self.upsample_kernel_sizes[1:],
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=self.upsample_kernel_sizes[2],
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=self.upsample_kernel_sizes[2],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=self.upsample_kernel_sizes[1],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=self.upsample_kernel_sizes[0],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=self.output_channel)

    def proj_feat(self, x, hidden_size, grid_size):
        new_view = (x.size(0), *grid_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(grid_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in: torch.Tensor, x: torch.Tensor, hidden_states_out: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass of UNETR decoder.

        Args:
            x_in (torch.Tensor): images in the shape of (batch, slice, time, height, width)
            x (torch.Tensor): latent features extracted from the encoder
            hidden_states_out (List[torch.Tensor]): the output of each layer of encoder

        Returns:
            torch.Tensor: segmentation probability in the same shape as x_in
        """
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[0]
        proj_x2 = self.proj_feat(x2, self.hidden_size//self.grid_size[0], self.grid_size) # (batch, hidden_size, s, 2, 16, 16)
        proj_x2 = proj_x2.view(proj_x2.shape[0], -1, *self.grid_size[1:])
        enc2 = self.encoder2(proj_x2)
        x3 = hidden_states_out[1]
        proj_x3 = self.proj_feat(x3, self.hidden_size//self.grid_size[0], self.grid_size) # (batch, hidden_size, s, 2, 16, 16)
        proj_x3 = proj_x3.view(proj_x3.shape[0], -1, *self.grid_size[1:])
        enc3 = self.encoder3(proj_x3)
        
        proj_x = self.proj_feat(x, self.hidden_size//self.grid_size[0], self.grid_size)
        dec3 = proj_x.view(proj_x.shape[0], -1, *self.grid_size[1:])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        seg_out = self.out(out)
        seg_pred = seg_out.view(seg_out.shape[0], -1, self.slice_num, *seg_out.shape[2:]) # (B, 4, slice, T, H, W)
        return seg_pred