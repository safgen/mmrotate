import math
from numpy.core.fromnumeric import resize, shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from .token_transformer import Token_transformer
from .token_performer import Token_performer
from .SELayer import SELayer
from .swin import SwinTransformerBlock, window_partition, window_reverse

class PRM(nn.Module):
    def __init__(self, img_size=224, kernel_size=4, downsample_ratio=4, dilations=[1,6,12], in_chans=3, embed_dim=64, share_weights=False, op='cat'):
        super().__init__()
        self.dilations = dilations
        self.embed_dim = embed_dim
        self.downsample_ratio = downsample_ratio
        self.op = op
        self.kernel_size = kernel_size
        self.stride = downsample_ratio
        self.share_weights = share_weights
        self.outSize = img_size // downsample_ratio

        if share_weights:
            # Multiple dilated convolutions with different dilation rates that share weights
            self.convolution = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=self.kernel_size, \
                stride=self.stride, padding=3*dilations[0]//2, dilation=dilations[0])

        else:
            self.convs = nn.ModuleList()
            # Multiple parallel dilated convolutions with different dilation rates
            for dilation in self.dilations:
                padding = math.ceil(((self.kernel_size-1)*dilation + 1 - self.stride) / 2)
                self.convs.append(nn.Sequential(*[nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=self.kernel_size, \
                    stride=self.stride, padding=padding, dilation=dilation),
                    nn.GELU()]))

        if self.op == 'sum':
            self.out_chans = embed_dim
        elif op == 'cat':
            self.out_chans = embed_dim * len(self.dilations)

    def forward(self, x):
        B, C, W, H = x.shape
        if self.share_weights:
            # Multiple dilated convolutions with different dilation rates that share weights
            padding = math.ceil(((self.kernel_size-1)*self.dilations[0] + 1 - self.stride) / 2)
            y = nn.functional.conv2d(x, weight=self.convolution.weight, bias=self.convolution.bias, \
                stride=self.downsample_ratio, padding=padding, dilation=self.dilations[0]).unsqueeze(dim=-1)
            for i in range(1, len(self.dilations)):
                padding = math.ceil(((self.kernel_size-1)*self.dilations[i] + 1 - self.stride) / 2)
                _y = nn.functional.conv2d(x, weight=self.convolution.weight, bias=self.convolution.bias, \
                    stride=self.downsample_ratio, padding=padding, dilation=self.dilations[i]).unsqueeze(dim=-1)
                y = torch.cat((y, _y), dim=-1)
        else:
            # Multiple parallel dilated convolutions with different dilation rates
            y = self.convs[0](x).unsqueeze(dim=-1)
            for i in range(1, len(self.dilations)):
                _y = self.convs[i](x).unsqueeze(dim=-1)
                y = torch.cat((y, _y), dim=-1)
        B, C, W, H, N = y.shape
        # Merge the results of these convolutions into 1D features
        if self.op == 'sum':
            # Summing and merging
            y = y.sum(dim=-1).flatten(2).permute(0,2,1).contiguous()
        elif self.op == 'cat':
            # Concatenating and merging
            y = y.permute(0,4,1,2,3).flatten(3).reshape(B, N*C, W*H).permute(0,2,1).contiguous()
        else:
            raise NotImplementedError('no such operation: {} for multi-levels!'.format(self.op))
        return y, (W, H)

class ReductionCell(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=64, token_dims=64, downsample_ratios=4, kernel_size=7,
                 num_heads=1, dilations=[1,2,3,4], share_weights=False, op='cat', tokens_type='performer', group=1,
                 relative_pos=False, drop=0., attn_drop=0., drop_path=0., mlp_ratio=1.0, gamma=False, init_values=1e-4, SE=False, window_size=7):
        super().__init__()

        self.img_size = img_size
        self.window_size = window_size
        self.op = op
        self.dilations = dilations
        self.num_heads = num_heads
        self.embed_dims = embed_dims
        self.token_dims = token_dims
        self.in_chans = in_chans
        self.downsample_ratios = downsample_ratios
        self.kernel_size = kernel_size
        self.outSize = img_size
        self.relative_pos = relative_pos
        PCMStride = []
        residual = downsample_ratios // 2
        # When downsample_ratios=4, PCMstride is [2,2,1], reducing the resolution by a factor of 4.
        for _ in range(3):
            PCMStride.append((residual > 0) + 1)
            residual = residual // 2
        assert residual == 0
        self.pool = None
        self.tokens_type = tokens_type
        if tokens_type == 'pooling':
            PCMStride = [1, 1, 1]
            self.pool = nn.MaxPool2d(downsample_ratios, stride=downsample_ratios, padding=0)
            tokens_type = 'transformer'
            self.outSize = self.outSize // downsample_ratios
            downsample_ratios = 1
        # PCM: three convolution layers with dilation rates of 2, 2, and 1, respectively, with the first two layers down-sampling the resolution.
        self.PCM = nn.Sequential(
                        nn.Conv2d(in_chans, embed_dims, kernel_size=(3, 3), stride=PCMStride[0], padding=(1, 1), groups=group),  # the 1st convolution
                        nn.BatchNorm2d(embed_dims),
                        nn.SiLU(inplace=True),
                        nn.Conv2d(embed_dims, embed_dims, kernel_size=(3, 3), stride=PCMStride[1], padding=(1, 1), groups=group),  # the 2nd convolution
                        nn.BatchNorm2d(embed_dims),
                        nn.SiLU(inplace=True),
                        nn.Conv2d(embed_dims, token_dims, kernel_size=(3, 3), stride=PCMStride[2], padding=(1, 1), groups=group),  # the 3rd convolution
                    )
        ## PRM performs down-sampling, with the down-sampling ratio determined by the dilation rates of the expandable convolution.
        self.PRM = PRM(img_size=img_size, kernel_size=kernel_size, downsample_ratio=downsample_ratios, dilations=self.dilations,
            in_chans=in_chans, embed_dim=embed_dims, share_weights=share_weights, op=op)
        self.outSize = self.outSize // downsample_ratios

        in_chans = self.PRM.out_chans
        if tokens_type == 'performer':
            # assert num_heads == 1
            self.attn = Token_performer(dim=in_chans, in_dim=token_dims, head_cnt=num_heads, kernel_ratio=0.5, gamma=gamma, init_values=init_values)
        elif tokens_type == 'performer_less':
            self.attn = None
            self.PCM = None
        elif tokens_type == 'transformer':
            # Ordinary vision transformer
            self.attn = Token_transformer(dim=in_chans, in_dim=token_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop, 
                                        attn_drop=attn_drop, drop_path=drop_path, gamma=gamma, init_values=init_values)
        elif tokens_type == 'swin':
            # Add a control to use relative position encoding or not. The window size is fixed and there is no shift operation.
            # token_dim is the dimension output by swint
            self.attn = SwinTransformerBlock(in_dim=in_chans, out_dim=token_dims, input_resolution=(self.img_size//self.downsample_ratios, self.img_size//self.downsample_ratios), 
                                            num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop,
                                            attn_drop=attn_drop, drop_path=drop_path, window_size=window_size, shift_size=0, relative_pos=relative_pos)

        if gamma:
            self.gamma2 = nn.Parameter(init_values * torch.ones((token_dims)),requires_grad=True)
            self.gamma3 = nn.Parameter(init_values * torch.ones((token_dims)),requires_grad=True)
        else:
            self.gamma2 = 1
            self.gamma3 = 1

        if SE:
            self.SE = SELayer(token_dims)
        else:
            self.SE = nn.Identity()

        self.num_patches = (img_size // 2) * (img_size // 2)  # there are 3 soft splits, strides are 4, 2, 2 respectively

    def forward(self, x, H, W):
        if len(x.shape) < 4:
            # B,N,C -> B,H,W,C
            B, N, C  = x.shape
            #n = int(np.sqrt(N))
            x = x.view(B, H, W, C).contiguous()
            x = x.permute(0, 3, 1, 2)
        if self.pool is not None:
            x = self.pool(x)
        shortcut = x
        PRM_x, _ = self.PRM(x)
        H, W = H // self.downsample_ratios, W // self.downsample_ratios
        # PRM's output is B, N, D*C, D: the number of parallel branches of the expanded convolution
        # PRM performs downsampling, and the downsampling rate is determined by the expansion rate of the convolution
        if self.tokens_type == 'swin':
            pass
            B, N, C = PRM_x.shape
            #H, W = self.img_size // self.downsample_ratios, self.img_size // self.downsample_ratios
            b, _, c = PRM_x.shape
            assert N == H*W
            # Get multi-scale features, normalize and use fixed-window MHSA directly
            x = self.attn.norm1(PRM_x)
            x = x.view(B, H, W, C)

            # pad feature maps to multiples of window size
            pad_l = pad_t = 0
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            x = torch.nn.functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, Hp, Wp, _ = x.shape

            x_windows = window_partition(x, self.window_size) # Partition features into windows
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
            attn_windows = self.attn.attn(x_windows, mask=self.attn.attn_mask)  # nW*B, window_size*window_size, C， window-level attention.
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.token_dims) # Convert window-level features from 1D to 2D
            shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp) # Reverse window features to restore the original feature dimensions
            x = shifted_x

            # If there is padding, remove it
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :].contiguous()

            # 'token_dim' is the output dimension of SwinT
            x = x.view(B, H * W, self.token_dims)

            # Pass shortcut tensor through PCM layer
            convX = self.PCM(shortcut)
            # Convert tensor shape to match 'x' tensor shape for addition
            convX = convX.permute(0, 2, 3, 1).view(*x.shape).contiguous()

            # Combine PCM and WMSA features
            x = x + self.attn.drop_path(convX * self.gamma2)
            # x = shortcut + self.attn.drop_path(x)
            # x = x + self.attn.drop_path(self.attn.mlp(self.attn.norm2(x)))
            x = x + self.attn.drop_path(self.gamma3 * self.attn.mlp(self.attn.norm2(x)))
        else:
            if self.attn is None:
                return PRM_x
            convX = self.PCM(shortcut) # Perform Convolution operation on shortcut
            x = self.attn.attn(self.attn.norm1(PRM_x)) # Apply attention on PRM_x features
            convX = convX.permute(0, 2, 3, 1).view(*x.shape).contiguous() # Reshape Convolution output to (B, H*W, C)
            x = x + self.attn.drop_path(convX * self.gamma2) # Fuse PCM features with attention output
            x = x + self.attn.drop_path(self.gamma3 * self.attn.mlp(self.attn.norm2(x))) # Apply drop-path and MLP on attention output

            x = self.SE(x) # Apply Squeeze-and-Excitation on features

        return x, H, W

    def train(self, mode=True, tag='default'):
        self.training = mode
        if tag == 'default':
            for module in self.children():
                module.train(mode)
        elif tag == 'linear':
            for module in self.children():
                module.eval()
        elif tag == 'linearLN':
            for module in self.children():
                module.train(False, tag=tag)
        return self