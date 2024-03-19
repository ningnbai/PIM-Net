import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum
from math import sqrt
from .Res2Net import res2net50_v1b_26w_4s
import torch
import math
from torch.nn.functional import upsample
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, \
    PairwiseDistance
from torch.nn import functional as F
from torch.autograd import Variable

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m, nn.AdaptiveMaxPool2d):
            pass
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.Unfold):
            pass
        elif isinstance(m, GELU):
            pass
        elif isinstance(m, Softmax):
            pass
        elif isinstance(m, Sigmoid):
            pass
        else:
            m.initialize()

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

class dilatedComConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(dilatedComConv4, self).__init__()
        self.conv2_1 = conv(inplans, planes//4, kernel_size=3, padding=pyconv_kernels[0]//2,
                            stride=stride, groups=pyconv_groups[0],dilation=1)
        self.conv2_2 = conv(inplans, planes//4, kernel_size=3, padding=pyconv_kernels[1]//2,
                            stride=stride, groups=pyconv_groups[1],dilation=2)
        self.conv2_3 = conv(inplans, planes//4, kernel_size=3, padding=pyconv_kernels[2]//2,
                            stride=stride, groups=pyconv_groups[2],dilation=3)
        self.conv2_4 = conv(inplans, planes//4, kernel_size=3, padding=pyconv_kernels[3]//2,
                            stride=stride, groups=pyconv_groups[3],dilation=4)

    def forward(self, x):
        conv2_1 = self.conv2_1(x)
        conv2_2 = self.conv2_2(x)
        conv2_3 = self.conv2_3(x)
        conv2_4 = self.conv2_4(x)
        return torch.cat((conv2_1, conv2_2, conv2_3, conv2_4), dim=1)

    def initialize(self):
        weight_init(self)

class MOA(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4):
        super(MOA, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes//reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True),
            dilatedComConv4(inplanes // reduction1, inplanes // reduction1),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // reduction1, planes, kernel_size=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.layers(x)

    def initialize(self):
        weight_init(self)

LayerNorm = partial(nn.InstanceNorm2d, affine = True)

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads
        self.reduction_ratio = reduction_ratio

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x):
        h, w = x.shape[-2:] # h:64,w:64
        heads, r = self.heads, self.reduction_ratio # heads:1 r:8
        # to_qkv:Conv2d(32, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        q, k, v = self.to_qkv(x).chunk(3, dim = 1) # x:[1,32,64,64]->[1,96,64,64]->3*[1,32,64,64]
        # k, v = map(lambda t: reduce(t, 'b c (h r1) (w r2) -> b c h w', 'mean', r1 = r, r2 = r), (k, v))
        # k,v : [1,32,64,64] -> [1,32,8,8]

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))
        # 分为multi-head 此时head数为1
        # q:[1,32,64,64] -> [1,4096,32]
        # k:[1,32,8,8] -> [1,64,32]
        # v:[1,32,8,8] -> [1,64,32]
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # q*k: [1,4096,32]*[1,64,32]->[1,4096,64]
        attn = sim.softmax(dim = -1)
        # attention:[1,4096,64]
        out = einsum('b i j, b j d -> b i d', attn, v)
        # attn*v: [1,4096,64]*[1,64,32]->[1,4096,32]
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        # out: [1,4096,32]->[1,32,64,64]
        # to_out:Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # output:[1,32,64,64]->[1,32,64,64]
        return self.to_out(out)
    def initialize(self):
        weight_init(self)

class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1),
            GELU(),
            # nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1)
        )


    def forward(self, x):
        return self.net(x)
    def initialize(self):
        weight_init(self)

class MSA4(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4):
        super(MSA4, self).__init__()
        self.reductionLayers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True)
        )
        self.get_overlap_patches = nn.Unfold(3, dilation=1, stride=2, padding=1)

        self.overlap_embed = nn.Conv2d(4608, planes, kernel_size=(1, 1), stride=(1, 1))
        self.SelfAttention = EfficientSelfAttention(dim=planes, heads=4, reduction_ratio=2)
        self.ffd = MixFeedForward(dim=planes, expansion_factor=4)
        self.LN = LayerNorm(planes)

    def forward(self, x):
        x_size = x.size()
        x = self.reductionLayers(x)
        h, w = x.shape[-2:]
        x = self.get_overlap_patches(x)
        num_patches = x.shape[-1]
        ratio = int(sqrt((h * w) / num_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)
        x = self.overlap_embed(x)
        x = self.SelfAttention(self.LN(x)) + x
        x = self.ffd(self.LN(x)) + x
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

    def initialize(self):
        weight_init(self)

class MSA3(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4):
        super(MSA3, self).__init__()
        self.reductionLayers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True)
        )
        self.get_overlap_patches = nn.Unfold(3, dilation=1, stride=2, padding=1)

        self.overlap_embed = nn.Conv2d(2304, planes, kernel_size=(1, 1), stride=(1, 1))
        self.SelfAttention = EfficientSelfAttention(dim=planes, heads=4, reduction_ratio=2)
        self.ffd = MixFeedForward(dim=planes, expansion_factor=4)
        self.LN = LayerNorm(planes)

    def forward(self, x):
        x_size = x.size()
        x = self.reductionLayers(x)
        h, w = x.shape[-2:]
        x = self.get_overlap_patches(x)
        num_patches = x.shape[-1]
        ratio = int(sqrt((h * w) / num_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)
        x = self.overlap_embed(x)
        x = self.SelfAttention(self.LN(x)) + x
        x = self.ffd(self.LN(x)) + x
        # x = self.SelfAttention(self.LN(x)) + x
        # x = self.ffd(self.LN(x)) + x
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

    def initialize(self):
        weight_init(self)

class MSA2(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4):
        super(MSA2, self).__init__()
        self.reductionLayers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True)
        )
        self.get_overlap_patches = nn.Unfold(3, dilation=1, stride=2, padding=1)

        self.overlap_embed = nn.Conv2d(1152, planes, kernel_size=(1, 1), stride=(1, 1))
        self.SelfAttention = EfficientSelfAttention(dim=planes, heads=4, reduction_ratio=2)
        self.ffd = MixFeedForward(dim=planes, expansion_factor=4)
        self.LN = LayerNorm(planes)

    def forward(self, x):
        x_size = x.size()
        x = self.reductionLayers(x)
        h, w = x.shape[-2:]
        x = self.get_overlap_patches(x)
        num_patches = x.shape[-1]
        ratio = int(sqrt((h * w) / num_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)
        x = self.overlap_embed(x)
        x = self.SelfAttention(self.LN(x)) + x
        x = self.ffd(self.LN(x)) + x
        # x = self.SelfAttention(self.LN(x)) + x
        # x = self.ffd(self.LN(x)) + x
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

    def initialize(self):
        weight_init(self)

class MSA1(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4):
        super(MSA1, self).__init__()
        self.reductionLayers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True)
        )
        self.get_overlap_patches = nn.Unfold(3, dilation=1, stride=2, padding=1)

        self.overlap_embed = nn.Conv2d(576, planes, kernel_size=(1, 1), stride=(1, 1))
        self.SelfAttention = EfficientSelfAttention(dim=planes, heads=4, reduction_ratio=2)
        self.ffd = MixFeedForward(dim=planes, expansion_factor=4)
        self.LN = LayerNorm(planes)

    def forward(self, x):
        x_size = x.size()
        x = self.reductionLayers(x)
        h, w = x.shape[-2:]
        x = self.get_overlap_patches(x)
        num_patches = x.shape[-1]
        ratio = int(sqrt((h * w) / num_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)
        x = self.overlap_embed(x)
        x = self.SelfAttention(self.LN(x)) + x
        x = self.ffd(self.LN(x)) + x
        # x = self.SelfAttention(self.LN(x)) + x
        # x = self.ffd(self.LN(x)) + x
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

    def initialize(self):
        weight_init(self)

class MergePR(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(MergePR, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(inplanes, planes,  kernel_size=3, padding=1, groups=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, local_context, global_context):
        x = torch.cat((local_context, global_context), dim=1)
        x = self.features(x)
        return x

    def initialize(self):
        weight_init(self)

class PRIM4(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(PRIM4, self).__init__()

        out_size_local_context = int(inplanes/4)

        out_size_global_max_context = int(inplanes/4)

        self.MOA = MOA(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.MSA = MSA4(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.merge_context = MergePR(out_size_local_context + out_size_global_max_context, planes, BatchNorm)
    def forward(self, x):

        x = self.merge_context(self.MOA(x),  self.MSA(x))
        return x

    def initialize(self):
        weight_init(self)

class PRIM3(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(PRIM3, self).__init__()

        out_size_local_context = int(inplanes/4)

        out_size_global_max_context = int(inplanes/4)

        self.MOA = MOA(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.MSA = MSA3(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.merge_context = MergePR(out_size_local_context + out_size_global_max_context, planes, BatchNorm)
    def forward(self, x):

        x = self.merge_context(self.MOA(x),  self.MSA(x))
        return x

    def initialize(self):
        weight_init(self)

class PRIM2(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(PRIM2, self).__init__()

        out_size_local_context = int(inplanes/4)

        out_size_global_max_context = int(inplanes/4)

        self.MOA = MOA(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.MSA = MSA2(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.merge_context = MergePR(out_size_local_context + out_size_global_max_context, planes, BatchNorm)
    def forward(self, x):

        x = self.merge_context(self.MOA(x),  self.MSA(x))
        return x

    def initialize(self):
        weight_init(self)

class PRIM1(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(PRIM1, self).__init__()

        out_size_local_context = int(inplanes/4)

        out_size_global_max_context = int(inplanes/4)

        self.MOA = MOA(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.MSA = MSA1(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.merge_context = MergePR(out_size_local_context + out_size_global_max_context, planes, BatchNorm)
    def forward(self, x):

        x = self.merge_context(self.MOA(x),  self.MSA(x))
        return x

    def initialize(self):
        weight_init(self)


class FAPAEnc(Module):

    def __init__(self, in_channels, ksize):
        super(FAPAEnc, self).__init__()
        self.pool1 = DAP(in_channels, ksize, ksize)
        self.pool2 = DAP(in_channels, ksize//2, ksize//2)
        self.pool3 = DAP(in_channels, ksize//3, ksize//3)
        self.pool4 = DAP(in_channels, ksize//6, ksize//6)

        self.conv1 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
                                nn.BatchNorm2d(in_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
                                nn.BatchNorm2d(in_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
                                nn.BatchNorm2d(in_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
                                nn.BatchNorm2d(in_channels),
                                ReLU(True))

    def forward(self, x):
        b, c, h, w = x.size()

        feat1 = self.conv1(self.pool1(x)).view(b, c, -1)
        feat2 = self.conv2(self.pool2(x)).view(b, c, -1)
        feat3 = self.conv3(self.pool3(x)).view(b, c, -1)
        feat4 = self.conv4(self.pool4(x)).view(b, c, -1)

        return torch.cat((feat1, feat2, feat3, feat4), 2)

    def initialize(self):
        weight_init(self)


class FAPA(Module):

    def __init__(self, in_channels, ksize):
        super(FAPA, self).__init__()
        self.softmax = Softmax(dim=-1)
        self.scale = Parameter(torch.zeros(1))
        self.FAPAEnc = FAPAEnc(in_channels, ksize)

        self.conv_query = Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)  # query_conv2
        self.conv_key = Linear(in_channels, in_channels // 4)  # key_conv2
        self.conv_value = Linear(in_channels, in_channels)  # value2

    def forward(self, x):
        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,M)
            returns :
                out : compact position attention feature
                attention map: (H*W)*M
        """
        m_batchsize, C, width, height = x.size()
        y = self.FAPAEnc(x)
        y = y.permute(0, 2, 1)
        m_batchsize, K, M = y.size()

        proj_query = self.conv_query(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # BxNxd
        proj_key = self.conv_key(y).view(m_batchsize, K, -1).permute(0, 2, 1)  # BxdxK
        energy = torch.bmm(proj_query, proj_key)  # BxNxK
        attention = self.softmax(energy)  # BxNxk

        proj_value = self.conv_value(y).permute(0, 2, 1)  # BxCxK
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # BxCxN
        out = out.view(m_batchsize, C, width, height)
        out = self.scale * out + x
        return out
    def initialize(self):
        weight_init(self)


class CAM(Module):


    def __init__(self, in_channels):
        super(CAM, self).__init__()
        self.softmax = Softmax(dim=-1)
        self.scale = Parameter(torch.zeros(1))
        self.conv1 = Conv2d(in_channels, in_channels//4, kernel_size=1)

    def forward(self, x):
        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,H,W)
            returns :
                out : compact channel attention feature
                attention map: K*C
        """
        m_batchsize, C, width, height = x.size()
        x_reshape = x.view(m_batchsize, C, -1)
        y = self.conv1(x)

        B, K, W, H = y.size()
        y_reshape = y.view(B, K, -1)
        proj_query = x_reshape  # BXC1XN
        proj_key = y_reshape.permute(0, 2, 1)  # BX(N)XC
        energy = torch.bmm(proj_query, proj_key)  # BXC1XC
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = y.view(B, K, -1)  # BCN

        out = torch.bmm(attention, proj_value)  # BC1N
        out = out.view(m_batchsize, C, width, height)

        out = x + self.scale * out
        return out
    def initialize(self):
        weight_init(self)


class DAP(nn.Module):
    def __init__(self, input_c=64, kernel_size=2, stride=2):
        super(DAP, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv1x1 = nn.Conv2d(input_c, 1, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()

        weights = self.sigmoid(self.bn(self.conv1x1(x)))
        # print(weights.shape)
        _, _, H_w, W_w = weights.size()

        assert H == H_w and W == W_w, "Input size and weights size must be the same."

        H1 = (H - self.kernel_size) // self.stride + 1
        W1 = (W - self.kernel_size) // self.stride + 1

        # Unfold the input tensor and weights tensor
        x_unfolded = F.unfold(x, self.kernel_size, stride=self.stride)
        weights_unfolded = F.unfold(weights, self.kernel_size, stride=self.stride)

        # Reshape unfolded tensors for element-wise multiplication
        x_unfolded = x_unfolded.view(B, C, self.kernel_size * self.kernel_size, H1 * W1)
        weights_unfolded = weights_unfolded.view(B, 1, self.kernel_size * self.kernel_size, H1 * W1)

        # Element-wise multiplication and sum along the patch dimension
        weighted_feature = x_unfolded * weights_unfolded
        output = weighted_feature.sum(dim=2)

        # Reshape the output tensor to the final size
        output = output.view(B, C, H1, W1)

        return output

    def initialize(self):
        weight_init(self)


class PRM3(nn.Module):
    def __init__(self):
        super(PRM3, self).__init__()


        self.above_conv2 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.above_bn2 = nn.BatchNorm2d(512)

        self.right_conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.right_bn1   = nn.BatchNorm2d(512)

        self.fuse_conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.fuse_bn   = nn.BatchNorm2d(512)

        self.CAM = CAM(512)
        self.FAPA = FAPA(512, 24)


    def forward(self, backboneabove, right):

        above_2 = F.relu(self.above_bn2(self.above_conv2(backboneabove)), inplace=True)

        right = F.interpolate(right, size=backboneabove.size()[2:], mode='bilinear')
        right_1 = F.relu(self.right_bn1(self.right_conv1(right)), inplace=True)


        fuse2 = above_2 * right_1

        fuse = F.relu(self.fuse_bn(self.fuse_conv(fuse2)), inplace=True)

        fuse_cam = self.CAM(fuse)
        fuse_FAPA = self.FAPA(fuse)

        fuse = fuse_cam + fuse_FAPA + right_1
        # print(fuse.shape)

        return  fuse

    def initialize(self):
        weight_init(self)

class PRM2(nn.Module):
    def __init__(self):
        super(PRM2, self).__init__()

        self.above_conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.above_bn2 = nn.BatchNorm2d(256)

        self.right_conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.right_bn1 = nn.BatchNorm2d(256)

        self.fuse_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fuse_bn = nn.BatchNorm2d(256)

        self.CAM = CAM(256)
        self.FAPA = FAPA(256, 48)

    def forward(self, backboneabove, right):
        above_2 = F.relu(self.above_bn2(self.above_conv2(backboneabove)), inplace=True)

        right = F.interpolate(right, size=backboneabove.size()[2:], mode='bilinear')
        right_1 = F.relu(self.right_bn1(self.right_conv1(right)), inplace=True)

        fuse2 = above_2 * right_1

        fuse = F.relu(self.fuse_bn(self.fuse_conv(fuse2)), inplace=True)

        fuse_cam = self.CAM(fuse)
        fuse_FAPA = self.FAPA(fuse)

        fuse = fuse_cam + fuse_FAPA + right_1
        # print(fuse.shape)

        return fuse

    def initialize(self):
        weight_init(self)

class PRM1(nn.Module):
    def __init__(self):
        super(PRM1, self).__init__()

        self.above_conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.above_bn2 = nn.BatchNorm2d(128)

        self.right_conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.right_bn1 = nn.BatchNorm2d(128)

        self.fuse_conv = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.fuse_bn = nn.BatchNorm2d(128)

        self.CAM = CAM(128)
        self.FAPA = FAPA(128, 48)

    def forward(self, backboneabove, right):
        above_2 = F.relu(self.above_bn2(self.above_conv2(backboneabove)), inplace=True)

        right = F.interpolate(right, size=backboneabove.size()[2:], mode='bilinear')
        right_1 = F.relu(self.right_bn1(self.right_conv1(right)), inplace=True)

        fuse2 = above_2 * right_1

        fuse = F.relu(self.fuse_bn(self.fuse_conv(fuse2)), inplace=True)

        fuse_cam = self.CAM(fuse)
        fuse_FAPA = self.FAPA(fuse)

        fuse = fuse_cam + fuse_FAPA + right_1
        # print(fuse.shape)

        return fuse

    def initialize(self):
        weight_init(self)

class PIMNet(nn.Module):
    def __init__(self, cfg):
        super(Res2Net50, self).__init__()
        self.cfg      = cfg
        self.bkbone = res2net50_v1b_26w_4s()
        self.PRIM4 = PRIM4(2048, 512, nn.BatchNorm2d)
        self.PRM3 = PRM3()
        self.PRM2 = PRM2()
        self.PRM1 = PRM1()
        self.linearrpred = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        if cfg.mode == 'train':
            self.PRIM3 = PRIM3(1024, 256, nn.BatchNorm2d)
            self.PRIM2 = PRIM2(512, 128, nn.BatchNorm2d)
            self.PRIM1 = PRIM1(256, 64, nn.BatchNorm2d)
            self.linearr1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
            self.linearr2 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
            self.linearr3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            self.linearr4 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        self.initialize()

    def forward(self, x, shape=None):
        if self.cfg.mode=='train':
            shape = x.size()[2:] if shape is None else shape
            out1_bk, out2_bk, out3_bk, out4_bk = self.bkbone(x)
            PRIM4_feature = self.PRIM4(out4_bk)
            PRIM3_feature = self.PRIM3(out3_bk)
            PRIM2_feature = self.PRIM2(out2_bk)
            PRIM1_feature = self.PRIM1(out1_bk)
            PRM3_out = self.PRM3(out3_bk,PRIM4_feature)
            PRM2_out = self.PRM2(out2_bk,PRM3_out)
            PRM1_out = self.PRM1(out1_bk,PRM2_out)
            PRIM1_out = F.interpolate(self.linearr1(PRIM1_feature), size=shape, mode='bilinear')
            PRIM2_out = F.interpolate(self.linearr2(PRIM2_feature), size=shape, mode='bilinear')
            PRIM3_out = F.interpolate(self.linearr3(PRIM3_feature), size=shape, mode='bilinear')
            PRIM4_out = F.interpolate(self.linearr4(PRIM4_feature), size=shape, mode='bilinear')
            pred = F.interpolate(self.linearrpred(PRM1_out), size=shape, mode='bilinear')
            return pred, PRIM1_out, PRIM2_out, PRIM3_out, PRIM4_out
        else:
            shape = x.size()[2:] if shape is None else shape
            out1_bk, out2_bk, out3_bk, out4_bk = self.bkbone(x)
            PRIM4_feature = self.PRIM4(out4_bk)
            PRM3_out = self.PRM3(out3_bk, PRIM4_feature)
            PRM2_out = self.PRM2(out2_bk, PRM3_out)
            PRM1_out = self.PRM1(out1_bk, PRM2_out)
            pred = F.interpolate(self.linearrpred(PRM1_out), size=shape, mode='bilinear')
            return pred

    def initialize(self):
        if self.cfg.mode=='test':
            param = {}
            model_dict = self.state_dict()
            checkpoint = torch.load(self.cfg.snapshot,map_location='cpu')
            for k, v in checkpoint.items():
                if k in model_dict:
                    param[k] = v
            model_dict.update(param)
            self.load_state_dict(model_dict)
        else:
            weight_init(self)
