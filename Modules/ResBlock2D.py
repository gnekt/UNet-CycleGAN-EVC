from torch import nn
import torch 
import math 
import torch.nn.functional as F



#### STARGANv2-VC ####


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'freqpreserve':
            return F.avg_pool2d(x, (1, 2))
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)
        
class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, sampling=DownSample("half")):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.sampling = sampling
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_out, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut_wo_sampling(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        return x    
    
    def _residual_wo_sampling(self, x):
        x = self.conv1(x)
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv2(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        return x
    
    def _shortcut_w_downampling(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.sampling:
            x = self.sampling(x)
        return x

    def _residual_w_downsampling(self, x):
        x = self.conv1(x)
        if self.sampling is not None:
            x = self.sampling(x)
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv2(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        return x

    def forward(self, x):
        if self.sampling is None:
            x = self._shortcut_wo_sampling(x) + self._residual_wo_sampling(x)
        elif type(self.sampling) == DownSample: 
            x = self._shortcut_w_downampling(x) + self._residual_w_downsampling(x)
        else:
            raise RuntimeError('Got unexpected sampling type %s, expected is [None, DownSample]' % self.sampling)
        return x / math.sqrt(2)  # unit variance 