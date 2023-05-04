from torch import nn
import torch
import torch.nn.functional as F
class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'freqpreserve':
            return F.interpolate(x, scale_factor=(1,2), mode='bilinear')
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='bilinear')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='bilinear')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)