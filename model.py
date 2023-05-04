import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.DownSample import DownSample
from Modules.ResBlock2D import ResBlk

class Discriminator(nn.Module):
    """
    StarganV2-VC discriminator
    """
    def __init__(self, dim_in=64, max_conv_dim=512, repeat_num=2):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        for lid in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, sampling=DownSample("half"))]
            dim_in = dim_out

        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.Conv2d(dim_out, 1, 1, 1, 0)]
        blocks += [nn.Sigmoid()]
        self.main = nn.Sequential(*blocks)

    def forward(self, x): # (batch)
        return self.main(x)[:,0,0,0]
        
class ResidualLayer1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualLayer1D, self).__init__()

        self.conv1d_layer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1,
                                                    padding=padding),
                                          nn.InstanceNorm1d(num_features=out_channels,
                                                            affine=True))

        self.conv_layer_gates = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=out_channels,
                                                                affine=True))

        self.conv1d_out_layer = nn.Sequential(nn.Conv1d(in_channels=out_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=in_channels,
                                                                affine=True))

    def forward(self, input):
        h1_norm = self.conv1d_layer(input)
        h1_gates_norm = self.conv_layer_gates(input)

        # GLU
        h1_glu = h1_norm * torch.sigmoid(h1_gates_norm)

        h2_norm = self.conv1d_out_layer(h1_glu)
        return input + h2_norm
    
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(DownConv, self).__init__()
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm2d(num_features=out_channels,
                                                         affine=True))
        self.convLayer_gates = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding),
                                             nn.InstanceNorm2d(num_features=out_channels,
                                                               affine=True))


    def forward(self, x):
        return self.convLayer(x) * torch.sigmoid(self.convLayer_gates(x))

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), ps_factor=2):
        super(UpConv, self).__init__()
        self.convLayer = nn.Sequential(
                                        nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm2d(num_features=out_channels,
                                                         affine=True))
        self.convLayer_gates = nn.Sequential(
                                            nn.Conv2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding),
                                             nn.InstanceNorm2d(num_features=out_channels,
                                                               affine=True))
        self.ps_factor = ps_factor
    def forward(self, x, skip):
        x = torch.functional.F.pixel_shuffle(x, self.ps_factor)
        
        # input is CHW
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]

        skip = F.pad(skip, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        return self.convLayer(x) * torch.sigmoid(self.convLayer_gates(x))
    

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList() 
        self.downs = nn.ModuleList()
        
        self.Conv1 = DownConv(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.DownConv1 = DownConv(64, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        
        self.Conv2 = DownConv(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.DownConv2 = DownConv(128, 128, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        
        self.Conv3 = DownConv(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.DownConv3 = DownConv(256, 256, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        
        self.in_bottleneck = nn.Conv1d(in_channels=2560, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.bottleneck = nn.Sequential(
            ResidualLayer1D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            ResidualLayer1D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            ResidualLayer1D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            ResidualLayer1D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            ResidualLayer1D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            ResidualLayer1D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        )
        self.out_1d = nn.Conv1d(in_channels=256, out_channels=2560, kernel_size=1, stride=1, padding=0)
        self.out_bottleneck = DownConv(256, 256 * 4, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        
        self.UpConv1 = UpConv(512, 128 * 4, kernel_size=(3,3), stride=(1,1), padding=(1,1), ps_factor=2)
        self.UpConv2 = UpConv(256, 64 * 4, kernel_size=(3,3), stride=(1,1), padding=(1,1), ps_factor=2)
        self.UpConv3 = UpConv(128, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), ps_factor=2)
        
        self.OutConv = nn.Conv2d(64, 1, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        
        
    def forward(self, x):
        # torch.Size([5, 1, 80, 192])
        s1 = self.Conv1(x)
        x = self.DownConv1(s1)

        # torch.Size([5, 64, 80, 192])
        s2 = self.Conv2(x)
        x = self.DownConv2(s2)
        
        # torch.Size([5, 128, 40, 96])
        s3 = self.Conv3(x)
        x = self.DownConv3(s3)

        b, c, h, w = x.size()
        in_bottleneck = self.in_bottleneck(x.view(b, c*h, -1))
        bottleneck = self.bottleneck(in_bottleneck)
        end_1d_bottleneck = self.out_1d(bottleneck)
        end_bottleneck = self.out_bottleneck(end_1d_bottleneck.view(b, c, h, -1))
        x = end_bottleneck
        
        x = self.UpConv1(x, s3)
        
        x = self.UpConv2(x, s2)
        
        x = self.UpConv3(x, s1)
        # torch.Size([5, 256, 20, 48])

        x = self.OutConv(x)

        return x