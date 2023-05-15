import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.ResBlock2D import ResBlk


######### 
#########
### SOME OF THE IMPLEMENTATION COMING FROM CYCLEGAN-VC2 + MODIFICATIONS


class Discriminator(nn.Module):
    """
    StarganV2-VC discriminator
    """
    def __init__(self, dim_in: int = 64, max_conv_dim: int = 512, repeat_num: int = 2):
        """_summary_

        Args:
            dim_in (int, optional): _description_. Defaults to 64.
            max_conv_dim (int, optional): _description_. Defaults to 512.
            repeat_num (int, optional): _description_. Defaults to 2.
        """        
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        for lid in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, normalize=True)]
            dim_in = dim_out

        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.Conv2d(dim_out, 1, 1, 1, 0)]
        blocks += [nn.Sigmoid()]
        self.main = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for discriminator

        Args:
            x (torch.Tensor): Input tensor -> (Batch, 1, MelBand, T_Mel)

        Returns:
            torch.Tensor: Output tensor -> (Batch, 1)
        """        
        return self.main(x)[:,0,0,0] # Out (Batch, 1, 1, 1) -> Squeeze -> (Batch, 1)
        
class ResidualLayer1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        """_summary_

        Args:
            in_channels (int): Input Channels
            out_channels (int): Output Channels
            kernel_size (int): Kernel Size
            stride (int): Stride
            padding (int): Padding
        """        
        
        super(ResidualLayer1D, self).__init__()
        self.conv1d_layer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding),
                                          nn.InstanceNorm1d(num_features=out_channels,
                                                            affine=True))

        self.conv_layer_gates = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        stride=stride,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=out_channels,
                                                                affine=True))

        self.conv1d_out_layer = nn.Sequential(nn.Conv1d(in_channels=out_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        stride=stride,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=in_channels,
                                                                affine=True))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """forward method for residual layer

        Args:
            input (torch.Tensor): Input tensor -> (Batch, InChannels, T_Mel)

        Returns:
            torch.Tensor: (Batch, OutChannels, T_Mel)
        """        
        h1_out = self.conv1d_layer(input) # Out: No Shape change
        h1_gates = self.conv_layer_gates(input) # Out: No Shape change

        # GLU
        h1_glu = h1_out * torch.sigmoid(h1_gates) # Out: No Shape change

        h2_out = self.conv1d_out_layer(h1_glu) # Out: No Shape change
        return input + h2_out
    
class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple = (3,3), stride: tuple = (1,1), padding: tuple = (1,1)):
        """_summary_

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            kernel_size (tuple, optional): _description_. Defaults to (3,3).
            stride (tuple, optional): _description_. Defaults to (1,1).
            padding (tuple, optional): _description_. Defaults to (1,1).
        """        
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for down convolution

        Args:
            x (torch.Tensor): Input tensor -> (Batch, InChannels, MelBand, T_Mel)

        Returns:
            torch.Tensor: Output tensor -> (Batch, OutChannels, MelBand, T_Mel)
        """        
        return self.convLayer(x) * torch.sigmoid(self.convLayer_gates(x))

class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple = (3,3), stride: tuple = (1,1), padding: tuple = (1,1), ps_factor: int = 2):
        """_summary_

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            kernel_size (tuple, optional): _description_. Defaults to (3,3).
            stride (tuple, optional): _description_. Defaults to (1,1).
            padding (tuple, optional): _description_. Defaults to (1,1).
            ps_factor (int, optional): _description_. Defaults to 2.
        """        
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
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): Input tensor 
            skip (torch.Tensor): Skip Tensor

        Returns:
            torch.Tensor: Output tensor
        """        
        
        # Pixel Shuffle decrease the number of channels by a factor of 4 and increse the spatial dimensions by a factor of 2
        # So x has a shape 4 times bigger than skip in the channel dimension, and 2 times smaller than skip in the spatial dimension
        x = torch.functional.F.pixel_shuffle(x, self.ps_factor)
        
        # Check if we have some difference in the spatial dimension between x and skip
        #  If we code correctly all the dimension not
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]

        # Pad skip tensor to match the spatial dimension of x
        skip = F.pad(skip, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip and x in the channel dimension
        x = torch.cat([x, skip], dim=1)
        
        return self.convLayer(x) * torch.sigmoid(self.convLayer_gates(x))
    

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList() 
        self.downs = nn.ModuleList()
        
        #### Encoder Net 2D
        self.Conv1 = DownConv(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.DownConv1 = DownConv(64, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        
        self.Conv2 = DownConv(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.DownConv2 = DownConv(128, 128, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        
        self.Conv3 = DownConv(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.DownConv3 = DownConv(256, 256, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        
        # BottleNeck 1D
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
        # Out Bottleneck
        self.out_bottleneck = DownConv(256, 256 * 4, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        
        # Decoder Net 2D
        self.UpConv1 = UpConv(512, 128 * 4, kernel_size=(3,3), stride=(1,1), padding=(1,1), ps_factor=2)
        self.UpConv2 = UpConv(256, 64 * 4, kernel_size=(3,3), stride=(1,1), padding=(1,1), ps_factor=2)
        self.UpConv3 = UpConv(128, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), ps_factor=2)
        
        # Output
        self.OutConv = nn.Conv2d(64, 1, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): Input Tensor -> (Batch, 1, MelBand, T_Mel)

        Returns:
            torch.Tensor: Output Tensor -> (Batch, 1, MelBand, T_Mel)
        """      
          
        # IN (Batch, 1, MelBand, T_Mel)
        
        s1 = self.Conv1(x) # Out: (Batch, 64, MelBand, T_Mel)
        x = self.DownConv1(s1) # Out: (Batch, 64, MelBand/2, T_Mel/2)

        s2 = self.Conv2(x) # Out: (Batch, 128, MelBand/2, T_Mel/2)
        x = self.DownConv2(s2) # Out: (Batch, 128, MelBand/4, T_Mel/4)
        
        s3 = self.Conv3(x) # Out: (Batch, 256, MelBand/4, T_Mel/4)
        x = self.DownConv3(s3) # Out: (Batch, 256, MelBand/8, T_Mel/8)

        b, c, h, w = x.size()
        x = x.view(b, c*h, -1) # Out (Batch, 256 * (MelBand/8), T_Mel/8)
        in_bottleneck = self.in_bottleneck(x) # Out (Batch, 256, T_Mel/8)
        bottleneck = self.bottleneck(in_bottleneck) # Out (Batch, 256, T_Mel/8)
        end_1d_bottleneck = self.out_1d(bottleneck) # Out (Batch, 256 * (MelBand/8), T_Mel/8)
        end_bottleneck = self.out_bottleneck(end_1d_bottleneck.view(b, c, h, -1)) # Out (Batch, 1024, MelBand/8, T_Mel/8)
        x = end_bottleneck
        
        x = self.UpConv1(x, s3) # Out: (Batch, 512, MelBand/4, T_Mel/4)
        
        x = self.UpConv2(x, s2) # Out: (Batch, 256, MelBand/2, T_Mel/2)
        
        x = self.UpConv3(x, s1) # Out: (Batch, 64, MelBand, T_Mel)

        x = self.OutConv(x) # Out: (Batch, 1, MelBand, T_Mel)

        return x