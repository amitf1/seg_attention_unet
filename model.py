import torch
import torch.nn as nn
from torch.nn import init


def deep_supervision_loss(preds, label, base_loss_function):

    if preds.dim() > label.dim():
        preds = torch.unbind(preds, dim=1)
    else:
        return base_loss_function(preds, label)
    n_preds = len(preds)
    weights = 2**torch.arange(n_preds - 1, -1, -1) # for n_preds=3 [2^2, 2^1, 2^0]
    weights = weights / weights.sum() 
    # for n_preds=3 3 layers to account in the loss the weights will be: [4/7 , 2/7, 1/7] each layer is twice as important as it's lower neigbour
   
    loss = None
    for weight, pred in zip(weights, preds):
        if loss is None:
            loss = base_loss_function(pred, label)*weight
        else:
            loss += base_loss_function(pred, label)*weight
    if loss < 0:
        raise(ValueError(f"loss should be non negative but got loss={loss}"))

    return loss

class DeepSuperHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int):
        """
        Deep supervision output layer, takes one of the unet levels' outputs and applies conv + upsample to the final image spatial dim
        apply before before stacking the layers and calculating deep suprvision loss
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels of the final output of the net 
        scale_factor (int): fator to scale up the Spatial dim to match the outputs spatial dim, should be 2**n where n is the inputs number of levels below the output final layer  
        """
        super(DeepSuperHead, self).__init__()
        self.dsh = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'))

    def forward(self, input):
        return self.dsh(input).unsqueeze(1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear')


    def forward(self, inputs1, inputs2):
        inputs2 = self.up(inputs2)
        return torch.cat([inputs1, inputs2], 1)
    
class AtentionBlock(nn.Module):
    """
    Grid Attention Block - inputs are x and g (where g denoting gating signal) feature maps from two sequential levels of a unet architecture with a factor of two in saptial dimension and n_channels 
     (b, c, h, w, d) and (b, 2c, h//2, w//2, d//2) the output is x multiplied by the attention weight for x with dim (b, c, h, w, d)
    
    """
    def __init__(self, in_channels, gating_channels, inter_channels, conv_mapping=False):
        """
        in_channels: number of channels in x - the upper layer input
        gating_channels: number of channels of the query/gating signal from the lower level of unet
        inter_channels: num of channels to map x and gatong signal
        conv_mapping (bool): if True use convolution as a weighted aggregation of x and g instead of regular sum 
                             use False for the originl implementation
        """
        super(AtentionBlock, self).__init__()
        self.conv_mapping = conv_mapping
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels
        self.W = nn.Sequential(nn.Conv3d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                               nn.BatchNorm3d(self.in_channels),
        )
        # according to the paper no bias on x, bias exists in g and psi
        # W_x: stride of 2 to reduce the saptial dim by a factor of 2 to match the upper layer's spatial dim to the gating signal's spatial dim
        self.W_x = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels, 
                             kernel_size=1, stride=2, padding=0, bias=False)
        self.W_g = nn.Conv3d(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        if self.conv_mapping:
            self.aggragate = nn.Conv3d(in_channels=2*self.inter_channels, out_channels=self.inter_channels, 
                             kernel_size=1, stride=1, padding=0, bias=False)
        self.sigma1 = nn.ReLU()
        self.psi = nn.Conv3d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigma2 = nn.Sigmoid()
        self.resampler = nn.Upsample(scale_factor=2, mode='trilinear')

    def forward(self, x, g):
        x1 = self.W_x(x) # strided conv to reduce x size to match the gating signal
        g1 = self.W_g(g)
        x1_g1 = x1 + g1 if not self.conv_mapping else self.aggragate(torch.cat((x1, g1), dim=1)) 
        q_att = self.sigma1(x1_g1)
        q_att = self.psi(q_att)
        alpha = self.sigma2(q_att)
        alpha = self.resampler(alpha) # resmaple spatial dim to match x
        x_out = alpha.expand_as(x) * x # expand attention weight's channels to match x
        return x_out, alpha

class AttentionUNET(nn.Module):
    """
    Implementation of https://arxiv.org/pdf/1804.03999.pdf
    """
    def __init__(self, in_channels: int, out_channels: int, n_deep_supervision: int, conv_mapping: bool=False):
        """
            in_channels: number of input channels to the net
            out_cahnnels: number of out cahnnels in the output for the net
            n_deep_supervision: num layers to stack at the output of the network for the deep supervision loss.
              allowed range (1-3) the input will be clipped if it is out of the range
            conv_mapping (bool): if True use convolution as a weighted aggregation of x and g instead of regular sum in each attention gate
                                 use False for the originl implementation
        """
        super(AttentionUNET, self).__init__()
        
        self.n_deep_supervision = torch.clip(torch.tensor(n_deep_supervision), min=1, max=3).item()
        self.conv_mapping = conv_mapping
        ## down path ##
        self.conv1 = ConvBlock(in_channels, 64)
        self.downsample1 = nn.MaxPool3d(kernel_size=2,stride=2)

        self.conv2 = ConvBlock(64, 128)
        self.downsample2 = nn.MaxPool3d(kernel_size=2,stride=2)

        self.conv3 = ConvBlock(128, 256)
        self.downsample3 = nn.MaxPool2d(kernel_size=2,stride=2)

        ## bottleneck ##
        self.conv4 = ConvBlock(256, 512)
        
        ## up path ##
        self.attn1 = AtentionBlock(256, 512, 256, conv_mapping) # out 256
        self.upsample1 = UpBlock() # out 512+256
        self.conv5 = ConvBlock(512+256, 256)
        
        self.attn2 = AtentionBlock(128, 256, 128, conv_mapping) # out 128
        self.upsample2 = UpBlock() # out 256+128
        self.conv6 = ConvBlock(256+128, 128)

        
        self.attn3 = AtentionBlock(64, 128, 64, conv_mapping) #out 64
        self.upsample3 = UpBlock() # out 128+64
        self.conv7 = ConvBlock(128+64, out_channels)

        self.dsh1 = DeepSuperHead(128, out_channels, 2)
        self.dsh2 = DeepSuperHead(256, out_channels, 4)

        self.apply(weights_init)

    def forward(self, x):
        ## down path ##
        x1 = self.conv1(x)
        x1_d = self.downsample1(x1)
        x2 = self.conv2(x1_d)
        x2_d = self.downsample1(x2)
        x3 = self.conv3(x2_d)
        x3_d = self.downsample1(x3)
        ## bottleneck ##
        x4 = self.conv4(x3_d)
        ## up path ##
        attn1, _ = self.attn1(x3, x4)
        x5 = self.conv5(self.upsample1(attn1, x4))
        attn2, _ = self.attn2(x2, x5)
        x6 = self.conv6(self.upsample2(attn2, x5))
        attn3, _ = self.attn3(x1, x6)
        x7 = self.conv7(self.upsample3(attn3, x6))
        if self.training: # deep-supervision output
            out = torch.cat([x7.unsqueeze(1), self.dsh1(x6), self.dsh2(x5)][:self.n_deep_supervision], 1)
        else: # only the final output for inference/val/test
            out = x7
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and getattr(m, 'weight') is not None:
        if classname.find('Conv') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and getattr(m, 'bias') is not None:
        init.constant_(m.bias.data, 0.0)


class DeeperAttentionUNET(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_deep_supervision: int, conv_mapping: bool=False):
        """
            in_channels: number of input channels to the net
            out_cahnnels: number of out cahnnels in the output for the net
            n_deep_supervision: num layers to stack at the output of the network for the deep supervision loss.
              allowed range (1-3) the input will be clipped if it is out of the range
            conv_mapping (bool): if True use convolution as a weighted aggregation of x and g instead of regular sum in each attention gate
                                 use False for the originl implementation
        """
        super(DeeperAttentionUNET, self).__init__()
        self.n_deep_supervision = torch.clip(torch.tensor(n_deep_supervision), min=1, max=3).item()
        self.conv_mapping = conv_mapping

        self.conv1 = ConvBlock(in_channels, 64)
        self.downsample1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = ConvBlock(64, 128)
        self.downsample2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(128, 256)
        self.downsample3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = ConvBlock(256, 512)
        self.downsample4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = ConvBlock(512, 1024)

        self.attn1 = AtentionBlock(512, 1024, 512, conv_mapping)  # out 512

        self.upsample1 = UpBlock()  # out 1024+512
        self.conv6 = ConvBlock(1024 + 512, 512)

        self.attn2 = AtentionBlock(256, 512, 256, conv_mapping)  # out 256
        self.upsample2 = UpBlock()  # out 512+256
        self.conv7 = ConvBlock(512 + 256, 256)

        self.attn3 = AtentionBlock(128, 256, 128, conv_mapping)  # out 128
        self.upsample3 = UpBlock()  # out 256+128
        self.conv8 = ConvBlock(256 + 128, 128)

        self.attn4 = AtentionBlock(64, 128, 64, conv_mapping)  # out 64
        self.upsample4 = UpBlock()  # out 128+64
        self.conv9 = ConvBlock(128 + 64, out_channels)

        self.dsh1 = DeepSuperHead(128, out_channels, 2)
        self.dsh2 = DeepSuperHead(256, out_channels, 4)

        self.apply(weights_init)

    def forward(self, x):
        ## down path ##
        x1 = self.conv1(x)
        x1_d = self.downsample1(x1)
        x2 = self.conv2(x1_d)
        x2_d = self.downsample1(x2)
        x3 = self.conv3(x2_d)
        x3_d = self.downsample1(x3)
        x4 = self.conv4(x3_d)
        x4_d = self.downsample1(x4)

        ## bottleneck ##
        x5 = self.conv5(x4_d)

        ## up path ##
        attn1, _ = self.attn1(x4, x5)
        x6 = self.conv6(self.upsample1(attn1, x5))
        attn2, _ = self.attn2(x3, x6)
        x7 = self.conv7(self.upsample2(attn2, x6))
        attn3, _ = self.attn3(x2, x7)
        x8 = self.conv8(self.upsample3(attn3, x7))
        attn4, _ = self.attn4(x1, x8)
        x9 = self.conv9(self.upsample4(attn4, x8))
        if self.training: # deep-supervision output
            out = torch.cat([x9.unsqueeze(1), self.dsh1(x8), self.dsh2(x7)][:self.n_deep_supervision], 1)
        else: # only the final output for inference/val/test
            out = x9
        return out

