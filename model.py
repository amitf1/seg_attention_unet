import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

def deep_supervision_loss(preds, label, base_loss_function):

    if preds.dim() > label.dim():
        preds = torch.unbind(preds, dim=1)
    else:
        return base_loss_function(preds, label)
    n_preds = len(preds)
    weights = 2**torch.arange(n_preds - 1, -1, -1)
    weights = weights / weights.sum() # for n_preds=3: [4/7 , 2/7, 1/7] each layer is twice as important as it's lower neigbour
   
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
    def __init__(self, in_size, out_size, scale_factor):
        super(DeepSuperHead, self).__init__()
        self.dsh = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
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
        # offset = outputs2.size()[2] - inputs1.size()[2]
        # padding = 2 * [offset // 2, offset // 2, 0]
        # outputs1 = F.pad(inputs1, padding)
        return torch.cat([inputs1, inputs2], 1)
    
class AtentionBlock(nn.Module):
    """
    Grid Attention Block - inputs are x and g (where g denoting gating signal) feature maps from two sequential levels of a unet architecture with a factor of two in saptial dimension and n_channels 
     (b, c, h, w, d) and (b, 2c, h//2, w//2, d//2) the output is the attention weight for x with dim (b, c, h, w, d)
    
    """
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AtentionBlock, self).__init__()

        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels
        self.W = nn.Sequential(nn.Conv3d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                               nn.BatchNorm3d(self.in_channels),
        )
        # according to the paper no bias on x, bias exists in g and psi
        self.W_x = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=2, padding=0, bias=False)
        self.W_g = nn.Conv3d(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv3d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigma1 = nn.ReLU()
        self.sigma2 = nn.Sigmoid()
        self.resampler = nn.Upsample(scale_factor=2, mode='trilinear')

    def forward(self, x, g):
        x1 = self.W_x(x)
        g1 = self.W_g(g)
        q_att = self.sigma1(x1 + g1)
        q_att = self.psi(q_att)
        alpha = self.sigma2(q_att)
        alpha = self.resampler(alpha)
        x_out = alpha.expand_as(x) * x
        return x_out, alpha

class AttentionUNET(nn.Module):
    def __init__(self, in_channels, out_channels, n_deep_suprvision):
        super(AttentionUNET, self).__init__()
        self.n_deep_suprvision = n_deep_suprvision
        self.conv1 = ConvBlock(in_channels, 64)
        self.downsample1 = nn.MaxPool3d(kernel_size=2,stride=2)

        self.conv2 = ConvBlock(64, 128)
        self.downsample2 = nn.MaxPool3d(kernel_size=2,stride=2)

        self.conv3 = ConvBlock(128, 256)
        self.downsample3 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv4 = ConvBlock(256, 512)
        
        self.attn1 = AtentionBlock(256, 512, 256) # out 256
        self.upsample1 = UpBlock() # out 512+256
        self.conv5 = ConvBlock(512+256, 256)
        
        self.attn2 = AtentionBlock(128, 256, 128) # out 128
        self.upsample2 = UpBlock() # out 256+128
        self.conv6 = ConvBlock(256+128, 128)

        
        self.attn3 = AtentionBlock(64, 128, 64) #out 64
        self.upsample3 = UpBlock() # out 128+64
        self.conv7 = ConvBlock(128+64, out_channels)

        self.dsh1 = DeepSuperHead(128, out_channels, 2)
        self.dsh2 = DeepSuperHead(256, out_channels, 4)

        self.apply(weights_init)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_d = self.downsample1(x1)
        x2 = self.conv2(x1_d)
        x2_d = self.downsample1(x2)
        x3 = self.conv3(x2_d)
        x3_d = self.downsample1(x3)
        x4 = self.conv4(x3_d)
        attn1, _ = self.attn1(x3, x4)
        x5 = self.conv5(self.upsample1(attn1, x4))
        attn2, _ = self.attn2(x2, x5)
        x6 = self.conv6(self.upsample2(attn2, x5))
        attn3, _ = self.attn3(x1, x6)
        x7 = self.conv7(self.upsample3(attn3, x6))
        if self.training:
            out = torch.cat([x7.unsqueeze(1), self.dsh1(x6), self.dsh2(x5)][:self.n_deep_suprvision], 1)
        else:
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
    def __init__(self, in_channels, out_channels, n_deep_suprvision):
        super(DeeperAttentionUNET, self).__init__()
        self.n_deep_suprvision = n_deep_suprvision
        self.conv1 = ConvBlock(in_channels, 64)
        self.downsample1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = ConvBlock(64, 128)
        self.downsample2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(128, 256)
        self.downsample3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = ConvBlock(256, 512)
        self.downsample4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = ConvBlock(512, 1024)

        self.attn1 = AtentionBlock(512, 1024, 512)  # out 512

        self.upsample1 = UpBlock()  # out 1024+512
        self.conv6 = ConvBlock(1024 + 512, 512)

        self.attn2 = AtentionBlock(256, 512, 256)  # out 256
        self.upsample2 = UpBlock()  # out 512+256
        self.conv7 = ConvBlock(512 + 256, 256)

        self.attn3 = AtentionBlock(128, 256, 128)  # out 128
        self.upsample3 = UpBlock()  # out 256+128
        self.conv8 = ConvBlock(256 + 128, 128)

        self.attn4 = AtentionBlock(64, 128, 64)  # out 64
        self.upsample4 = UpBlock()  # out 128+64
        self.conv9 = ConvBlock(128 + 64, out_channels)

        self.dsh1 = DeepSuperHead(128, out_channels, 2)
        self.dsh2 = DeepSuperHead(256, out_channels, 4)

        self.apply(weights_init)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_d = self.downsample1(x1)
        x2 = self.conv2(x1_d)
        x2_d = self.downsample1(x2)
        x3 = self.conv3(x2_d)
        x3_d = self.downsample1(x3)
        x4 = self.conv4(x3_d)
        x4_d = self.downsample1(x4)
        x5 = self.conv5(x4_d)
        attn1, _ = self.attn1(x4, x5)
        x6 = self.conv6(self.upsample1(attn1, x5))
        attn2, _ = self.attn2(x3, x6)
        x7 = self.conv7(self.upsample2(attn2, x6))
        attn3, _ = self.attn3(x2, x7)
        x8 = self.conv8(self.upsample3(attn3, x7))
        attn4, _ = self.attn4(x1, x8)
        x9 = self.conv9(self.upsample4(attn4, x8))
        if self.training:
            out = torch.cat([x9.unsqueeze(1), self.dsh1(x8), self.dsh2(x7)][:self.n_deep_suprvision], 1)
        else:
            out = x9
        return out

