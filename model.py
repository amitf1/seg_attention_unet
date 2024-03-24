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
    interpolated_label = label
    for weight, pred in zip(weights, preds):
        if loss is None:
            loss = base_loss_function(pred, interpolated_label)*weight
        else:
            loss += base_loss_function(pred, interpolated_label)*weight
        interpolated_label = F.interpolate(interpolated_label, scale_factor=0.5)
    if loss < 0:
        raise(ValueError(f"loss should be non negative but got loss={loss}"))

    return loss

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
        q_att = self.sigma1(x1 + g1, inplace=True)
        q_att = self.psi(q_att)
        alpha = self.sigma2(q_att)
        alpha = self.resampler(alpha)
        x_out = alpha.expand_as(x) * x
        return x_out, alpha

class AttentionUNET(nn.Module):
    def __init__(self, in_channels, out_channel, n_deep_suprvision):
        super(AttentionUNET, self).__init__()
        self.n_deep_suprvision = n_deep_suprvision
        self.conv1 = ConvBlock(in_channels, 64)
        self.downsample1 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2 = ConvBlock(64, 128)
        self.downsample2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3 = ConvBlock(128, 256)
        self.downsample3 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv4 = ConvBlock(256, 512)
        
        self.attn1 = AtentionBlock(512, 256, 128)
        self.upsample1 = UpBlock()
        self.conv5 = ConvBlock(256, 128)
        
        self.attn2 = AtentionBlock(256, 128, 64)
        self.upsample2 = UpBlock()
        self.conv6 = ConvBlock(128, 64)

        self.upsample3 = UpBlock()
        self.attn3 = AtentionBlock(128, 64, 32)
        self.conv7 = ConvBlock(128, out_channel)

        self.apply(weights_init)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_d = self.downsample1(x1)
        x2 = self.conv2(x1_d)
        x2_d = self.downsample1(x2)
        x3 = self.conv3(x2_d)
        x3_d = self.downsample1(x3)
        x4 = self.conv4(x3_d)
        attn1 = self.attn1(x3, x4)
        x5 = self.conv5(self.upsample1(attn1, x4))
        attn2 = self.attn2(x2, x5)
        x6 = self.conv6(self.upsample2(attn2, x5))
        attn3 = self.attn3(x1, x6)
        x7 = self.conv7(self.upsample3(attn3, x6))
        return torch.cat([x7, x6, x5][:self.n_deep_suprvision], 1)

def weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)