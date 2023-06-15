import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class residualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(residualBlock, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False))
        else:
          self.skip = None

        self.block = nn.Sequential(nn.BatchNorm2d(in_channels, track_running_stats=False),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                   nn.BatchNorm2d(out_channels, track_running_stats=False),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels, 3, padding=1)
                                   )   

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out
    
class EncoderConv(nn.Module):
    def __init__(self, latents = 64, hw = 32):
        super(EncoderConv, self).__init__()
        
        self.latents = latents
        self.c = 4
        
        self.size = self.c * np.array([2,4,8,16,32], dtype = np.intc)
        
        self.maxpool = nn.MaxPool2d(2)
        
        self.dconv_down1 = residualBlock(1, self.size[0])
        self.dconv_down2 = residualBlock(self.size[0], self.size[1])
        self.dconv_down3 = residualBlock(self.size[1], self.size[2])
        self.dconv_down4 = residualBlock(self.size[2], self.size[3])
        self.dconv_down5 = residualBlock(self.size[3], self.size[4])
        self.dconv_down6 = residualBlock(self.size[4], self.size[4])
        
        self.fc_mu = nn.Linear(in_features=self.size[4]*hw*hw, out_features=self.latents)
        self.fc_logvar = nn.Linear(in_features=self.size[4]*hw*hw, out_features=self.latents)

    def forward(self, x):
        x = self.dconv_down1(x)
        x = self.maxpool(x)

        x = self.dconv_down2(x)
        x = self.maxpool(x)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        
        conv5 = self.dconv_down5(x)
        x = self.maxpool(conv5)
        
        conv6 = self.dconv_down6(x)
        
        x = conv6.view(conv6.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
                
        return x_mu, x_logvar

    
class SiameseReg(nn.Module):
    def __init__(self, config):
        super(SiameseReg, self).__init__()
        
        self.config = config
        hw = config['inputsize'] // 32  
        self.z = config['latents']

        self.encoder = EncoderConv(latents = self.z, hw = hw)
        self.fc1 = nn.Linear(in_features=self.z*2, out_features=self.z*2)
        self.fc_out = nn.Linear(in_features=self.z*2, out_features=6)


    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) 
    
        
    def forward(self, x1, x2):
        self.mu_1, self.log_var_1 = self.encoder(x1)
        self.mu_2, self.log_var_2 = self.encoder(x2)

        if self.training and self.config['sampling']:
            z1 = self.sampling(self.mu_1, self.log_var_1)
            z2 = self.sampling(self.mu_2, self.log_var_2)
        else:
            z1 = self.mu_1
            z2 = self.mu_2
        
        z = torch.cat((z1, z2), dim = 1)
        
        z = F.relu(self.fc1(z))
        out = self.fc_out(z)
        
        return out