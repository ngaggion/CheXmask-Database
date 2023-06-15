import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal


class DeformableNet(nn.Module):
    def __init__(self, input_size, batchnorm=True):
        super().__init__()
        self.unet = UNet(in_channels=2, batchnorm=batchnorm)
        self.grid_sampler = GridSampler(input_size)

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        flow = self.unet(x)
        reg_img = self.grid_sampler(src, flow)

        return reg_img, flow


class GridSampler(nn.Module):
    """ https://github.com/voxelmorph/voxelmorph """

    def __init__(self, input_size, mode='bilinear'):
        super().__init__()

        self.input_size = input_size
        self.mode = mode

        vectors = [torch.arange(0, s) for s in self.input_size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow

        for i in range(len(self.input_size)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (self.input_size[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True)


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels=None, batchnorm=True):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
            nn.ELU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32) if batchnorm else nn.Identity(),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32) if batchnorm else nn.Identity(),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
            nn.ELU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
            nn.ELU(),
            nn.Dropout(p=0.5)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32) if batchnorm else nn.Identity(),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32) if batchnorm else nn.Identity(),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32) if batchnorm else nn.Identity(),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
            nn.ELU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
            nn.ELU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
        )

        self.flow = nn.Conv2d(16, out_channels, 3, 1, 1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.flow(x)

        return x
