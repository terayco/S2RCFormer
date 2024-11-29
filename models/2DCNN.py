import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from einops import rearrange, repeat


class TDCNN(nn.Module):
    def cal(self, inn, k, p, s):
        return int((inn + 2 * p - (k - 1) - 1) / s + 1)

    def __init__(self, c1, c2, classes, patch_size):
        super(TDCNN, self).__init__()
        # c1+c2
        dim = c1 + c2
        self.conv2d1 = nn.Sequential(
            nn.Conv3d(in_channels=c1, out_channels=20, kernel_size=3, stride=1, padding=(0, 0, 1)),
            nn.ReLU(),
        )
        dim = self.cal(dim, k=3, p=1, s=1)
        self.conv2d2 = nn.Sequential(
            nn.Conv3d(in_channels=20, out_channels=2, kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1)),
            nn.ReLU(),
        )
        dim = self.cal(dim, k=3, p=1, s=2)
        # p-2
        self.conv2d3 = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=35, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1)),
            nn.ReLU(),
        )
        dim = self.cal(dim, k=3, p=1, s=1)
        # p-2
        self.conv2d4 = nn.Sequential(
            nn.Conv3d(in_channels=35, out_channels=2, kernel_size=(1, 1, 2), stride=(1, 1, 2), padding=(0, 0, 1)),
            nn.ReLU(),
        )
        dim = self.cal(dim, k=2, p=1, s=2)

        dim = (patch_size - 2) ** 2 * dim * 2
        self.fc = nn.Linear(dim, classes)

    def forward(self, x1, x2):
        _, _, pp = x1.shape
        x1 = rearrange(x1, 'b c (h w) -> b 1 h w c', h=int(pp ** 0.5))
        x2 = rearrange(x2, 'b c (h w) -> b 1 h w c', h=int(pp ** 0.5))
        x1 = torch.concat((x1, x2), dim=4)
        x1 = self.conv3d1(x1)
        x1 = self.conv3d2(x1)
        x1 = self.conv3d3(x1)
        x1 = self.conv3d4(x1)
        x1 = rearrange(x1, 'b n c h w -> b (n c h w)')
        x1 = self.fc(x1)
        return x1
