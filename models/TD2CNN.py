import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from einops import rearrange, repeat


class TD2CNN(nn.Module):
    def cal(self, inn, k, p, s):
        return int((inn + 2 * p - (k - 1) - 1) / s + 1)

    def __init__(self, c1, c2, classes,patchsize):
        super(TD2CNN, self).__init__()
        # c1+c2
        dim = c1 + c2
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv2d2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv2d3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv2d4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Linear(patchsize**2, classes)

    def forward(self, x1, x2):
        _, _, pp = x1.shape
        x1 = rearrange(x1, 'b c (h w) -> b c h w', h=int(pp ** 0.5))
        x2 = rearrange(x2, 'b c (h w) -> b c h w', h=int(pp ** 0.5))
        x1 = torch.concat((x1, x2), dim=1)
        x1 = self.conv2d1(x1)
        x1 = self.conv2d2(x1)
        x1 = self.conv2d3(x1)
        x1 = self.conv2d4(x1)
        x1 = rearrange(x1, 'b c h w -> b (c h w)')
        x1 = self.fc(x1)
        return x1

if __name__ == '__main__':
    model = TD2CNN(30,1,7,11)
    model.eval()
    print(model)
    input1 = torch.randn(64, 30, 121)
    input2 = torch.randn(64, 1, 121)
    x = model(input1, input2)
    print(x.size())
    # summary(model, ((64, 1, 30, 11, 11), (64, 1, 11, 11)))