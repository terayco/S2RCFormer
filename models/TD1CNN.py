import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from einops import rearrange, repeat


class TD1CNN(nn.Module):
    def cal(self, inn, k, p, s):
        return int((inn + 2 * p - (k - 1) - 1) / s + 1)

    def __init__(self, c1, c2, classes,patchsize):
        super(TD1CNN, self).__init__()
        # c1+c2
        dim = c1 + c2
        self.conv1d1 = nn.Sequential(
            nn.Conv1d(in_channels=patchsize**2, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv1d2 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv1d3 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv1d4 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(dim, classes)

    def forward(self, x1, x2):
        _, _, pp = x1.shape
        x1 = rearrange(x1, 'b c n -> b n c')
        x2 = rearrange(x2, 'b c n -> b n c')
        x1 = torch.concat((x1, x2), dim=2)
        x1 = self.conv1d1(x1)
        x1 = self.conv1d2(x1)
        x1 = self.conv1d3(x1)
        x1 = self.conv1d4(x1)
        x1 = rearrange(x1, 'b n c -> b (n c)')
        x1 = self.fc(x1)
        return x1

if __name__ == '__main__':
    model = TD1CNN(30,1,7,11)
    model.eval()
    print(model)
    input1 = torch.randn(64, 30, 121)
    input2 = torch.randn(64, 1, 121)
    x = model(input1, input2)
    print(x.size())
    # summary(model, ((64, 1, 30, 11, 11), (64, 1, 11, 11)))