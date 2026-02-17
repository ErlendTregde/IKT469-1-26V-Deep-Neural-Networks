import torch
import torch.nn as nn
from .modules import InceptionBlock

class InceptionNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(InceptionNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.inception1 = InceptionBlock(64, 32, 48, 64, 8, 16, 16)
        self.inception2 = InceptionBlock(128, 64, 64, 96, 16, 48, 32)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(240, num_classes)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.inception1(out)
        out = self.inception2(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
