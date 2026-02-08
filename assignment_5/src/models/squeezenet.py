import torch
import torch.nn as nn
from models.modules import FireModule

class SqueezeNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 96, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(96)
        self.fire2 = FireModule(96, 16, 64, 64)
        self.fire3 = FireModule(128, 16, 64, 64)
        self.fire4 = FireModule(128, 32, 128, 128)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.fire2(out)
        out = self.fire3(out)
        out = self.fire4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
