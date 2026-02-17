from .modules import InceptionBlock, ResidualBlock, FireModule
import torch
import torch.nn as nn


class custom_model(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(custom_model, self).__init__()

        # Stem: input_channels → 64 channels
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Inception: 64 → 128 channels (32+64+16+16)
        self.inception = InceptionBlock(64, 32, 48, 64, 8, 16, 16)

        # Skip connection: 64 → 128 to match inception output
        self.skip = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )

        # Residual: 128 → 128 channels
        self.residual = ResidualBlock(128, 128)

        # Fire: 128 → 128 channels (64+64)
        self.fire = FireModule(128, 16, 64, 64)

        # Output layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(384, num_classes)  # 128+128+128 = 384

    def forward(self, x):
        x = self.stem(x)                        # 1 → 64 channels
        identity = x                            # Save AFTER stem (64 channels)
        skip = self.skip(identity)              # Prepare skip (64 → 128)

        incep_out = self.inception(x)            # 64 → 128
        x = incep_out + skip                     # Add skip (64 → 128)

        res_out = self.residual(x)               # 128 → 128
        x = res_out + skip                       # Add skip again

        fire_out = self.fire(x)                  # 128 → 128

        # Concat all outputs
        out = torch.cat((incep_out, res_out, fire_out), dim=1)  # 384 channels

        out = self.avg_pool(out)
        out = out.flatten(1)
        out = self.fc(out)

        return out
