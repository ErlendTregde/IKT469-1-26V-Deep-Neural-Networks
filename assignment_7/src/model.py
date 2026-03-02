import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W).transpose(1, 2)


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = max(1, math.ceil(d_model / 16))

        A_log = torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float32)
            .unsqueeze(0)
            .expand(d_model, -1)
        )
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(d_model))

        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        A = -torch.exp(self.A_log.float())

        proj = self.x_proj(x)
        dt, B_in, C_in = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        h = torch.zeros(B, self.d_model, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            dA = torch.exp(dt[:, i].unsqueeze(-1) * A.unsqueeze(0))
            dB = dt[:, i].unsqueeze(-1) * B_in[:, i].unsqueeze(1)
            h = h * dA + dB * x[:, i].unsqueeze(-1)
            ys.append((h * C_in[:, i].unsqueeze(1)).sum(-1))

        y = torch.stack(ys, dim=1)
        return y + x * self.D


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, expand: int):
        super().__init__()
        d_inner = expand * d_model
        kernel_size = 4

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=kernel_size, padding=kernel_size - 1, groups=d_inner)
        self.ssm = SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        L = x.shape[1]

        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_branch = F.silu(x_branch)

        y = self.ssm(x_branch)
        y = y * F.silu(z)

        return self.out_proj(y) + residual


class MambaClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        d_model: int = 128,
        d_state: int = 16,
        num_layers: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, expand) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(dim=1)
        return self.classifier(x)
