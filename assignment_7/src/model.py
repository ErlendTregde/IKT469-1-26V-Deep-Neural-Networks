import sys
import types
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Workaround: selective_scan_cuda.so has a symbol mismatch with the installed
# PyTorch (undefined _ZN3c104cuda29c10_cuda_check_implementationE...).
# Pre-stub the broken extension so the import doesn't crash, then redirect
# Mamba to the pure-PyTorch reference scan that mamba_ssm ships alongside it.
# ---------------------------------------------------------------------------
if "selective_scan_cuda" not in sys.modules:
    sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")

# Load the interface module (safe now that the stub is registered).
import mamba_ssm.ops.selective_scan_interface as _si

# Pull out the pure-PyTorch reference implementations.
_ref_scan = _si.selective_scan_ref
_ref_inner = _si.mamba_inner_ref

# Patch at the interface level (affects anything that imports the module).
_si.selective_scan_fn = _ref_scan
_si.mamba_inner_fn = _ref_inner

# Also patch the names already bound inside mamba_simple (it uses
# `from ... import selective_scan_fn` so we must fix the module namespace).
import mamba_ssm.modules.mamba_simple as _ms

_ms.selective_scan_fn = _ref_scan
_ms.mamba_inner_fn = _ref_inner

# Now a safe import of Mamba using the reference path.
from mamba_ssm.modules.mamba_simple import Mamba  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Token embeddings — patch-based linear projection
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Splits an image into non-overlapping patches and projects each to d_model."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, num_patches, d_model)
        x = self.proj(x)
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W).transpose(1, 2)


# ---------------------------------------------------------------------------
# 2. SSM layer — thin wrapper: LayerNorm + Mamba() + residual
# ---------------------------------------------------------------------------
class MambaLayer(nn.Module):
    """Pre-norm residual wrapper around mamba_ssm.Mamba."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # mamba_ssm.Mamba handles in_proj, depthwise conv, SSM scan, out_proj internally
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm + residual: x + Mamba(LN(x))
        return x + self.mamba(self.norm(x))


# ---------------------------------------------------------------------------
# 3. Full classifier: PatchEmbed → SSM layers → mean pool → Linear head
# ---------------------------------------------------------------------------
class MambaClassifier(nn.Module):
    """
    SSM-only baseline.

    Pipeline
    --------
    1. Token embeddings  (PatchEmbed)
    2. SSM               (stack of MambaLayer, each wrapping mamba_ssm.Mamba)
    3. Mean pool         (average over sequence dimension)
    4. Classifier        (linear head)
    """

    def __init__(
        self,
        num_classes: int,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        num_layers: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        # 1. Token embeddings
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, d_model)
        # 2. SSM (stacked Mamba layers)
        self.layers = nn.ModuleList(
            [MambaLayer(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
             for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        # 4. Classifier
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Token embeddings
        x = self.patch_embed(x)           # (B, num_patches, d_model)
        # 2. SSM
        for layer in self.layers:
            x = layer(x)                  # (B, num_patches, d_model)
        # 3. Mean pool
        x = self.norm(x).mean(dim=1)      # (B, d_model)
        # 4. Classifier
        return self.classifier(x)         # (B, num_classes)
