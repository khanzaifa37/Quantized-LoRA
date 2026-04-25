from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


# Approximate NF4 levels used for blockwise normal-float quantization.
NF4_CODEBOOK = torch.tensor(
    [
        -1.0,
        -0.6961928,
        -0.52507305,
        -0.3949175,
        -0.28444138,
        -0.18477343,
        -0.09105004,
        0.0,
        0.0795803,
        0.1609302,
        0.2461123,
        0.33791524,
        0.44070983,
        0.562617,
        0.72295684,
        1.0,
    ],
    dtype=torch.float32,
)


@dataclass
class NF4Tensor:
    codes: torch.ByteTensor
    scales: torch.Tensor
    original_shape: Tuple[int, ...]
    block_size: int = 64

    def dequantize(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        codebook = NF4_CODEBOOK.to(device=device, dtype=torch.float32)
        codes = self.codes.to(device=device, dtype=torch.long)
        scales = self.scales.to(device=device, dtype=torch.float32)

        values = codebook[codes]
        flat = values.view(scales.numel(), self.block_size) * scales.unsqueeze(-1)
        total_values = int(torch.tensor(self.original_shape).prod().item())
        return flat.reshape(-1)[:total_values].view(self.original_shape).to(dtype=dtype)


def dequantize_nf4(
    codes: torch.Tensor,
    scales: torch.Tensor,
    original_shape: Tuple[int, ...],
    *,
    block_size: int = 64,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    codebook = NF4_CODEBOOK.to(device=codes.device, dtype=torch.float32)
    values = codebook[codes.long()]
    flat = values.view(scales.numel(), block_size) * scales.to(torch.float32).unsqueeze(-1)
    total_values = int(torch.tensor(original_shape).prod().item())
    return flat.reshape(-1)[:total_values].view(original_shape).to(dtype=dtype)


def _pad_to_block_size(flat_weight: torch.Tensor, block_size: int) -> torch.Tensor:
    remainder = flat_weight.numel() % block_size
    if remainder == 0:
        return flat_weight
    padding = block_size - remainder
    return torch.nn.functional.pad(flat_weight, (0, padding))


def quantize_nf4(weight: torch.Tensor, block_size: int = 64) -> NF4Tensor:
    if weight.ndim < 2:
        raise ValueError("NF4 quantization expects at least a matrix-shaped tensor.")

    flat = weight.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    padded = _pad_to_block_size(flat, block_size)
    blocks = padded.view(-1, block_size)

    scales = blocks.abs().amax(dim=1).clamp_min(1e-8)
    normalized = blocks / scales.unsqueeze(-1)

    codebook = NF4_CODEBOOK.view(1, 1, -1)
    distances = (normalized.unsqueeze(-1) - codebook).abs()
    codes = distances.argmin(dim=-1).to(torch.uint8)

    return NF4Tensor(
        codes=codes.contiguous(),
        scales=scales.contiguous(),
        original_shape=tuple(weight.shape),
        block_size=block_size,
    )
