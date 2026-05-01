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


def dequantize_row_chunk(
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    in_features: int,
    block_size: int,
    row_start: int,
    row_end: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    blocks_per_row = in_features // block_size
    block_start = row_start * blocks_per_row
    block_end = row_end * blocks_per_row

    chunk_codes = codes[block_start:block_end]
    chunk_scales = scales[block_start:block_end].to(compute_dtype)
    codebook = NF4_CODEBOOK.to(device=codes.device, dtype=compute_dtype)
    chunk_values = codebook[chunk_codes.long()]
    chunk_weight = chunk_values * chunk_scales.unsqueeze(-1)
    return chunk_weight.reshape(row_end - row_start, in_features)


class _ChunkedNF4MatMul(torch.autograd.Function):
    """
    Streaming x @ W.T where W is stored as NF4 codes + per-block scales.

    Forward and backward both rebuild W one row-chunk at a time, so the full
    fp16 weight never lives in memory and is not retained in the autograd graph.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        codes: torch.Tensor,
        scales: torch.Tensor,
        out_features: int,
        in_features: int,
        block_size: int,
        chunk_size: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, codes, scales)
        ctx.out_features = out_features
        ctx.in_features = in_features
        ctx.block_size = block_size
        ctx.chunk_size = chunk_size

        compute_dtype = (
            x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16
        )
        out_shape = x.shape[:-1] + (out_features,)
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        x_compute = x.to(compute_dtype)

        for row_start in range(0, out_features, chunk_size):
            row_end = min(row_start + chunk_size, out_features)
            chunk_weight = dequantize_row_chunk(
                codes,
                scales,
                in_features=in_features,
                block_size=block_size,
                row_start=row_start,
                row_end=row_end,
                compute_dtype=compute_dtype,
            )
            out[..., row_start:row_end] = torch.nn.functional.linear(x_compute, chunk_weight).to(x.dtype)
            del chunk_weight

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, codes, scales = ctx.saved_tensors
        out_features = ctx.out_features
        in_features = ctx.in_features
        block_size = ctx.block_size
        chunk_size = ctx.chunk_size

        compute_dtype = (
            x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16
        )
        grad_x = torch.zeros_like(x, dtype=compute_dtype)
        grad_output_compute = grad_output.to(compute_dtype)

        for row_start in range(0, out_features, chunk_size):
            row_end = min(row_start + chunk_size, out_features)
            chunk_weight = dequantize_row_chunk(
                codes,
                scales,
                in_features=in_features,
                block_size=block_size,
                row_start=row_start,
                row_end=row_end,
                compute_dtype=compute_dtype,
            )
            grad_chunk = grad_output_compute[..., row_start:row_end]
            grad_x.add_(grad_chunk @ chunk_weight)
            del chunk_weight

        return grad_x.to(x.dtype), None, None, None, None, None, None


def chunked_nf4_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    original_shape: Tuple[int, ...],
    bias: torch.Tensor | None,
    *,
    block_size: int = 64,
    chunk_size: int = 128,
) -> torch.Tensor:
    out_features, in_features = original_shape

    if in_features % block_size != 0:
        weight = dequantize_nf4(
            codes,
            scales,
            original_shape,
            block_size=block_size,
            dtype=x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16,
        )
        out = torch.nn.functional.linear(x, weight, bias)
        return out

    out = _ChunkedNF4MatMul.apply(x, codes, scales, out_features, in_features, block_size, chunk_size)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


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
