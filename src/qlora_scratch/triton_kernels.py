"""
Triton-fused NF4 dequant + matmul kernels for QLoRA.

The forward and backward kernels rebuild the NF4 weight one tile at a time
inside GPU registers / shared memory, feed it directly into the tensor cores,
and discard it before the next tile. The full fp16 weight never lives in
HBM and is never retained in autograd's saved tensors, so peak VRAM stays
near the NF4 footprint while throughput stays close to a regular fp16
linear.

If Triton is unavailable (no CUDA, or the package is not installed), the
autograd Function transparently falls back to the chunked Python path
defined in ``quantization.py``.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .quantization import NF4_CODEBOOK, dequantize_row_chunk

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the runtime environment
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _nf4_matmul_fwd_kernel(
        X_ptr,
        CODES_ptr,
        SCALES_ptr,
        CODEBOOK_ptr,
        Y_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_ym,
        stride_yn,
        BLOCK_SIZE: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k_local = tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_block in range(0, BLOCKS_PER_ROW):
            offs_k = k_block * BLOCK_K + offs_k_local

            x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)

            block_idx = offs_n * BLOCKS_PER_ROW + k_block
            code_ptrs = (
                CODES_ptr + block_idx[:, None] * BLOCK_SIZE + offs_k_local[None, :]
            )
            codes = tl.load(code_ptrs, mask=mask_n[:, None], other=0)
            codes_int = codes.to(tl.int32)
            w_norm = tl.load(CODEBOOK_ptr + codes_int)

            scales = tl.load(SCALES_ptr + block_idx, mask=mask_n, other=0.0)
            scales_h = scales.to(tl.float16)
            w_tile = w_norm.to(tl.float16) * scales_h[:, None]

            acc += tl.dot(x.to(tl.float16), tl.trans(w_tile))

        y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        y_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(y_ptrs, acc.to(tl.float16), mask=y_mask)

    @triton.jit
    def _nf4_matmul_bwd_x_kernel(
        DOUT_ptr,
        CODES_ptr,
        SCALES_ptr,
        CODEBOOK_ptr,
        DX_ptr,
        M,
        N,
        K,
        stride_dout_m,
        stride_dout_n,
        stride_dx_m,
        stride_dx_k,
        BLOCK_SIZE: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k_local = tl.arange(0, BLOCK_K)
        offs_k = pid_k * BLOCK_K + offs_k_local

        mask_m = offs_m < M

        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            dout_ptrs = (
                DOUT_ptr
                + offs_m[:, None] * stride_dout_m
                + offs_n[None, :] * stride_dout_n
            )
            dout = tl.load(
                dout_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
            )

            block_idx = offs_n * BLOCKS_PER_ROW + pid_k
            code_ptrs = (
                CODES_ptr + block_idx[:, None] * BLOCK_SIZE + offs_k_local[None, :]
            )
            codes = tl.load(code_ptrs, mask=mask_n[:, None], other=0)
            codes_int = codes.to(tl.int32)
            w_norm = tl.load(CODEBOOK_ptr + codes_int)

            scales = tl.load(SCALES_ptr + block_idx, mask=mask_n, other=0.0)
            scales_h = scales.to(tl.float16)
            w_tile = w_norm.to(tl.float16) * scales_h[:, None]

            acc += tl.dot(dout.to(tl.float16), w_tile)

        dx_ptrs = DX_ptr + offs_m[:, None] * stride_dx_m + offs_k[None, :] * stride_dx_k
        tl.store(dx_ptrs, acc.to(tl.float16), mask=mask_m[:, None])


def _triton_fwd(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    original_shape: Tuple[int, ...],
    block_size: int,
) -> torch.Tensor:
    out_features, in_features = original_shape
    if in_features % block_size != 0:
        raise ValueError(
            "Triton NF4 kernel requires in_features % block_size == 0; "
            f"got in_features={in_features}, block_size={block_size}"
        )

    orig_shape = x.shape
    M = x.numel() // in_features
    K = in_features
    N = out_features

    x_2d = x.reshape(M, K).contiguous()
    y = torch.empty((M, N), dtype=torch.float16, device=x.device)

    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = block_size
    BLOCKS_PER_ROW = K // block_size

    codebook = NF4_CODEBOOK.to(device=x.device, dtype=torch.float16).contiguous()

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _nf4_matmul_fwd_kernel[grid](
        x_2d,
        codes,
        scales,
        codebook,
        y,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_SIZE=block_size,
        BLOCKS_PER_ROW=BLOCKS_PER_ROW,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return y.to(x.dtype).reshape(orig_shape[:-1] + (N,))


def _triton_bwd_x(
    grad_output: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    original_shape: Tuple[int, ...],
    block_size: int,
    x_shape: Tuple[int, ...],
    x_dtype: torch.dtype,
) -> torch.Tensor:
    out_features, in_features = original_shape
    K = in_features
    N = out_features
    M = grad_output.numel() // N

    grad_out_2d = grad_output.reshape(M, N).contiguous()
    grad_x = torch.empty((M, K), dtype=torch.float16, device=grad_output.device)

    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = block_size
    BLOCKS_PER_ROW = K // block_size

    codebook = NF4_CODEBOOK.to(device=grad_output.device, dtype=torch.float16).contiguous()

    grid = (triton.cdiv(M, BLOCK_M), BLOCKS_PER_ROW)
    _nf4_matmul_bwd_x_kernel[grid](
        grad_out_2d,
        codes,
        scales,
        codebook,
        grad_x,
        M,
        N,
        K,
        grad_out_2d.stride(0),
        grad_out_2d.stride(1),
        grad_x.stride(0),
        grad_x.stride(1),
        BLOCK_SIZE=block_size,
        BLOCKS_PER_ROW=BLOCKS_PER_ROW,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return grad_x.to(x_dtype).reshape(x_shape)


def _chunked_fwd(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    original_shape: Tuple[int, ...],
    block_size: int,
    chunk_size: int,
) -> torch.Tensor:
    out_features, in_features = original_shape
    compute_dtype = (
        x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16
    )
    out = torch.empty(x.shape[:-1] + (out_features,), dtype=x.dtype, device=x.device)
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


def _chunked_bwd_x(
    grad_output: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    original_shape: Tuple[int, ...],
    block_size: int,
    chunk_size: int,
    x_shape: Tuple[int, ...],
    x_dtype: torch.dtype,
) -> torch.Tensor:
    out_features, in_features = original_shape
    compute_dtype = (
        x_dtype if x_dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16
    )
    grad_x = torch.zeros(x_shape, dtype=compute_dtype, device=grad_output.device)
    grad_out_c = grad_output.to(compute_dtype)
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
        grad_chunk = grad_out_c[..., row_start:row_end]
        grad_x.add_(grad_chunk @ chunk_weight)
        del chunk_weight
    return grad_x.to(x_dtype)


def _can_use_triton(x: torch.Tensor, in_features: int, block_size: int) -> bool:
    return (
        TRITON_AVAILABLE
        and x.is_cuda
        and (in_features % block_size == 0)
    )


class _TritonNF4MatMul(torch.autograd.Function):
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
        ctx.used_triton = _can_use_triton(x, in_features, block_size)

        original_shape = (out_features, in_features)
        if ctx.used_triton:
            return _triton_fwd(x, codes, scales, original_shape, block_size)
        return _chunked_fwd(x, codes, scales, original_shape, block_size, chunk_size)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, codes, scales = ctx.saved_tensors
        original_shape = (ctx.out_features, ctx.in_features)

        if ctx.used_triton:
            grad_x = _triton_bwd_x(
                grad_output.contiguous(),
                codes,
                scales,
                original_shape,
                ctx.block_size,
                x.shape,
                x.dtype,
            )
        else:
            grad_x = _chunked_bwd_x(
                grad_output,
                codes,
                scales,
                original_shape,
                ctx.block_size,
                ctx.chunk_size,
                x.shape,
                x.dtype,
            )

        return grad_x, None, None, None, None, None, None


def triton_nf4_linear(
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
    out = _TritonNF4MatMul.apply(
        x, codes, scales, out_features, in_features, block_size, chunk_size
    )
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


# ----------------------------------------------------------------------------
# Autotuned variant
#
# The kernels above use static tile sizes chosen as a conservative default.
# Real cuBLAS / bitsandbytes implementations pick tile shape, num_warps, and
# software-pipeline depth (`num_stages`) per matmul shape per GPU.
#
# Below we re-declare the same forward and backward kernels but wrap them with
# `@triton.autotune` over a small grid of configurations. Triton benchmarks
# every config the first time a new (M, N, K) is seen, then caches the winner.
# This is the lowest-effort way to close the throughput gap to LoRA / cuBLAS.
#
# BLOCK_K stays equal to the NF4 block size (64) because each K-iteration
# corresponds to exactly one NF4 block. BLOCK_M, BLOCK_N, num_warps, and
# num_stages are tuned.
# ----------------------------------------------------------------------------


if TRITON_AVAILABLE:

    _AUTOTUNED_FWD_CONFIGS = [
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    ]

    _AUTOTUNED_BWD_X_CONFIGS = [
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    ]

    @triton.autotune(configs=_AUTOTUNED_FWD_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _autotuned_nf4_matmul_fwd_kernel(
        X_ptr,
        CODES_ptr,
        SCALES_ptr,
        CODEBOOK_ptr,
        Y_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_ym,
        stride_yn,
        BLOCK_SIZE: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k_local = tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_block in range(0, BLOCKS_PER_ROW):
            offs_k = k_block * BLOCK_K + offs_k_local

            x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)

            block_idx = offs_n * BLOCKS_PER_ROW + k_block
            code_ptrs = (
                CODES_ptr + block_idx[:, None] * BLOCK_SIZE + offs_k_local[None, :]
            )
            codes = tl.load(code_ptrs, mask=mask_n[:, None], other=0)
            codes_int = codes.to(tl.int32)
            w_norm = tl.load(CODEBOOK_ptr + codes_int)

            scales = tl.load(SCALES_ptr + block_idx, mask=mask_n, other=0.0)
            scales_h = scales.to(tl.float16)
            w_tile = w_norm.to(tl.float16) * scales_h[:, None]

            acc += tl.dot(x.to(tl.float16), tl.trans(w_tile))

        y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        y_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(y_ptrs, acc.to(tl.float16), mask=y_mask)

    @triton.autotune(configs=_AUTOTUNED_BWD_X_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _autotuned_nf4_matmul_bwd_x_kernel(
        DOUT_ptr,
        CODES_ptr,
        SCALES_ptr,
        CODEBOOK_ptr,
        DX_ptr,
        M,
        N,
        K,
        stride_dout_m,
        stride_dout_n,
        stride_dx_m,
        stride_dx_k,
        BLOCK_SIZE: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k_local = tl.arange(0, BLOCK_K)
        offs_k = pid_k * BLOCK_K + offs_k_local

        mask_m = offs_m < M

        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            dout_ptrs = (
                DOUT_ptr
                + offs_m[:, None] * stride_dout_m
                + offs_n[None, :] * stride_dout_n
            )
            dout = tl.load(
                dout_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
            )

            block_idx = offs_n * BLOCKS_PER_ROW + pid_k
            code_ptrs = (
                CODES_ptr + block_idx[:, None] * BLOCK_SIZE + offs_k_local[None, :]
            )
            codes = tl.load(code_ptrs, mask=mask_n[:, None], other=0)
            codes_int = codes.to(tl.int32)
            w_norm = tl.load(CODEBOOK_ptr + codes_int)

            scales = tl.load(SCALES_ptr + block_idx, mask=mask_n, other=0.0)
            scales_h = scales.to(tl.float16)
            w_tile = w_norm.to(tl.float16) * scales_h[:, None]

            acc += tl.dot(dout.to(tl.float16), w_tile)

        dx_ptrs = DX_ptr + offs_m[:, None] * stride_dx_m + offs_k[None, :] * stride_dx_k
        tl.store(dx_ptrs, acc.to(tl.float16), mask=mask_m[:, None])


def _autotuned_triton_fwd(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    original_shape: Tuple[int, ...],
    block_size: int,
) -> torch.Tensor:
    out_features, in_features = original_shape
    if in_features % block_size != 0:
        raise ValueError(
            "Triton NF4 kernel requires in_features % block_size == 0; "
            f"got in_features={in_features}, block_size={block_size}"
        )

    orig_shape = x.shape
    M = x.numel() // in_features
    K = in_features
    N = out_features

    x_2d = x.reshape(M, K).contiguous()
    y = torch.empty((M, N), dtype=torch.float16, device=x.device)

    BLOCKS_PER_ROW = K // block_size
    codebook = NF4_CODEBOOK.to(device=x.device, dtype=torch.float16).contiguous()

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    _autotuned_nf4_matmul_fwd_kernel[grid](
        x_2d,
        codes,
        scales,
        codebook,
        y,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_SIZE=block_size,
        BLOCKS_PER_ROW=BLOCKS_PER_ROW,
    )

    return y.to(x.dtype).reshape(orig_shape[:-1] + (N,))


def _autotuned_triton_bwd_x(
    grad_output: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    original_shape: Tuple[int, ...],
    block_size: int,
    x_shape: Tuple[int, ...],
    x_dtype: torch.dtype,
) -> torch.Tensor:
    out_features, in_features = original_shape
    K = in_features
    N = out_features
    M = grad_output.numel() // N

    grad_out_2d = grad_output.reshape(M, N).contiguous()
    grad_x = torch.empty((M, K), dtype=torch.float16, device=grad_output.device)

    BLOCKS_PER_ROW = K // block_size
    codebook = NF4_CODEBOOK.to(device=grad_output.device, dtype=torch.float16).contiguous()

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), BLOCKS_PER_ROW)
    _autotuned_nf4_matmul_bwd_x_kernel[grid](
        grad_out_2d,
        codes,
        scales,
        codebook,
        grad_x,
        M,
        N,
        K,
        grad_out_2d.stride(0),
        grad_out_2d.stride(1),
        grad_x.stride(0),
        grad_x.stride(1),
        BLOCK_SIZE=block_size,
        BLOCKS_PER_ROW=BLOCKS_PER_ROW,
    )

    return grad_x.to(x_dtype).reshape(x_shape)


class _AutotunedTritonNF4MatMul(torch.autograd.Function):
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
        ctx.used_triton = _can_use_triton(x, in_features, block_size)

        original_shape = (out_features, in_features)
        if ctx.used_triton:
            return _autotuned_triton_fwd(x, codes, scales, original_shape, block_size)
        return _chunked_fwd(x, codes, scales, original_shape, block_size, chunk_size)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, codes, scales = ctx.saved_tensors
        original_shape = (ctx.out_features, ctx.in_features)

        if ctx.used_triton:
            grad_x = _autotuned_triton_bwd_x(
                grad_output.contiguous(),
                codes,
                scales,
                original_shape,
                ctx.block_size,
                x.shape,
                x.dtype,
            )
        else:
            grad_x = _chunked_bwd_x(
                grad_output,
                codes,
                scales,
                original_shape,
                ctx.block_size,
                ctx.chunk_size,
                x.shape,
                x.dtype,
            )

        return grad_x, None, None, None, None, None, None


def autotuned_triton_nf4_linear(
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
    out = _AutotunedTritonNF4MatMul.apply(
        x, codes, scales, out_features, in_features, block_size, chunk_size
    )
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out
