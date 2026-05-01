from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from .quantization import chunked_nf4_linear, dequantize_nf4, quantize_nf4
from .triton_kernels import (
    TRITON_AVAILABLE,
    autotuned_triton_nf4_linear,
    triton_nf4_linear,
)


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
        "c_proj",
        "c_fc",
    )
    block_size: int = 64
    adapter_dtype: torch.dtype = torch.float16
    chunk_size: int = 128

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


class QuantizedLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.config = config

        quantized_weight = quantize_nf4(
            base_linear.weight.data,
            block_size=config.block_size,
        )
        self.register_buffer("qweight_codes", quantized_weight.codes, persistent=True)
        self.register_buffer("qweight_scales", quantized_weight.scales, persistent=True)
        self.original_shape = quantized_weight.original_shape

        if base_linear.bias is None:
            self.register_parameter("bias", None)
        else:
            bias = nn.Parameter(base_linear.bias.detach().to(config.adapter_dtype), requires_grad=False)
            self.register_parameter("bias", bias)

        self.lora_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(
            torch.empty(config.rank, self.in_features, dtype=config.adapter_dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, config.rank, dtype=config.adapter_dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_weight = dequantize_nf4(
            self.qweight_codes,
            self.qweight_scales,
            self.original_shape,
            block_size=self.config.block_size,
            dtype=x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16,
        )
        result = torch.nn.functional.linear(x, base_weight, self.bias)

        adapter_input = self.lora_dropout(x).to(self.lora_A.dtype)
        adapter_hidden = torch.nn.functional.linear(adapter_input, self.lora_A)
        adapter_output = torch.nn.functional.linear(adapter_hidden, self.lora_B)
        return result + adapter_output.to(result.dtype) * self.config.scaling


class ChunkedQuantizedLoRALinear(nn.Module):
    """
    Drop-in replacement for QuantizedLoRALinear that streams the base-weight
    matmul through ``chunked_nf4_linear`` so the full fp16 weight is never
    materialized in forward and is not retained in autograd's saved tensors.
    """

    def __init__(self, base_linear: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.config = config

        quantized_weight = quantize_nf4(
            base_linear.weight.data,
            block_size=config.block_size,
        )
        self.register_buffer("qweight_codes", quantized_weight.codes, persistent=True)
        self.register_buffer("qweight_scales", quantized_weight.scales, persistent=True)
        self.original_shape = quantized_weight.original_shape

        if base_linear.bias is None:
            self.register_parameter("bias", None)
        else:
            bias = nn.Parameter(base_linear.bias.detach().to(config.adapter_dtype), requires_grad=False)
            self.register_parameter("bias", bias)

        self.lora_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(
            torch.empty(config.rank, self.in_features, dtype=config.adapter_dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, config.rank, dtype=config.adapter_dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = chunked_nf4_linear(
            x,
            self.qweight_codes,
            self.qweight_scales,
            self.original_shape,
            self.bias,
            block_size=self.config.block_size,
            chunk_size=self.config.chunk_size,
        )

        adapter_input = self.lora_dropout(x).to(self.lora_A.dtype)
        adapter_hidden = torch.nn.functional.linear(adapter_input, self.lora_A)
        adapter_output = torch.nn.functional.linear(adapter_hidden, self.lora_B)
        return result + adapter_output.to(result.dtype) * self.config.scaling


class TritonQuantizedLoRALinear(nn.Module):
    """
    QLoRA layer backed by Triton-fused NF4 dequant + matmul kernels.

    Forward and backward both run through ``triton_nf4_linear``, which launches
    a custom CUDA kernel that rebuilds the NF4 weight one tile at a time inside
    the GPU's tensor cores. The fp16 weight is never materialized in HBM and
    is never saved by autograd, so peak VRAM stays close to the NF4 footprint
    while throughput stays close to a regular fp16 linear.

    Falls back to the chunked Python path when Triton or CUDA is unavailable
    (so the class can be instantiated and unit-tested on CPU).
    """

    def __init__(self, base_linear: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.config = config

        quantized_weight = quantize_nf4(
            base_linear.weight.data,
            block_size=config.block_size,
        )
        self.register_buffer("qweight_codes", quantized_weight.codes, persistent=True)
        self.register_buffer("qweight_scales", quantized_weight.scales, persistent=True)
        self.original_shape = quantized_weight.original_shape

        if base_linear.bias is None:
            self.register_parameter("bias", None)
        else:
            bias = nn.Parameter(base_linear.bias.detach().to(config.adapter_dtype), requires_grad=False)
            self.register_parameter("bias", bias)

        self.lora_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(
            torch.empty(config.rank, self.in_features, dtype=config.adapter_dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, config.rank, dtype=config.adapter_dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def using_triton_kernel(self) -> bool:
        return TRITON_AVAILABLE and self.qweight_codes.is_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = triton_nf4_linear(
            x,
            self.qweight_codes,
            self.qweight_scales,
            self.original_shape,
            self.bias,
            block_size=self.config.block_size,
            chunk_size=self.config.chunk_size,
        )

        adapter_input = self.lora_dropout(x).to(self.lora_A.dtype)
        adapter_hidden = torch.nn.functional.linear(adapter_input, self.lora_A)
        adapter_output = torch.nn.functional.linear(adapter_hidden, self.lora_B)
        return result + adapter_output.to(result.dtype) * self.config.scaling


class AutotunedTritonQuantizedLoRALinear(nn.Module):
    """
    QLoRA layer backed by autotuned Triton-fused NF4 dequant + matmul kernels.

    Identical to ``TritonQuantizedLoRALinear`` but routes through
    ``autotuned_triton_nf4_linear``, which lets Triton benchmark a small set of
    tile shapes / num_warps / num_stages per (M, N, K) and cache the winner.

    The first call for each new matmul shape pays a one-time tuning cost
    (Triton runs each config a few times). After that, the cached config is
    used and step time should drop noticeably below the static-tile Triton
    layer, ideally approaching the cuBLAS-backed dense-dequant path on
    throughput while keeping the chunked / Triton VRAM profile.
    """

    def __init__(self, base_linear: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.config = config

        quantized_weight = quantize_nf4(
            base_linear.weight.data,
            block_size=config.block_size,
        )
        self.register_buffer("qweight_codes", quantized_weight.codes, persistent=True)
        self.register_buffer("qweight_scales", quantized_weight.scales, persistent=True)
        self.original_shape = quantized_weight.original_shape

        if base_linear.bias is None:
            self.register_parameter("bias", None)
        else:
            bias = nn.Parameter(base_linear.bias.detach().to(config.adapter_dtype), requires_grad=False)
            self.register_parameter("bias", bias)

        self.lora_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(
            torch.empty(config.rank, self.in_features, dtype=config.adapter_dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, config.rank, dtype=config.adapter_dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def using_triton_kernel(self) -> bool:
        return TRITON_AVAILABLE and self.qweight_codes.is_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = autotuned_triton_nf4_linear(
            x,
            self.qweight_codes,
            self.qweight_scales,
            self.original_shape,
            self.bias,
            block_size=self.config.block_size,
            chunk_size=self.config.chunk_size,
        )

        adapter_input = self.lora_dropout(x).to(self.lora_A.dtype)
        adapter_hidden = torch.nn.functional.linear(adapter_input, self.lora_A)
        adapter_output = torch.nn.functional.linear(adapter_hidden, self.lora_B)
        return result + adapter_output.to(result.dtype) * self.config.scaling


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.config = config

        self.register_buffer(
            "base_weight",
            base_linear.weight.detach().to(config.adapter_dtype),
            persistent=True,
        )

        if base_linear.bias is None:
            self.register_parameter("bias", None)
        else:
            bias = nn.Parameter(base_linear.bias.detach().to(config.adapter_dtype), requires_grad=False)
            self.register_parameter("bias", bias)

        self.lora_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(
            torch.empty(config.rank, self.in_features, dtype=config.adapter_dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, config.rank, dtype=config.adapter_dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.nn.functional.linear(
            x,
            self.base_weight.to(dtype=x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16),
            self.bias,
        )
        adapter_input = self.lora_dropout(x).to(self.lora_A.dtype)
        adapter_hidden = torch.nn.functional.linear(adapter_input, self.lora_A)
        adapter_output = torch.nn.functional.linear(adapter_hidden, self.lora_B)
        return result + adapter_output.to(result.dtype) * self.config.scaling


def _iter_named_linears(model: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def _get_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _prepare_model(model: nn.Module, config: LoRAConfig, adapter_cls: type[nn.Module]) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "config"):
        model.config.use_cache = False

    replaced = 0
    for name, linear in list(_iter_named_linears(model)):
        if name.endswith("lm_head"):
            continue
        if not any(target in name for target in config.target_modules):
            continue

        parent, attr_name = _get_parent_module(model, name)
        setattr(parent, attr_name, adapter_cls(linear, config))
        replaced += 1

    if replaced == 0:
        raise RuntimeError(
            "No target linear layers were replaced. Update LoRAConfig.target_modules for this model."
        )

    return model


def prepare_model_for_kbit_training(model: nn.Module, config: LoRAConfig) -> nn.Module:
    return _prepare_model(model, config, QuantizedLoRALinear)


def prepare_model_for_chunked_kbit_training(model: nn.Module, config: LoRAConfig) -> nn.Module:
    return _prepare_model(model, config, ChunkedQuantizedLoRALinear)


def prepare_model_for_triton_kbit_training(model: nn.Module, config: LoRAConfig) -> nn.Module:
    return _prepare_model(model, config, TritonQuantizedLoRALinear)


def prepare_model_for_autotuned_triton_kbit_training(model: nn.Module, config: LoRAConfig) -> nn.Module:
    return _prepare_model(model, config, AutotunedTritonQuantizedLoRALinear)


def prepare_model_for_lora_training(model: nn.Module, config: LoRAConfig) -> nn.Module:
    return _prepare_model(model, config, LoRALinear)


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
