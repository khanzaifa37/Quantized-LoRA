from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from .quantization import dequantize_nf4, quantize_nf4


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


def prepare_model_for_kbit_training(model: nn.Module, config: LoRAConfig) -> nn.Module:
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
        setattr(parent, attr_name, QuantizedLoRALinear(linear, config))
        replaced += 1

    if replaced == 0:
        raise RuntimeError(
            "No target linear layers were replaced. Update LoRAConfig.target_modules for this model."
        )

    return model


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
