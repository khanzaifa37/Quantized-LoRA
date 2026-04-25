from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer


class PagedAdamW32bit(Optimizer):
    """
    Educational AdamW variant that keeps optimizer state on CPU in float32.
    During `step()`, parameter tensors are updated page by page so only a slice
    of optimizer state is handled at a time.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        page_size: int = 2**18,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            page_size=page_size,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            page_size = group["page_size"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("PagedAdamW32bit does not support sparse gradients.")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros(param.numel(), dtype=torch.float32, device="cpu")
                    state["exp_avg_sq"] = torch.zeros(param.numel(), dtype=torch.float32, device="cpu")

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                param_flat = param.data.view(-1)
                grad_cpu = grad.float().view(-1).cpu()

                for start in range(0, param.numel(), page_size):
                    end = min(start + page_size, param.numel())
                    grad_slice = grad_cpu[start:end]
                    exp_avg_slice = exp_avg[start:end]
                    exp_avg_sq_slice = exp_avg_sq[start:end]

                    exp_avg_slice.mul_(beta1).add_(grad_slice, alpha=1.0 - beta1)
                    exp_avg_sq_slice.mul_(beta2).addcmul_(grad_slice, grad_slice, value=1.0 - beta2)

                    bias_correction1 = 1.0 - beta1**step
                    bias_correction2 = 1.0 - beta2**step
                    denom = (exp_avg_sq_slice.sqrt() / bias_correction2**0.5).add_(eps)
                    update = (exp_avg_slice / bias_correction1) / denom

                    if weight_decay:
                        param_slice_cpu = param_flat[start:end].float().cpu()
                        update.add_(param_slice_cpu, alpha=weight_decay)

                    param_flat[start:end].add_(
                        update.to(device=param.device, dtype=param.dtype),
                        alpha=-lr,
                    )

        return loss
