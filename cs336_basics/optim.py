from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer.

    Adam keeps moving averages of gradients and squared gradients. AdamW adds
    weight decay separately from the gradient update, which is why it is called
    "decoupled" weight decay.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults: dict[str, Any] = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
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

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad
                state = self.state[parameter]

                # State is created lazily the first time a parameter receives a
                # gradient. This mirrors PyTorch optimizer style.
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay. PyTorch's AdamW applies this directly
                # to the parameter before the Adam moment update.
                if weight_decay != 0:
                    parameter.mul_(1 - lr * weight_decay)

                # First moment: moving average of gradients.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Second moment: moving average of squared gradients.
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction compensates for the moving averages starting
                # at zero during early training steps.
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # Adam update.
                denominator = exp_avg_sq.sqrt().add_(eps)
                parameter.addcdiv_(exp_avg, denominator, value=-step_size)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Linear warmup followed by cosine decay.

    The schedule has three regions:
    1. Warmup: linearly increase from 0 to max LR.
    2. Cosine: smoothly decay from max LR to min LR.
    3. After the cosine window: stay at min LR.
    """
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate

    if it > cosine_cycle_iters:
        return min_learning_rate

    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)
