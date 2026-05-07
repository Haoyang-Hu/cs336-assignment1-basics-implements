from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


def softmax(in_features: Tensor, dim: int) -> Tensor:
    """
    Numerically stable softmax.

    Softmax turns arbitrary numbers into probabilities along one dimension.
    Subtracting the maximum value first does not change the final probabilities,
    but it prevents very large exponentials from overflowing.
    """
    # Reference code to type:
    # shifted = in_features - torch.max(in_features, dim=dim, keepdim=True).values
    # exp_values = torch.exp(shifted)
    # return exp_values / torch.sum(exp_values, dim=dim, keepdim=True)
    shifted = in_features - torch.max(in_features, dim=dim, keepdim=True).values
    exp_values = torch.exp(shifted)
    return exp_values / torch.sum(exp_values, dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """
    Average cross-entropy loss for unnormalized logits.

    `inputs` has shape `(batch_size, vocab_size)`.
    `targets` has shape `(batch_size,)` and stores the correct class index for
    each row.
    """
    # Reference code to type:
    # shifted = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    # log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    # target_logits = shifted[torch.arange(targets.shape[0], device=targets.device), targets]
    # return torch.mean(log_sum_exp - target_logits)
    shifted = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    row_indices = torch.arange(targets.shape[0], device=targets.device)
    target_logits = shifted[row_indices, targets]
    return torch.mean(log_sum_exp - target_logits)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip all gradients together to have total L2 norm at most `max_l2_norm`.

    This modifies `parameter.grad` in place. Parameters with `grad is None` are
    ignored, which matches normal PyTorch optimizer behavior.
    """
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return

    # The total norm treats every gradient tensor as one long vector.
    total_norm = torch.sqrt(sum(torch.sum(grad.detach() ** 2) for grad in grads))
    if total_norm <= max_l2_norm:
        return

    # Multiplying every gradient by the same scale keeps the direction of the
    # combined gradient vector, but shrinks its length.
    scale = max_l2_norm / (total_norm + 1e-6)
    for grad in grads:
        grad.mul_(scale)
