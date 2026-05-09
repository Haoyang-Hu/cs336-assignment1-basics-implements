from __future__ import annotations

import numpy.typing as npt
import torch
from torch import Tensor


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """
    Sample random language-model training examples from a 1D token array.

    For each sampled start position `i`, the input is:

        dataset[i : i + context_length]

    and the label is the same sequence shifted one token to the right:

        dataset[i + 1 : i + context_length + 1]
    """
    max_start = len(dataset) - context_length
    if max_start <= 0:
        raise ValueError("dataset must be longer than context_length")

    starts = torch.randint(0, max_start, (batch_size,))

    # Build each row independently. This is easy to read and perfectly fine for
    # the assignment tests.
    x = torch.stack(
        [torch.as_tensor(dataset[start : start + context_length]) for start in starts.tolist()]
    )
    y = torch.stack(
        [torch.as_tensor(dataset[start + 1 : start + context_length + 1]) for start in starts.tolist()]
    )

    return x.to(device=device, dtype=torch.long), y.to(device=device, dtype=torch.long)
