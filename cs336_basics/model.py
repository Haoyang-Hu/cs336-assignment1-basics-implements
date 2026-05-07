from __future__ import annotations

import math

import torch
from torch import Tensor


def linear(weights: Tensor, in_features: Tensor) -> Tensor:
    """
    Linear layer without bias.

    PyTorch stores linear weights as `(output_dim, input_dim)`, so the forward
    pass multiplies by `weights.T`.
    """
    # Reference code to type:
    # return in_features @ weights.transpose(-1, -2)
    return in_features @ weights.transpose(-1, -2)


def embedding(weights: Tensor, token_ids: Tensor) -> Tensor:
    """
    Look up one embedding vector for each token ID.

    Tensor indexing handles any leading shape in `token_ids`. For example, a
    `(batch, sequence)` ID tensor returns `(batch, sequence, d_model)`.
    """
    # Reference code to type:
    # return weights[token_ids]
    return weights[token_ids]


def silu(in_features: Tensor) -> Tensor:
    """SiLU activation: x * sigmoid(x)."""
    # Reference code to type:
    # return in_features * torch.sigmoid(in_features)
    return in_features * torch.sigmoid(in_features)


def swiglu(w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor, in_features: Tensor) -> Tensor:
    """
    SwiGLU feed-forward network.

    The two "up" projections produce tensors with shape `(..., d_ff)`. One side
    goes through SiLU and gates the other side by elementwise multiplication.
    The final projection maps back to `d_model`.
    """
    # Reference code to type:
    # gate = silu(linear(w1_weight, in_features))
    # value = linear(w3_weight, in_features)
    # return linear(w2_weight, gate * value)
    gate = silu(linear(w1_weight, in_features))
    value = linear(w3_weight, in_features)
    return linear(w2_weight, gate * value)


def rmsnorm(weights: Tensor, in_features: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Root-mean-square layer normalization.

    RMSNorm normalizes by the RMS of the last dimension only. The learned
    `weights` then scale each feature.
    """
    # Reference code to type:
    # rms = torch.sqrt(torch.mean(in_features.float() ** 2, dim=-1, keepdim=True) + eps)
    # normalized = in_features / rms
    # return normalized.to(in_features.dtype) * weights
    rms = torch.sqrt(torch.mean(in_features.float() ** 2, dim=-1, keepdim=True) + eps)
    normalized = in_features / rms
    return normalized.to(in_features.dtype) * weights


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Scaled dot-product attention.

    Shape convention:
    - Q: `(..., queries, d_k)`
    - K: `(..., keys, d_k)`
    - V: `(..., keys, d_v)`
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        # In these tests, True means "this key is visible" and False means
        # "hide this key". A very negative score becomes nearly zero after
        # softmax.
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    attention_weights = torch.softmax(scores, dim=-1)
    return attention_weights @ V


def _split_heads(x: Tensor, num_heads: int) -> Tensor:
    """Convert `(..., seq, d_model)` into `(..., heads, seq, d_head)`."""
    *leading_dims, sequence_length, d_model = x.shape
    d_head = d_model // num_heads
    x = x.view(*leading_dims, sequence_length, num_heads, d_head)
    return x.transpose(-3, -2)


def _combine_heads(x: Tensor) -> Tensor:
    """Convert `(..., heads, seq, d_head)` back into `(..., seq, d_model)`."""
    *leading_dims, num_heads, sequence_length, d_head = x.shape
    x = x.transpose(-3, -2).contiguous()
    return x.view(*leading_dims, sequence_length, num_heads * d_head)


def _causal_mask(sequence_length: int, device: torch.device) -> Tensor:
    """
    Lower-triangular mask for autoregressive self-attention.

    Position `i` can see positions `0..i`, but not future positions.
    """
    return torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device))


def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    """Multi-head causal self-attention without RoPE."""
    q = _split_heads(linear(q_proj_weight, in_features), num_heads)
    k = _split_heads(linear(k_proj_weight, in_features), num_heads)
    v = _split_heads(linear(v_proj_weight, in_features), num_heads)

    sequence_length = in_features.shape[-2]
    mask = _causal_mask(sequence_length, in_features.device)
    attention_output = scaled_dot_product_attention(q, k, v, mask=mask)

    combined = _combine_heads(attention_output)
    return linear(o_proj_weight, combined)


def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Tensor,
    token_positions: Tensor,
) -> Tensor:
    """
    Rotary positional embedding.

    RoPE rotates each pair of hidden features by an angle determined by the
    token position. The first pair rotates slowly, later pairs rotate faster.
    """
    del max_seq_len
    device = in_query_or_key.device
    dtype = in_query_or_key.dtype

    half_dim = d_k // 2
    feature_pair_indices = torch.arange(half_dim, device=device, dtype=torch.float32)
    inv_freq = theta ** (-2 * feature_pair_indices / d_k)

    positions = token_positions.to(device=device, dtype=torch.float32)
    angles = positions[..., None] * inv_freq

    # If the input has a head dimension, insert a singleton axis so the same
    # positions can broadcast across all heads.
    while angles.ndim < in_query_or_key.ndim:
        angles = angles.unsqueeze(-3)

    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)

    x_even = in_query_or_key[..., 0::2]
    x_odd = in_query_or_key[..., 1::2]

    rotated = torch.empty_like(in_query_or_key)
    rotated[..., 0::2] = x_even * cos - x_odd * sin
    rotated[..., 1::2] = x_even * sin + x_odd * cos
    return rotated


def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Tensor | None = None,
) -> Tensor:
    """Multi-head causal self-attention with RoPE applied to Q and K."""
    q = _split_heads(linear(q_proj_weight, in_features), num_heads)
    k = _split_heads(linear(k_proj_weight, in_features), num_heads)
    v = _split_heads(linear(v_proj_weight, in_features), num_heads)

    sequence_length = in_features.shape[-2]
    if token_positions is None:
        token_positions = torch.arange(sequence_length, device=in_features.device)

    d_head = d_model // num_heads
    q = rope(d_head, theta, max_seq_len, q, token_positions)
    k = rope(d_head, theta, max_seq_len, k, token_positions)

    mask = _causal_mask(sequence_length, in_features.device)
    attention_output = scaled_dot_product_attention(q, k, v, mask=mask)
    combined = _combine_heads(attention_output)
    return linear(o_proj_weight, combined)


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Tensor,
) -> Tensor:
    """
    One pre-norm Transformer block.

    Pre-norm means each sublayer sees normalized input, then the sublayer result
    is added back to the residual stream.
    """
    del d_ff
    normed = rmsnorm(weights["ln1.weight"], in_features)
    attention_output = multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=normed,
    )
    residual = in_features + attention_output

    normed = rmsnorm(weights["ln2.weight"], residual)
    ffn_output = swiglu(
        weights["ffn.w1.weight"],
        weights["ffn.w2.weight"],
        weights["ffn.w3.weight"],
        normed,
    )
    return residual + ffn_output


def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Tensor,
) -> Tensor:
    """Transformer language model forward pass using the provided weights."""
    del context_length
    x = embedding(weights["token_embeddings.weight"], in_indices)

    for layer_index in range(num_layers):
        prefix = f"layers.{layer_index}."
        block_weights = {
            key.removeprefix(prefix): value
            for key, value in weights.items()
            if key.startswith(prefix)
        }
        x = transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=in_indices.shape[-1],
            theta=rope_theta,
            weights=block_weights,
            in_features=x,
        )

    x = rmsnorm(weights["ln_final.weight"], x)
    return linear(weights["lm_head.weight"], x)
