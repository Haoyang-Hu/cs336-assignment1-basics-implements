from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from einops import rearrange, einsum


@dataclass(frozen=True)
class TransformerLMConfig:
    """
    Hyperparameters for a trainable Transformer language model.

    The default values match the TinyStories model size requested in the
    assignment document: a 10k-token vocabulary, 256-token context, 4 layers,
    512 hidden size, 16 attention heads, and a 1344-wide feed-forward network.
    """

    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0
    init_std: float = 0.02


def linear(weights: Tensor, in_features: Tensor) -> Tensor:
    """
    Linear layer without bias.

    PyTorch stores linear weights as `(output_dim, input_dim)`, so the forward
    pass multiplies by `weights.T`.
    """
    # einops self‑documents the contraction: the input feature dimension `d_in`
    # is matched with the second axis of `weights`, and the output dimension
    # `d_out` appears in the result.
    return einsum(weights, in_features, "d_out d_in, ... d_in -> ... d_out")


def embedding(weights: Tensor, token_ids: Tensor) -> Tensor:
    """
    Look up one embedding vector for each token ID.

    Tensor indexing handles any leading shape in `token_ids`. For example, a
    `(batch, sequence)` ID tensor returns `(batch, sequence, d_model)`.
    """
    return weights[token_ids]


def silu(in_features: Tensor) -> Tensor:
    """SiLU activation: x * sigmoid(x)."""
    return in_features * torch.sigmoid(in_features)


def swiglu(w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor, in_features: Tensor) -> Tensor:
    """
    SwiGLU feed-forward network.

    The two "up" projections produce tensors with shape `(..., d_ff)`. One side
    goes through SiLU and gates the other side by elementwise multiplication.
    The final projection maps back to `d_model`.
    """
    gate = silu(linear(w1_weight, in_features))
    value = linear(w3_weight, in_features)
    return linear(w2_weight, gate * value)


def rmsnorm(weights: Tensor, in_features: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Root-mean-square layer normalization.

    RMSNorm normalizes by the RMS of the last dimension only. The learned
    `weights` then scale each feature.
    """
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

    # Compute attention logits with einops: per query, per key dot product.
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(d_k)

    if mask is not None:
        # True means "this key is visible", so we set False positions to -inf.
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    attention_weights = torch.softmax(scores, dim=-1)

    # Weighted sum of values, again with einops for clarity.
    return einsum(attention_weights, V, "... q k, ... k d -> ... q d")


def _split_heads(x: Tensor, num_heads: int) -> Tensor:
    """Convert `(..., seq, d_model)` into `(..., heads, seq, d_head)`."""
    # rearrange splits the last dimension into heads × d_head and moves heads
    # before the sequence axis – all in one self‑documenting line.
    return rearrange(x, "... seq (h d) -> ... h seq d", h=num_heads)


def _combine_heads(x: Tensor) -> Tensor:
    """Convert `(..., heads, seq, d_head)` back into `(..., seq, d_model)`."""
    # Reverse of _split_heads: merge the head dimension back into the feature dimension.
    return rearrange(x, "... h seq d -> ... seq (h d)")


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


class TransformerLMModule(torch.nn.Module):
    """
    Trainable wrapper around the functional `transformer_lm` above.

    The assignment tests use plain functions that receive a `weights` dict.
    Training needs a `torch.nn.Module` so PyTorch can find parameters,
    calculate gradients, and save/load checkpoints. This class bridges those
    two worlds: it owns `nn.Parameter`s, then rebuilds the same flat `weights`
    dictionary expected by `transformer_lm` during each forward pass.
    """

    def __init__(self, config: TransformerLMConfig) -> None:
        super().__init__()
        self.config = config

        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if (config.d_model // config.num_heads) % 2 != 0:
            raise ValueError("RoPE needs an even head dimension")

        self._name_to_safe_name: dict[str, str] = {}
        params: dict[str, torch.nn.Parameter] = {}

        def add_parameter(name: str, shape: tuple[int, ...]) -> None:
            # PyTorch module parameter names cannot contain dots, but the
            # functional model expects names like `layers.0.ln1.weight`.
            # We store a safe version internally and keep a map back to the
            # assignment-style name.
            safe_name = name.replace(".", "__")
            self._name_to_safe_name[name] = safe_name
            params[safe_name] = torch.nn.Parameter(torch.empty(shape))

        add_parameter("token_embeddings.weight", (config.vocab_size, config.d_model))

        for layer_index in range(config.num_layers):
            prefix = f"layers.{layer_index}."
            add_parameter(prefix + "ln1.weight", (config.d_model,))
            add_parameter(prefix + "attn.q_proj.weight", (config.d_model, config.d_model))
            add_parameter(prefix + "attn.k_proj.weight", (config.d_model, config.d_model))
            add_parameter(prefix + "attn.v_proj.weight", (config.d_model, config.d_model))
            add_parameter(prefix + "attn.output_proj.weight", (config.d_model, config.d_model))
            add_parameter(prefix + "ln2.weight", (config.d_model,))
            add_parameter(prefix + "ffn.w1.weight", (config.d_ff, config.d_model))
            add_parameter(prefix + "ffn.w2.weight", (config.d_model, config.d_ff))
            add_parameter(prefix + "ffn.w3.weight", (config.d_ff, config.d_model))

        add_parameter("ln_final.weight", (config.d_model,))
        add_parameter("lm_head.weight", (config.vocab_size, config.d_model))

        self.params = torch.nn.ParameterDict(params)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize model parameters before training.

        Linear/embedding weights start as small random numbers. RMSNorm scale
        weights start at 1 so the normalization layer initially leaves feature
        magnitudes unchanged.
        """
        for name, parameter in self.functional_weights().items():
            if "ln" in name and name.endswith(".weight"):
                torch.nn.init.ones_(parameter)
            else:
                torch.nn.init.normal_(parameter, mean=0.0, std=self.config.init_std)

    def functional_weights(self) -> dict[str, Tensor]:
        """Return parameters using the flat names expected by `transformer_lm`."""
        return {
            name: self.params[safe_name]
            for name, safe_name in self._name_to_safe_name.items()
        }

    def forward(self, in_indices: Tensor) -> Tensor:
        """
        Run the language model.

        `in_indices` has shape `(batch, sequence)`. The returned logits have
        shape `(batch, sequence, vocab_size)`.
        """
        if in_indices.shape[-1] > self.config.context_length:
            raise ValueError(
                f"sequence length {in_indices.shape[-1]} exceeds context length {self.config.context_length}"
            )

        return transformer_lm(
            vocab_size=self.config.vocab_size,
            context_length=self.config.context_length,
            d_model=self.config.d_model,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            d_ff=self.config.d_ff,
            rope_theta=self.config.rope_theta,
            weights=self.functional_weights(),
            in_indices=in_indices,
        )
