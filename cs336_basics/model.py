"""
Transformer building blocks for CS336 assignment 1.

There are two layers of API in this file:

1. Capitalized classes such as `Linear`, `RMSNorm`, and `TransformerLM`.
   These are real `torch.nn.Module` objects. They own parameters, appear in
   `model.state_dict()`, and are the right objects to use for training.

2. Lowercase functions such as `linear`, `rmsnorm`, and `transformer_lm`.
   These are stateless helpers. They are useful for tests because the tests
   pass in exact reference weights and ask for the output of one forward pass.

Keeping both APIs in one file makes it easier to compare the mathematical
operation with the trainable PyTorch module that wraps it.
"""

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


class Linear(torch.nn.Module):
    """
    A bias-free linear layer implemented from first principles.

    PyTorch's `nn.Linear` normally stores both `weight` and `bias`. The
    assignment asks for only the matrix multiply, so this module has one
    parameter:

        weight.shape == (out_features, in_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # The handout asks for truncated normal initialization with this
        # Xavier-style standard deviation. Truncating at 3 standard deviations
        # avoids rare very large initial weights.
        std = math.sqrt(2 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, in_features: Tensor) -> Tensor:
        return linear(self.weight, in_features)


class Embedding(torch.nn.Module):
    """
    Token embedding lookup table implemented without `torch.nn.Embedding`.

    `weight[token_id]` returns the learned vector for one token. If `token_ids`
    has shape `(batch, sequence)`, the output has shape
    `(batch, sequence, embedding_dim)`.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return embedding(self.weight, token_ids)


class SiLU(torch.nn.Module):
    """SiLU activation module."""

    def forward(self, in_features: Tensor) -> Tensor:
        return silu(in_features)


class RMSNorm(torch.nn.Module):
    """
    Root-mean-square normalization with a learned gain vector.

    RMSNorm rescales each token vector by the root mean square of its features,
    then multiplies by `weight`. Unlike LayerNorm, it does not subtract the
    feature mean.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def forward(self, in_features: Tensor) -> Tensor:
        return rmsnorm(self.weight, in_features, self.eps)


class SwiGLU(torch.nn.Module):
    """
    Position-wise feed-forward network with a SiLU gate.

    The transformer applies this independently at every token position. The
    shape path is:

        (..., d_model) -> (..., d_ff) -> (..., d_model)

    The gate branch and value branch both project up to `d_ff`; their
    elementwise product is projected back down to `d_model`.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, in_features: Tensor) -> Tensor:
        return swiglu(self.w1.weight, self.w2.weight, self.w3.weight, in_features)


class ScaledDotProductAttention(torch.nn.Module):
    """Scaled dot-product attention module."""

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
        return scaled_dot_product_attention(Q, K, V, mask)


class RoPE(torch.nn.Module):
    """
    Rotary positional embedding module.

    RoPE does not own trainable parameters. It uses deterministic sine/cosine
    rotations so attention can tell where a token is in the sequence.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        half_dim = d_k // 2
        feature_pair_indices = torch.arange(half_dim, device=device, dtype=torch.float32)
        inv_freq = theta ** (-2 * feature_pair_indices / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]
        # These buffers are cached for inspection/debugging and will move with
        # the module across devices, but they are not saved in checkpoints
        # because the forward helper can recompute them from `theta` and `d_k`.
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, in_query_or_key: Tensor, token_positions: Tensor) -> Tensor:
        return rope(self.d_k, self.theta, self.max_seq_len, in_query_or_key, token_positions)


RotaryPositionalEmbedding = RoPE


class MultiHeadSelfAttention(torch.nn.Module):
    """
    Multi-head causal self-attention without positional rotation.

    The four linear projections are stored as modules so their parameter names
    match the assignment state dict:

        q_proj.weight, k_proj.weight, v_proj.weight, output_proj.weight
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, in_features: Tensor) -> Tensor:
        return multihead_self_attention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            o_proj_weight=self.output_proj.weight,
            in_features=in_features,
        )


class MultiHeadSelfAttentionWithRoPE(MultiHeadSelfAttention):
    """Multi-head causal self-attention with RoPE applied to Q and K."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(d_model, num_heads, device=device, dtype=dtype)
        d_head = d_model // num_heads
        if d_head % 2 != 0:
            raise ValueError("RoPE needs an even head dimension")
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = RoPE(theta=theta, d_k=d_head, max_seq_len=max_seq_len, device=device)

    def forward(self, in_features: Tensor, token_positions: Tensor | None = None) -> Tensor:
        return multihead_self_attention_with_rope(
            d_model=self.d_model,
            num_heads=self.num_heads,
            max_seq_len=self.max_seq_len,
            theta=self.theta,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            o_proj_weight=self.output_proj.weight,
            in_features=in_features,
            token_positions=token_positions,
        )


class TransformerBlock(torch.nn.Module):
    """
    One pre-norm Transformer block.

    "Pre-norm" means the input is normalized before each sublayer. The residual
    connections then add the sublayer output back to the stream:

        x = x + attention(rmsnorm(x))
        x = x + feed_forward(rmsnorm(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, in_features: Tensor) -> Tensor:
        return transformer_block(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            theta=self.theta,
            weights=dict(self.named_parameters()),
            in_features=in_features,
        )


class TransformerLM(torch.nn.Module):
    """
    Decoder-only Transformer language model.

    The model turns token IDs into logits:

        token IDs -> token embeddings -> transformer blocks -> final logits

    Logits are unnormalized scores over the vocabulary. Training code passes
    them to cross-entropy, which applies the probability normalization.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def functional_weights(self) -> dict[str, Tensor]:
        """
        Return parameters using the flat names expected by `transformer_lm`.

        `named_parameters()` recursively walks submodules, so nested modules
        naturally produce names like `layers.0.attn.q_proj.weight`.
        """
        return dict(self.named_parameters())

    def forward(self, in_indices: Tensor) -> Tensor:
        if in_indices.shape[-1] > self.context_length:
            raise ValueError(
                f"sequence length {in_indices.shape[-1]} exceeds context length {self.context_length}"
            )

        return transformer_lm(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            weights=self.functional_weights(),
            in_indices=in_indices,
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Older checkpoints from the previous implementation stored parameters
        # in a `ParameterDict` whose keys could not contain dots, so dots were
        # replaced by double underscores. This small conversion keeps those
        # checkpoints loadable after switching to real nested modules.
        converted_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("params."):
                converted_key = key.removeprefix("params.").replace("__", ".")
            else:
                converted_key = key
            converted_state_dict[converted_key] = value
        return super().load_state_dict(converted_state_dict, strict=strict, assign=assign)


def linear(weights: Tensor, in_features: Tensor) -> Tensor:
    """
    Linear layer without bias.

    PyTorch stores linear weights as `(output_dim, input_dim)`, so the forward
    pass multiplies by `weights.T`.
    """
    # einops self-documents the contraction: the input feature dimension `d_in`
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
    # Compute the normalization in float32 for stability, then cast back to the
    # original dtype so mixed-precision callers keep the expected output dtype.
    in_dtype = in_features.dtype
    in_features_float = in_features.to(torch.float32)
    rms = torch.sqrt(torch.mean(in_features_float**2, dim=-1, keepdim=True) + eps)
    normalized = in_features_float / rms
    return (normalized * weights.to(torch.float32)).to(in_dtype)


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
    # rearrange splits the last dimension into heads x d_head and moves heads
    # before the sequence axis - all in one self-documenting line.
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
    """
    Multi-head causal self-attention without RoPE.

    Starting shape:

        in_features: (..., sequence_length, d_model)

    After projection and splitting:

        q/k/v: (..., num_heads, sequence_length, d_head)
    """
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

    # `token_positions` can be `(seq,)` or `(batch, seq)`, while q/k may also
    # include a head dimension. Unsqueezing at `-3` inserts singleton axes before
    # the sequence dimension until broadcasting lines up.
    while angles.ndim < in_query_or_key.ndim:
        angles = angles.unsqueeze(-3)

    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)

    x_even = in_query_or_key[..., 0::2]
    x_odd = in_query_or_key[..., 1::2]

    # Each adjacent pair `(x_even, x_odd)` is treated like a 2D vector and
    # rotated by the angle for that token position.
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
    # RoPE is applied to queries and keys only. Values are not rotated because
    # values carry content that attention weights will mix after scoring.
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
    # The weights dictionary is intentionally flat because the assignment tests
    # pass reference weights by name. Each module has matching parameter names,
    # and the functional path uses those same names here.
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
    """
    Transformer language model forward pass using the provided weights.

    This is the stateless version of `TransformerLM.forward`. It exists so the
    tests can inject exact reference weights without constructing an optimizer
    or mutating a module.
    """
    del context_length
    x = embedding(weights["token_embeddings.weight"], in_indices)

    for layer_index in range(num_layers):
        prefix = f"layers.{layer_index}."
        # Pull out only the parameters for this block and remove the
        # `layers.N.` prefix, giving keys like `ln1.weight`.
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


class TransformerLMModule(TransformerLM):
    """Config-based alias used by the training and generation scripts."""

    def __init__(self, config: TransformerLMConfig) -> None:
        super().__init__(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
        )
        self.config = config


LinearModule = Linear
EmbeddingModule = Embedding
SiLUModule = SiLU
RMSNormModule = RMSNorm
SwiGLUModule = SwiGLU
ScaledDotProductAttentionModule = ScaledDotProductAttention
RoPEModule = RoPE
MultiHeadSelfAttentionModule = MultiHeadSelfAttention
MultiHeadSelfAttentionWithRoPEModule = MultiHeadSelfAttentionWithRoPE
TransformerBlockModule = TransformerBlock
