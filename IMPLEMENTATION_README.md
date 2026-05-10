# Implementation Code Structure

This file explains where each assignment component lives.

## Beginner Reading Path

If you are new to the project, read it in this order:

1. `tests/adapters.py`: start here to see the exact functions the tests call.
   The adapters show the public interface expected by the assignment.
2. `cs336_basics/model.py`: read the top module comment, then compare a
   capitalized module class such as `Linear` with its lowercase helper
   function `linear`.
3. `cs336_basics/nn_utils.py`: the math is small and direct, so it is a good
   place to practice reading tensor code.
4. `cs336_basics/BPE.py`: read the overview comment first, then follow
   `train_bpe(...)` and `Tokenizer.encode(...)`.
5. `train_tinystories.py`: read this last. It combines tokenizer loading,
   batching, the model, optimizer, loss, and checkpointing into one training
   loop.

Useful mental model:

- Files in `cs336_basics/` are the implementation.
- Files in `tests/` describe the expected behavior.
- Adapter functions translate from the test API to the implementation API.
- Capitalized model classes are trainable `torch.nn.Module`s.
- Lowercase model functions are stateless math helpers used by tests and
  modules.

## Main Files

### `cs336_basics/BPE.py`

Byte-pair encoding tokenizer code.

Contains:

- `train_bpe`
- `Tokenizer`
- special-token splitting helpers
- pre-token counting helpers
- BPE pair counting and merge helpers

This file is the tokenizer section of the assignment.

### `cs336_basics/model.py`

Transformer and neural-network model code.

Contains:

- `TransformerLMConfig`
- `Linear`
- `Embedding`
- `SiLU`
- `RMSNorm`
- `SwiGLU`
- `ScaledDotProductAttention`
- `RoPE` / `RotaryPositionalEmbedding`
- `MultiHeadSelfAttention`
- `MultiHeadSelfAttentionWithRoPE`
- `TransformerBlock`
- `TransformerLM`
- `TransformerLMModule`
- `linear`
- `embedding`
- `silu`
- `swiglu`
- `rmsnorm`
- `scaled_dot_product_attention`
- `multihead_self_attention`
- `rope`
- `multihead_self_attention_with_rope`
- `transformer_block`
- `transformer_lm`

The comments explain tensor shapes and why each operation happens.

The capitalized classes are `torch.nn.Module` implementations that own
`torch.nn.Parameter`s and expose assignment-style state-dict names. `*Module`
aliases are also provided for the layer names where that convention is useful.
The lowercase functions remain as small functional helpers used by the modules
and for compatibility with older code.

`TransformerLMModule` is the config-based trainable wrapper used by
`train_tinystories.py`.

### `cs336_basics/nn_utils.py`

Small neural-network utility functions.

Contains:

- `softmax`
- `cross_entropy`
- `gradient_clipping`

These are written directly instead of calling PyTorch helper functions, so the
math is visible.

### `cs336_basics/data.py`

Data sampling for language-model training.

Contains:

- `get_batch`

It samples random token windows from a 1D token dataset and creates the shifted
labels.

### `cs336_basics/optim.py`

Optimization utilities.

Contains:

- `AdamW`
- `get_lr_cosine_schedule`

`AdamW` is implemented as a `torch.optim.Optimizer` subclass.

### `cs336_basics/serialization.py`

Checkpoint saving and loading.

Contains:

- `save_checkpoint`
- `load_checkpoint`

It saves model state, optimizer state, and the current iteration.

### `tests/adapters.py`

Adapter layer used by the tests.

The tests call functions in `tests/adapters.py`. Each adapter now imports the
real implementation from `cs336_basics/`. For module-based layers, the adapters
use `torch.func.functional_call` so the tests can run a module with exact
fixture weights without manually copying those weights into the module.

### `train_tinystories.py`

Section-5.3-style language-model training code.

Contains:

- tokenizer preparation for the TinyStories BPE tokenizer
- text-to-token caching
- train/validation splitting
- Transformer LM construction from laptop, smoke, or assignment presets
- AdamW training loop
- cosine learning-rate schedule
- cross-entropy loss
- gradient clipping
- checkpoint saving and resume support

### `generate_tinystories.py`

Decoding script for trained TinyStories checkpoints.

Contains:

- checkpoint loading
- tokenizer loading
- autoregressive generation
- temperature sampling
- top-k filtering

## Test Commands

Run the full test suite:

```bash
uv run pytest -q
```

Run only tokenizer/BPE tests:

```bash
uv run pytest tests/test_train_bpe.py tests/test_tokenizer.py -q
```

Train a BPE tokenizer on the TinyStories fixture:

```bash
uv run python -m cs336_basics.train_bpe_tinystories
```

By default this trains on `tests/fixtures/tinystories_sample_5M.txt` with
`vocab_size=1000` and writes:

```text
outputs/tinystories_bpe/vocab.json
outputs/tinystories_bpe/merges.txt
```

You can change the vocabulary size or output directory:

```bash
uv run python -m cs336_basics.train_bpe_tinystories --vocab-size 10000 --output-dir outputs/tinystories_bpe_10k
```

The script above is a small wrapper around this Python code:

```python
from pathlib import Path

from cs336_basics.BPE import Tokenizer, train_bpe
from cs336_basics.train_bpe_tinystories import save_tokenizer

input_path = Path("tests/fixtures/tinystories_sample_5M.txt")
output_dir = Path("outputs/tinystories_bpe")
special_tokens = ["<|endoftext|>"]

vocab, merges = train_bpe(
    input_path=input_path,
    vocab_size=1000,
    special_tokens=special_tokens,
)

vocab_path, merges_path = save_tokenizer(vocab, merges, output_dir)

tokenizer = Tokenizer.from_files(
    vocab_filepath=vocab_path,
    merges_filepath=merges_path,
    special_tokens=special_tokens,
)

text = "Once upon a time<|endoftext|>"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(ids)
print(decoded)
```

Offline BPE learning path:

```text
1. Read the top comments in cs336_basics/BPE.py for the full pipeline.
2. Read train_bpe(...) to see training call _count_pretokens, _count_adjacent_pairs,
   _choose_best_pair, and _apply_merge_to_counts.
3. Run the TinyStories command above and inspect outputs/tinystories_bpe/merges.txt.
4. Load vocab.json and merges.txt with Tokenizer.from_files(...) and try encode/decode.
```

Train the laptop-friendly TinyStories language model:

```bash
uv run python train_tinystories.py
```

The laptop preset is the default and is meant to be runnable without internet on
a normal laptop:

```text
vocab_size = 1000
context_length = 64
d_model = 128
num_layers = 4
num_heads = 4
d_ff = 512
rope_theta = 10000.0
batch_size = 16
max_iters = 1500
max_characters = 1000000
```

If it is still too slow on your machine:

```bash
uv run python train_tinystories.py --max-iters 500 --max-characters 300000
```

Run a tiny smoke test before a longer training run:

```bash
uv run python train_tinystories.py --preset smoke
```

Run the full assignment-shaped configuration only when you have enough compute:

```bash
uv run python train_tinystories.py --preset assignment
```

Generate from the trained model:

```bash
uv run python generate_tinystories.py --prompt "Once upon a time" --max-new-tokens 200
```

Run only model tests:

```bash
uv run pytest tests/test_model.py -q
```

Run only utility/data/optimizer/serialization tests:

```bash
uv run pytest tests/test_nn_utils.py tests/test_data.py tests/test_optimizer.py tests/test_serialization.py -q
```

## Current Verification

The full test suite passes:

```text
46 passed, 2 skipped
```
