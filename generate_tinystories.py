from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.BPE import Tokenizer
from cs336_basics.model import TransformerLMConfig, TransformerLMModule
from train_tinystories import choose_device

# ---------------------------------------------------------------------------
# Default paths used when the user does not override them on the command line.
# These are the same directories that the training script writes to by default
# when you use the "laptop" preset.
# ---------------------------------------------------------------------------
DEFAULT_RUN_DIR = Path("outputs/tinystories_lm_laptop")
DEFAULT_PROMPT = "Once upon a time"


def load_training_config(run_dir: Path) -> tuple[TransformerLMConfig, Path, list[str]]:
    """
    Read the training configuration that was saved by `train_tinystories.py`.

    When the training script finishes (or saves a checkpoint) it writes a
    small JSON file called `config.json` inside the run directory.  That file
    contains everything we need to reconstruct the exact same model shape and
    tokenizer so that the checkpoint can be loaded correctly.

    Returns three items:
        * model_config  – a TransformerLMConfig object (holds d_model, num_layers, etc.)
        * tokenizer_dir – path to the directory that contains vocab.json and merges.txt
        * special_tokens – list of strings that the tokenizer treats as atomic tokens
    """
    # ------------------------------------------------------------------
    # Step 1: Open the JSON configuration file.
    # The training script writes it as a human-readable dictionary, so we
    # can load it just like any other JSON file.
    # ------------------------------------------------------------------
    config_path = run_dir / "config.json"
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)

    # ------------------------------------------------------------------
    # Step 2: Rebuild the model configuration from the dictionary.
    # `config["model"]` contains keys like `vocab_size`, `d_model`, etc.
    # The `**` syntax unpacks the dictionary into keyword arguments.
    # ------------------------------------------------------------------
    model_config = TransformerLMConfig(**config["model"])

    # ------------------------------------------------------------------
    # Step 3: Get the tokenizer directory and special tokens.
    # These were stored as strings in the JSON, so we convert the
    # tokenizer directory back to a Path object for easier file handling.
    # ------------------------------------------------------------------
    tokenizer_dir = Path(config["tokenizer_dir"])
    special_tokens = list(config["special_tokens"])
    return model_config, tokenizer_dir, special_tokens


def apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """
    Zero out all logits except the `top_k` largest ones.

    This is a simple way to avoid sampling extremely unlikely tokens.
    Setting `top_k = 0` or `top_k = None` disables the filtering entirely,
    so the model can pick from the full vocabulary.

    How it works:
        1. Find the k-th largest logit value (the "threshold").
        2. Any logit *smaller* than that threshold is replaced with -inf.
        3. After softmax, -inf entries become 0, so they are never sampled.

    Args:
        logits: Raw output from the final LM head, shape (vocab_size,).
        top_k:  Maximum number of tokens to keep.  Can be None.

    Returns:
        A tensor with the same shape as `logits`, but with low values zeroed.
    """
    # If top_k is None or 0, we don't do any filtering – just return logits as is.
    if top_k is None or top_k <= 0:
        return logits

    # Make sure we don't try to keep more tokens than the vocabulary size.
    k = min(top_k, logits.shape[-1])

    # `torch.topk(logits, k).values` returns the k largest logits.
    # Taking [..., -1, None] grabs the *smallest* of those top k values
    # (the "threshold") and adds a trailing singleton dimension for broadcasting.
    threshold = torch.topk(logits, k).values[..., -1, None]

    # Replace every logit that is *below* the threshold with negative infinity.
    # `torch.finfo(logits.dtype).min` gives the smallest representable number
    # for the tensor's data type (e.g., about -3.4e38 for float32).
    return logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)


@torch.no_grad()
def generate(
    model: TransformerLMModule,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
) -> str:
    """
    Generate text by sampling one token at a time (autoregressive decoding).

    The process:
        1. Encode the prompt string into a list of token IDs.
        2. Loop for `max_new_tokens` steps:
           a) Feed the most recent `context_length` tokens to the model.
           b) Take the logits for the **last** position only (we only care
              about the next token after the current sequence).
           c) Divide by `temperature` to sharpen (T < 1) or flatten (T > 1)
              the distribution.
           d) Optionally apply top‑k filtering.
           e) Convert logits to probabilities with softmax.
           f) Sample a single token ID from those probabilities.
           g) Append the new token ID to the sequence.
        3. Decode the whole list of token IDs back into a human‑readable string.

    Args:
        model:         The trained TransformerLM model (in eval mode).
        tokenizer:     The BPE tokenizer that matches the model's vocabulary.
        prompt:        The initial text that the model should continue from.
        max_new_tokens: How many additional tokens to generate.
        temperature:   Scaling factor for logits before softmax.
                       * temperature < 1.0 → model becomes more "confident"
                                             (sharper distribution)
                       * temperature > 1.0 → model becomes more "creative"
                                             (flatter distribution)
        top_k:         If set, only the top‑k tokens are kept; others are
                       zeroed out before sampling.  None or 0 disables this.
        device:        The device the model lives on (cpu, cuda, or mps).

    Returns:
        The full generated text (prompt + continuation) as a single string.
    """
    # ------------------------------------------------------------------
    # Put the model into evaluation mode.
    # This disables things like dropout that are only used during training.
    # ------------------------------------------------------------------
    model.eval()

    # ------------------------------------------------------------------
    # Encode the prompt: a string → a list of integer token IDs.
    # Example: "Once upon a time" → [45, 23, 890, 12]
    # ------------------------------------------------------------------
    token_ids = tokenizer.encode(prompt)

    # ------------------------------------------------------------------
    # Main generation loop – we add one new token per iteration.
    # ------------------------------------------------------------------
    for _ in range(max_new_tokens):
        # The model has a fixed `context_length` (e.g., 256 tokens).
        # If our sequence is longer than that, we must discard the oldest
        # tokens and only keep the most recent `context_length` tokens.
        # This is called a "sliding window" over the sequence.
        context_ids = token_ids[-model.config.context_length :]

        # Convert the list of token IDs into a PyTorch tensor.
        # The shape becomes (1, seq_len)  – 1 batch, seq_len tokens.
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)

        # ----- Forward pass through the model -----
        # `model(input_ids)` returns a tensor of shape (1, seq_len, vocab_size).
        # We only need the prediction for the very last position,
        # because that tells us what the *next* token should be.
        # `[0, -1]` means: batch index 0, last sequence position.
        logits = model(input_ids)[0, -1]          # shape: (vocab_size,)

        # ----- Temperature scaling -----
        # Dividing logits by temperature changes how "peaked" the softmax
        # distribution will be.  We clamp temperature to at least 1e-8 to
        # avoid division by zero (which would produce NaN/inf).
        logits = logits / max(temperature, 1e-8)

        # ----- Optional top‑k filtering -----
        logits = apply_top_k(logits, top_k)

        # ----- Turn logits into probabilities -----
        # softmax exponentiates the logits and normalises them so they
        # sum to 1.  The result is a proper probability distribution over
        # every token in the vocabulary.
        probabilities = torch.softmax(logits, dim=-1)

        # ----- Sample the next token -----
        # `torch.multinomial` draws one sample from the probability
        # distribution.  Higher-probability tokens are more likely to be
        # chosen.  `.item()` extracts the single integer from the tensor.
        next_id = torch.multinomial(probabilities, num_samples=1).item()

        # Append the newly-generated token ID to our running list.
        token_ids.append(next_id)

    # ------------------------------------------------------------------
    # After generating all the tokens, decode the full list of token IDs
    # back into a Unicode string and return it.
    # ------------------------------------------------------------------
    return tokenizer.decode(token_ids)


def parse_args() -> argparse.Namespace:
    """
    Build the command‑line interface for the generation script.

    Every option has a sensible default, so a bare `python generate_tinystories.py`
    will produce a story using the latest "laptop" checkpoint and the
    default prompt "Once upon a time".
    """
    parser = argparse.ArgumentParser(
        description="Generate TinyStories‑like text from a trained checkpoint."
    )
    # ------------------------------------------------------------------
    # --run-dir: Directory that contains config.json and the checkpoints/ folder.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
    )
    # ------------------------------------------------------------------
    # --checkpoint: Specific checkpoint file to load. If omitted, uses the
    #               'final.pt' checkpoint.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    # ------------------------------------------------------------------
    # --prompt: Starting text that the model will continue from.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
    )
    # ------------------------------------------------------------------
    # --max-new-tokens: How many additional tokens to generate after the prompt.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
    )
    # ------------------------------------------------------------------
    # --temperature: Temperature for logit scaling. Lower = more focused,
    #                higher = more diverse.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )
    # ------------------------------------------------------------------
    # --top-k: Keep only the top‑k most likely tokens before sampling.
    #          0 disables this filtering.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
    )
    # ------------------------------------------------------------------
    # --device: Device to run on: 'cpu', 'cuda', 'mps', or 'auto' to pick
    #           the best available.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )
    # ------------------------------------------------------------------
    # --seed: Random seed for reproducibility of the generated text.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point of the script.

    Steps:
        1. Parse command‑line arguments.
        2. Choose the device (CPU / GPU / MPS).
        3. Load the training configuration and tokenizer.
        4. Build the model and load the trained checkpoint.
        5. Call `generate()` and print the resulting text.
    """
    # ----- Parse arguments and set up the device & random seed -----
    args = parse_args()
    device = choose_device(args.device)
    torch.manual_seed(args.seed)

    # ----- Recover the model configuration and tokenizer -----
    model_config, tokenizer_dir, special_tokens = load_training_config(args.run_dir)
    tokenizer = Tokenizer.from_files(
        vocab_filepath=tokenizer_dir / "vocab.json",
        merges_filepath=tokenizer_dir / "merges.txt",
        special_tokens=special_tokens,
    )

    # ----- Build the model and load the trained weights -----
    model = TransformerLMModule(model_config).to(device)

    # If the user did not specify a checkpoint, default to the final one.
    checkpoint_path = args.checkpoint or (args.run_dir / "checkpoints" / "final.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}. "
            f"Run `uv run python train_tinystories.py` first."
        )

    # `torch.load` reads the saved checkpoint file.
    # `map_location=device` makes sure the tensors are placed on the correct device
    # (e.g., if the checkpoint was saved on GPU but we're now running on CPU).
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # `load_state_dict` copies all the saved model parameters (weights and biases)
    # into the model we just created.  This is the step that restores the trained
    # model from disk.
    model.load_state_dict(checkpoint["model"])

    # ----- Generate and print the text -----
    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(text)


# ---------------------------------------------------------------------------
# Python convention: if this file is run directly (not imported), call main().
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()