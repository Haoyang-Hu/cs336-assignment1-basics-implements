from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# These imports come from your own cs336_basics package.  Each one is a
# building block you implemented earlier in the assignment.
# ---------------------------------------------------------------------------
from cs336_basics.BPE import Tokenizer, train_bpe
from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLMConfig, TransformerLMModule
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optim import AdamW, get_lr_cosine_schedule
from cs336_basics.serialization import load_checkpoint, save_checkpoint
from cs336_basics.train_bpe_tinystories import save_tokenizer

# ---------------------------------------------------------------------------
# Default file paths.  These are used when the user doesn't override them on
# the command line.  The script is designed to work straight out of the box
# with the fixture data included in the repository.
# ---------------------------------------------------------------------------
DEFAULT_TEXT_PATH = Path("tests/fixtures/tinystories_sample_5M.txt")
DEFAULT_TOKENIZER_DIR = Path("outputs/tinystories_bpe")
DEFAULT_RUN_DIR = Path("outputs/tinystories_lm_laptop")
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]

# ---------------------------------------------------------------------------
# PRESETS
# ---------------------------------------------------------------------------
# The script can run in three different modes, selected by ``--preset``.
# Each preset is a dictionary containing values for every configurable
# hyper‑parameter.  When you type ``--preset laptop``, the script fills in
# any arguments you didn't explicitly provide on the command line.
#
# `laptop` – small model + small dataset so you can complete a full training
#            run on an ordinary laptop in a reasonable amount of time.
# `assignment` – uses the exact model size and token budget from the
#                assignment handout.  Requires a strong GPU or a lot of
#                patience.
# `smoke` – a tiny run (2 iterations) that only checks whether the code
#           starts without crashing.
# ---------------------------------------------------------------------------
PRESETS = {
    "laptop": {
        "tokenizer_dir": Path("outputs/tinystories_bpe"),
        "run_dir": Path("outputs/tinystories_lm_laptop"),
        "vocab_size": 1000,
        "context_length": 64,
        "d_model": 128,
        "num_layers": 4,
        "num_heads": 4,
        "d_ff": 512,
        "rope_theta": 10000.0,
        "batch_size": 16,
        "max_iters": 1500,
        "max_lr": 3e-4,
        "min_lr": 3e-5,
        "warmup_iters": 100,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "val_fraction": 0.02,
        "eval_interval": 100,
        "eval_iters": 5,
        "checkpoint_interval": 500,
        "max_characters": 1_000_000,
    },
    "assignment": {
        "tokenizer_dir": Path("outputs/tinystories_bpe_10k"),
        "run_dir": Path("outputs/tinystories_lm_assignment"),
        "vocab_size": 10000,
        "context_length": 256,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "rope_theta": 10000.0,
        "batch_size": 128,
        "max_iters": 10000,
        "max_lr": 3e-4,
        "min_lr": 3e-5,
        "warmup_iters": 1000,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "val_fraction": 0.01,
        "eval_interval": 100,
        "eval_iters": 10,
        "checkpoint_interval": 1000,
        "max_characters": None,
    },
    "smoke": {
        "tokenizer_dir": Path("outputs/tinystories_bpe"),
        "run_dir": Path("outputs/tinystories_lm_smoke"),
        "vocab_size": 1000,
        "context_length": 32,
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 128,
        "rope_theta": 10000.0,
        "batch_size": 4,
        "max_iters": 2,
        "max_lr": 3e-4,
        "min_lr": 3e-5,
        "warmup_iters": 1,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "val_fraction": 0.02,
        "eval_interval": 1,
        "eval_iters": 1,
        "checkpoint_interval": 2,
        "max_characters": 100_000,
    },
}


def choose_device(requested_device: str) -> torch.device:
    """
    Choose where training should run.

    `auto` uses CUDA on NVIDIA GPUs, then Apple Silicon MPS, then CPU. On a
    Mac laptop, MPS is usually much faster than CPU if it is available.
    """
    # If the user explicitly asked for a specific device (e.g., "cpu"), use it.
    if requested_device != "auto":
        return torch.device(requested_device)

    # Otherwise, automatically pick the fastest available device.
    # CUDA is for NVIDIA GPUs, MPS is for Apple Silicon (M1/M2/M3/M4) GPUs.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenizer_matches_vocab_size(vocab_path: Path, merges_path: Path, vocab_size: int) -> bool:
    """Return True when an existing tokenizer has the requested vocab size."""
    # The tokenizer directory must contain both a vocab.json and a merges.txt.
    if not vocab_path.exists() or not merges_path.exists():
        return False

    # Load the JSON vocabulary and check how many tokens it contains.
    with open(vocab_path, encoding="utf-8") as vocab_file:
        vocab = json.load(vocab_file)
    return len(vocab) == vocab_size


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    """
    Fill unspecified command‑line arguments from the selected preset.

    argparse stores every optional override as `None` by default. That lets us
    distinguish "the user did not set this" from "the user intentionally set
    this to a value." For example, `--preset laptop --max-iters 3000` keeps all
    laptop defaults except `max_iters`.
    """
    # Get the dictionary of values for the chosen preset (e.g. "laptop").
    preset = PRESETS[args.preset]

    # For every key in the preset, if the user didn't provide a value (still
    # None), fill it with the preset's default.
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    # Fill in the text path and special tokens with their own defaults.
    if args.text_path is None:
        args.text_path = DEFAULT_TEXT_PATH
    if args.special_tokens is None:
        args.special_tokens = DEFAULT_SPECIAL_TOKENS

    return args


def prepare_tokenizer(args: argparse.Namespace) -> Tokenizer:
    """
    Load the TinyStories tokenizer, training it first if needed.

    The assignment model uses `vocab_size=10000`. Your earlier BPE demo may
    have produced a smaller 1000‑token vocab, so this function checks the size
    before reusing files.
    """
    # Paths where the vocab and merge files are stored.
    vocab_path = args.tokenizer_dir / "vocab.json"
    merges_path = args.tokenizer_dir / "merges.txt"

    # If the tokenizer files are missing or have the wrong size, train a new
    # BPE tokenizer from scratch.  This can take several minutes for larger
    # vocabularies.
    if not tokenizer_matches_vocab_size(vocab_path, merges_path, args.vocab_size):
        print(f"training BPE tokenizer with vocab_size={args.vocab_size} ...")
        vocab, merges = train_bpe(
            input_path=args.text_path,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
        )
        # Save the trained tokenizer so we don't have to retrain next time.
        vocab_path, merges_path = save_tokenizer(vocab, merges, args.tokenizer_dir)
        print(f"saved tokenizer to {args.tokenizer_dir}")

    # Return a Tokenizer object that can encode/decode text.
    return Tokenizer.from_files(vocab_path, merges_path, special_tokens=args.special_tokens)


def load_or_create_token_cache(args: argparse.Namespace, tokenizer: Tokenizer) -> np.ndarray:
    """
    Convert the text file to token IDs once and cache the result as ``.npy``.

    Training repeatedly samples from token IDs, not raw text. Saving the encoded
    array makes later runs start much faster.
    """
    # Create the output directory if it doesn't exist yet.
    args.run_dir.mkdir(parents=True, exist_ok=True)

    # Build a unique filename for this combination of vocab size and text
    # partition.  This way different presets don't overwrite each other.
    partition_name = "all" if args.max_characters is None else f"chars{args.max_characters}"
    token_cache_path = args.run_dir / f"tinystories_tokens_vocab{args.vocab_size}_{partition_name}.npy"

    # If we already encoded this data before, just load it from disk.
    if token_cache_path.exists():
        return np.load(token_cache_path)

    # ---- Encoding pass ----
    print(f"encoding {args.text_path} to token IDs ...")
    if args.max_characters is not None:
        print(f"using only the first {args.max_characters:,} characters for this run")

    # `encode_iterable` processes text chunk by chunk, so very large files
    # never need to be entirely in memory at once.
    token_ids = list(tokenizer.encode_iterable(iter_text_chunks(args.text_path, args.max_characters)))

    # Choose a compact integer type.  `uint16` can hold token IDs up to 65535,
    # which is plenty for our 10k vocabulary, and uses half the space of
    # `int64`.  For larger vocabularies we fall back to `int64`.
    token_dtype = np.uint16 if args.vocab_size <= np.iinfo(np.uint16).max else np.int64
    tokens = np.asarray(token_ids, dtype=token_dtype)

    # Save the encoded array so next time we can skip the encoding step.
    np.save(token_cache_path, tokens)
    print(f"saved {len(tokens):,} token IDs to {token_cache_path}")
    return tokens


def iter_text_chunks(text_path: Path, max_characters: int | None):
    """
    Yield text chunks from the dataset, optionally stopping early.

    This is how the laptop preset uses a smaller partition of TinyStories. We
    read line by line so memory stays small and so the code works offline with
    the local fixture file already in ``tests/fixtures/``.
    """
    characters_seen = 0
    with open(text_path, encoding="utf-8") as text_file:
        for line in text_file:
            # If no limit is set, just pass through every line.
            if max_characters is None:
                yield line
                continue

            # If we have already read enough characters, stop.
            remaining = max_characters - characters_seen
            if remaining <= 0:
                break

            # If the line is longer than the remaining budget, truncate it.
            chunk = line[:remaining]
            characters_seen += len(chunk)
            yield chunk


def split_train_val(
    tokens: np.ndarray,
    val_fraction: float,
    context_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Use the beginning for training and the end for validation.

    We keep at least ``context_length + 2`` tokens in both splits. This prevents
    tiny smoke‑test splits from being too short for ``get_batch``, which needs an
    input window plus the next‑token labels.
    """
    # The smallest split that still allows us to sample a valid batch.
    minimum_split_tokens = context_length + 2
    if len(tokens) < 2 * minimum_split_tokens:
        raise ValueError(
            f"not enough tokens ({len(tokens)}) for context_length={context_length}; "
            "increase --max-characters or lower --context-length"
        )

    # Compute how many tokens should go to validation.
    val_size = max(minimum_split_tokens, int(len(tokens) * val_fraction))
    # Make sure we still leave enough for training.
    val_size = min(val_size, len(tokens) - minimum_split_tokens)

    # Split point: the last `val_size` tokens become validation, the rest train.
    split_index = len(tokens) - val_size
    train_tokens = tokens[:split_index]
    val_tokens = tokens[split_index:]
    return train_tokens, val_tokens


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """
    Move optimizer momentum tensors after loading a checkpoint.

    ``torch.load(..., map_location="cpu")`` is safe, but optimizer states must
    live on the same device as the parameters before the next optimizer step.
    """
    # The optimizer keeps extra tensors for each parameter (e.g., AdamW's m and v).
    # We need to move those tensors to the same device the model is on.
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


@torch.no_grad()
def estimate_loss(
    model: TransformerLMModule,
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Average a few random batches so the validation number is less noisy."""
    # Switch the model to evaluation mode – this turns off dropout etc.
    model.eval()
    losses: dict[str, float] = {}

    # Evaluate on both train and val splits.
    for split_name, tokens in [("train", train_tokens), ("val", val_tokens)]:
        split_losses = []
        # We average over several random batches to get a stable estimate.
        for _ in range(args.eval_iters):
            x, y = get_batch(tokens, args.batch_size, args.context_length, str(device))
            logits = model(x)
            # Cross‑entropy expects (batch*seq, vocab) and (batch*seq,).
            loss = cross_entropy(
                logits.reshape(-1, args.vocab_size),
                y.reshape(-1),
            )
            split_losses.append(loss.item())
        losses[split_name] = sum(split_losses) / len(split_losses)

    # Put the model back in training mode before returning.
    model.train()
    return losses


def save_run_config(args: argparse.Namespace, model_config: TransformerLMConfig) -> None:
    """Write the exact settings needed by ``generate_tinystories.py``."""
    # Create the run directory if it doesn't exist.
    args.run_dir.mkdir(parents=True, exist_ok=True)

    # Gather all the information that a generation script would need.
    config = {
        "model": asdict(model_config),
        "preset": args.preset,
        "tokenizer_dir": str(args.tokenizer_dir),
        "special_tokens": args.special_tokens,
        "text_path": str(args.text_path),
        "max_characters": args.max_characters,
    }

    # Save it as a human‑readable JSON file.
    with open(args.run_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")


def print_run_summary(args: argparse.Namespace, device: torch.device, token_count: int) -> None:
    """Print the settings that matter most before training starts."""
    tokens_per_step = args.batch_size * args.context_length
    planned_tokens = tokens_per_step * args.max_iters
    print("training summary:")
    print(f"  preset: {args.preset}")
    print(f"  device: {device}")
    print(f"  tokenizer dir: {args.tokenizer_dir}")
    print(f"  run dir: {args.run_dir}")
    print(f"  cached token IDs: {token_count:,}")
    print(f"  model: layers={args.num_layers}, d_model={args.d_model}, heads={args.num_heads}, d_ff={args.d_ff}")
    print(f"  context={args.context_length}, batch={args.batch_size}, steps={args.max_iters}")
    print(f"  planned training tokens: {planned_tokens:,}")


def parse_args() -> argparse.Namespace:
    """Build the command‑line interface and apply the selected preset."""
    parser = argparse.ArgumentParser(
        description="Train a laptop‑friendly Transformer LM on TinyStories."
    )

    # --preset: which predefined configuration to use as defaults.
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="laptop",
    )
    # --text-path: path to the raw text file (UTF‑8).
    parser.add_argument("--text-path", type=Path, default=None)
    # --tokenizer-dir: where to save/load the BPE tokenizer.
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    # --run-dir: where to write checkpoints, logs, and config.
    parser.add_argument("--run-dir", type=Path, default=None)
    # --special-token: special tokens that the tokenizer must keep intact.
    parser.add_argument("--special-token", action="append", dest="special_tokens", default=None)

    # Model architecture parameters – if not given, the preset fills them in.
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=None)

    # Training hyper‑parameters – if not given, the preset fills them in.
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--max-lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)

    # --max-characters: use only the first N characters of the text file.
    parser.add_argument(
        "--max-characters",
        type=int,
        default=None,
    )

    # Logging and checkpointing intervals.
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)

    # --resume-from: path to a checkpoint file to continue training.
    parser.add_argument("--resume-from", type=Path, default=None)

    # Runtime settings.
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1337)

    # Parse the actual command line and then fill any missing values from the
    # chosen preset.
    return apply_preset(parser.parse_args())


def main() -> None:
    """
    Main entry point of the training script.

    This function orchestrates all the steps:
        1. Parse arguments and select device.
        2. Prepare tokenizer and token cache.
        3. Build the model and optimizer.
        4. (Optionally) resume from a checkpoint.
        5. Run the training loop: sample batches, forward/backward, log, save.
    """
    # ----- Step 1: Parse command‑line arguments and apply the preset -----
    args = parse_args()
    device = choose_device(args.device)
    print(f"using device: {device}")

    # ----- Step 2: Set random seeds for reproducibility -----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----- Step 3: Tokenizer -----
    # Load (or train) the BPE tokenizer.
    tokenizer = prepare_tokenizer(args)
    # Encode the whole text into a NumPy array of token IDs, or load a cached copy.
    tokens = load_or_create_token_cache(args, tokenizer)
    # Split into training and validation sets.
    train_tokens, val_tokens = split_train_val(tokens, args.val_fraction, args.context_length)
    print_run_summary(args, device, len(tokens))

    # ----- Step 4: Model -----
    # Build the Transformer configuration object from the preset/args values.
    model_config = TransformerLMConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    # Create the model and move it to the chosen device (GPU, MPS, or CPU).
    model = TransformerLMModule(model_config).to(device)

    # ----- Step 5: Optimizer -----
    # Create the AdamW optimizer.  It will update all learnable model parameters.
    optimizer = AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)

    # ----- Step 6: (Optional) resume from a checkpoint -----
    start_iteration = 0
    if args.resume_from is not None:
        # `load_checkpoint` restores both model weights and optimizer state.
        start_iteration = load_checkpoint(args.resume_from, model, optimizer)
        # The optimizer's tensors may still be on CPU, so move them to the correct device.
        move_optimizer_state_to_device(optimizer, device)
        print(f"resumed from {args.resume_from} at iteration {start_iteration}")

    # ----- Step 7: Save the run configuration for future generation -----
    save_run_config(args, model_config)

    # ----- Step 8: Prepare checkpoint directory -----
    checkpoint_dir = args.run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ----- Step 9: Training loop -----
    model.train()
    last_log_time = time.time()

    # Iterate from `start_iteration` to `max_iters - 1`.
    for iteration in range(start_iteration, args.max_iters):
        # --- Learning rate schedule ---
        # Compute the learning rate for this iteration using a cosine schedule
        # that first warms up linearly, then decays.
        lr = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- Sample a batch of data ---
        # x: input token IDs, shape (batch_size, context_length)
        # y: target token IDs, same shape, shifted by one position.
        x, y = get_batch(train_tokens, args.batch_size, args.context_length, str(device))

        # --- Forward pass ---
        logits = model(x)   # shape: (batch_size, context_length, vocab_size)

        # --- Compute loss ---
        # Cross‑entropy compares the predicted logits against the true next tokens.
        loss = cross_entropy(
            logits.reshape(-1, args.vocab_size),
            y.reshape(-1),
        )

        # --- Backward pass ---
        optimizer.zero_grad(set_to_none=True)   # reset gradients from previous step
        loss.backward()                         # compute gradients w.r.t. loss

        # --- Gradient clipping ---
        # Prevents individual gradient updates from being too large, which can
        # destabilise training.
        gradient_clipping(model.parameters(), args.grad_clip)

        # --- Optimizer step ---
        # Update model parameters using the computed gradients.
        optimizer.step()

        # --- Logging ---
        step = iteration + 1  # human‑friendly step counter (1‑based)
        if step == 1 or step % args.eval_interval == 0:
            elapsed = time.time() - last_log_time
            last_log_time = time.time()
            losses = estimate_loss(model, train_tokens, val_tokens, args, device)
            print(
                f"iter {step:6d} | "
                f"lr {lr:.2e} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"{elapsed:.1f}s"
            )

        # --- Save checkpoint ---
        if step % args.checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{step:06d}.pt"
            save_checkpoint(model, optimizer, step, checkpoint_path)
            print(f"saved checkpoint: {checkpoint_path}")

    # ----- After training: save the final checkpoint -----
    final_checkpoint_path = checkpoint_dir / "final.pt"
    save_checkpoint(model, optimizer, args.max_iters, final_checkpoint_path)
    print(f"saved final checkpoint: {final_checkpoint_path}")


# ---------------------------------------------------------------------------
# Standard Python entry point.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()