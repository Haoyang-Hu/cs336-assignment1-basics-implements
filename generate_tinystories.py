from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.BPE import Tokenizer
from cs336_basics.model import TransformerLMConfig, TransformerLMModule
from train_tinystories import choose_device


# These defaults match the laptop-friendly training preset.
DEFAULT_RUN_DIR = Path("outputs/tinystories_lm_laptop")
DEFAULT_PROMPT = "Once upon a time"


def load_training_config(run_dir: Path) -> tuple[TransformerLMConfig, Path, list[str]]:
    """
    Read the `config.json` file saved by `train_tinystories.py`.

    A checkpoint only stores tensors. The config file stores the information
    needed to rebuild the model and tokenizer around those tensors: vocabulary
    size, model width, number of layers, tokenizer path, and special tokens.
    """
    config_path = run_dir / "config.json"
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)

    model_config = TransformerLMConfig(**config["model"])
    tokenizer_dir = Path(config["tokenizer_dir"])
    special_tokens = list(config["special_tokens"])
    return model_config, tokenizer_dir, special_tokens


def apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """
    Keep only the `top_k` most likely tokens before sampling.

    The model produces one logit per vocabulary token. Top-k sampling masks out
    every token except the k highest-logit choices. After softmax, masked tokens
    have probability 0, so they cannot be sampled.
    """
    if top_k is None or top_k <= 0:
        return logits

    k = min(top_k, logits.shape[-1])

    # The smallest value among the top-k logits is the cutoff. Anything below
    # it is outside the top-k set.
    threshold = torch.topk(logits, k).values[..., -1, None]

    # Use the smallest finite number for this dtype as a practical "-infinity".
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
    Generate text one token at a time.

    This is autoregressive decoding: each new token is appended to the prompt,
    then the longer sequence is fed back into the model to predict the next
    token. Temperature and top-k control how random the sampling is.
    """
    model.eval()
    token_ids = tokenizer.encode(prompt)

    for _ in range(max_new_tokens):
        # The model has a fixed context window. If the generated text becomes
        # longer than that, keep only the most recent tokens.
        context_ids = token_ids[-model.config.context_length :]
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)

        # `model(input_ids)` has shape `(1, sequence, vocab_size)`. The final
        # position is the model's prediction for the next token.
        logits = model(input_ids)[0, -1]

        # Temperature changes how sharp the distribution is:
        # - lower than 1.0: safer and more repetitive
        # - equal to 1.0: unchanged
        # - higher than 1.0: more random and surprising
        logits = logits / max(temperature, 1e-8)
        logits = apply_top_k(logits, top_k)

        probabilities = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probabilities, num_samples=1).item()
        token_ids.append(next_id)

    return tokenizer.decode(token_ids)


def parse_args() -> argparse.Namespace:
    """Build the command-line interface for text generation."""
    parser = argparse.ArgumentParser(description="Generate TinyStories-like text from a trained checkpoint.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Lower is safer; higher is more random.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Keep only the top-k token choices before sampling. Use 0 to disable.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def main() -> None:
    """
    Load a trained model, generate a continuation, and print it.

    The training script writes both `config.json` and `checkpoints/final.pt`.
    This script reads those files from `--run-dir` by default.
    """
    args = parse_args()
    device = choose_device(args.device)
    torch.manual_seed(args.seed)

    model_config, tokenizer_dir, special_tokens = load_training_config(args.run_dir)
    tokenizer = Tokenizer.from_files(
        vocab_filepath=tokenizer_dir / "vocab.json",
        merges_filepath=tokenizer_dir / "merges.txt",
        special_tokens=special_tokens,
    )

    model = TransformerLMModule(model_config).to(device)
    checkpoint_path = args.checkpoint or (args.run_dir / "checkpoints" / "final.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}. "
            "Run `uv run python train_tinystories.py` first."
        )

    # `map_location=device` lets a CPU laptop load a checkpoint that may have
    # been saved on a GPU, or the other way around.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

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


if __name__ == "__main__":
    main()
