from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cs336_basics.BPE import _gpt2_bytes_to_unicode, train_bpe


DEFAULT_INPUT_PATH = Path("tests/fixtures/tinystories_sample_5M.txt")
DEFAULT_OUTPUT_DIR = Path("outputs/tinystories_bpe")
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]


def _bytes_to_gpt2_text(token_bytes: bytes) -> str:
    byte_encoder = _gpt2_bytes_to_unicode()
    return "".join(byte_encoder[byte] for byte in token_bytes)


def save_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"

    gpt2_vocab = {
        _bytes_to_gpt2_text(token_bytes): token_id
        for token_id, token_bytes in vocab.items()
    }
    with open(vocab_path, "w", encoding="utf-8") as vocab_file:
        json.dump(gpt2_vocab, vocab_file, ensure_ascii=False, indent=2)
        vocab_file.write("\n")

    with open(merges_path, "w", encoding="utf-8") as merges_file:
        for left, right in merges:
            merges_file.write(f"{_bytes_to_gpt2_text(left)} {_bytes_to_gpt2_text(right)}\n")

    return vocab_path, merges_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on TinyStories.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Training text path. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="Final vocabulary size, including byte tokens and special tokens.",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        default=None,
        help="Special token to keep indivisible. Repeat this flag for multiple tokens.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for vocab.json and merges.txt. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    special_tokens = args.special_tokens if args.special_tokens is not None else DEFAULT_SPECIAL_TOKENS

    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
    )
    elapsed_seconds = time.time() - start_time

    vocab_path, merges_path = save_tokenizer(vocab, merges, args.output_dir)

    print(f"trained vocab size: {len(vocab)}")
    print(f"learned merges: {len(merges)}")
    print(f"elapsed seconds: {elapsed_seconds:.2f}")
    print(f"vocab: {vocab_path}")
    print(f"merges: {merges_path}")


if __name__ == "__main__":
    main()
