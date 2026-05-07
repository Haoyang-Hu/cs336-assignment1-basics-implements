from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Iterator
from functools import lru_cache
from typing import BinaryIO

import regex


# ---------------------------------------------------------------------------
# BPE overview
# ---------------------------------------------------------------------------
#
# The BPE pipeline has two major parts:
#
# 1. Training:
#    - Read a text corpus.
#    - Split around special tokens so special tokens are never learned as normal
#      byte merges.
#    - Pre-tokenize regular text with the GPT-2 regex below.
#    - Convert each pre-token into UTF-8 bytes.
#    - Repeatedly find the most frequent adjacent byte/token pair and merge it.
#    - Record each chosen merge in order.
#
# 2. Tokenization:
#    - Split input text around special tokens.
#    - Pre-tokenize non-special text with the same GPT-2 regex.
#    - Convert each pre-token to bytes.
#    - Apply the learned merges in training order.
#    - Convert final byte tokens to integer token IDs.
#
# Implementation details:
# - Vocabulary maps token_id -> token_bytes.
# - Merges are ordered pairs of bytes: list[tuple[bytes, bytes]].
# - Raw bytes 0..255 must be available as base tokens.
# - Special tokens are indivisible tokens.
# - When training BPE, frequency ties should be broken by choosing the
#   lexicographically greatest pair.


# GPT-2 pre-tokenization regex.
#
# This regex creates "pre-tokens" such as words, punctuation groups, numbers,
# and whitespace groups. BPE merges happen inside each pre-token, not across
# pre-token boundaries.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_RE = regex.compile(PAT)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.

    This helper is useful when you later parallelize pre-token counting. It
    returns byte offsets. Each adjacent pair `(start, end)` describes one chunk:

        file.seek(start)
        chunk = file.read(end - start)

    The boundaries are first guessed by file size, then moved forward until the
    next `split_special_token`. That keeps examples separated by the special
    token from being split across chunks.

    The BPE training and tokenizer framework begins below.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Work in byte offsets. `seek` and `tell` operate on bytes, not decoded text
    # characters.
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial evenly spaced guesses. The first boundary is always 0. The final
    # boundary is always EOF.
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    # Read ahead in small windows while searching for the next special token.
    mini_chunk_size = 4096

    # Only adjust interior boundaries. The first and last boundaries are fixed.
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            # If we hit EOF before finding the split token, this boundary
            # collapses to the end of the file.
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Move the boundary to the beginning of the next special token.
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            # Continue scanning in the next small window.
            initial_position += mini_chunk_size

    # Duplicate boundaries can happen if several guesses move to the same
    # special token or to EOF.
    return sorted(set(chunk_boundaries))


# ---------------------------------------------------------------------------
# Special-token helpers
# ---------------------------------------------------------------------------


def _build_special_token_regex(special_tokens: list[str] | None) -> regex.Pattern | None:
    """Build a regex that finds configured special tokens."""
    if not special_tokens:
        return None

    unique_tokens = {token for token in special_tokens if token}
    if not unique_tokens:
        return None

    # Match longer tokens first so overlapping tokens are kept whole.
    sorted_tokens = sorted(unique_tokens, key=lambda token: (-len(token), token))

    # Escape tokens so characters like "|" are treated literally.
    escaped_tokens = [regex.escape(token) for token in sorted_tokens]
    return regex.compile("(?:" + "|".join(escaped_tokens) + ")")


def _split_on_special_tokens(text: str, special_tokens: list[str] | None) -> list[tuple[str, bool]]:
    """
    Split text into `(piece, is_special)` records.
    """
    # Special tokens act as hard boundaries. The tokenizer should keep them
    # whole, while normal text around them is handled by pre-tokenization later.
    special_re = _build_special_token_regex(special_tokens)
    if special_re is None:
        return [(text, False)] if text else []

    pieces: list[tuple[str, bool]] = []

    # `last_end` is the first character index that has not been copied into
    # `pieces` yet.
    last_end = 0

    for match in special_re.finditer(text):
        # Text between `last_end` and `match.start()` is normal text before this
        # special token. Skip this append when the span is empty.
        if match.start() > last_end:
            pieces.append((text[last_end:match.start()], False))

        # `match.group()` is the actual special token string that was found.
        pieces.append((match.group(), True))

        # Everything through this special token has now been handled.
        last_end = match.end()

    # Add any normal text after the final special token.
    if last_end < len(text):
        pieces.append((text[last_end:], False))

    return pieces



# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _bytes_to_tuple(token_bytes: bytes) -> tuple[bytes, ...]:
    """
    Convert raw bytes into one-byte BPE tokens.

    Example:

        b"cat" -> (b"c", b"a", b"t")
    """
    # Each integer from iterating over `bytes` is wrapped back into a length-1
    # bytes object, because BPE tokens are stored as `bytes`, not as integers.
    return tuple(bytes([byte]) for byte in token_bytes)


def _pretoken_to_byte_tuple(pretoken: str) -> tuple[bytes, ...]:
    """
    Convert one regex pre-token to its initial byte-token sequence.

    A visible non-ASCII character may be multiple UTF-8 bytes, and byte-level
    BPE starts from those individual bytes.
    """
    # Python strings are Unicode text. Encoding turns the visible text into the
    # raw UTF-8 bytes that byte-level BPE actually trains on.
    token_bytes = pretoken.encode("utf-8")
    return _bytes_to_tuple(token_bytes)


def _count_pretokens(
    text: str,
    special_tokens: list[str] | None,
) -> Counter[tuple[bytes, ...]]:
    """
    Count how many times each pre-token byte sequence appears.

    The returned Counter maps:

        tuple_of_current_tokens -> frequency

    At the beginning of training, each tuple element is a one-byte token. Later,
    after merges, tuple elements can be multi-byte tokens.

    Special tokens are skipped so BPE does not learn merges inside their
    spelling, such as b"<|" from "<|endoftext|>".
    """
    counts: Counter[tuple[bytes, ...]] = Counter()

    for piece, is_special in _split_on_special_tokens(text, special_tokens):
        # Special tokens already have their own vocab entries. Skipping them
        # here prevents normal BPE merges from being learned inside them.
        if is_special:
            continue

        for match in PRETOKEN_RE.finditer(piece):
            pretoken = match.group()
            # The Counter key is the byte-token tuple, not the function object.
            counts[_pretoken_to_byte_tuple(pretoken)] += 1

    return counts



def _count_adjacent_pairs(
    word_counts: Counter[tuple[bytes, ...]],
) -> Counter[tuple[bytes, bytes]]:
    """
    Count adjacent token pairs across all pre-token types.

    Example:

        word_counts[(b"l", b"o", b"w")] = 3

    contributes:

        (b"l", b"o") += 3
        (b"o", b"w") += 3
    """
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()

    # `.items()` gives both the pre-token tuple and how often it occurred.
    for word, frequency in word_counts.items():
        # `zip(word, word[1:])` creates adjacent pairs:
        # (a, b, c) -> (a, b), (b, c)
        for pair in zip(word, word[1:]):
            pair_counts[pair] += frequency

    return pair_counts



def _choose_best_pair(pair_counts: Counter[tuple[bytes, bytes]]) -> tuple[bytes, bytes]:
    """
    Choose the next pair to merge.

    Ties are resolved by choosing the lexicographically greatest pair.
    """
    # `max` compares tuple items from left to right, so this chooses the largest
    # count first. If counts tie, it chooses the greatest byte pair.
    return max((count, pair) for pair, count in pair_counts.items())[1]



def _merge_pair_in_word(
    word: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    """
    Merge every non-overlapping occurrence of `pair` inside one pre-token.

    Example:

        word = (b"a", b"b", b"c", b"b", b"c")
        pair = (b"b", b"c")
        result = (b"a", b"bc", b"bc")
    """

    merged_word: list[bytes] = []
    i = 0

    while i < len(word):
        # If the current token and next token match the target pair, combine
        # them into one larger token and skip over both old tokens.
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged_word.append(pair[0] + pair[1])
            i += 2
        else:
            # Otherwise keep the current token unchanged.
            merged_word.append(word[i])
            i += 1

    return tuple(merged_word)


def _apply_merge_to_counts(
    word_counts: Counter[tuple[bytes, ...]],
    pair: tuple[bytes, bytes],
) -> Counter[tuple[bytes, ...]]:
    """
    Apply one merge to every pre-token type in the Counter.

    A new Counter is easier than mutating keys in the old Counter, because tuple
    keys are immutable and multiple old words may become the same new word.
    """
    updated_counts: Counter[tuple[bytes, ...]] = Counter()

    for word, frequency in word_counts.items():
        merged_word = _merge_pair_in_word(word, pair)
        updated_counts[merged_word] += frequency

    return updated_counts



def _initial_vocab(special_tokens: list[str] | None) -> dict[int, bytes]:
    """
    Build the starting vocabulary.

    Special tokens come first, followed by all 256 single-byte tokens.
    """

    vocab: dict[int, bytes] = {}
    seen_values: set[bytes] = set()

    # Special tokens get the first IDs so they can be encoded as single tokens.
    for special_token in special_tokens or []:
        token_bytes = special_token.encode("utf-8")
        if token_bytes not in seen_values:
            vocab[len(vocab)] = token_bytes
            seen_values.add(token_bytes)

    # Add every possible byte as a base token. Later BPE merges build larger
    # byte strings from these 256 atomic byte tokens.
    for byte in range(256):
        token_bytes = bytes([byte])
        if token_bytes not in seen_values:
            vocab[len(vocab)] = token_bytes
            seen_values.add(token_bytes)

    return vocab



def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Returns `(vocab, merges)`, where `vocab` maps token IDs to bytes and
    `merges` records the chosen merge pairs in creation order.
    """

    special_tokens = special_tokens or []
    vocab = _initial_vocab(special_tokens)

    with open(input_path, encoding="utf-8") as input_file:
        text = input_file.read()

    word_counts = _count_pretokens(text, special_tokens)

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        # Each iteration learns exactly one new merge.
        pair_counts = _count_adjacent_pairs(word_counts)
        if not pair_counts:
            break

        best_pair = _choose_best_pair(pair_counts)
        merges.append(best_pair)

        merged_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged_token

        # After learning the merge, rewrite the training counts so future pair
        # counts see the merged token as one unit.
        word_counts = _apply_merge_to_counts(word_counts, best_pair)

    return vocab, merges



# ---------------------------------------------------------------------------
# Tokenizer framework
# ---------------------------------------------------------------------------


class Tokenizer:
    """
    Byte-level BPE tokenizer framework.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """
        Store tokenizer data structures.

        `merge_ranks` stores training order, so lower rank means higher priority
        during encoding.
        """
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens or []

        # Make sure each requested special token has an ID. Some tests pass a
        # vocab that already contains the special token; others may not.
        existing_values = set(self.vocab.values())
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in existing_values:
                self.vocab[len(self.vocab)] = token_bytes
                existing_values.add(token_bytes)

        # Encoding needs bytes -> ID, while decoding uses the original ID -> bytes.
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

        # A smaller rank means this merge was learned earlier.
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        # Special tokens are input as strings, but the vocab stores bytes.
        self.special_token_to_id = {
            special_token: self.token_to_id[special_token.encode("utf-8")]
            for special_token in self.special_tokens
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """
        Construct a tokenizer from GPT-2-style vocab and merges files.

        GPT-2 fixture files store bytes as printable unicode strings, so this
        method maps them back to raw bytes before constructing the tokenizer.
        """
        # Reverse the GPT-2 display mapping so each printable character maps back
        # to its original byte value.
        byte_decoder = {value: key for key, value in _gpt2_bytes_to_unicode().items()}

        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            gpt2_vocab = json.load(vocab_file)

        vocab = {
            token_id: bytes(byte_decoder[character] for character in token)
            for token, token_id in gpt2_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as merges_file:
            for line in merges_file:
                parts = line.rstrip().split(" ")
                if len(parts) != 2:
                    continue

                left = bytes(byte_decoder[character] for character in parts[0])
                right = bytes(byte_decoder[character] for character in parts[1])
                merges.append((left, right))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs."""
        token_ids: list[int] = []

        for piece, is_special in _split_on_special_tokens(text, self.special_tokens):
            # Special tokens skip regex pre-tokenization and BPE merging.
            if is_special:
                token_ids.append(self.special_token_to_id[piece])
                continue

            for match in PRETOKEN_RE.finditer(piece):
                # Normal text is pre-tokenized first, then each pre-token is
                # separately converted to bytes and BPE-merged.
                pretoken_bytes = match.group().encode("utf-8")
                for token_bytes in self._apply_bpe(pretoken_bytes):
                    token_ids.append(self.token_to_id[token_bytes])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of text chunks.

        This is useful for large files because the caller does not need to keep
        the entire encoded result in memory.
        """
        for chunk in iterable:
            # `yield from` forwards each ID produced by `encode(chunk)` one by one.
            yield from self.encode(chunk)

    def decode(self, ids: Iterable[int]) -> str:
        """
        Convert token IDs back into text.

        Invalid UTF-8 byte sequences are replaced instead of raising an error.
        """
        # Decoding is simple because each token ID maps to the exact bytes that
        # were produced during encoding.
        text_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def _apply_bpe(self, token_bytes: bytes) -> tuple[bytes, ...]:
        """
        Apply learned BPE merges to one pre-token.

        Lower merge rank means the pair was learned earlier and has higher
        priority during encoding.
        """
        parts = _bytes_to_tuple(token_bytes)

        while len(parts) >= 2:
            best_rank = None
            best_pair = None

            for pair in zip(parts, parts[1:]):
                rank = self.merge_ranks.get(pair)
                # Choose the available adjacent pair with the earliest training
                # rank. Pairs not in `merge_ranks` cannot be merged.
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            parts = _merge_pair_in_word(parts, best_pair)

        return parts


# ---------------------------------------------------------------------------
# GPT-2 byte-unicode helper
# ---------------------------------------------------------------------------


@lru_cache
def _gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Return GPT-2's reversible byte-to-unicode display mapping.

    This is not the BPE algorithm itself. It is only needed when reading GPT-2
    fixture files whose vocab/merges are stored as printable unicode strings
    instead of raw bytes.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]

    n = 0
    for byte in range(2**8):
        if byte not in bs:
            bs.append(byte)
            cs.append(2**8 + n)
            n += 1

    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))
