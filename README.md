# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

## Train TinyStories LM Offline On A Laptop

This repo now includes the section-5.3-and-beyond training path:

- `train_tinystories.py` trains a Transformer language model.
- `generate_tinystories.py` loads a checkpoint and samples TinyStories-like text.
- `cs336_basics/model.py` includes `TransformerLMModule`, a trainable wrapper
  around the assignment-tested functional Transformer code.

The default training command is designed for a normal laptop and does not need
internet. It uses the local file `tests/fixtures/tinystories_sample_5M.txt`, the
already-created 1000-token BPE tokenizer in `outputs/tinystories_bpe/`, and only
the first 1,000,000 characters of the fixture.

```sh
uv run python train_tinystories.py
```

Laptop preset:

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

If this is still too slow, reduce the amount of work:

```sh
uv run python train_tinystories.py --max-iters 500 --max-characters 300000
```

Run a very quick smoke test first if you want to check everything before a
longer run:

```sh
uv run python train_tinystories.py --preset smoke
```

The laptop run creates or reuses:

```text
outputs/tinystories_bpe/vocab.json
outputs/tinystories_bpe/merges.txt
outputs/tinystories_lm_laptop/config.json
outputs/tinystories_lm_laptop/checkpoints/final.pt
```

After training, generate text with:

```sh
uv run python generate_tinystories.py \
  --prompt "Once upon a time" \
  --max-new-tokens 200
```

The assignment-sized configuration is still available, but it is not the laptop
default:

```sh
uv run python train_tinystories.py --preset assignment
```

### Download data
Download the TinyStories data and a subsample of OpenWebText

This download step is optional for the laptop workflow above. The laptop
workflow already uses the local fixture in `tests/fixtures/`, so it keeps
working when you have no internet connection.

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
