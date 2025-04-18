# Computer Vision Environment

Python UV environment with the main dependencies for computer vision.

## Requirements

- [UV](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.12 (can be installed with `uv python install 3.12`)
- PyCharm `Recommended` or VSCodium (with Python extensions)

### Windows / Linux

- Run PyTorch on GPU: [CUDA](https://developer.nvidia.com/cuda-downloads) >= 12.6

## Setup

```bash
uv sync
```

## Usage

### Run a python script

```bash
uv run [filename]
```

### Add a new dependency

```bash
uv add [package]
```

### Run Tensorboard

```bash
tensorboard --logdir runs
```