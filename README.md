# Adaptive KV Cache on LongBench


> **Graduation Thesis Project** — Evaluating the effects of Adaptive KV Cache compression methods on long-context LLM performance using the [LongBench](https://github.com/THUDM/LongBench) benchmark.

Built upon the original [AdaKV](https://github.com/FFY0/AdaKV) framework and extended with a **Dynamic Cross-Layer** budget allocation strategy.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Quick Start with Docker](#quick-start-with-docker)
- [Manual Setup](#manual-setup)
- [Running Experiments](#running-experiments)
  - [Step 1 — Generate Predictions](#step-1--generate-predictions)
  - [Step 2 — Evaluate Results](#step-2--evaluate-results)
- [Compression Modes](#compression-modes)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

Large Language Models (LLMs) with long-context capabilities accumulate large **Key-Value (KV) Caches** during inference, which becomes a significant memory and latency bottleneck. This project investigates how **Adaptive KV Cache** compression methods affect model accuracy on diverse long-context tasks.

We benchmark three compression strategies against a full-cache baseline across 16 LongBench tasks at multiple budget sizes (128, 256, 512 tokens per head):

| Method | Description |
|---|---|
| **`dyn`** (Dynamic Cross-Layer) | Budget allocated dynamically across transformer layers based on attention entropy |
| **`ada`** (AdaKV) | Adaptive per-head budget from the original AdaKV paper |
| **`fix`** (SnapKV Fixed) | Fixed uniform budget per head |
| *Base* | No compression — full KV cache |

---

## Project Structure

```
G-AdaKV/
├── adaptive_snapkv/            # Core KV-cache compression library
│   └── monkeypatch/
│       ├── monkeypatch.py              # Entry point: model replacement functions
│       ├── adaptive_llama_hijack.py    # AdaKV (per-head adaptive) for LLaMA
│       ├── dynamic_llama_hijack.py     # Dynamic cross-layer mode for LLaMA
│       ├── snapkv_utils.py             # Core SnapKV attention utilities
│       └── dynamic_snapkv_utils.py     # Dynamic cross-layer attention utilities
│
├── csrc/                       # C++/CUDA extension source (compiled via `make i`)
│
├── experiments/
│   └── LongBench/
│       ├── config/
│       │   ├── dataset2prompt.json     # Prompt templates for each LongBench task
│       │   └── dataset2maxlen.json     # Max generation length per task
│       ├── run_budgets.sh      # Main experiment runner (loops over budget sizes)
│       ├── pred.py             # Prediction script (called by run_budgets.sh)
│       ├── eval.py             # Evaluation script (computes scores from pred/)
│       └── metrics.py          # Metric functions imported by eval.py
│
├── Dockerfile                  # Reproducible GPU environment (CUDA 11.8 + PyTorch 2.0)
├── pyproject.toml              # Package metadata
├── makefile                    # `make i` → builds csrc + installs package
└── requirements.txt            # Full pinned dependency list
```

---

## Key Components

### `adaptive_snapkv/` — Core Library

The `monkeypatch` subpackage **replaces internal attention forward functions** of HuggingFace Transformers models at runtime (no model file changes required). The entry point is [`monkeypatch.py`](adaptive_snapkv/monkeypatch/monkeypatch.py), which exposes:

```python
from adaptive_snapkv.monkeypatch.monkeypatch import (
    config_compress,          # Set hyperparameters on the model config
    replace_llama_dynamic,    # Activate dynamic cross-layer mode for LLaMA
    replace_llama_adaptive,   # Activate AdaKV mode for LLaMA
)
```

### `experiments/LongBench/` — Evaluation Pipeline

| File | Role |
|---|---|
| `run_budgets.sh` | Outer loop: runs `pred.py` for budgets 128, 256, 512 sequentially |
| `pred.py` | Loads model, applies monkeypatch, iterates over all 16 LongBench datasets |
| `eval.py` | Reads `pred/<run_name>/*.jsonl` and writes `result.json` with task scores |
| `metrics.py` | F1, ROUGE, retrieval, code similarity metric implementations |
| `config/` | JSON configs required by `pred.py` at runtime |

---

## Quick Start with Docker

The recommended way to run experiments without managing CUDA / Python dependencies manually.

### Prerequisites
- Docker with NVIDIA Container Toolkit installed
- A GPU with ≥ 24 GB VRAM (tested on **NVIDIA GeForce RTX 3090 GPU**, 24 GB VRAM)
- HuggingFace model weights downloaded (e.g., `meta-llama/Llama-3.1-8B-Instruct`)

### 1. Build the image

```bash
docker build -t g-adakv:latest .
```

### 2. Run the container

```bash
docker run --gpus all -it \
  -e HF_HOME=/models \
  -v /path/to/your/hf_models:/models \
  g-adakv:latest bash
```

> **Note:** Mount your HuggingFace model cache so the container can access model weights without re-downloading them.
> - On Linux/Mac: `-v ~/.cache/huggingface:/models`
> - On Windows: `-v C:\Users\<Your_Username>\.cache\huggingface:/models`

> **Windows Users:** If you cloned this repository on Windows, bash scripts might have `CRLF` line endings. Before running them inside the container, convert them to `LF` format using `sed` (e.g., `sed -i 's/\r$//' run_budgets.sh`).

### 3. Inside the container, run experiments

```bash
cd /app/G-AdaKV/experiments/LongBench
bash run_budgets.sh
```

---

## Manual Setup

If you prefer a local environment (requires CUDA 11.8 and Python 3.10):

> **Note:** The manual setup is designed for **Linux** or **WSL (Windows Subsystem for Linux)**. Native Windows is not officially supported due to `flash-attention` wheel compatibility and `make` commands. Windows users are strongly encouraged to use the Docker method above.

### 1. Clone the repository

```bash
git clone https://github.com/emir-oztunc/Global---AdaKV.git
cd Global---AdaKV
```

### 2. Install dependencies

It is highly recommended to use a **Conda** environment (Python 3.10) to avoid version conflicts and easily manage the CUDA toolkit if you don't have it installed system-wide:

```bash
conda create -n global-adakv python=3.10 -y
conda activate global-adakv
# Install CUDA toolkit (if nvcc is not available on your system)
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
```

Install the required Python packages. **Note:** `setuptools<70.0.0` is required because newer versions remove `pkg_resources` which breaks PyTorch's C++ extension builder.

```bash
pip install setuptools==69.5.1
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

pip install packaging ninja transformers==4.44.2 datasets tiktoken jieba rouge_score

# Flash Attention 2 (prebuilt wheel for CUDA 11.8 + PyTorch 2.0)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.0.post1/flash_attn-2.4.0.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 3. Build and install the package

```bash
make i
# This runs: cd csrc && make   (builds CUDA extensions)
#            pip install -e .  (installs adaptive_snapkv in editable mode)
```

---

## Running Experiments

All experiment commands are run from the `experiments/LongBench/` directory.

```bash
cd experiments/LongBench
```

### Step 1 — Generate Predictions

#### Option A: Automated multi-budget run (recommended)

Edit the variables at the top of `run_budgets.sh` to set your model path and mode, then:

```bash
bash run_budgets.sh
```

This will run `pred.py` for **budget = 128, 256, 512** sequentially and save results to:
```
pred/
└── <PREFIX>-budget128/   ← one .jsonl per LongBench task
└── <PREFIX>-budget256/
└── <PREFIX>-budget512/
```

Key parameters in `run_budgets.sh`:

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model ID or local path |
| `MAX_LENGTH` | `60000` | Maximum input context length (tokens) |
| `MODE` | `dyn` | Compression mode: `dyn`, `ada`, or leave empty for base |
| `PREFIX` | `llama_all_datasets-dyn` | Output folder name prefix |

#### Option B: Single run

```bash
python pred.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --max_length 60000 \
    --out_name my_run_budget256 \
    --mode dyn \
    --budget 256
```

### Step 2 — Evaluate Results

After predictions are complete, compute scores for all runs in `pred/`:

```bash
python eval.py
```

This reads every subfolder under `pred/`, scores each task using the appropriate metric, and writes:
- `pred/<run_name>/result.json` — aggregate scores per task
- `pred/<run_name>/list_result.json` — per-sample score lists

**Example `result.json` output:**
```json
{
    "qasper": 28.4,
    "narrativeqa": 19.7,
    "hotpotqa": 41.2,
    ...
}
```

---

## Compression Modes

| Mode flag | Strategy | Suitable for |
|---|---|---|
| `dyn` | Dynamic cross-layer budget allocation | **Recommended** — best accuracy/memory tradeoff |
| `ada` | Per-head adaptive budget (original AdaKV) | Reproducing AdaKV paper results |
| `fix` | Uniform fixed budget per head | SnapKV baseline |
| *(none)* | No compression — full KV cache | Accuracy upper bound |

Additional hyperparameters for `pred.py`:

| Flag | Default | Description |
|---|---|---|
| `--budget` | `1024` | Total KV tokens budget per layer |
| `--floor_alpha` | `0.2` | Minimum fraction of budget guaranteed per head |
| `--pyram` | off | Enable pyramid-shaped budget distribution across layers |
| `--pyram_beta` | `20` | Pyramid decay factor (from AdaKV paper) |
| `--gqa_support` | off | Enable Grouped Query Attention support |

---

## Acknowledgements

This project is built upon the following excellent works:

- **AdaKV** — [FFY0/AdaKV](https://github.com/FFY0/AdaKV)  
  *AdaKV: Optimizing KV Cache Eviction in LLMs with Adaptive Budget Allocation*  
  The core monkeypatching architecture and adaptive budget allocation logic originate from this work.

- **SnapKV** — [FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV)  
  Efficient KV cache compression via observation-based token selection.

- **LongBench** — [THUDM/LongBench](https://github.com/THUDM/LongBench)  
  A bilingual, multitask benchmark for long context understanding in LLMs.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.  
The `csrc/` directory contains a separate component under its own license (see [`csrc/LICENSE`](csrc/LICENSE)).
