# Nemotron Harmony Dataset Pipeline & Benchmarks

This project implements a scalable data processing and benchmarking pipeline for NVIDIA's Nemotron distilled datasets. It is designed to normalize large-scale datasets into a unified "Harmony" format, extract rich metadata, and facilitate efficient training and evaluation (NLL scoring) using Modal.

## 🚀 High-Level Architecture

The workflow is split into two distinct stages to optimize for cost and efficiency:

1.  **CPU Stage (Data Engineering)**: Run on high-core CPU servers (e.g., EPYC).
    *   **Normalize**: Converts heterogeneous source datasets (Math, Proofs, Science, Agentic) into a standard "text-first" Harmony format (Parquet).
    *   **Extract Metadata**: Parses correctness signals, difficulty bins, and tool usage from raw logs.
    *   **Filter & Dedup**: Removes invalid records, exact duplicates, and empty completions.
    *   **Build Pools**: Groups data into "candidate pools" (e.g., `high_correctness`, `tool_use`) for targeted scoring/training.

2.  **GPU Stage (Modal)**: Run on cloud GPUs via Modal.
    *   **NLL Scoring**: Computes completion-only perplexity on candidate pools to identify high-quality data subsets.
    *   **Training Benchmarks**: Validates training performance using FSDP + Unsloth on the processed data.

---

## 📂 Project Structure

```text
.
├── cpu_normalize_dataset.py       # Core logic: Normalizes raw HF datasets -> Harmony Parquet shards
├── cpu_build_candidate_pools.py   # Organization: Groups normalized data into specific pools
├── run_cpu_pipeline_all.sh        # Master script: Runs normalization + pooling for all defined datasets
├── harmony_text.py                # Utilities: Parsing/rendering Harmony format (<|start|>, <|call|>, etc.)
├── modal/
│   ├── fsdp_unsloth_training_benchmark.py  # Modal app: GPT-OSS + Unsloth on 8x B200 benchmark
│   ├── nll_scoring.py                      # (Referenced) NLL scoring logic
│   └── ...                                 # Other Modal utilities
├── DATA_FILTERING_SPEC.md         # detailed specification of the data filtering logic
└── scripts/
    ├── upload_folder_to_hf_dataset_repo.sh # Helper to upload artifacts to Hugging Face
    └── ...
```

---

## 🛠️ Setup & Prerequisites

### Environment
-   **Linux**: Recommended for optimal performance.
-   **Python 3.10+**
-   **Modal**: For running GPU workloads (`pip install modal && modal setup`).

### Dependencies
Install the required python packages (see individual scripts for specifics, but generally):
```bash
pip install pandas pyarrow huggingface_hub requests pyyaml
```

### Environment Variables
-   `HF_TOKEN`: Required for downloading restricted datasets (Nemotron) and uploading artifacts.
-   `HF_HOME`: (Optional) Set this to a path with ample storage (perfomant disk/RAM) for caching large models/datasets.

---

## 🖥️ CPU Pipeline (Data Preparation)

The CPU pipeline is responsible for ingesting `nvidia/Nemotron-*` datasets and producing clean, standardized Parquet files.

### Quick Start
To process all datasets (Math-v2, Proofs-v1, Science-v1, Agentic-v1, Chat-v1):

```bash
# Run a smoke test (small subset)
./run_cpu_pipeline_all.sh --max_records 1000

# Run full pipeline (ensure you have sufficient disk space!)
./run_cpu_pipeline_all.sh
```

### Key Scripts

#### 1. `cpu_normalize_dataset.py`
Reads a raw HF dataset and outputs "Harmony" formatted Parquet files.
-   **Input**: Hugging Face Dataset Repos (e.g., `nvidia/Nemotron-Math-v2`)
-   **Output**: `cpu_out/<dataset_tag>/data/<split>/*.parquet`
-   **Preserves**:
    -   `text`: Full prompt + completion in Harmony format.
    -   `meta_*`: Correctness (pass@k), Difficulty, Domain.
    -   `quality_*`: Validity flags.

**Example**:
```bash
python cpu_normalize_dataset.py \
  --dataset nvidia/Nemotron-Math-v2 \
  --out_dir cpu_out/nvidia_math_v2 \
  --hf_layout \
  --write_readme
```

#### 2. `cpu_build_candidate_pools.py`
Scans normalized output and organizes it into semantic pools for downstream tasks.
-   **Example Pools**: `candidates_high_correctness`, `candidates_tool_use`, `candidates_proof`.

---

## ⚡ GPU Pipeline (Modal)

The Modal scripts consume the artifacts produced by the CPU pipeline.

### Training Benchmark
Run a distributed training benchmark (FSDP + Unsloth) on the processed data.

```bash
# Deploys/Runs the App defined in modal/fsdp_unsloth_training_benchmark.py
modal run modal/fsdp_unsloth_training_benchmark.py
```

**Features**:
-   **Persistent Volumes**: Caches models and datasets to avoid repeated downloads.
-   **Unsloth Integration**: Uses Unsloth optimizations for faster training.
-   **FSDP**: Distributed execution on multiple GPUs.

---

## 📊 Data Schema (Harmony Normalized)

The normalized Parquet files contain the following key columns:

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | string | Stable SHA1 hash of the text content. |
| `text` | string | Full prompt and completion formatted with Harmony tags. |
| `dataset` | string | Source dataset ID. |
| `split` | string | Original split name. |
| `meta_correctness` | float | Normalized correctness score (0.0 - 1.0). |
| `meta_difficulty_bin`| string | `low`, `medium`, `high`. |
| `loss_mode` | string | Instruction for loss masking (e.g., `assistant_all`). |

---

## 📝 Specifications

For a deep dive into the filtering logic, metadata extraction rules, and schema definitions, please refer to:
👉 **[DATA_FILTERING_SPEC.md](./DATA_FILTERING_SPEC.md)**
