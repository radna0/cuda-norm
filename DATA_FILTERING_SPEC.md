# Dataset Filtering + Metadata Indexing Spec (EPYC CPU) + NLL Scoring Interface (Modal GPU)

This is the handoff document for the engineer maintaining dataset formatting/publishing for Harmony-formatted Nemotron distilled datasets (e.g. `radna0/nemotron-math-v2-harmony-tools`).

## 0) What we’re doing (plain English)

We have very large teacher-distilled datasets (Nemotron-*). Training on everything is wasteful and can hurt quality (duplicates, garbage, unreliable examples).

We split the work into two stages:

### CPU stage (EPYC machine — your responsibility)
You **do not run models**. You:
- Inspect schemas across datasets/splits.
- Extract/normalize all useful metadata (correctness, difficulty bins, tool-use, etc).
- Validate formatting and remove obvious garbage.
- Deduplicate (exact + near-dupe).
- Produce **candidate pools** (Parquet) for later GPU scoring.
- Publish CPU-produced artifacts (indexes, candidate shards) back to Hugging Face.

### GPU stage (Modal — not your responsibility)
Modal jobs compute **completion-only NLL/PPL** for a chosen student checkpoint on your candidate pools. This gives a “how surprised is the student by the teacher completion?” score that we use to build:
- Anchor/Stability subsets (prevent regressions)
- Restoration subsets (recover capability after architecture changes)
- Learning subsets (high-value training)
- Challenge subsets (hard-but-trustworthy)

## 1) Key recommendation: “text-first, structure-preserved”

Yes: the best interface for trainers is a single `text` column that is already Harmony-rendered (exactly what the model sees).

But we must **also** preserve enough structure/metadata to:
- filter correctly (without expensive model scoring),
- mask loss correctly (completion-only), and
- debug issues later.

### Minimum columns to include in any CPU output shard
- `id` (string, stable): prefer `uuid` from source; otherwise a deterministic hash of canonical source fields.
- `dataset` (string): source dataset repo, e.g. `nvidia/Nemotron-Math-v2`
- `subset` (string): config name (usually `default`)
- `split` (string): e.g. `low`, `high_part01`, `tool_calling`
- `text` (string): Harmony-formatted prompt+completion
- `loss_mode` (string): how Modal should compute loss spans. Start with:
  - `assistant_all`: loss on all assistant message bodies; exclude system/user/tool messages
  - (optional) `assistant_last`: loss only on final assistant message body

### Strongly recommended columns
- `meta_correctness` (float, nullable): normalized correctness signal when available (e.g. pass@k / k)
- `meta_pass_k` (int, nullable)
- `meta_correct_count` (int, nullable)
- `meta_difficulty_bin` (string, nullable): `low|medium|high` if applicable
- `meta_domain` (string): `math|proof|science|agentic|chat_if`
- `meta_tags` (list[string]) if available
- `quality_valid_harmony` (bool)
- `quality_completion_nonempty` (bool)
- `quality_valid_tool_schema` (bool)
- `quality_is_duplicate` (bool)
- `stats_char_len` (int)
- `stats_approx_tokens` (int, optional CPU-side estimate)

### Completion-only loss boundary (critical)
If we rely on `text` alone, Modal must know where “labels begin”.

For Harmony-formatted `text` like `radna0/nemotron-math-v2-harmony-tools`, boundaries are marked by tags such as:
- `<|start|>assistant<|message|>` … `<|end|>`
- `<|start|>tool<|message|>` … `<|end|>`

So we can do completion-only loss without storing token offsets by using `loss_mode=assistant_all` and parsing these tags.

If you change the template, you must version it and update the parser on the Modal side.

## 2) Dataset-specific metadata notes (what to extract)

### `nvidia/Nemotron-Math-v2`
Fields observed in samples include:
- `problem`, `expected_answer`, `messages`, `metadata`, `license`, `data_source`, optional `tools`, optional `url/user_*`

Important: `metadata` contains per-record evaluation results like:
- `reason_low_with_tool: {count, pass, accuracy}`
- `reason_high_no_tool: {count, pass, accuracy}`

Normalize this into:
- `meta_pass_k`, `meta_correct_count`, `meta_correctness` (choose a policy; suggested below)
- keep the full original evaluation block as `meta_eval_json` **if** you can afford it.

Suggested policy for `meta_correctness`:
- `meta_correctness = max(accuracy over available eval conditions)` (best-case solvability)
- also store `meta_correctness_high = max(high_with_tool, high_no_tool)` when present

Why: we often want “hard but solvable / trustworthy” examples.

### `nvidia/Nemotron-Math-Proofs-v1`
Has Lean fields:
- `formal_statement`, `lean_header`, plus `problem`
Often no correctness metadata. Focus on:
- format validity
- dedup by `problem` and/or `formal_statement`

### `nvidia/Nemotron-Science-v1`
Splits like `MCQ`, `RQA` with:
- `messages`, `uuid`, `license`, `used_in`, `tools`
Likely no correctness metadata; use length/format/tool-use tags.

### `nvidia/Nemotron-Agentic-v1`
`tool_calling` can contain heterogeneous `messages[].content` types (string/dict/list).
Do **not** assume message content is always a string. If rendering to `text`, stringify non-strings deterministically (e.g. `json.dumps(..., sort_keys=True)`).

### `nvidia/Nemotron-Instruction-Following-Chat-v1`
Splits like `chat_if`, `structured_outputs`.
Has `capability_target`, `reasoning` toggles, etc. Preserve as metadata.

## 3) CPU pipeline (what you implement)

### Step A — Inventory + schema report
Deliver:
- `inventory.json`: dataset → splits → counts → column names/types
- `metadata_coverage.md`: per split, % with correctness metadata

### Step B — Normalize to canonical rows
For each source row:
1) Create/choose `id`
2) Extract/normalize `meta_*`
3) Render `text` (Harmony) OR reference existing Harmony text if already produced
4) Compute `quality_*` flags and cheap stats

### Step C — Filter & deduplicate (CPU-only)
Hard drop:
- invalid Harmony render
- empty/near-empty completion
- tool schema malformed (when tool calls exist)
- exact duplicates

Soft tag (keep but label):
- extreme length (too long/too short)
- suspicious repeated patterns

### Step D — Produce candidate pools for Modal
Write Parquet shards grouped by purpose, e.g.:
- `candidates_missing_correctness_meta/*.parquet`
- `candidates_tool_use/*.parquet`
- `candidates_long_context/*.parquet`
- `candidates_verified_correct/*.parquet`

Each row must include at least: `id`, `text`, `loss_mode`, `meta_*`, `quality_*`.

### Step E — Publish artifacts to HF
Use batch commits (avoid 1 commit per file).

## 4) Modal interface (what your outputs must enable)

Modal scoring job expects:
- `text` (Harmony formatted)
- `loss_mode` (how to mask)
- optionally `meta_correctness` (so it can stratify/compare score distributions)

It will write back:
- `id`
- `nll_mean`, `nll_sum`
- `ppl = exp(nll_mean)`
- `completion_token_count`
- `percentile_within_split` (computed after scoring)

### Modal implementation in this repo

- `modal/nll_scoring.py` (GPU): downloads candidate Parquet files from HF, masks loss to assistant message bodies, and writes Parquet score shards to the persistent Modal data volume.

Environment variables (Modal):
- `CANDIDATE_DATASET_ID`: HF dataset repo_id containing candidate parquet files
- `CANDIDATE_SUBDIR` (optional): subfolder inside the repo (e.g. `high_correctness/`)
- `MODEL_NAME`: checkpoint to score (default: `unsloth/gpt-oss-120b`)
- `MAX_LENGTH`, `BATCH_SIZE`, `MAX_RECORDS` (optional knobs)

## 5) Acceptance criteria

You are “done” when:
- We have schema + metadata coverage reports for all datasets/splits we care about.
- We have published candidate pools (Parquet) ready for Modal NLL scoring.
- Every candidate row has stable `id`, `text`, and a well-defined `loss_mode`.
- We can trace any training example back to its source dataset + split + original metadata.

## 6) What’s implemented in this repo (run these on the EPYC CPU box)

### Normalize + extract metadata to Parquet

This script produces *text-first* Parquet shards with `id`, `text`, `loss_mode`, `meta_*`, `quality_*`, and `stats_*`:

- `cpu_normalize_dataset.py`

Examples:

```bash
# Nemotron-Math-v2 (renders Harmony text + extracts correctness metadata)
python cpu_normalize_dataset.py \
  --dataset nvidia/Nemotron-Math-v2 \
  --out_dir cpu_out/nvidia_math_v2 \
  --hf_layout --write_readme \
  --drop_invalid_harmony --drop_empty_completion

# Existing Harmony dataset (keeps text, adds id/quality flags; useful for join-by-id)
python cpu_normalize_dataset.py \
  --dataset radna0/nemotron-math-v2-harmony-tools \
  --out_dir cpu_out/radna0_harmony_tools \
  --hf_layout --write_readme \
  --drop_invalid_harmony --drop_empty_completion

# Nemotron-Math-Proofs-v1 (builds a prompt->Lean completion conversation)
python cpu_normalize_dataset.py \
  --dataset nvidia/Nemotron-Math-Proofs-v1 \
  --out_dir cpu_out/nvidia_math_proofs \
  --hf_layout --write_readme \
  --drop_invalid_harmony --drop_empty_completion
```

Other supported datasets (same interface):

```bash
python cpu_normalize_dataset.py --dataset nvidia/Nemotron-Science-v1 --out_dir cpu_out/nvidia_science --hf_layout --write_readme
python cpu_normalize_dataset.py --dataset nvidia/Nemotron-Agentic-v1 --out_dir cpu_out/nvidia_agentic --hf_layout --write_readme
python cpu_normalize_dataset.py --dataset nvidia/Nemotron-Instruction-Following-Chat-v1 --out_dir cpu_out/nvidia_if_chat --hf_layout --write_readme
```

Note: when a dataset provides tool schemas via a top-level `tools` field, the normalizer writes a `*__tools_catalog.json` mapping `source_tools_id -> tools_schema` and stores `source_tools_id/source_tools_count` per row.

### Harmony parsing utilities

- `harmony_text.py` provides:
  - `parse_harmony(text)` (understands `<|end|>`, `<|call|>`, `<|return|>`)
  - `basic_quality_flags_from_text(text)` (validity, tool presence, nonempty completion)
  - `sha1_text_id(text)` (stable join key)

### Build candidate pools (CPU-only)

This script takes the normalized shards and produces simple pool folders you can hand to Modal for NLL scoring:

- `cpu_build_candidate_pools.py`

Example:

```bash
python cpu_build_candidate_pools.py \
  --in_dir cpu_out/nvidia_math_v2 \
  --out_dir cpu_pools/nvidia_math_v2
```

It creates pools opportunistically based on available columns (e.g. `tool_use`, `high_correctness/low_correctness`, `reasoning_on/off`, `capability_chat`, etc.).

### One-command runner (optional)

For a quick smoke test across all the datasets in this project:

```bash
./run_cpu_pipeline_all.sh --max_records 1000
```

### Upload artifacts to HF (dataset repo)

To publish the CPU outputs (normalized shards / pools) to a Hugging Face *dataset* repo:

- `upload_folder_to_hf_dataset_repo.sh`

Example:

```bash
HF_TOKEN=... ./upload_folder_to_hf_dataset_repo.sh \
  radna0/nemotron-math-v2-harmony-tools-meta \
  cpu_out/nvidia_math_v2
```
