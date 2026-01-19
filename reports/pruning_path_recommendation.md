# Pruning path recommendation (MoE repo vs. NeMo/TensorRT)

## Recommendation (now)

Use the **MoE expert pruning repo path** (expert profiling → soft prune → structural prune) as the **first implementation track**.

## Why this matches our constraints

- **Independent value without training**: profiling + soft pruning give immediate cost/quality signals; structural expert pruning yields deployable variants without doing full distillation.
- **Directly answers the roadmap questions**:
  1) “Does MoE expert pruning work cleanly in our tooling (20B)?” → yes/no via `modal/gpt_oss_pruning_track.py`.
  2) “How expensive is it on 120B?” → answerable with config+index analysis + partial per-layer dry run, no training.
- **Compatible with analysis-first**: routing histograms, co-activation patterns, and confidence stats are the right inputs to decide which experts to keep and what top-k to run.

## NeMo/TensorRT path (later)

The NVIDIA pruning + distillation workflow is powerful, but it assumes:

- HF → **NeMo checkpoint conversion**
- pruning via `/opt/NeMo/scripts/llm/gpt_prune.py`
- **distillation/retraining** via `/opt/NeMo/scripts/llm/gpt_train.py` to recover quality
- TensorRT-LLM inference benchmarking (often FP8)

That’s heavier operationally and, importantly, ties the pruning loop to a later-stage **training + NeMo tooling** commitment.

## Practical next step

Complete Task 1A + 1B (20B) and Task 2A (120B estimate) first, then reassess whether NeMo conversion is worth the integration cost for our next phase.

