# Pruning eval parity (PPL rules)

This repo treats PPL as *next-token perplexity* computed over packed token blocks,
and for Harmony chat data we compute **completion-only** loss:

## Packing

- Read Harmony-formatted examples from HF dataset `text`.
- Tokenize with `add_special_tokens=False`.
- Concatenate examples in-order, appending `eos_token_id` between examples.
- Chunk into fixed blocks of length `seq_len+1` token ids.

## Loss (next-token NLL)

- For each block, compute logits on `input_ids[:-1]` and score targets `input_ids[1:]`.
- Exclude the first token in each block (no context).

## Completion-only masking

- Parse Harmony tags and select **assistant message body spans** (not system/user/tool).
- Build a per-token keep mask via `return_offsets_mapping=True` (token overlaps assistant span).
- PPL is computed over *kept tokens only*: `ppl = exp(total_nll / kept_tokens)`.

## Reference

- The methodology matches the project's SGLang truth harness shape (packing + logprob NLL),
  but this implementation uses Transformers for convenience with soft-prune patching.
