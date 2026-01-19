# 20B soft-prune eval (PPL parity)

- Model: `openai/gpt-oss-20b`
- Dataset: `radna0/nemotron-math-v2-harmony-tools` split `high_part00` col `text`
- seq_len: 1024 | blocks: 64 | batch_size: 1
- rows_seen: 5 | pack_wall_s: 5.3s

| keep_frac | keep_n | top_k | ppl | ppl_delta | kept_tokens | pred_tokens | tok/s(pred) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 32 | 4 | 2.805128 | +0.0000 | 64612 | 65536 | 5011 |
| 1.00 | 32 | 2 | 3.099321 | +0.2942 | 64612 | 65536 | 6037 |
| 0.50 | 16 | 4 | 3.921340 | +1.1162 | 64612 | 65536 | 14017 |
| 0.50 | 16 | 2 | 4.325094 | +1.5200 | 64612 | 65536 | 11971 |

- Baseline (keep_frac=1.0, top_k=4) ppl=2.805128