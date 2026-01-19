# EAFT parity (long-seq tokenbudget ~10M, partial) — keep24/32 EAFT-REAP

- Base: `openai/gpt-oss-20b`
- Pruned: `calib_union_keep24of32_k75_eaftreap`
- top_k: 4 (unchanged)
- Status: Kaggle server stopped; **seq=16384 did not complete** (we only have base JSON for that config).

## Completed results

### seq=4096, blocks=2560, bs=4

| pack | seq | ppl_L | ppl_R | Δppl | cc_L | cc_R | Δcc (pp) | mean_p_L | mean_p_R | Δmean_p | JS2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNION | 4096 | 3.189 | 3.321 | +0.132 | 0.041 | 0.037 | -0.004 | 0.5905 | 0.5808 | -0.0097 | 0.0005 |
| calib_prompt_10000_v2 | 4096 | 2.767 | 2.817 | +0.050 | 0.039 | 0.035 | -0.004 | 0.6373 | 0.6324 | -0.0050 | 0.0003 |
| reasoning_style_10k_v2 | 4096 | 2.228 | 2.341 | +0.113 | 0.025 | 0.026 | +0.001 | 0.6944 | 0.6841 | -0.0103 | 0.0005 |
| tool_agentic_10k_v6 | 4096 | 4.004 | 4.097 | +0.093 | 0.082 | 0.076 | -0.006 | 0.5147 | 0.5079 | -0.0068 | 0.0004 |

Source: `harmony/cuda-norm/reports/_tmp_runs/eaftreap75_longseq_tokenbudget10m_keep24_uniform_20260118_200453/seq4096_blocks2560_bs4.md`

### seq=8192, blocks=1280, bs=2

| pack | seq | ppl_L | ppl_R | Δppl | cc_L | cc_R | Δcc (pp) | mean_p_L | mean_p_R | Δmean_p | JS2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNION | 8192 | 2.998 | 3.112 | +0.114 | 0.046 | 0.042 | -0.004 | 0.6049 | 0.5959 | -0.0090 | 0.0005 |
| calib_prompt_10000_v2 | 8192 | 2.623 | 2.667 | +0.045 | 0.042 | 0.038 | -0.004 | 0.6502 | 0.6455 | -0.0047 | 0.0003 |
| reasoning_style_10k_v2 | 8192 | 2.089 | 2.186 | +0.097 | 0.027 | 0.028 | +0.001 | 0.7101 | 0.7006 | -0.0095 | 0.0005 |
| tool_agentic_10k_v6 | 8192 | 3.718 | 3.800 | +0.082 | 0.051 | 0.049 | -0.002 | 0.5308 | 0.5243 | -0.0065 | 0.0004 |

Source: `harmony/cuda-norm/reports/_tmp_runs/eaftreap75_longseq_tokenbudget10m_keep24_uniform_20260118_200453/seq8192_blocks1280_bs2.md`

## Missing / incomplete

### seq=16384, blocks=640, bs=1

- Base JSON exists: `harmony/cuda-norm/artifacts/eaft_models/20260118_230214/openai_gpt-oss-20b.json`
- Pruned JSON missing (Kaggle stopped mid-run; no local copy).

