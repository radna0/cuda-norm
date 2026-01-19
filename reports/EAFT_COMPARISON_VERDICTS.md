# EAFT comparison verdicts

## 20B vs 120B (base=20B, pruned=120B)
- base: `openai/gpt-oss-20b`
- pruned: `openai/gpt-oss-120b`
- top_k: 4 | blocks: 8 | entropy_topk: 20
- seq_len=1024 avg ΔPPL (non-UNION packs) = +1.142 (+20.8%)
  - worst pack: `tool_agentic_10k_v6` ΔPPL=+4.238 (+76.9%) base=5.512 pruned=9.749
  - best pack: `reasoning_style_10k_v2` ΔPPL=-0.525 (-9.2%) base=5.715 pruned=5.190
  - UNION: ΔPPL=-0.445 (-7.7%) base=5.807 pruned=5.362
- seq_len=2048 avg ΔPPL (non-UNION packs) = +1.284 (+21.2%)
  - worst pack: `tool_agentic_10k_v6` ΔPPL=+4.750 (+83.9%) base=5.664 pruned=10.414
  - best pack: `reasoning_style_10k_v2` ΔPPL=-0.650 (-14.0%) base=4.645 pruned=3.995
  - UNION: ΔPPL=+0.030 (+0.6%) base=4.611 pruned=4.641

## 20B base vs REAP-0.5 (pruned)
- base: `openai/gpt-oss-20b`
- pruned: `sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4`
- top_k: 4 | blocks: 32 | entropy_topk: 20
- seq_len=1024 avg ΔPPL (non-UNION packs) = +2.310 (+41.5%)
  - worst pack: `reasoning_style_10k_v2` ΔPPL=+2.822 (+53.5%) base=5.272 pruned=8.094
  - best pack: `calib_prompt_10000_v2` ΔPPL=+1.808 (+31.0%) base=5.831 pruned=7.639
  - UNION: ΔPPL=+2.041 (+36.2%) base=5.640 pruned=7.681
- seq_len=2048 avg ΔPPL (non-UNION packs) = +1.940 (+40.7%)
  - worst pack: `tool_agentic_10k_v6` ΔPPL=+2.260 (+45.4%) base=4.979 pruned=7.240
  - best pack: `calib_prompt_10000_v2` ΔPPL=+1.507 (+31.6%) base=4.773 pruned=6.280
  - UNION: ΔPPL=+1.798 (+41.7%) base=4.312 pruned=6.110

## 20B vs 20B sanity
- base: `openai/gpt-oss-20b`
- pruned: `openai/gpt-oss-20b`
- top_k: 4 | blocks: 8 | entropy_topk: 20
- seq_len=1024 avg ΔPPL (non-UNION packs) = +0.000 (+0.0%)
  - worst pack: `reasoning_style_10k_v2` ΔPPL=+0.000 (+0.0%) base=5.715 pruned=5.715
  - best pack: `reasoning_style_10k_v2` ΔPPL=+0.000 (+0.0%) base=5.715 pruned=5.715
  - UNION: ΔPPL=+0.000 (+0.0%) base=5.807 pruned=5.807
- seq_len=2048 avg ΔPPL (non-UNION packs) = +0.000 (+0.0%)
  - worst pack: `reasoning_style_10k_v2` ΔPPL=+0.000 (+0.0%) base=4.642 pruned=4.642
  - best pack: `reasoning_style_10k_v2` ΔPPL=+0.000 (+0.0%) base=4.642 pruned=4.642
  - UNION: ΔPPL=+0.000 (+0.0%) base=4.609 pruned=4.609
