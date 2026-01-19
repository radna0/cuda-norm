# DFLASH offline accept/accuracy summaries

These are *offline* accept checks (draft vs target greedy) from `harmony/cuda-norm/logs/tpu_dflash/*.json`.

| file | ctx_len | block_size | num_samples | accept_len_mean | accept_rate_tokens | token_acc_mean | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| offline_cache_eval_run2000_20260117_120959.json | 1023 | 8 | 256 | 6.9766 | 0.9967 | 0.9983 | /dev/shm/dflash-checkpoints/gptoss20b_dflash_ctx1024_b8_k4_bs64_s2000_resume200_v1_20260117_110932/run-2000 |
| offline_cache_eval_run2000_20260117_121319.json | 1023 | 8 | 256 | 6.9766 | 0.9967 | 0.9983 | /dev/shm/dflash-checkpoints/gptoss20b_dflash_ctx1024_b8_k4_bs64_s2000_resume200_v1_20260117_110932/run-2000 |
| offline_cache_eval_run2000_20260117_121701.json | 1023 | 8 | 64 | 7.0000 | 1.0000 | 1.0000 | /dev/shm/dflash-checkpoints/gptoss20b_dflash_ctx1024_b8_k4_bs64_s2000_resume200_v1_20260117_110932/run-2000 |
| offline_eval_run10000_curcache_first64_20260118_013948.json | 1023 | 8 | 64 | 5.6719 | 0.8103 | 0.8817 | /dev/shm/dflash-checkpoints/gptoss20b_dflash_long_20260117_215726/run-10000 |
