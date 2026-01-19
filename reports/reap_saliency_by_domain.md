# 20B REAP-lite saliency by domain

- Base model: `openai/gpt-oss-20b`
- Dataset: `radna0/harmony-nemotron-cpu-artifacts` split `train` col `text`
- Domain column: `meta_domain`
- Domains: agentic, general, math
- Rows/domain: 200 | Seed: 3407 | Max scan rows: 600000
- Max seq length: 2048 | Batch size: 1

## agentic

- source: `radna0/harmony-nemotron-cpu-artifacts` split `train` col `text` domain_filter=`agentic`
- scanned=600001 matched=335062 sample_path=`/root/data/reap_domain_samples/radna0__harmony-nemotron-cpu-artifacts/train/agentic_seed3407_n200.jsonl`
- kept_tokens=277136 total_tokens=337046 tok_s_pred=3581
- parquet: `artifacts/reap_saliency_by_domain/agentic.parquet`

- layer_0: top4=0.375 top8=0.585 top16=0.800
- layer_1: top4=0.395 top8=0.638 top16=0.865
- layer_10: top4=0.494 top8=0.695 top16=0.902
- layer_23: top4=0.480 top8=0.748 top16=0.901

## general

- source: `radna0/harmony-nemotron-cpu-artifacts` split `train` col `text` domain_filter=`chat_if`
- scanned=600001 matched=264938 sample_path=`/root/data/reap_domain_samples/radna0__harmony-nemotron-cpu-artifacts/train/chat_if_seed3407_n200.jsonl`
- kept_tokens=301820 total_tokens=335022 tok_s_pred=3602
- parquet: `artifacts/reap_saliency_by_domain/general.parquet`

- layer_0: top4=0.323 top8=0.549 top16=0.762
- layer_1: top4=0.341 top8=0.609 top16=0.865
- layer_10: top4=0.486 top8=0.688 top16=0.907
- layer_23: top4=0.518 top8=0.680 top16=0.865

## math

- source: `radna0/nemotron-math-v2-harmony-tools` split `high_part00` col `text` domain_filter=``
- scanned=600001 matched=600000 sample_path=`/root/data/reap_domain_samples/radna0__nemotron-math-v2-harmony-tools/high_part00/all_seed3407_n200.jsonl`
- kept_tokens=405512 total_tokens=409316 tok_s_pred=3603
- parquet: `artifacts/reap_saliency_by_domain/math.parquet`

- layer_0: top4=0.411 top8=0.615 top16=0.826
- layer_1: top4=0.346 top8=0.582 top16=0.853
- layer_10: top4=0.498 top8=0.717 top16=0.976
- layer_23: top4=0.593 top8=0.804 top16=0.933
