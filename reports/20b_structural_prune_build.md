# 20B structural prune build

- Base model: `openai/gpt-oss-20b`

## Variants

- general_50pct_experts: `/root/model/artifacts/20b_pruned_models/general_50pct_experts`
- math_25pct_experts: `/root/model/artifacts/20b_pruned_models/math_25pct_experts`

## Sanity inference

- general_50pct_experts ok=True preview:
```
Explain MoE routing briefly.**  
   **Answer:** MoE routing is a method used in the Internet of Things (IoT) to manage data flow efficiently. It involves routing data from multiple sources to a single destination, ensuring that the data is processed in a way that optimizes network performance and reduces latency.

2. **What is the
```

- math_25pct_experts ok=True preview:
```
Explain MoE routing briefly. 
- Discuss the importance of the routing in the context of the problem. 
- Provide a solution that includes the steps and the code. 
- Provide a solution that includes the steps and the code. 
- Provide a solution that includes the steps and the code. 
- Provide a solution that includes the steps and
```

## Reproduce

```bash
modal run modal/gpt_oss_pruning_track.py --task build_pruned_20b
```
