# Real AMD Benchmark Summary

## Environment

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Hardware: `AMD Instinct MI300X`
- Backend: `ROCm + vLLM`
- Workload: `Customer-support chat prompts`
- Captured at: `2026-05-09T08:27:31.469700+00:00`

## Results

| Concurrency | Tokens/sec | Requests/min | Avg latency | P95 latency | VRAM | Error rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 211.92 | 50.33 | 1.192s | 1.188s | 172.8 GB | 0% |
| 4 | 645.29 | 151.31 | 1.585s | 1.621s | 172.8 GB | 0% |
| 8 | 959.15 | 226.82 | 2.101s | 4.191s | 172.8 GB | 0% |
| 16 | 2775.85 | 652.66 | 1.463s | 1.471s | 172.8 GB | 0% |

## ROCm Monitor Evidence

- ROCm SMI samples parsed: `142`
- Peak GPU utilization: `100%`
- Peak allocated VRAM: `90%`
- Peak GPU memory read/write activity: `59%`

## ROCmBench Recommendation

For the balanced optimization goal, ROCmBench Agent recommends `concurrency-16`.

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 4096 --max-num-seqs 16 --gpu-memory-utilization 0.90
```

## Demo Talking Point

The result shows why ROCmBench Agent is useful: the best measured configuration was not merely the lowest-concurrency safe default. The benchmark found a high-concurrency setting that delivered 2,775.85 tokens/sec with 1.471s p95 latency and zero observed errors.
