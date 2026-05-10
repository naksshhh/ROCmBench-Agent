---
title: ROCmBench Agent
emoji: ⚡
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# ROCmBench Agent

ROCmBench Agent is an AI deployment copilot for profiling open-source LLM workloads on AMD Instinct GPUs. It ranks serving configurations, explains the tradeoffs, and exports an AMD deployment readiness report.

## Hackathon Thesis

Most projects in the AMD Developer Hackathon consume AMD compute. ROCmBench Agent helps developers use that compute better. It turns AMD Developer Cloud, ROCm, vLLM, and Qwen into a practical optimization workflow for LLM deployment.

## Features

- Use the included real AMD Developer Cloud benchmark run or upload a new benchmark JSON.
- Attach a ROCm monitor log and surface proof: peak GPU use, VRAM allocation, memory activity, and sample count.
- Chart ROCm SMI telemetry over time so judges can see GPU utilization, VRAM reservation, and memory activity during the benchmark.
- Rank configurations by latency, throughput, memory, or balanced readiness.
- Visualize the performance frontier.
- Explain the recommendation with a score breakdown across throughput, latency, reliability, and memory.
- Score GPU saturation and turn raw ROCm evidence into a production gate.
- Produce a deployment gate matrix for reliability, latency, ROCm proof, sustained saturation, VRAM headroom, and AMD readiness.
- Plan the next benchmark experiments automatically, including concurrency probes, long-context VRAM tests, soak tests, and second-model comparisons.
- Show before/after impact versus the lowest-concurrency baseline.
- Estimate serving cost from GPU hourly price, traffic, and output-token assumptions.
- Generate a deployment report with the recommended vLLM launch command, ROCm proof, impact, and cost estimate.
- Switch to a long-context tradeoff sample to show memory and latency decisions.
- Run a live OpenAI-compatible vLLM endpoint benchmark directly from the UI.
- Search Hugging Face models by task/category and select a model from autocomplete suggestions.
- Auto-load the selected Hugging Face model on the AMD droplet with vLLM before live benchmarking.
- Store benchmark history in SQLite and compare current runs against previous baselines.
- Detect deployment regressions with APPROVE / WARN / BLOCK decisions.
- Compare available model/scenario runs in one decision matrix.
- Generate production deployment artifacts: `vllm_start.sh`, `docker-compose.yml`, and CI workflow.

## Real AMD Run

The current repo includes a measured AMD Developer Cloud run for `Qwen/Qwen2.5-7B-Instruct` served with vLLM on an AMD Instinct MI300X.

| Concurrency | Tokens/sec | Avg latency | P95 latency | Error rate |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 211.92 | 1.192s | 1.188s | 0% |
| 4 | 645.29 | 1.585s | 1.621s | 0% |
| 8 | 959.15 | 2.101s | 4.191s | 0% |
| 16 | 2775.85 | 1.463s | 1.471s | 0% |

ROCm monitor evidence: peak GPU utilization hit 100%, vLLM reserved 90% VRAM, and peak memory read/write activity reached 59%.
The saturation gate scores the run at 87.0/100 because 76.1% of ROCm samples were at 90%+ GPU utilization.

ROCmBench recommends:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 4096 --max-num-seqs 16 --gpu-memory-utilization 0.90
```

Compared with the concurrency-1 baseline, the recommended configuration delivers about 13.1x higher throughput while keeping the observed error rate at 0%.

## Follow-Up GPU Validation

After the first benchmark, ROCmBench Agent planned and ran additional AMD GPU tests:

- Concurrency ceiling: `concurrency-24` reached 3,392.42 tokens/sec with 1.667s p95 latency and 0% errors. `concurrency-32` was only slightly faster at 3,423.27 tokens/sec, but p95 latency rose to 3.249s, so the agent rejects it for balanced serving.
- 8192 context boundary: `concurrency-16` reached 2,684.62 tokens/sec with 1.523s p95 latency and 0% errors.
- Soak stability: `concurrency-16` held 2,689.75 tokens/sec with 1.557s p95 latency and 0% errors.
- Second-model comparison: `Qwen/Qwen2.5-3B-Instruct` reached 4,219.56 tokens/sec at concurrency-32, but p95 latency was 2.368s.

See `docs/gpu_validation_suite.md` for the full validation table and judge story.

## Local Demo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL and click **Analyze AMD Run**.

The app defaults to the real AMD benchmark. Use the scenario dropdown to switch between the measured run, a sample MI300X demo, and a long-context tradeoff sample.

For auto-loading live Hugging Face models on a remote AMD droplet, set:

```bash
export ROCMBENCH_SSH_TARGET="root@YOUR_DROPLET_IP"
export ROCMBENCH_SSH_KEY="~/.ssh/your_key"
export ROCMBENCH_CONTAINER="rocm"
```

## Production Ops

The **Production Ops** tab adds a deployment workflow around the benchmark:

- Run a live benchmark against an OpenAI-compatible vLLM endpoint.
- Choose model task/category, fetch Hugging Face model suggestions, and select the model to benchmark.
- Use auto-load to SSH into the AMD droplet, start the selected HF model with vLLM, open/reuse a local tunnel, and wait until `/v1/models` confirms the selected model.
- Save runs to local SQLite history.
- Check regressions against the previous matching model/workload baseline.
- Refresh ROCm monitor proof from a log.
- Inspect the ROCm SMI proof chart and production gate matrix.
- Use the generated next-experiment plan to decide which AMD GPU run to spend credits on next.
- Generate a deployment bundle for the recommended config.

## AMD Developer Cloud Benchmark Flow

Start a vLLM server on an AMD GPU instance:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.90
```

Then run the benchmark client:

```bash
python scripts/run_vllm_benchmark.py \
  --endpoint http://localhost:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --workload "Customer-support chat prompts" \
  --concurrency 1 4 8 16 \
  --requests 32 \
  --output benchmark_results/amd_qwen25_7b.json
```

Upload the generated JSON in the app.

## Submission Assets

- Hugging Face Space: deploy this repo as a Gradio Space.
- GitHub repo: include the benchmark script, sample result, and generated report.
- Demo video: show upload, chart, recommendation, and report export.
- Build in Public: post screenshots of the AMD run and the final readiness report.
