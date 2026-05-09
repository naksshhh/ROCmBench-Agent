# ROCmBench Agent Pitch

## One-Liner

ROCmBench Agent turns AMD Developer Cloud into an AI deployment optimizer for open-source LLMs.

## Problem

Developers can spin up powerful AMD GPUs, but they still have to guess the right serving configuration. Wrong choices waste credits, increase latency, and make AMD deployment feel harder than it should.

## Solution

ROCmBench Agent runs benchmark experiments, ranks configurations, verifies ROCm GPU evidence, and generates an AMD readiness report with a recommended vLLM launch command and the next benchmark plan.

## Demo Flow

1. Select or upload an AMD benchmark run.
2. Choose an optimization goal: latency, throughput, memory, or balanced.
3. Show the performance frontier chart.
4. Show the production gate matrix and GPU saturation score.
5. Show the agent-planned next experiments.
6. Open the generated deployment report and point to the recommended vLLM command.

## Why AMD

The project is built around AMD Developer Cloud, ROCm, MI300X, and vLLM. It does not simply run on AMD; it helps other teams deploy better on AMD.

## Real Benchmark Proof

We ran `Qwen/Qwen2.5-7B-Instruct` on AMD Instinct MI300X with ROCm and vLLM. ROCmBench Agent found `concurrency-16` as the best balanced config: 2,775.85 tokens/sec, 1.463s average latency, 1.471s p95 latency, and 0% error rate. ROCm SMI showed the benchmark was using the GPU: peak GPU utilization hit 100%, 76.1% of samples were at 90%+ GPU utilization, and vLLM reserved 90% VRAM.

Then the agent planned follow-up tests and found the real boundary: `concurrency-24` reached 3,392.42 tokens/sec with 1.667s p95 latency, while `concurrency-32` only improved throughput to 3,423.27 tokens/sec but raised p95 latency to 3.249s. It also validated an 8192-context serving run and compared `Qwen/Qwen2.5-3B-Instruct`, which reached 4,219.56 tokens/sec with 2.368s p95 latency.

## Why It Wins

- Clear business value: lower inference cost and faster deployment decisions.
- Strong hackathon fit: useful to anyone building AI apps on AMD GPUs.
- End-to-end demo: benchmark input, analysis, ROCm proof chart, deployment gates, recommendation, report.
- Agentic behavior: the app does not stop at "best row"; it proposes the next AMD GPU experiments to run.
- Expandable product: can become CI for AI deployment readiness.

## 2-Minute Script

Most AI teams can launch a model, but they cannot easily answer: which serving settings should I use on AMD GPUs?

ROCmBench Agent solves that. I upload a benchmark run from AMD Developer Cloud. The app scores each configuration across latency, throughput, memory use, and error rate. Then it recommends the best setup for my goal and generates a deployment report.

Here the balanced configuration wins. We can see the performance frontier, the AMD readiness score, the GPU saturation gate, and the exact vLLM launch command.

In our real AMD Developer Cloud run, the first recommended configuration reached 2,775.85 tokens/sec with 1.471s p95 latency and zero errors. Then the agent ran the next tests and discovered the deployment boundary: concurrency-24 gives 3,392.42 tokens/sec at 1.667s p95, while concurrency-32 barely improves throughput but pushes p95 to 3.249s. ROCm SMI showed 100% peak GPU utilization, so this is a real GPU-backed benchmark, not a mock.

The important point is that this is not another chatbot or charting app. It is an agentic workflow that runs experiments, reasons over the results, checks production gates, and tells me what AMD GPU experiment to run next.

Our next step is to connect this to CI so every model change gets an AMD readiness check before production.
