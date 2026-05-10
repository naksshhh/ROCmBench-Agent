# ROCmBench Agent Video Script

Target length: 90 seconds.

## 0:00-0:10

Hi, I’m showing ROCmBench Agent, an AMD GPU deployment copilot for open-source LLMs.

I started with a simple question: if I pick a Hugging Face model, what config would I actually trust in production?

## 0:10-0:25

The app can auto-load a selected Hugging Face model on an AMD Developer Cloud droplet using vLLM and ROCm.

It verifies the exact served model through `/v1/models`, then runs concurrency benchmarks instead of guessing a deployment setting.

## 0:25-0:45

ROCmBench scores each config across throughput, p95 latency, error rate, cost per 1M tokens, VRAM pressure, and GPU saturation.

It also parses ROCm SMI logs, so the report includes real AMD GPU evidence, not just API-level numbers.

## 0:45-1:10

The most interesting result was Qwen2.5-7B on MI300X.

Concurrency 24 delivered 3,392 tokens per second at 1.667 seconds p95 latency.

Concurrency 32 delivered 3,423 tokens per second, but p95 latency jumped to 3.249 seconds.

So the fastest-looking config was not the best production config. ROCmBench picked concurrency 24.

## 1:10-1:30

The final output is a deploy, warn, or block decision, a recommended vLLM command, a production gate matrix, and a downloadable report.

The goal is to turn AMD GPU benchmarking into a repeatable deployment workflow, not a one-off notebook.
