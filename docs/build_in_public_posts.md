# Build In Public Posts

## Post 1

Building for the AMD Developer Hackathon with @AIatAMD and @lablab.

We are making ROCmBench Agent: an AI deployment copilot that benchmarks open-source LLM workloads on AMD Developer Cloud, ranks vLLM configs, and generates an AMD readiness report.

First milestone: local Gradio demo + sample MI300X benchmark flow is live.

## Post 2

ROCmBench Agent update for the AMD Developer Hackathon:

The app now takes benchmark JSON from AMD Developer Cloud, scores serving configs by latency / throughput / memory, visualizes the performance frontier, and exports a deployment report with a recommended vLLM command.

Real MI300X run with Qwen + ROCm + vLLM is in: 2,775.85 tok/s at concurrency 16, 1.471s p95 latency, 0% errors.

## Post 3

What we learned building on AMD Developer Cloud:

The hardest part is not only running the model. It is choosing the right deployment settings for a real workload. ROCmBench Agent turns those experiments into a repeatable workflow: benchmark, rank, explain, deploy.

Built with ROCm, vLLM, Qwen, and AMD Instinct GPUs. ROCm SMI proof from our run showed 100% peak GPU utilization and 90% VRAM allocation.

## Post 4

New ROCmBench Agent features:

- ROCm proof panel
- 13.1x before/after throughput lift
- readiness score breakdown
- cost-per-1M-token estimate
- long-context tradeoff scenario
- richer deployment report

This is now much closer to the actual product thesis: not a benchmark viewer, an AMD deployment copilot.
